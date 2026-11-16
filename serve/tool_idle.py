"""STRATA_TOOL_IDLE=1 (MORI idleness prediction) and STRATA_TOOL_IDLE_PREWARM=1 (TokenCake predictive upload);
opt-in, unset/empty/0 = nothing is recorded, scheduled or run.

LEADS.md "MORI: idleness from tool durations, cross-tier KV migration" and "TokenCake: offload idle KV during
function calls, predictive upload".  Both papers treat the time a client spends running a tool as a known idle window
of that conversation's KV: MORI predicts the window from the tool's past durations and migrates the KV across tiers
by it; TokenCake offloads the idle KV during the call and uploads it back predictively, just before the call is
expected to return.

On this engine the offload half already exists: when another conversation (a title request, a sub-agent) runs while
the agent's tool runs, the engine parks the agent's KV in host RAM (the conversation cache) and restores it when the
agent's next request arrives, on that request's clock.  What was missing is the prediction and the upload ahead of
time.  This module adds both:

  * Predictor (MORI): per tool name, an exponential moving average of the observed idle time (the end of a reply
    that made tool calls -> the arrival of the request that carries their results, matched by tool call id).
    STRATA_TOOL_IDLE=1 records and reports it (`report()`, the "[strata] tool idle" log line); it changes nothing.
  * Prewarm (TokenCake): STRATA_TOOL_IDLE_PREWARM=1.  When another request ran during the tool window (so the
    agent's KV was parked), a timer fires STRATA_TOOL_IDLE_LEAD_S (default 1.0) before the predicted return and runs
    the agent's last prompt again for one token in the background, which restores its KV into the engine before the
    real request arrives.  Any request that arrives cancels a prewarm that is still running.  Exact for the tokens:
    only the engine's cache state moves.  Needs the engine idle (the service FIFO free); otherwise nothing runs.

Knobs (read once): STRATA_TOOL_IDLE, STRATA_TOOL_IDLE_PREWARM, STRATA_TOOL_IDLE_ALPHA (EMA weight of a new sample,
default 0.3), STRATA_TOOL_IDLE_DEFAULT_S (prediction for a tool never seen, default 5.0), STRATA_TOOL_IDLE_LEAD_S,
STRATA_TOOL_IDLE_MIN_S (no prewarm for a predicted window shorter than this, default 2.0).
"""
from __future__ import annotations

import os
import threading
import time

_CFG: dict | None = None
_LOCAL = threading.local()                     # .background: a run this module started (not a client turn)


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def config() -> dict:
    global _CFG
    if _CFG is None:
        prewarm = os.environ.get("STRATA_TOOL_IDLE_PREWARM", "").strip() == "1"
        # STRATA_ASYNC_COMPACT (serve/async_compact.py) runs on this module's turn-end and request hooks
        compact = (os.environ.get("STRATA_ASYNC_COMPACT", "").strip() or "0") not in ("0", "")
        _CFG = {"on": prewarm or compact or os.environ.get("STRATA_TOOL_IDLE", "").strip() == "1",
                "prewarm": prewarm,
                "alpha": min(1.0, max(0.01, _f("STRATA_TOOL_IDLE_ALPHA", 0.3))),
                "default_s": max(0.0, _f("STRATA_TOOL_IDLE_DEFAULT_S", 5.0)),
                "lead_s": max(0.0, _f("STRATA_TOOL_IDLE_LEAD_S", 1.0)),
                "min_s": max(0.0, _f("STRATA_TOOL_IDLE_MIN_S", 2.0))}
    return _CFG


def enabled() -> bool:
    return config()["on"]


def background() -> bool:
    """True inside a run this module (or a module built on its hooks) started."""
    return bool(getattr(_LOCAL, "background", False))


class Predictor:
    """MORI-style idleness from tool durations: EMA of the observed window per tool name."""

    def __init__(self, alpha: float = 0.3, default_s: float = 5.0):
        self.alpha, self.default_s = alpha, default_s
        self.ema: dict[str, float] = {}
        self.samples: dict[str, int] = {}

    def observe(self, names, seconds: float) -> None:
        for n in set(names) or {""}:
            old = self.ema.get(n)
            self.ema[n] = seconds if old is None else old + self.alpha * (seconds - old)
            self.samples[n] = self.samples.get(n, 0) + 1

    def predict(self, names) -> float:
        """The window of a turn that called `names`: the longest of their predictions (they run, the slowest
        decides)."""
        return max((self.ema.get(n, self.default_s) for n in (set(names) or {""})), default=self.default_s)


def tool_result_ids(req) -> set:
    """The tool call ids a request answers: OpenAI role "tool" tool_call_id, Anthropic tool_result tool_use_id,
    Responses function_call_output call_id."""
    out: set = set()
    if not isinstance(req, dict):
        return out
    for m in req.get("messages") or req.get("input") or []:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "tool" and m.get("tool_call_id"):
            out.add(str(m["tool_call_id"]))
        if m.get("type") == "function_call_output" and m.get("call_id"):
            out.add(str(m["call_id"]))
        c = m.get("content")
        if isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and part.get("type") == "tool_result" and part.get("tool_use_id"):
                    out.add(str(part["tool_use_id"]))
    return out


class Tracker:
    """The last reply that made tool calls, and what happened while its tools ran."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or config()
        self.pred = Predictor(self.cfg["alpha"], self.cfg["default_s"])
        self.lock = threading.Lock()
        self.pending: dict | None = None        # {"ids": set, "names": [...], "t": end time, "prompt": [...], ...}
        self.timer: threading.Timer | None = None
        self.job_cancel: threading.Event | None = None
        self.stats = {"turns": 0, "observed": 0, "interleaved": 0, "prewarms": 0, "prewarm_cancelled": 0,
                      "prewarm_skipped": 0, "last_predicted_s": None, "last_observed_s": None}
        self.turn_end_hooks: list = []          # f(svc, info) for modules that use the idle window (async compaction)
        self.request_hooks: list = []           # f(req) when a request arrives (they cancel their background work)

    # -- events -----------------------------------------------------------------------------------------------------
    def on_turn_end(self, svc, prompt_ids, calls, info: dict | None = None) -> None:
        """A client turn ended.  `calls`: [(id, name)] of the tool calls it made (empty: a plain answer)."""
        info = dict(info or {})
        info.update(prompt=list(prompt_ids or []), calls=list(calls))
        for hook in list(self.turn_end_hooks):
            try:
                hook(svc, info)
            except Exception as e:              # noqa: BLE001 - an optional helper must never fail a request
                print(f"[strata] tool idle: hook failed ({type(e).__name__}: {e})", flush=True)
        if not calls:
            return
        names = [n for _, n in calls]
        predicted = self.pred.predict(names)
        with self.lock:
            self._cancel_locked()
            self.pending = {"ids": {str(i) for i, _ in calls}, "names": names, "t": time.monotonic(),
                            "prompt": list(prompt_ids or []), "interleaved": False, "predicted": predicted}
            self.stats["turns"] += 1
            self.stats["last_predicted_s"] = round(predicted, 3)
            if self.cfg["prewarm"] and predicted >= self.cfg["min_s"] and self.pending["prompt"]:
                delay = max(0.0, predicted - self.cfg["lead_s"])
                self.timer = threading.Timer(delay, self._fire, args=(svc, self.pending))
                self.timer.daemon = True
                self.timer.start()

    def on_request(self, req) -> None:
        """A request arrived (before it runs): close the window it answers, or mark the window interleaved."""
        answered = tool_result_ids(req)
        for hook in list(self.request_hooks):
            try:
                hook(req)
            except Exception as e:              # noqa: BLE001
                print(f"[strata] tool idle: request hook failed ({type(e).__name__}: {e})", flush=True)
        with self.lock:
            p = self.pending
            if p is not None and answered & p["ids"]:
                seconds = time.monotonic() - p["t"]
                self.pred.observe(p["names"], seconds)
                self.stats["observed"] += 1
                self.stats["last_observed_s"] = round(seconds, 3)
                print(f"[strata] tool idle: {','.join(sorted(set(p['names'])))} {seconds:.1f} s "
                      f"(predicted {p['predicted']:.1f} s{', interleaved' if p['interleaved'] else ''})", flush=True)
                self.pending = None
            elif p is not None and not p["interleaved"]:
                p["interleaved"] = True          # another conversation runs inside the tool window: KV parked
                self.stats["interleaved"] += 1
            self._cancel_locked()

    # -- the prewarm (TokenCake predictive upload) ------------------------------------------------------------------
    def _cancel_locked(self) -> None:
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None
        if self.job_cancel is not None and not self.job_cancel.is_set():
            self.job_cancel.set()
            self.stats["prewarm_cancelled"] += 1
        self.job_cancel = None

    def _fire(self, svc, pending: dict) -> None:
        with self.lock:
            if self.pending is not pending or not pending["interleaved"]:
                return                           # answered already, or the KV never left: nothing to upload
            fifo = getattr(svc, "fifo", None)
            if fifo is None or fifo.locked():
                self.stats["prewarm_skipped"] += 1
                return
            cancel = self.job_cancel = threading.Event()
        threading.Thread(target=self._prewarm, args=(svc, pending["prompt"], cancel), daemon=True).start()

    def _prewarm(self, svc, prompt: list, cancel: threading.Event) -> None:
        _LOCAL.background = True
        try:
            for _ in svc.run(prompt, False, None, 1, {"temperature": 0.0}, cancel):
                if cancel.is_set():
                    break
            if not cancel.is_set():
                with self.lock:
                    self.stats["prewarms"] += 1
        except Exception as e:                  # noqa: BLE001 - background work must never take the server down
            print(f"[strata] tool idle: prewarm failed ({type(e).__name__}: {e})", flush=True)
        finally:
            _LOCAL.background = False

    def report(self) -> dict:
        with self.lock:
            return {**self.stats, "predicted_s": {k: round(v, 3) for k, v in self.pred.ema.items()},
                    "samples": dict(self.pred.samples)}

    # -- the Service.run wrapper ------------------------------------------------------------------------------------
    def wrap(self, svc, prompt_ids, events, info: dict | None = None):
        """Service.run's events, unchanged; at the turn's end on_turn_end() is told which tool calls it made."""
        calls, content = [], []
        for kind, obj in events:
            if kind == "event" and getattr(obj, "kind", None) == "tool_call" and getattr(obj, "call", None):
                calls.append((obj.call.id, obj.call.name))
            elif kind == "event" and getattr(obj, "kind", None) == "content":
                content.append(obj.text)
            elif kind == "done" and isinstance(obj, dict) and obj.get("finish") in ("stop", "tool_calls"):
                self.on_turn_end(svc, prompt_ids, calls, {**(info or {}), "content": "".join(content)})
            yield kind, obj


TRACKER: Tracker | None = None


def tracker() -> Tracker | None:
    """The process tracker when enabled, else None."""
    global TRACKER
    if not enabled():
        return None
    if TRACKER is None:
        TRACKER = Tracker()
    return TRACKER


def on_request(req) -> None:
    t = tracker()
    if t is not None:
        t.on_request(req)
