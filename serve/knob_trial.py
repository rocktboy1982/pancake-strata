"""STRATA_KNOB_TRIAL=1 (opt-in, default off): the knob trial of MOD_PLAN.md section 6 item 36 (LEADS.md Part I row I10,
include/strata/program/knob_trial.hpp), run BETWEEN REQUESTS by the server.

The engine has no per-window live-knob path (generate.cpp applies the per-request keys at a request's start), so the
interleaving of the C++ rule is done per request: one answer is one 'window'.  A candidate setting (`strata_tune` keys:
spec_min_p, pcie_frac, aux_cpus, prefill_pipe; all exact: they change speed, not the tokens a given sampler state
gives) is timed against the incumbent settings in the order I0 C1 I1 C2 I2 C3 I3, each round's ratio = candidate rate /
the mean of the incumbent rates on both sides (a linear drift cancels), and the candidate is adopted for every later
request that sends no `strata_tune` of its own only when the MEDIAN ratio wins by the margin (10%).  The decision rule is
the C++ one (same constants, same early stops), pinned by serve/test_knob_trial.py with the same cases as
src/program/knob_trial_test.cpp.

HONEST LIMIT: windows of one engine are the same text under two settings; requests of one server are DIFFERENT prompts
and answers, so the rate also moves with the content.  The bracket cancels a drift, not a content difference: the
margin and the 3 rounds are the only guard.  A request that sends its own `strata_tune`, ends before MIN_TOKENS tokens,
is cancelled, or reports no speed is neither assigned a candidate nor counted.

    STRATA_KNOB_TRIAL=1
    STRATA_KNOB_TRIAL_CANDIDATES='[{"spec_min_p": 0.3}, {"spec_min_p": 0.6}]'   (default: this example list: values are NOT measured)
    STRATA_KNOB_TRIAL_MIN_TOKENS=32      (an answer shorter than this is not a window)

The log line is the C++ one: `strata knob trial: spec_min_p default -> 0.3: WIN ratio 1.123 (rounds ...); ...`.
"""
from __future__ import annotations

import json
import math
import threading
from typing import Callable, Optional

INCUMBENT, CANDIDATE = "incumbent", "candidate"
RUNNING, WIN, LOSE, UNDECIDED = "running", "win", "lose", "undecided"
TUNE_KEYS = ("spec_min_p", "pcie_frac", "aux_cpus", "prefill_pipe")
DEFAULT_CANDIDATES = [{"spec_min_p": 0.3}, {"spec_min_p": 0.6}]


class KnobTrial:
    """The decision rule (a port of include/strata/program/knob_trial.hpp)."""

    def __init__(self, window_tokens: int = 8, rounds: int = 3, margin: float = 0.10, settle_hit: float = 0.02):
        self.window_tokens, self.rounds, self.margin, self.settle_hit = window_tokens, rounds, margin, settle_hit
        self.settled = False
        self.last_hit: float = -1.0
        self.last_hit_known = False
        self.windows = 0
        self.settle_windows = 0
        self.incumbents: list[float] = []
        self.cand: list[float] = []
        self.bracket: list[float] = []
        self.ratios: list[float] = []
        self.verdict = RUNNING
        self.reason = ""

    def next_arm(self) -> str:
        if not self.settled:
            return INCUMBENT
        return CANDIDATE if len(self.bracket) % 2 == 1 else INCUMBENT

    def add(self, arm: str, tokens: int, seconds: float, hit: float = -1.0) -> None:
        if self.verdict != RUNNING:
            return
        self.windows += 1
        if tokens < self.window_tokens or not seconds > 0.0 or not math.isfinite(seconds):
            self.verdict, self.reason = UNDECIDED, "the answer ended inside a window"
            return
        rate = tokens / seconds
        if not self.settled:
            if arm != INCUMBENT:
                return
            self.settle_windows += 1
            agree = (hit < 0.0 or abs(hit - self.last_hit) <= self.settle_hit) if self.last_hit_known else True
            second = self.settle_windows >= 2
            self.last_hit = hit
            self.last_hit_known = hit >= 0.0
            self.incumbents.append(rate)
            if second and agree:
                self.settled = True
                self.bracket = [rate]
            return
        if arm == INCUMBENT:
            if len(self.bracket) % 2 == 1:
                return
            self.incumbents.append(rate)
            self.bracket.append(rate)
            self._close_round()
        else:
            if len(self.bracket) % 2 == 0:
                return
            mn, mx = min(self.incumbents), max(self.incumbents)
            if rate < (1.0 - self.margin) * mn and rate < mn - (mx - mn):
                self.cand.append(rate)
                self.verdict, self.reason = LOSE, "a candidate window ran apart from the incumbent band"
                return
            self.cand.append(rate)
            self.bracket.append(rate)

    def median_ratio(self) -> float:
        if not self.ratios:
            return 0.0
        r = sorted(self.ratios)
        n = len(r)
        return r[n // 2] if n % 2 else 0.5 * (r[n // 2 - 1] + r[n // 2])

    def _close_round(self) -> None:
        n = len(self.bracket)
        c = self.bracket[n - 2]
        base = 0.5 * (self.bracket[n - 3] + self.bracket[n - 1])
        self.ratios.append(c / base if base > 0.0 else 0.0)
        lost = sum(1 for r in self.ratios if r < 1.0)
        if 2 * lost > self.rounds:
            self.verdict, self.reason = LOSE, "more than half of the rounds lost"
            return
        if len(self.ratios) >= self.rounds:
            self.verdict = WIN if self.median_ratio() >= 1.0 + self.margin else LOSE
            if self.verdict == LOSE:
                self.reason = "median ratio below the margin"

    def line(self, knob: str, frm: str, to: str) -> str:
        v = {WIN: "WIN", LOSE: "LOSE", UNDECIDED: "UNDECIDED"}.get(self.verdict, "RUNNING")
        s = f"strata knob trial: {knob} {frm} -> {to}: {v}"
        if self.ratios:
            s += f" ratio {self.median_ratio():.3f} (rounds" + "".join(f" {r:.3f}" for r in self.ratios) + ")"
        s += f"; {self.windows} windows of {self.window_tokens} tokens, {self.settle_windows} to settle"
        if self.reason:
            s += "; " + self.reason
        return s


def parse_candidates(text: Optional[str]) -> list[dict]:
    """The candidate list: objects of TUNE_KEYS (numbers in range) only; anything else is dropped.  None / invalid: the
    default example list."""
    if text is None or not text.strip():
        return [dict(c) for c in DEFAULT_CANDIDATES]
    try:
        raw = json.loads(text)
    except ValueError:
        return [dict(c) for c in DEFAULT_CANDIDATES]
    out = []
    for c in raw if isinstance(raw, list) else []:
        if not isinstance(c, dict):
            continue
        d = {}
        for k, v in c.items():
            if k in TUNE_KEYS and isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0:
                if k in ("spec_min_p", "pcie_frac") and v > 1:
                    continue
                d[k] = v
        if d:
            out.append(d)
    return out


def _fmt(settings: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(settings.items())) if settings else "default"


class KnobTrialRunner:
    """Assigns the arm of each request and feeds the finished ones back; adopts a winner.  Thread safe."""

    def __init__(self, candidates: list[dict], min_tokens: int = 32, log: Callable[[str], None] = print):
        self.queue = [dict(c) for c in candidates]
        self.min_tokens = max(1, int(min_tokens))
        self.log = log
        self.adopted: dict = {}            # the incumbent's settings (the winners so far)
        self.trial: Optional[KnobTrial] = None
        self.current: Optional[dict] = None
        self.lock = threading.Lock()
        self.history: list[str] = []

    def active(self) -> bool:
        with self.lock:
            return bool(self.queue) or self.trial is not None

    def assign(self, sampling: Optional[dict]) -> tuple[Optional[dict], Optional[str]]:
        """(sampling for the engine, arm or None).  A request with its own strata_tune is left as it is (arm None); with
        none, the adopted settings and, in a trial's candidate window, the candidate's go in as `strata_tune`."""
        smp = sampling or {}
        if isinstance(smp.get("strata_tune"), dict):
            return sampling, None
        with self.lock:
            if self.trial is None and self.queue:
                self.current = self.queue.pop(0)
                self.trial = KnobTrial(window_tokens=self.min_tokens)
            arm = self.trial.next_arm() if self.trial is not None else None
            tune = dict(self.adopted)
            if arm == CANDIDATE and self.current is not None:
                tune.update(self.current)
        if not tune:
            return sampling, arm
        return {**smp, "strata_tune": tune}, arm

    def feed(self, arm: Optional[str], tokens: int, tok_per_s: Optional[float], hit: Optional[float] = None) -> Optional[str]:
        """A finished request that was assigned `arm`.  Returns the log line when this request ended the trial."""
        if arm is None or tok_per_s is None or not tok_per_s > 0 or tokens < self.min_tokens:
            return None
        with self.lock:
            t = self.trial
            if t is None or t.verdict != RUNNING:
                return None
            t.add(arm, tokens, tokens / tok_per_s, -1.0 if hit is None else float(hit))
            if t.verdict == RUNNING:
                return None
            cand = self.current or {}
            knob = "+".join(sorted(cand)) or "?"
            line = t.line(knob, _fmt(self.adopted), _fmt({**self.adopted, **cand}))
            if t.verdict == WIN:
                self.adopted.update(cand)
            self.history.append(line)
            self.trial, self.current = None, None
        self.log(line)
        return line


def from_env(env: dict, log: Callable[[str], None] = print) -> Optional[KnobTrialRunner]:
    """None unless STRATA_KNOB_TRIAL=1: the server's behaviour is then exactly the old one."""
    if (env.get("STRATA_KNOB_TRIAL") or "") != "1":
        return None
    try:
        mt = int(env.get("STRATA_KNOB_TRIAL_MIN_TOKENS") or 32)
    except ValueError:
        mt = 32
    cands = parse_candidates(env.get("STRATA_KNOB_TRIAL_CANDIDATES"))
    return KnobTrialRunner(cands, mt, log) if cands else None
