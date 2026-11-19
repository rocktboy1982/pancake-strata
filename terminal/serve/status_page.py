"""The project status page: GET /status for a browser (Accept: text/html, or ?html=1), GET /pancake (always the page) and
GET /status.json (the same data as JSON).  Ported from pancake-strata (port/rdna3-decode 214c8cd31, 19ffe37ab) and
extended for this branch: besides the engine's counters it shows the project's state.

What the page shows, all read-only (GET only; nothing here starts, stops, measures or writes anything):

* the project: branch and head commit of this checkout (read from .git, no git process), the engine's version and build
  (its BUILD.json), backend, model, context, run config, uptime;
* the detected hardware: gfx architecture, ROCm version, VRAM, PCIe link, CPU, RAM, read by the tuned-flags code's own
  read-only detection (terminal/strata_control/tuned_flags.py: rocminfo, /opt/rocm*/.info/version, sysfs, /proc; no
  privileges), once, in the background;
* the tuned flags: the profile of terminal/strata_control/tuned_flags.py that matches this machine (the user file
  $STRATA_TUNED_FLAGS_FILE or ~/.strata/tuned-flags.json first, then the shipped terminal/data/tuned-flags.json), and for
  each flag whether it was applied at start, overridden by the user (config or environment) or changed by a saved tuning;
* the user's own settings (the run config's env and args, the STRATA_ / HIP_ / HSA_ variables of the environment);
* the flag diagnostics' verdicts when a verdicts file is present ($STRATA_FLAG_DIAG_VERDICTS, or
  ~/.strata/flag-diag/verdicts.json: what tools/pancake/flag_diag.sh run writes into its OUT_DIR);
* the live counters: decode / prompt tok/s (now, run average, last request), prefix reuse, expert cache hit rate, PCIe
  share, drafts, the conversation cache from the engine's log, and every INFO fact the engine reported (expert slots,
  windows, ...), and the hardware's live readings.

Numbers and settings only: no prompt, answer or tool text.  Every value that is not known prints as an em dash (U+2014).
One formatter (`view`) makes the table cells for both the server-rendered page (works with JavaScript off) and the
page's script (serve/web/status.js), which only redraws what /status.json returns.
"""
from __future__ import annotations

import html
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any

DASH = "—"
PROJECT = "updated-strata"
SERVE = Path(__file__).resolve().parent
TERMINAL = SERVE.parent
REPO = TERMINAL.parent
TEMPLATE = SERVE / "web" / "status.html"
MARK = "<!--status-sections-->"
# the process environment as the user started the server, before anything of the server's own start is added
START_ENV = dict(os.environ)
USER_ENV_PREFIXES = ("STRATA_", "HIP_", "HSA_", "ROCR_", "ROCBLAS_", "HIPBLASLT_", "MIOPEN_")


def verdict_files(env: dict | None = None) -> list[Path]:
    env = START_ENV if env is None else env
    own = env.get("STRATA_FLAG_DIAG_VERDICTS")
    return [Path(own).expanduser()] if own else [Path.home() / ".strata" / "flag-diag" / "verdicts.json"]


# --------------------------------------------------------------------------- formatting

def missing(v) -> bool:
    return v is None or v == "" or (isinstance(v, float) and v != v)


def fmt(v, digits: int = 1, unit: str = "") -> str:
    """A cell: the em dash when the value is not known, else the number (thousands separated) or the text."""
    if missing(v):
        return DASH
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        s = f"{v:,}"
    elif isinstance(v, float):
        s = f"{v:,.{digits}f}"
    elif isinstance(v, (list, tuple)):
        s = ", ".join(str(x) for x in v) or DASH
    elif isinstance(v, dict):
        s = json.dumps(v, sort_keys=True)
    else:
        s = str(v)
    return f"{s} {unit}" if unit and s != DASH else s


def pct(v, digits: int = 1) -> str:
    return DASH if missing(v) else f"{v:.{digits}f} %"


def duration(s) -> str:
    if missing(s):
        return DASH
    s = int(s)
    if s < 90:
        return f"{s} s"
    m = s // 60
    return f"{m} min {s % 60} s" if m < 90 else f"{m // 60} h {m % 60} min"


# --------------------------------------------------------------------------- the checkout and the engine

def git_head(start: Path = REPO) -> dict:
    """{"branch", "head"} of the checkout that holds `start`, read from the .git files (a worktree's .git file, packed
    refs), with no git process.  {} when there is no checkout (an installed copy)."""
    start = Path(start)
    for d in (start, *start.parents):
        g = d / ".git"
        if g.exists():
            break
    else:
        return {}
    try:
        if g.is_file():
            text = g.read_text(encoding="utf-8").strip()
            if not text.startswith("gitdir:"):
                return {}
            gitdir = (d / text[len("gitdir:"):].strip()).resolve()
        else:
            gitdir = g
        common = gitdir
        if (gitdir / "commondir").is_file():
            common = (gitdir / (gitdir / "commondir").read_text(encoding="utf-8").strip()).resolve()
        head = (gitdir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return {"branch": None, "head": head[:12] or None}
        ref = head[4:].strip()
        sha = None
        for base in (gitdir, common):
            if (base / ref).is_file():
                sha = (base / ref).read_text(encoding="utf-8").strip()
                break
        if sha is None and (common / "packed-refs").is_file():
            for line in (common / "packed-refs").read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    sha = parts[0]
                    break
        return {"branch": ref[len("refs/heads/"):] if ref.startswith("refs/heads/") else ref, "head": sha[:12] if sha else None}
    except OSError:
        return {}


def spawn_parts(engine) -> tuple[str | None, list | None, dict | None]:
    """(binary path, arguments, environment) the engine was started with; None for what this engine does not keep (the
    mock engine of the tests, a CPU reader)."""
    spawn = getattr(engine, "spawn", None)
    if not isinstance(spawn, tuple) or len(spawn) < 5:
        return None, None, None
    args = [str(a) for a in spawn[1]] if isinstance(spawn[1], (list, tuple)) else None
    env = spawn[4] if isinstance(spawn[4], dict) else None
    return (str(spawn[0]) if spawn[0] else None), args, env


def engine_build(exe: str | None) -> dict:
    """The engine's BUILD.json (next to the binary): version, build id or source, backend, archs, ROCm."""
    if not exe:
        return {}
    try:
        data = json.loads((Path(exe).parent / "BUILD.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def user_config(svc) -> dict:
    """The run config as the user wrote it (the file; the tuned flags and saved tunings are applied in memory only)."""
    path = getattr(svc, "config_path", None)
    if not path:
        return {}
    try:
        text = Path(path).read_bytes().decode("utf-8-sig")
    except OSError:
        return {}
    try:
        from serve.server_parts.http_api import parse_config_text
        cfg = parse_config_text(text, path)
    except (Exception, SystemExit):  # noqa: BLE001 - a page never fails on the config
        try:
            cfg = json.loads(text)
        except ValueError:
            return {}
    return cfg if isinstance(cfg, dict) else {}


def arg_value(args: list, name: str):
    """The value of `name` in an argument list ("--x v" or "--x=v"; "" for a switch with no value), None when absent."""
    for i, a in enumerate(args):
        a = str(a)
        if a == name:
            nxt = str(args[i + 1]) if i + 1 < len(args) else ""
            return "" if nxt.startswith("--") else nxt
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


# --------------------------------------------------------------------------- hardware (read once, in the background)

_HW: dict[str, Any] = {"state": "unread", "hw": None}
_HW_LOCK = threading.Lock()


def _read_hardware() -> None:
    try:
        from strata_control import tuned_flags
        hw = tuned_flags.detect_hardware()
    except Exception:  # noqa: BLE001
        hw = None
    with _HW_LOCK:
        _HW.update(state="read", hw=hw)


def hardware(svc) -> tuple[str, dict | None]:
    """("read" | "reading", the tuned-flags detection's dict or None).  A test (or a caller that already has it) sets
    svc.status_hardware."""
    given = getattr(svc, "status_hardware", None)
    if given is not None:
        return "read", (given or None)
    with _HW_LOCK:
        if _HW["state"] == "unread":
            _HW["state"] = "reading"
            threading.Thread(target=_read_hardware, name="status-hw", daemon=True).start()
        return _HW["state"], _HW["hw"]


# --------------------------------------------------------------------------- tuned flags

def tuned_flags_report(svc, hw: dict | None, hw_state: str = "read", env: dict | None = None,
                       argv: list | None = None) -> dict:
    """The profile of terminal/strata_control/tuned_flags.py for this machine and each flag's state at start."""
    from strata_control import tuned_flags as tf
    env = START_ENV if env is None else env
    argv = sys.argv if argv is None else argv
    rep: dict[str, Any] = {"enabled": not tf.disabled(env, argv), "file": None, "source": None, "match": None,
                           "flags": [], "state": "", "rule": "a measured GAIN only: exact >= 2 % end to end, or lossy "
                           "with a passing quality gate >= 3 %; your own setting always wins"}
    if not rep["enabled"]:
        rep["state"] = "off (STRATA_TUNED_FLAGS=0 or --no-tuned-flags): plain upstream defaults"
        return rep
    if hw is None:
        rep["state"] = "reading the hardware" if hw_state == "reading" else \
            "no supported graphics card read here: plain defaults"
        return rep
    prof, path = tf.select(hw, tf.candidate_files(env))
    if prof is None:
        rep["state"] = (f"no measured profile for {hw.get('gfx')} / ROCm {hw.get('rocm') or '?'} / "
                        f"{hw.get('vram_gib')} GiB: plain defaults")
        return rep
    rep.update(file=str(path), source=prof.get("source"), match=prof.get("match"))
    cfg = user_config(svc)
    u_env = {str(k): str(v) for k, v in (cfg.get("env") or {}).items()}
    u_args = [str(a) for a in cfg.get("args") or []]
    _, eff_args, eff_env = spawn_parts(getattr(svc, "engine", None))
    applied = 0
    for f in prof["flags"]:
        value = str(f.get("value", "")).replace("<ROOT>", str(tf.ROOT))
        name, kind = f["name"], f["kind"]
        if kind == "env":
            if name in u_env:
                status, now = "yours (run config)", u_env[name]
            elif name in env:
                status, now = "yours (environment)", env[name]
            elif eff_env is None:
                status, now = "engine not started yet", None
            elif name in eff_env:
                status, now = ("applied" if eff_env[name] == value else "changed by a saved tuning"), eff_env[name]
            else:
                status, now = "not applied", None
        else:
            mine = arg_value(u_args, name)
            eff = arg_value(eff_args, name) if eff_args is not None else None
            if mine is not None:
                status, now = "yours (run config)", mine
            elif eff_args is None:
                status, now = "engine not started yet", None
            elif eff is not None:
                status, now = ("applied" if eff == value else "changed by a saved tuning"), eff
            else:
                status, now = "not applied", None
        applied += status == "applied"
        rep["flags"].append({"kind": kind, "name": name, "value": value, "status": status, "in_effect": now,
                             "exact": f.get("exact"), "gain_pct": f.get("gain_pct"),
                             "quality_gate": f.get("quality_gate"), "ledger": f.get("ledger")})
    rep["state"] = f"{applied} of {len(rep['flags'])} applied at start"
    return rep


def user_settings(svc, env: dict | None = None) -> list[dict]:
    """The user's own settings: the run config's env and args, and the engine-related variables of the environment."""
    env = START_ENV if env is None else env
    cfg = user_config(svc)
    out = [{"name": str(k), "value": str(v), "from": "run config env"} for k, v in sorted((cfg.get("env") or {}).items())]
    args = [str(a) for a in cfg.get("args") or []]
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--") and "=" not in a and i + 1 < len(args) and not args[i + 1].startswith("--"):
            out.append({"name": a, "value": args[i + 1], "from": "run config args"})
            i += 2
            continue
        name, _, value = a.partition("=")
        out.append({"name": name, "value": value, "from": "run config args"})
        i += 1
    out += [{"name": k, "value": v, "from": "environment"} for k, v in sorted(env.items()) if k.startswith(USER_ENV_PREFIXES)]
    return out


def flag_verdicts(env: dict | None = None) -> dict | None:
    """The newest flag-diagnostics verdicts file (tools/pancake/flag_diag.py, schema strata.flag-diag/1), or None."""
    for path in verdict_files(env):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or not isinstance(data.get("verdicts"), list):
            continue
        rows = [{"name": v.get("name"), "kind": v.get("kind"), "verdict": v.get("verdict"), "gain_pct": v.get("gain_pct"),
                 "noise_pct": v.get("noise_pct"), "reason": v.get("reason"), "ledger": v.get("ledger")}
                for v in data["verdicts"] if isinstance(v, dict)]
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = None
        return {"file": str(path), "complete": data.get("complete"), "minutes": data.get("minutes"),
                "hardware": data.get("hardware"), "written": mtime, "rows": rows}
    return None


# --------------------------------------------------------------------------- live counters

def live_counters(svc) -> dict:
    with svc.status_lock:
        s = dict(svc.status)
        if getattr(svc, "live_reqs", None):
            s = {**s, **dict(list(svc.live_reqs.values())[-1][0]), "queued": s.get("queued", 0)}
        hist = list(svc.history)
        t = dict(svc.totals)
    busy = bool(s.get("busy"))
    loaded = svc.loaded() if hasattr(svc, "loaded") else True
    state = ("reading" if s.get("first_token") is None else "generating") if busy else ("idle" if loaded else "unloaded")
    last = hist[-1] if hist else {}
    read = t.get("prompt_tokens", 0) - t.get("reused", 0)
    hits = [h["hit_rate"] for h in hist if isinstance(h.get("hit_rate"), (int, float))]
    eng = getattr(svc, "engine", None)
    cc = {}
    conv = getattr(svc, "conv_log", None)
    if conv is not None:
        try:
            cc = conv.poll(getattr(eng, "log_path", None), getattr(eng, "log_start", None))
        except Exception:  # noqa: BLE001
            cc = {}
    lp_ms, lp_read = last.get("prompt_ms"), (last.get("prompt_tokens") or 0) - (last.get("reused") or 0)
    return {
        "state": state, "queued": s.get("queued", 0),
        "decode_tok_s_now": round(svc._tok_s(), 1) if busy and s.get("first_token") else None,
        "prefill_tok_s_now": getattr(eng, "prefill_tok_s_mean", None) if busy else None,
        "decode_tok_s_run": round(t["output_tokens"] / (t["decode_ms"] / 1000), 1)
        if t.get("decode_ms", 0) > 0 and t.get("output_tokens", 0) > 0 else None,
        "prefill_tok_s_run": round(read / (t["prompt_ms"] / 1000), 1) if t.get("prompt_ms", 0) > 0 and read > 0 else None,
        "decode_tok_s_last": last.get("decode_tok_s"),
        "prefill_tok_s_last": round(lp_read / (lp_ms / 1000), 1) if isinstance(lp_ms, (int, float)) and lp_ms > 0 and lp_read > 0 else None,
        "requests": t.get("requests"), "prompt_tokens": t.get("prompt_tokens"), "output_tokens": t.get("output_tokens"),
        "reused_tokens": t.get("reused"),
        "prefix_reuse_pct": round(100 * t["reused"] / t["prompt_tokens"], 1) if t.get("prompt_tokens") else None,
        "expert_hit_pct_last": round(100 * last["hit_rate"], 1) if isinstance(last.get("hit_rate"), (int, float)) else None,
        "expert_hit_pct_mean": round(100 * statistics.fmean(hits), 1) if hits else None,
        "pcie_share_pct_last": round(100 * last["pcie_share"], 1) if isinstance(last.get("pcie_share"), (int, float)) else None,
        "drafts_offered": t.get("drafts_offered"), "drafts_accepted": t.get("drafts_accepted"),
        "draft_accept_pct": round(100 * t["drafts_accepted"] / t["drafts_offered"], 1) if t.get("drafts_offered") else None,
        "conversation_cache": {k: cc.get(k) for k in ("parked", "parks", "restores", "evictions", "refused_total")},
    }


def hardware_now(svc) -> dict:
    tel = svc.telemetry.snapshot() if getattr(svc, "telemetry", None) else {"now": {}, "static": {}}
    now, static = dict(tel.get("now") or {}), dict(tel.get("static") or {})

    def mib(v):
        return v >> 20 if isinstance(v, int) else None
    return {"gpu_name": static.get("gpu_name"), "gpu_util_pct": now.get("gpu_util"), "gpu_temp_c": now.get("gpu_temp"),
            "gpu_power_w": now.get("gpu_power"), "vram_used_mib": mib(now.get("gpu_mem_used")),
            "vram_total_mib": mib(now.get("gpu_mem_total")) or static.get("gpu_vram_total_mib"),
            "ram_used_mib": mib(now.get("ram_used")), "ram_total_mib": mib(now.get("ram_total")), "cpu_pct": now.get("cpu")}


# --------------------------------------------------------------------------- the whole status

def build(svc) -> dict:
    """GET /status.json: the data and the view (the table cells the page shows)."""
    eng = getattr(svc, "engine", None)
    exe, _, _ = spawn_parts(eng)
    info = dict(getattr(eng, "info", {}) or {})
    build_json = engine_build(exe)
    git = git_head()
    project = {"name": PROJECT, "branch": git.get("branch"), "head": git.get("head"),
               "engine_version": info.get("version") or build_json.get("version"),
               "engine_build": build_json.get("build_id") or build_json.get("source"),
               "engine_binary": exe, "backend": (getattr(svc, "backend", None) or build_json.get("backend") or "").upper() or None,
               "model": getattr(svc, "model", None),
               "max_context": svc.reported_ctx() if hasattr(svc, "reported_ctx") else getattr(eng, "max_context", None),
               "config_path": getattr(svc, "config_path", None),
               "uptime_s": int(time.time() - svc.started_at), "started": svc.started_at}
    hw_state, hw = hardware(svc)
    facts = {k: v for k, v in sorted(info.items()) if k != "version" and isinstance(v, (str, int, float, bool))}
    data = {"project": project, "hardware": {"state": hw_state, "detected": hw},
            "tuned_flags": tuned_flags_report(svc, hw, hw_state), "settings": user_settings(svc),
            "verdicts": flag_verdicts(), "live": live_counters(svc), "engine_facts": facts,
            "hardware_now": hardware_now(svc), "time": time.time()}
    data["view"] = view(data)
    return data


def _kv(sid, title, note, rows):
    return {"id": sid, "title": title, "note": note, "kind": "kv", "columns": ["Item", "Value"], "rows": rows}


def _table(sid, title, note, columns, rows, empty):
    return {"id": sid, "title": title, "note": note, "kind": "table", "columns": columns, "rows": rows, "empty": empty}


def view(d: dict) -> list[dict]:
    """The sections of the page, every cell already a string (missing values the em dash)."""
    p, hw, tf, lv, now = d["project"], d["hardware"]["detected"] or {}, d["tuned_flags"], d["live"], d["hardware_now"]
    started = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p["started"])) if p.get("started") else None
    out = [_kv("project", "Project", "This checkout and the engine it runs", [
        ["Project", fmt(p["name"])], ["Branch", fmt(p["branch"])], ["Head", fmt(p["head"])],
        ["Engine version", fmt(p["engine_version"])], ["Engine build", fmt(p["engine_build"])],
        ["Engine binary", fmt(p["engine_binary"])], ["Backend", fmt(p["backend"])], ["Model", fmt(p["model"])],
        ["Context", fmt(p["max_context"], unit="tokens")], ["Run config", fmt(p["config_path"])],
        ["Server started", fmt(started)], ["Uptime", duration(p["uptime_s"])]])]
    hw_note = "being read (rocminfo, sysfs)" if d["hardware"]["state"] == "reading" else \
        "read-only detection of strata_control/tuned_flags.py" if hw else "no AMD card read here (rocminfo, sysfs)"
    out.append(_kv("hardware", "Hardware", hw_note, [
        ["GPU", fmt(now.get("gpu_name"))], ["Graphics architecture", fmt(hw.get("gfx"))], ["ROCm", fmt(hw.get("rocm"))],
        ["VRAM (largest card)", fmt(hw.get("vram_gib") or None, unit="GiB")], ["PCIe link", fmt(hw.get("pcie_link"))],
        ["CPU", fmt(hw.get("cpu"))], ["RAM", fmt(hw.get("ram_gib") or None, unit="GiB")]]))
    out.append(_kv("tuned-state", "Tuned flags", "terminal/strata_control/tuned_flags.py, applied by every start", [
        ["State", fmt(tf["state"])], ["Profile file", fmt(tf["file"])], ["Profile source", fmt(tf["source"])],
        ["Profile match", fmt(tf["match"])], ["Rule", fmt(tf["rule"])]]))
    out.append(_table("tuned-flags", "Tuned flags of this machine", "what the profile sets, and what is in effect",
                      ["Flag", "Kind", "Profile value", "State", "In effect", "Exact", "Gain", "Ledger"],
                      [[fmt(f["name"]), fmt(f["kind"]), fmt(f["value"]), fmt(f["status"]), fmt(f["in_effect"]),
                        fmt(f["exact"]), pct(f["gain_pct"]), fmt(f["ledger"])] for f in tf["flags"]],
                      "No tuned flag applies here"))
    out.append(_table("settings", "Your settings", "the run config and the environment; yours always win",
                      ["Name", "Value", "From"], [[fmt(s["name"]), fmt(s["value"]), fmt(s["from"])] for s in d["settings"]],
                      "None"))
    v = d["verdicts"]
    if v:
        written = time.strftime("%Y-%m-%d %H:%M", time.localtime(v["written"])) if v.get("written") else None
        note = f"{v['file']} ({'complete' if v.get('complete') else 'incomplete'}, {fmt(v.get('minutes'))} min, {fmt(written)})"
        rows = [[fmt(r["name"]), fmt(r["verdict"]), pct(r["gain_pct"]), pct(r["noise_pct"]), fmt(r["reason"])] for r in v["rows"]]
    else:
        note = "no verdicts file (" + ", ".join(str(x) for x in verdict_files()) + "; flag_diag.sh run writes one)"
        rows = []
    out.append(_table("verdicts", "Flag diagnostics", note, ["Flag", "Verdict", "Gain", "Noise floor", "Reason"], rows,
                      "No verdicts"))
    cc = lv["conversation_cache"]
    drafts = DASH if not lv["drafts_offered"] else \
        f"{fmt(lv['drafts_accepted'])} / {fmt(lv['drafts_offered'])} ({pct(lv['draft_accept_pct'])})"
    out.append(_kv("speed", "Speed", "since the server started; tokens per second", [
        ["State", fmt(lv["state"])], ["Queued", fmt(lv["queued"])],
        ["Decode now", fmt(lv["decode_tok_s_now"])], ["Decode, run average", fmt(lv["decode_tok_s_run"])],
        ["Decode, last request", fmt(lv["decode_tok_s_last"])], ["Prompt now", fmt(lv["prefill_tok_s_now"])],
        ["Prompt, run average", fmt(lv["prefill_tok_s_run"])], ["Prompt, last request", fmt(lv["prefill_tok_s_last"])],
        ["Requests finished", fmt(lv["requests"])], ["Prompt tokens", fmt(lv["prompt_tokens"])],
        ["Output tokens", fmt(lv["output_tokens"])]]))
    out.append(_kv("caches", "Caches", "hit rates from the engine's DONE lines and its log", [
        ["Prefix reuse", pct(lv["prefix_reuse_pct"]) + ("" if not lv["reused_tokens"] else f" ({fmt(lv['reused_tokens'])} tokens)")],
        ["Expert cache hit, last request", pct(lv["expert_hit_pct_last"])],
        ["Expert cache hit, mean", pct(lv["expert_hit_pct_mean"])],
        ["PCIe share, last request", pct(lv["pcie_share_pct_last"])], ["Drafts accepted", drafts],
        ["Conversations parked now", fmt(cc.get("parked"))], ["Conversation parks", fmt(cc.get("parks"))],
        ["Conversation restores", fmt(cc.get("restores"))], ["Conversation evictions", fmt(cc.get("evictions"))]]))
    out.append(_table("facts", "Engine facts", "what the engine reported at start (INFO lines: expert slots, windows, ...)",
                      ["Name", "Value"], [[fmt(k), fmt(x)] for k, x in d["engine_facts"].items()], "None reported"))
    out.append(_kv("hardware-now", "Hardware now", "live readings", [
        ["GPU load", pct(now.get("gpu_util_pct"), 0)], ["GPU temperature", fmt(now.get("gpu_temp_c"), 0, "°C")],
        ["GPU power", fmt(now.get("gpu_power_w"), 0, "W")], ["VRAM used", fmt(now.get("vram_used_mib"), unit="MiB")],
        ["VRAM total", fmt(now.get("vram_total_mib"), unit="MiB")], ["RAM used", fmt(now.get("ram_used_mib"), unit="MiB")],
        ["RAM total", fmt(now.get("ram_total_mib"), unit="MiB")], ["CPU", pct(now.get("cpu_pct"), 0)]]))
    return out


# --------------------------------------------------------------------------- HTML

def render_sections(sections: list[dict]) -> str:
    e = html.escape
    parts = []
    for s in sections:
        head = "".join(f'<th scope="col">{e(c)}</th>' for c in s["columns"])
        if s["kind"] == "kv":
            body = "".join(f'<tr><th scope="row">{e(r[0])}</th><td>{e(r[1])}</td></tr>' for r in s["rows"])
        elif s["rows"]:
            body = "".join("<tr>" + "".join(f"<td>{e(c)}</td>" for c in r) + "</tr>" for r in s["rows"])
        else:
            body = f'<tr><td colspan="{len(s["columns"])}" class="empty">{e(s.get("empty") or DASH)}</td></tr>'
        parts.append(f'<section class="panel" id="s-{e(s["id"])}"><div class="panel-bar"><h2>{e(s["title"])}</h2>'
                     f'<span class="small">{e(s.get("note") or "")}</span></div><div class="scroll"><table class="{e(s["kind"])}">'
                     f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div></section>")
    return "\n".join(parts)


def page(svc, allowed: bool) -> bytes:
    """The page with the status drawn in on the server, so it reads with JavaScript off.  A server with an API key and a
    request without it gets the page with no values (the script asks for the key)."""
    template = TEMPLATE.read_text(encoding="utf-8")
    if allowed:
        inner = render_sections(build(svc)["view"])
    else:
        inner = ('<p class="notice" id="locked">This server has an API key: press API key and enter it to see the status '
                 '(needs JavaScript).</p>')
    return template.replace(MARK, inner, 1).encode("utf-8")


def wants_html(accept: str, query: str) -> bool:
    """A browser asked for /status: the Accept header names text/html, or the query says ?html=1."""
    if "text/html" in (accept or ""):
        return True
    from urllib.parse import parse_qs
    return parse_qs(query or "").get("html", [""])[0] not in ("", "0", "false")
