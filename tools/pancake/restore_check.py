"""restore_check.py - the restore-equivalence test (MOD_PLAN items 35 / 83).

A conversation that resumes from the conversation cache (a parked state or a message-boundary checkpoint) must answer
token for token like the same conversation read from position 0.  The engine has the switch for the comparison:
STRATA_CKPT_REREAD=1 reads a checkpoint's tokens again instead of restoring it.  With the VRAM expert set fixed
(--adapt-swaps 0) and greedy decoding, both runs must give the same text; any difference is something the checkpoint
or the restore missed (the UNSW finding on hybrid-state caches: a full-hit restore off by one token).

Each run starts a fresh server (the preset expanded as run-pancake.sh does, abtest.make_config overrides, its own port), sends the same scripted agent conversation turn by turn
- a long system prompt, a user turn, a tool call and its result, a follow-up user turn, then a branch back to an earlier
point (a partial hit) - and records every greedy answer.  Arm "restore" uses the cache as served; arm "reread" sets
STRATA_CKPT_REREAD=1.  STRATA_SNAPSHOT_VERIFY=1 is set on both (the engine also checks the drafter's snapshot).

    python tools/pancake/restore_check.py [--preset flashnext-iq2_xs-hip] [--port 18437] [--max-tokens 48] [--out FILE.json]
        [--tool-boundary]   # MOD_PLAN 16: both arms with STRATA_CACHE_TOOL_BOUNDARY=1; "tool-retry" resumes at the call

Exit 0 when every turn matches, 1 on a difference, 2 on a setup error.  Needs the GPU for itself: run it only when no
other engine serves (it refuses when something listens on the port) and outside a benchmark.
"""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SYSTEM = ("You are a coding agent working in a C++ repository. Answer briefly. " * 60).strip()   # a long, cached prefix
TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file",
                                            "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                                                           "required": ["path"]}}}]


def turns():
    """The scripted conversation: (label, messages) per request, each one a continuation of an earlier one."""
    base = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "What does the function draft() in src/core/mtp.cpp return? Read the file."}]
    call = {"role": "assistant", "content": "",
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "read_file", "arguments": json.dumps({"path": "src/core/mtp.cpp"})}}]}
    result = {"role": "tool", "tool_call_id": "c1",
              "content": (ROOT / "src/core/mtp.cpp").read_text(errors="replace")[:6000]}
    follow = {"role": "user", "content": "Summarize the min_p handling in two sentences."}
    branch = {"role": "user", "content": "Instead, list three function names you saw."}
    # MOD_PLAN 16: the same call answered with another result (a retried step).  With STRATA_CACHE_TOOL_BOUNDARY the
    # engine resumes it at the tool-call boundary checkpoint taken in "after-tool-result" (--tool-boundary)
    retry = {"role": "tool", "tool_call_id": "c1",
             "content": (ROOT / "src/core/mtp.cpp").read_text(errors="replace")[6000:9000]}
    return [("first", base),
            ("after-tool-result", base + [call, result]),
            ("follow-up", base + [call, result, {"role": "assistant", "content": "It returns true on success."}, follow]),
            ("branch-back", base + [call, result, branch]),
            ("tool-retry", base + [call, retry])]


def port_free(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def wait_ready(port, proc, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(3)
    return False


def chat(port, messages, max_tokens):
    body = {"model": "m", "messages": messages, "tools": TOOLS, "max_tokens": max_tokens, "temperature": 0,
            "reasoning_effort": "low", "stream": False}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.loads(r.read())
    m = out["choices"][0]["message"]
    return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "",
            "tool_calls": m.get("tool_calls") or [], "usage": out.get("usage", {})}


def expand_preset(preset, out):
    """configs/pancake/<preset>.json with run-pancake.sh's placeholders filled (HIP build)."""
    data = os.environ.get("STRATA_DATA", "/home/dan/clone-of-d/Strata-data")
    models = os.environ.get("MODELS", "/home/dan/clone-of-d/models")
    raw = (ROOT / "configs/pancake" / f"{preset}.json").read_text()
    raw = raw.replace("<STRATA_DATA>", data).replace("<REPO>", str(ROOT)).replace("<MODELS>", models)
    raw = raw.replace("/build-cuda/", "/build-hip/")
    Path(out).write_text(raw)


def run_arm(name, env_extra, a):
    sys.path.insert(0, str(ROOT / "tools/pancake"))
    import abtest                                        # make_config: the same arm overrides abtest uses
    base = ROOT / f"restore-check-base.json"
    expand_preset(a.preset, base)
    cfg_path = ROOT / f"restore-check-{name}.json"
    cfg = abtest.make_config(str(base), {"env": {"STRATA_SNAPSHOT_VERIFY": "1", **env_extra},
                                         "args": {"--adapt-swaps": "0"}}, str(cfg_path))
    cfg_path.write_text(json.dumps({**cfg, "turn_log": None}, indent=1))   # keep the daily turn log clean
    log_out = ROOT / f"restore-check-{name}.log"
    env = dict(os.environ, LD_LIBRARY_PATH="/opt/rocm/core-10.0/lib", HIP_VISIBLE_DEVICES="0")
    # (STRATA_CACHE_TOOL_BOUNDARY is read by the server too: a config's env reaches only the engine)
    env.update({k: v for k, v in env_extra.items() if k.startswith("STRATA_CACHE_TOOL_BOUNDARY")})
    with open(log_out, "w") as lf:
        proc = subprocess.Popen([str(ROOT / "tools/pancake/capped.sh"), str(ROOT / ".venv/bin/python"), "serve/server.py",
                                 "--engine", "strata", "--config", str(cfg_path), "--port", str(a.port)],
                                cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    try:
        if not wait_ready(a.port, proc):
            print(f"[{name}] server did not come up (see {log_out})", file=sys.stderr)
            return None
        answers = []
        for label, msgs in turns():
            ans = chat(a.port, msgs, a.max_tokens)
            answers.append({"turn": label, **ans})
            print(f"[{name}] {label}: cached={ans['usage'].get('prompt_tokens_details', {}).get('cached_tokens')} "
                  f"out={ans['usage'].get('completion_tokens')}", flush=True)
        return answers
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=120)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", default="flashnext-iq2_xs-hip",
                    help="configs/pancake preset (the exact HIP preset keeps decoding deterministic)")
    ap.add_argument("--port", type=int, default=18437)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tool-boundary", action="store_true",
                    help="MOD_PLAN 16: both arms with STRATA_CACHE_TOOL_BOUNDARY=1 (gap 16, so the short scripted call "
                         "is checkpointed)")
    a = ap.parse_args()
    if not port_free(a.port):
        print(f"port {a.port} is in use: refusing (one engine at a time on the GPU)", file=sys.stderr)
        return 2
    tb = {"STRATA_CACHE_TOOL_BOUNDARY": "1", "STRATA_CACHE_TOOL_BOUNDARY_GAP": "16"} if a.tool_boundary else {}
    restore = run_arm("restore", dict(tb), a)            # both arms: --adapt-swaps 0 (fixed VRAM expert set)
    reread = run_arm("reread", {"STRATA_CKPT_REREAD": "1", **tb}, a)
    if restore is None or reread is None:
        return 2
    diffs = []
    for r, q in zip(restore, reread):
        same = r["content"] == q["content"] and r["reasoning"] == q["reasoning"] and r["tool_calls"] == q["tool_calls"]
        print(f"{r['turn']}: {'MATCH' if same else 'DIFFERENT'}")
        if not same:
            diffs.append({"turn": r["turn"], "restore": r, "reread": q})
    if a.out:
        Path(a.out).write_text(json.dumps({"restore": restore, "reread": reread, "diffs": diffs}, indent=1))
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
