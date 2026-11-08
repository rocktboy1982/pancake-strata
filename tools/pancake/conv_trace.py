"""conv_trace.py - pancake-strata: a routing trace of interleaved multi-turn conversations, for MOD_PLAN phase 11 items 3
(conversation working set) and 4 (per-domain profiles).  It runs one `--serve` engine with --dump-routing, plays several
conversations of different kinds turn by turn in an interleaved order (code edit A, prose B, A again, tools C, ...), and
writes where each request's records end, so tools/pancake/replay_seeding.py --requests can replay the exact sequence:

    .venv/bin/python tools/pancake/conv_trace.py pancake-flashnext-iq2_xs-fast-hip.json bench/conv-trace \\
        [--max-new 384] [--domain-keys] [--tag base]

CONFIG is the resolved preset run-pancake.sh writes (pancake-<preset>.json: exe, args, env, cwd, lib_dirs, tokenizer).
Output: OUT/<tag>.trace (the routing), OUT/<tag>.requests.json ([{end_record, conv, domain, prompt, new, decode_tok_s}]),
OUT/<tag>.engine.log.  --domain-keys sends each request's domain=<name> (as serve/server.py does with
STRATA_DOMAIN_PROFILES), so the same script is the live A/B: run it once plain and once with STRATA_DOMAIN_PROFILES /
STRATA_CONV_WORKSET in the environment, and compare the decode rates and the engine log's hit counts.

The next turn's prompt is the previous prompt + the reply's tokens + the next user message (the template's own
rendering with thinking off), so a conversation's later turns start with its earlier ones as a server would send them.
About 12 requests of up to --max-new tokens each: ~10 minutes with the engine's start-up.  Do not run it beside a
benchmark: it loads the GPU and the CPU like one.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "qwen35"))
sys.path.insert(0, str(ROOT / "tools" / "pancake"))
from q36_harness import Eng  # noqa: E402
from bench_engines import load_tok  # noqa: E402

TOOLS_SYS = ("You can call these tools. Answer with a JSON object {\"tool\": NAME, \"arguments\": {...}} when one is "
             "needed.\n" + json.dumps([
                 {"name": "read_file", "parameters": {"path": "string"}},
                 {"name": "grep", "parameters": {"pattern": "string", "path": "string"}},
                 {"name": "run_tests", "parameters": {"target": "string"}},
                 {"name": "web_search", "parameters": {"query": "string"}}], indent=1))


def src(path, chars=14000):
    t = (ROOT / path).read_text(encoding="utf-8", errors="replace")
    return t[:chars]


def conversations():
    """name -> (domain, system or None, [user turns])."""
    return {
        "A": ("code", None, [
            "Here is a file.\n\n" + src("src/core/expert_cache.cpp") +
            "\n\nAdd a function that returns how many experts of each layer are resident, and show the whole changed "
            "part of the file.",
            "Now write unit tests for that function, as a separate C++ file.",
            "Rename the function to resident_per_layer everywhere and show the diff."]),
        "B": ("prose", None, [
            "Write a short story (about 300 words) about a lighthouse keeper who finds a message in a bottle.",
            "Rewrite the ending so that it is hopeful, and give the story a title."]),
        "C": ("tool", TOOLS_SYS, [
            "Find where the adaptive expert tier decays its usage counts in this repository and tell me the factor.",
            "The tests for it fail on my machine. What do you call next, and why?"]),
        "D": ("code", None, [
            "Review this Python code for bugs and style problems; be specific.\n\n" + src("tools/replay_cache_policy.py"),
            "Fix the two most important problems and show the corrected functions."]),
    }


ORDER = ["A", "B", "A", "C", "B", "D", "A", "C", "D"]


def render(system, user):
    s = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    return s + "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--domain-keys", action="store_true")
    ap.add_argument("--tag", default="conv")
    a = ap.parse_args()
    cfg = json.load(open(a.config, encoding="utf-8-sig"))
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    trace = out / f"{a.tag}.trace"
    if trace.exists():
        trace.unlink()
    tok = load_tok(cfg["tokenizer"])
    env = {k: str(v) for k, v in (cfg.get("env") or {}).items()}
    libs = ":".join(cfg.get("lib_dirs") or [])
    if libs:
        env["LD_LIBRARY_PATH"] = libs + (":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else "")
    args = [x for x in cfg["args"] if x != "--serve"] + ["--dump-routing", str(trace)]
    if cfg.get("cwd"):
        os.chdir(cfg["cwd"])
    e = Eng(cfg["exe"], args, env=env, log=str(out / f"{a.tag}.engine.log"))
    print(f"loaded in {e.load_s:.1f}s", flush=True)
    convs = conversations()
    hist = {n: None for n in convs}   # the token ids of the last prompt + its reply
    turn = {n: 0 for n in convs}
    rows = []
    for n in ORDER:
        dom, system, users = convs[n]
        u = users[turn[n]]
        if hist[n] is None:
            ids = tok.encode(render(system, u), parse_special=True)
        else:
            ids = hist[n] + tok.encode("<|im_end|>\n<|im_start|>user\n" + u +
                                       "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", parse_special=True)
        keys = f"domain={dom}" if a.domain_keys else ""
        toks, done, ttft, wall = e.solo(ids, a.max_new, keys=keys)
        hist[n] = ids + toks
        turn[n] += 1
        rec_bytes = 8 + 8 * 10
        end = trace.stat().st_size // rec_bytes if trace.exists() else 0
        dec = (len(toks) - 1) / (wall - ttft) if len(toks) > 1 and ttft and wall > ttft else 0.0
        rows.append(dict(end_record=end, conv=n, domain=dom, prompt=len(ids), new=len(toks), decode_tok_s=round(dec, 2),
                         ttft_s=round(ttft or 0.0, 3), done=done, t=time.strftime("%Y-%m-%dT%H:%M:%S")))
        print(f"{n}{turn[n]} ({dom}): prompt {len(ids)}, {len(toks)} new, decode {dec:.1f} tok/s, records to {end}",
              flush=True)
        (out / f"{a.tag}.requests.json").write_text(json.dumps(rows, indent=1))
    e.p.stdin.close()
    e.p.wait(timeout=120)
    print(f"wrote {trace} and {out / (a.tag + '.requests.json')}")


if __name__ == "__main__":
    main()
