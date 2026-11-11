#!/usr/bin/env python3
"""park_sim.py - offline judge of the conversation-cache eviction rules on a replay plan (MOD_PLAN A15 park TTL, F1 STRATA_PARK_TRUNCATE).

replay_turns.py replays a recorded agent session against a live server (27 min per pair of arms, GPU).  This tool takes the same
PLAN (`replay_turns.py SPEC --dry-run --out DIR` writes DIR/plan.json; no server starts) and runs it through the real ConversationCache
header via the `park_replay_sim` binary (src/core/park_replay_sim.cpp, CMake target park_replay_sim): no GPU, no model, seconds.
It answers "which parked conversation does each rule evict, and how many cold re-reads does that cost", not decode speed.

    python tools/pancake/park_sim.py DIR/plan.json --sim build-cpu/park_replay_sim [--shared-system 1024] [--budgets 1280,2048]
           [--out result.json]

What it models (src/core/park_replay_sim.cpp has the details): the engine's request flow (live session kept for the same conversation,
best parked match taken or borrowed, the outgoing conversation parked with drop_superseded / make_room / put, F1's truncated park when the
whole image does not fit), snapshot sizes from the fitted log model (111 MiB + 14.9 KiB/token + 113 MiB per checkpoint), prompt time from
replay_turns.est_prompt_s.  Not modelled: the RAM floor, the disk tier, batch slots, decode.  The clock is the recording's own timeline.
The variants run: the default rule, F1, --conversation-cache-min-tokens 8192, A15 (TTL 30 s, with and without the tool hold), and combinations.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPLAY_SHARED_MIN_TAIL = 64   # a conversation shares at most (its first prompt - 64) leading tokens with the shared system prompt


def plan_to_trace(plan, shared_system=0, tool_agent=False):
    """The replay plan (a list of entries, replay_turns.plan_from_rows) -> trace lines for park_replay_sim.

    Fields: conv kind target out start_s dur_s tool shared.  The timeline is the recording's (rec.time, rec.duration_s); a plan without one
    (the synthetic session) gets one from the plan's own gaps and the prompt-time model.  `shared_system` N: every conversation starts with
    the same N-token system prompt (replay_turns --shared-system), capped for a conversation shorter than N + 64 tokens.
    `tool_agent`: every agent turn ends in a tool call (the recorded finish is "stop" for them, the turn log does not say).
    """
    lines, first_target, clock = [], {}, 0.0
    t0 = None
    for e in plan:
        rec = e.get("rec") or {}
        out = int(rec.get("output_tokens") or e.get("max_new") or 1)
        if rec.get("time") is not None:
            if t0 is None:
                t0 = rec["time"]
            start, dur = rec["time"] - t0, float(rec.get("duration_s") or 0.0)
        else:
            start = clock + float(e.get("gap_s") or 0.0)
            dur = 0.2 + 0.0005 * e["target"]
        clock = start + dur
        conv = e["conv"]
        first_target.setdefault(conv, e["target"])
        shared = max(0, min(int(shared_system), first_target[conv] - REPLAY_SHARED_MIN_TAIL)) if shared_system else 0
        kind = "A" if e["kind"] == "agent" else "S"
        tool = 1 if (tool_agent and kind == "A") else 0
        lines.append(f"{conv} {kind} {int(e['target'])} {max(1, out)} {start:.3f} {dur:.3f} {tool} {shared}")
    return lines


VARIANTS = [
    ("default (oldest first)", []),
    ("F1 park-truncate", ["--truncate"]),
    ("min-tokens 8192", ["--min-tokens", "8192"]),
    ("F1 + min-tokens 8192", ["--truncate", "--min-tokens", "8192"]),
    ("A15 ttl30, hold 0 (branches only)", ["--ttl-s", "30", "--hold-s", "0"]),
    ("A15 ttl30, hold 60", ["--ttl-s", "30", "--hold-s", "60"]),
    ("A15 ttl30 hold 60 + F1", ["--ttl-s", "30", "--hold-s", "60", "--truncate"]),
]


def run_variant(sim, trace_path, budget, extra):
    cmd = [str(sim), "--trace", str(trace_path), "--budget-mib", str(budget)] + extra
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if out.returncode != 0:
        raise SystemExit(f"{' '.join(cmd)}: {out.stderr.strip() or out.stdout.strip()}")
    return out.stdout


def parse_result(text):
    """The `result ...` line of park_replay_sim -> dict; the `cold requests:` line -> list."""
    import re
    res = {}
    for line in text.splitlines():
        m = re.search(r"agent prompt\s+([\d.]+) s\s+cold (\d+) \(\s*([\d.]+) s\)\s+all\s+([\d.]+) s \| parks (\d+) \(\+(\d+) truncated\) "
                      r"skips (\d+) restores (\d+) borrows (\d+) \| evictions (\d+) \(ttl (\d+)\) truncations (\d+)", line)
        if m:
            k = ["agent_prompt_s", "cold", "cold_s", "all_prompt_s", "parks", "truncated_parks", "skips", "restores", "borrows",
                 "evictions", "ttl_evictions", "truncations"]
            res.update({a: (float(b) if "." in b else int(b)) for a, b in zip(k, m.groups())})
        elif line.strip().startswith("cold requests:"):
            res["cold_requests"] = [int(x) for x in line.split(":", 1)[1].split()]
    return res


def evaluate(plan, sim, budgets=(1280,), shared_system=0, tool_agent=False):
    results = {}
    with tempfile.TemporaryDirectory() as d:
        tp = Path(d) / "trace.txt"
        tp.write_text("\n".join(plan_to_trace(plan, shared_system, tool_agent)) + "\n")
        for b in budgets:
            for name, extra in VARIANTS:
                results[f"{b} MiB | {name}"] = parse_result(run_variant(sim, tp, b, extra))
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("plan", help="plan.json written by replay_turns.py --dry-run")
    ap.add_argument("--sim", required=True, help="the park_replay_sim binary")
    ap.add_argument("--budgets", default="1280,2048", help="conversation cache budgets, MiB (1280: the 11-09 engine's)")
    ap.add_argument("--shared-system", type=int, default=0, help="tokens of system prompt every conversation shares (0: none)")
    ap.add_argument("--tool-agent", action="store_true", help="agent turns end in a tool call (A15's hold applies to them)")
    ap.add_argument("--out", help="write the results as JSON")
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    plan = plan["plan"] if isinstance(plan, dict) else plan
    budgets = [int(x) for x in a.budgets.split(",")]
    res = evaluate(plan, a.sim, budgets, a.shared_system, a.tool_agent)
    print(f"{len(plan)} requests; shared system {a.shared_system} tokens; agent turns {'end in a tool call' if a.tool_agent else 'do not'}")
    base = None
    for key, r in res.items():
        if key.endswith("default (oldest first)"):
            base = r
        d = "" if base is None or r is base else f"  vs default: agent {r['agent_prompt_s'] - base['agent_prompt_s']:+.1f} s, all {r['all_prompt_s'] - base['all_prompt_s']:+.1f} s"
        print(f"{key:48s} agent {r['agent_prompt_s']:7.1f} s  cold {r['cold']} ({r['cold_s']:5.1f} s)  all {r['all_prompt_s']:7.1f} s"
              f"  skips {r['skips']} evict {r['evictions']} (ttl {r['ttl_evictions']}) trunc {r['truncated_parks']}+{r['truncations']}{d}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
