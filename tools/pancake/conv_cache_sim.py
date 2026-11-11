#!/usr/bin/env python3
"""conv_cache_sim.py - offline model of the conversation cache on a recorded agent session (MOD_PLAN 4.0.1 / A12 side, 16.6).

Question: how much prompt time do `--conversation-cache-mib` / `--conversation-cache-min-tokens` / `--conversation-cache-slots`
save on a recorded session, and what do they cost on the requests they touch?  No GPU, no server: arithmetic over
the per-turn log (`pancake-turns.jsonl`) with the same conversation grouping as tools/pancake/replay_turns.py.

    python tools/pancake/conv_cache_sim.py [--turns pancake-turns.jsonl] [--session 2026-11-09T16:13/16:43]
           [--avail-gib 18 --keep-gib 12]                  the RAM rule's inputs (MOD_PLAN 86), for the "ram-rule" budget
           [--budget MIB --slots N --min-tokens N]         one scenario, else the built-in table
           [--ms-per-token 0.414 --restore-s 0.9 --snapshot-mib 111,14.9,113]

The cache follows include/strata/core/conversation_cache.hpp and src/program/generate.cpp (parking at a switch):
  - a request first takes its own parked copy out of the cache (`incoming`, still held in RAM), then the conversation that
    was live is parked if it has at least min-tokens tokens: it must fit in budget - held bytes, and a parked copy
    is evicted, oldest first, while the cache holds `slots` entries or the bytes do not fit (make_room); a snapshot larger
    than budget - held is skipped, so the next turn of that conversation is a cold read;
  - a request continues the live conversation (read = prompt - live tokens), restores its parked copy (read = prompt - the
    parked prefix, plus the copy-back time) or reads everything.
Costs: ms-per-token x tokens read (0.414: least-squares fit over the 11-09 log, long reads 0.39-0.43) and restore-s per 1257 MiB
restored (the log parks 55,424 tokens = 1258 MiB in 706 ms).  A fixed per-request overhead is the same in every scenario and
left out, so the numbers are DIFFERENCES between scenarios; the "recorded" line is the log's own prompt time for scale.
Snapshot size = a + b KiB x tokens + c MiB x checkpoints (default 111, 14.9, 113: fits the log's park / skip lines within
1 MiB, bench/results/2026-11-10-offline/RESULTS.md); a checkpoint sits every 16384 tokens read, at most 6 kept.

Known limits of the model: the text of side requests is not known, so a "repeat" side request (reused > 0 and the same size as an earlier one
in the log) is taken to have the same prefix as that earlier request; parked sizes are a fit, not the engine's own bytes; the pinned
shared prefix, the park TTL (A15) and PARK_TRUNCATE (F1) are not modeled.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import replay_turns as R  # noqa: E402

CKPT_EVERY = 16384
MAX_CKPTS = 6
RESTORE_REF_MIB = 1257


def ram_rule_budget_mib(avail_mib, keep_mib, min_free_mib=2560, margin_mib=512, cap_mib=8192):
    """ram_budget_conversation_cache_mib (include/strata/core/conversation_memory.hpp): available - max(keep, floor) - margin,
    capped, in steps of 256, and 0 (off) under 1024."""
    mib = avail_mib - max(keep_mib, min_free_mib, 0) - max(margin_mib, 0)
    if mib <= 0:
        return 0
    if cap_mib > 0:
        mib = min(mib, cap_mib)
    mib -= mib % 256
    return mib if mib >= 1024 else 0


def legacy_budget_mib(avail_mib, min_free_mib=2560):
    """auto_conversation_cache_mib: 40% of what is left above the parking floor, at most 8192, steps of 256, off under 1024."""
    usable = avail_mib - min_free_mib
    if usable <= 0:
        return 0
    mib = min(usable * 2 // 5, 8192)
    mib -= mib % 256
    return mib if mib >= 1024 else 0


def requests_from_log(rows, conv_min=8192, max_growth=16384):
    """The recorded requests as the simulation's input: conversation key, prompt tokens, output tokens, agent or side, and what the
    log measured (reused tokens, prompt_ms), grouped as replay_turns.plan_from_rows does."""
    plan = R.plan_from_rows(rows, conv_min, max_growth, 2.0, "recorded")
    return [{"conv": e["conv"], "tot": e["target"], "gen": int(e["rec"].get("output_tokens") or 0), "agent": e["kind"] == "agent",
             "rec_reused": int(e["rec"].get("reused") or 0), "rec_s": float(e["rec"].get("prompt_ms") or 0) / 1000.0} for e in plan]


class Params:
    def __init__(self, budget_mib, slots=4, min_tokens=0, ms_per_token=0.414, restore_s=0.9, snap=(111.0, 14.9, 113.0)):
        self.budget, self.slots, self.min_tokens = budget_mib, slots, min_tokens
        self.ms_per_token, self.restore_s, self.snap = ms_per_token, restore_s, snap

    def snapshot_mib(self, tokens, ckpts):
        a, b, c = self.snap
        return a + b / 1024.0 * tokens + c * ckpts


def simulate(reqs, p):
    """Runs the requests through the cache.  Returns {"rows": per request, "skipped": [snapshot MiB of each skipped park], ...}."""
    cache = {}                       # conv -> {tok, ck, size, t}
    live = None                      # {conv, tok, ck}
    rows, skipped, evictions, parks = [], [], 0, 0
    seen = set()
    for t, r in enumerate(reqs):
        conv, tot = r["conv"], r["tot"]
        incoming = None if (live and live["conv"] == conv) else cache.pop(conv, None)
        held = incoming["size"] if incoming else 0
        if live and live["conv"] != conv:
            if live["tok"] >= p.min_tokens:
                size = p.snapshot_mib(live["tok"], live["ck"])
                if size > p.budget - held:
                    skipped.append(round(size))
                else:
                    cache.pop(live["conv"], None)
                    while cache and (len(cache) >= p.slots or sum(e["size"] for e in cache.values()) > p.budget - held - size):
                        del cache[min(cache, key=lambda k: cache[k]["t"])]
                        evictions += 1
                    if len(cache) < p.slots:
                        cache[live["conv"]] = {"tok": live["tok"], "ck": live["ck"], "size": size, "t": t}
                        parks += 1
            live = None
        if live:                                          # warm: the session still holds this conversation
            how, read, cost, ck = "warm", max(tot - live["tok"], 0), 0.0, min(MAX_CKPTS, live["ck"] + 1)
        elif incoming:                                    # its parked copy comes back
            how, read = "restore", tot - min(incoming["tok"], tot)
            cost, ck = p.restore_s * incoming["size"] / RESTORE_REF_MIB, min(MAX_CKPTS, incoming["ck"] + 1)
        else:
            how, read, cost = "cold", tot, 0.0
            ck = max(1, tot // CKPT_EVERY) if r["agent"] else 1
        first = conv not in seen
        seen.add(conv)
        cost += read * p.ms_per_token / 1000.0
        rows.append({"conv": conv, "agent": r["agent"], "tot": tot, "read": read, "how": how, "cost_s": cost, "first": first,
                     "rec_s": r["rec_s"], "rec_cold": r["rec_reused"] == 0})
        live = {"conv": conv, "tok": tot + r["gen"], "ck": ck}
    return {"rows": rows, "skipped": skipped, "evictions": evictions, "parks": parks}


def summarize(res, cold_min=8192):
    """Totals; a cold re-read is a non-first agent turn above cold_min tokens that read more than half of its prompt (replay_turns)."""
    rows = res["rows"]
    cold = [x for x in rows if x["agent"] and not x["first"] and x["tot"] > cold_min and x["read"] > x["tot"] * 0.5]
    return {"prompt_s": sum(x["cost_s"] for x in rows), "agent_s": sum(x["cost_s"] for x in rows if x["agent"]),
            "side_s": sum(x["cost_s"] for x in rows if not x["agent"]), "cold": len(cold), "cold_s": sum(x["cost_s"] for x in cold),
            "restores": sum(1 for x in rows if x["how"] == "restore"), "skipped": len(res["skipped"]),
            "evictions": res["evictions"]}


def scenarios(avail_mib, keep_mib):
    ram = ram_rule_budget_mib(avail_mib, keep_mib)
    leg = legacy_budget_mib(avail_mib)
    return [("S0 legacy budget, no min-tokens (the 11-09 run: 1280 MiB)", Params(1280)),
            ("S1 --conversation-cache-mib 2048", Params(2048)),
            ("S2 --conversation-cache-min-tokens 8192 at 1280", Params(1280, min_tokens=8192)),
            ("S3 4.0.1 flags: 2048 MiB + min-tokens 8192", Params(2048, min_tokens=8192)),
            ("S4 RAM rule (86), %d MiB, no flags" % ram, Params(ram)),
            ("S5 RAM rule (86), %d MiB + min-tokens 8192" % ram, Params(ram, min_tokens=8192)),
            ("S6 RAM rule (86), %d MiB + min-tokens 8192 + 8 slots" % ram, Params(ram, slots=8, min_tokens=8192)),
            ("S7 RAM rule (86), %d MiB, 8 slots, no min-tokens" % ram, Params(ram, slots=8)),
            ("S8 legacy 40%% rule at the same RAM, %d MiB" % leg, Params(leg))]


def report(reqs, table, show_rows=False):
    rec = sum(r["rec_s"] for r in reqs)
    rec_cold = sum(r["rec_s"] for r in reqs if r["agent"] and r["rec_reused"] == 0 and r["tot"] > 8192)
    print("session: %d requests (%d agent turns, %d side), %.2fM prompt tokens; recorded prompt time %.1f s, of it %.1f s on "
          "agent turns read from token 0" % (len(reqs), sum(r["agent"] for r in reqs), sum(not r["agent"] for r in reqs),
                                            sum(r["tot"] for r in reqs) / 1e6, rec, rec_cold))
    print("%-62s %8s %8s %8s %5s %7s %6s %5s %5s" % ("scenario", "prompt s", "agent s", "side s", "cold", "cold s", "skip", "evict", "saved"))
    base = None
    for name, p in table:
        s = summarize(simulate(reqs, p))
        if base is None:
            base = s
        print("%-62s %8.1f %8.1f %8.1f %5d %7.1f %6d %5d %+8.1f" % (name, s["prompt_s"], s["agent_s"], s["side_s"], s["cold"], s["cold_s"],
                                                                    s["skipped"], s["evictions"], base["prompt_s"] - s["prompt_s"]))
        if show_rows:
            for x in simulate(reqs, p)["rows"]:
                print("   %s %6d read %6d %-8s %6.1f s" % (x["conv"], x["tot"], x["read"], x["how"], x["cost_s"]))
    return base


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--turns", default=str(R.ROOT / "pancake-turns.jsonl"))
    ap.add_argument("--session", default="2026-11-09T16:13/16:43")
    ap.add_argument("--session-gap", type=float, default=600)
    ap.add_argument("--conv-min", type=int, default=8192)
    ap.add_argument("--max-growth", type=int, default=16384)
    ap.add_argument("--avail-gib", type=float, default=18.0, help="MemAvailable once the arena is loaded (this box: ~51 - 33)")
    ap.add_argument("--keep-gib", type=float, default=12.0, help="STRATA_RAM_KEEP_GIB (run-pancake.sh sets 12)")
    ap.add_argument("--budget", type=int, default=None, help="one scenario: budget in MiB (with --slots, --min-tokens)")
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--min-tokens", type=int, default=0)
    ap.add_argument("--ms-per-token", type=float, default=0.414)
    ap.add_argument("--restore-s", type=float, default=0.9)
    ap.add_argument("--snapshot-mib", default="111,14.9,113")
    ap.add_argument("--rows", action="store_true", help="print every request of every scenario")
    a = ap.parse_args(argv)
    snap = tuple(float(x) for x in a.snapshot_mib.split(","))
    rows = R.load_turns(a.turns)
    got, label = R.select_session(rows, a.session, a.session_gap)
    print(label)
    reqs = requests_from_log(got, a.conv_min, a.max_growth)
    if a.budget is not None:
        table = [("budget %d MiB, %d slots, min-tokens %d" % (a.budget, a.slots, a.min_tokens),
                  Params(a.budget, a.slots, a.min_tokens, a.ms_per_token, a.restore_s, snap))]
    else:
        table = scenarios(int(a.avail_gib * 1024), int(a.keep_gib * 1024))
        for _, p in table:
            p.ms_per_token, p.restore_s, p.snap = a.ms_per_token, a.restore_s, snap
    report(reqs, table, a.rows)


if __name__ == "__main__":
    main()
