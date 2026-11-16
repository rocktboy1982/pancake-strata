#!/usr/bin/env python3
"""state_cost.py - pancake-strata MOD_PLAN.md section 6 item 41: what saving conversation state costs, from an engine log.

    python tools/pancake/state_cost.py engine.log [--turns pancake-turns.jsonl] [--json OUT.json]

Offline, stdlib only.  Reads the lines the engine really prints (src/program/generate.cpp, stderr), sums them and says what share of
the served time they are.  It invents nothing: a quantity the log does not carry is printed as "not in the log".

Lines used (always printed unless noted):
  strata serve: conversation cache: parked|skipped N tokens... in X ms; parked=P bytes=B evictions=E snapshot_bytes=S reused_kv_bytes=R
      one park: the outgoing conversation copied to host RAM at the next request's start (inside that request's prompt time).
      S = the image's bytes; R = K/V bytes kept from the previous parked copy (not copied again).
  strata serve: conversation cache: restored|borrowed N tokens (live|checkpoint) in X ms ...        a parked conversation coming back
  strata serve: conversation cache: skip parking (...)                                             a park the rules refused (no copy)
  strata serve: conversation disk: read back a N-token match (S MiB) in X ms                       disk tier read (opt-in tier)
  strata serve: conversation disk: disk-restored N tokens (live|checkpoint, S MiB) in X ms (read pass Y ms)
  strata trace: checkpoint at L: sync X ms, save Y ms ...        ONLY with STRATA_TRACE set: one prompt checkpoint (the GDN state to RAM)
A prompt checkpoint's bytes are not in the log; they are the GDN state of every GDN layer plus small vectors, derived here from the
Flash-Next geometry (include/strata/core/geometry.hpp: gdn_state_bytes()).  The disk tier's saves (background) print no ms and are
not counted.

With --turns (the server's turn log) the totals are put against the summed request time (duration_s) and prompt time (prompt_ms) of
the turns in the log's span, and per request.  Without it, only the totals and per-event means.
"""
import argparse
import json
import re
import sys

# Flash-Next (qwen4exp) geometry, include/strata/core/geometry.hpp defaults
N_LAYERS, QSA_INTERVAL = 48, 4
SSM_STATE, SSM_V_HEADS, SSM_CONV_CHANNELS, SSM_D_CONV = 128, 48, 10240, 4
N_HEAD_KV, HEAD_DIM, IDX_KEY_DIM, IDX_BLOCK = 2, 256, 128, 4


def gdn_state_bytes():
    """The recurrent + convolution state of every GDN layer in one image (conversation_state.cpp conversation_state_sizes)."""
    n_gdn = N_LAYERS - N_LAYERS // QSA_INTERVAL
    per_layer = (SSM_STATE * SSM_V_HEADS * SSM_STATE + SSM_CONV_CHANNELS * (SSM_D_CONV - 1)) * 4
    return n_gdn * per_layer


def kv_bytes_per_token():
    """--kv int8 K/V of the full-attention layers plus the indexer's pooled rows (conversation_snapshot.cpp layout())."""
    per_layer = (N_HEAD_KV * HEAD_DIM * 1) * 2 + (N_HEAD_KV * (HEAD_DIM // 64) * 2) * 2 + (IDX_KEY_DIM * 4) // IDX_BLOCK
    return (N_LAYERS // QSA_INTERVAL) * per_layer


NUM = r"([\d.]+)"
RE_PARK = re.compile(r"conversation cache: (parked|skipped) (\d+) tokens.*? in " + NUM + r" ms; parked=(\d+) bytes=(\d+)")
RE_SNAP = re.compile(r"snapshot_bytes=(\d+)")
RE_REUSED = re.compile(r"reused_kv_bytes=(\d+)")
RE_RESTORE = re.compile(r"conversation cache: (restored|borrowed) (\d+) tokens(?: \((live|checkpoint)\))? .*?in " + NUM + r" ms")
RE_SKIP = re.compile(r"conversation cache: skip parking \(([^)]*)\)")
RE_DISK_READ = re.compile(r"conversation disk: (?:read back|read back but could not park) a (\d+)-token match \((\d+) MiB\) in " + NUM + r" ms")
RE_DISK_REST = re.compile(r"conversation disk: disk-restored (\d+) tokens \((live|checkpoint), (\d+) MiB\) in " + NUM + r" ms")
RE_CKPT = re.compile(r"strata trace: checkpoint at (\d+): sync " + NUM + r" ms, save " + NUM + r" ms")
RE_EVICT = re.compile(r"strata trace: checkpoint at (\d+): (\d+) evicted in " + NUM + r" ms")


def parse_log(lines):
    ev = {"parks": [], "skipped_parks": [], "restores": [], "borrows": [], "skip_reasons": {}, "disk_reads": [],
          "disk_restores": [], "checkpoints": [], "ckpt_evictions": []}
    for line in lines:
        m = RE_CKPT.search(line)
        if m:
            ev["checkpoints"].append({"at": int(m.group(1)), "sync_ms": float(m.group(2)), "save_ms": float(m.group(3))})
            continue
        m = RE_EVICT.search(line)
        if m:
            ev["ckpt_evictions"].append({"at": int(m.group(1)), "n": int(m.group(2)), "ms": float(m.group(3))})
            continue
        m = RE_PARK.search(line)
        if m:
            snap = RE_SNAP.search(line)
            reu = RE_REUSED.search(line)
            row = {"tokens": int(m.group(2)), "ms": float(m.group(3)), "snapshot_bytes": int(snap.group(1)) if snap else None,
                   "reused_kv_bytes": int(reu.group(1)) if reu else None}
            ev["parks" if m.group(1) == "parked" else "skipped_parks"].append(row)
            continue
        m = RE_RESTORE.search(line)
        if m:
            ev["restores" if m.group(1) == "restored" else "borrows"].append(
                {"tokens": int(m.group(2)), "kind": m.group(3) or "", "ms": float(m.group(4))})
            continue
        m = RE_DISK_REST.search(line)
        if m:
            ev["disk_restores"].append({"tokens": int(m.group(1)), "kind": m.group(2), "bytes": int(m.group(3)) << 20, "ms": float(m.group(4))})
            continue
        m = RE_DISK_READ.search(line)
        if m:
            ev["disk_reads"].append({"tokens": int(m.group(1)), "bytes": int(m.group(2)) << 20, "ms": float(m.group(3))})
            continue
        m = RE_SKIP.search(line)
        if m:
            why = re.sub(r"\d+", "N", m.group(1))
            ev["skip_reasons"][why] = ev["skip_reasons"].get(why, 0) + 1
    return ev


def load_turns(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def summarize(ev, turns=None):
    s = {}
    parks = ev["parks"]
    s["parks"] = {"n": len(parks), "ms": sum(p["ms"] for p in parks),
                  "snapshot_bytes": sum(p["snapshot_bytes"] or 0 for p in parks),
                  "reused_kv_bytes": sum(p["reused_kv_bytes"] or 0 for p in parks),
                  "bytes_in_log": all(p["snapshot_bytes"] is not None for p in parks) if parks else None,
                  "max_ms": max((p["ms"] for p in parks), default=0.0)}
    s["parks"]["copied_upper_bytes"] = s["parks"]["snapshot_bytes"] - s["parks"]["reused_kv_bytes"]
    s["skipped_parks"] = {"n": len(ev["skipped_parks"]), "ms": sum(p["ms"] for p in ev["skipped_parks"])}
    s["skip_reasons"] = ev["skip_reasons"]
    s["restores"] = {"n": len(ev["restores"]), "ms": sum(r["ms"] for r in ev["restores"]),
                     "from_checkpoint": sum(1 for r in ev["restores"] if r["kind"] == "checkpoint"),
                     "max_ms": max((r["ms"] for r in ev["restores"]), default=0.0)}
    s["borrows"] = {"n": len(ev["borrows"]), "ms": sum(r["ms"] for r in ev["borrows"])}
    s["disk_reads"] = {"n": len(ev["disk_reads"]), "ms": sum(r["ms"] for r in ev["disk_reads"]),
                       "bytes": sum(r["bytes"] for r in ev["disk_reads"])}
    s["disk_restores"] = {"n": len(ev["disk_restores"]), "ms": sum(r["ms"] for r in ev["disk_restores"]),
                          "bytes": sum(r["bytes"] for r in ev["disk_restores"])}
    ck = ev["checkpoints"]
    s["checkpoints"] = {"n": len(ck), "sync_ms": sum(c["sync_ms"] for c in ck), "save_ms": sum(c["save_ms"] for c in ck),
                        "ms_in_log": bool(ck), "bytes_derived": len(ck) * gdn_state_bytes()}
    s["ckpt_evictions"] = {"n": len(ev["ckpt_evictions"]), "ms": sum(c["ms"] for c in ev["ckpt_evictions"])}
    s["state_ms"] = (s["parks"]["ms"] + s["skipped_parks"]["ms"] + s["restores"]["ms"] + s["borrows"]["ms"] +
                     s["disk_reads"]["ms"] + s["disk_restores"]["ms"] + s["checkpoints"]["save_ms"] +
                     s["checkpoints"]["sync_ms"] + s["ckpt_evictions"]["ms"])
    if turns:
        n = len(turns)
        dur = sum(float(t.get("duration_s") or 0.0) for t in turns)
        prm = sum(float(t.get("prompt_ms") or 0.0) for t in turns) / 1000.0
        s["turns"] = {"n": n, "duration_s": dur, "prompt_s": prm,
                      "state_share_of_duration": (s["state_ms"] / 1000.0) / dur if dur > 0 else None,
                      "state_share_of_prompt": (s["state_ms"] / 1000.0) / prm if prm > 0 else None,
                      "state_ms_per_request": s["state_ms"] / n if n else None,
                      "state_bytes_per_request": ((s["parks"]["copied_upper_bytes"] + s["checkpoints"]["bytes_derived"]) / n) if n else None}
    return s


def render(s):
    def mib(b):
        return "%.1f MiB" % (b / 1048576.0)
    out = []
    p, r = s["parks"], s["restores"]
    out.append("parks:        %d, %.1f ms total, mean %.1f, max %.1f; snapshot %s, of which K/V kept from the last copy %s%s" % (
        p["n"], p["ms"], p["ms"] / p["n"] if p["n"] else 0.0, p["max_ms"], mib(p["snapshot_bytes"]), mib(p["reused_kv_bytes"]),
        "" if p["bytes_in_log"] in (True, None) else "  (some parks print no bytes: older log format)"))
    out.append("  skipped:    %d (%.1f ms)  reasons: %s" % (s["skipped_parks"]["n"], s["skipped_parks"]["ms"],
                                                           json.dumps(s["skip_reasons"]) if s["skip_reasons"] else "none"))
    out.append("restores:     %d (%d from a checkpoint), %.1f ms total, mean %.1f, max %.1f; borrows %d (%.1f ms)" % (
        r["n"], r["from_checkpoint"], r["ms"], r["ms"] / r["n"] if r["n"] else 0.0, r["max_ms"], s["borrows"]["n"], s["borrows"]["ms"]))
    c = s["checkpoints"]
    if c["ms_in_log"]:
        out.append("checkpoints:  %d, sync %.1f ms + save %.1f ms; bytes DERIVED %s (GDN state %s each); evictions %d (%.1f ms)" % (
            c["n"], c["sync_ms"], c["save_ms"], mib(c["bytes_derived"]), mib(gdn_state_bytes()), s["ckpt_evictions"]["n"], s["ckpt_evictions"]["ms"]))
    else:
        out.append("checkpoints:  not in the log (the engine prints them only with STRATA_TRACE set); one is %s of GDN state" % mib(gdn_state_bytes()))
    out.append("disk tier:    reads %d (%.1f ms, %s), restores %d (%.1f ms, %s); the background saves print no ms" % (
        s["disk_reads"]["n"], s["disk_reads"]["ms"], mib(s["disk_reads"]["bytes"]), s["disk_restores"]["n"],
        s["disk_restores"]["ms"], mib(s["disk_restores"]["bytes"])))
    out.append("state time:    %.1f ms in all" % s["state_ms"])
    t = s.get("turns")
    if t:
        out.append("turn log:     %d requests, %.1f s request time, %.1f s prompt time" % (t["n"], t["duration_s"], t["prompt_s"]))
        out.append("  share:      %.2f%% of request time, %.2f%% of prompt time; %.1f ms and %s per request" % (
            100 * (t["state_share_of_duration"] or 0.0), 100 * (t["state_share_of_prompt"] or 0.0), t["state_ms_per_request"] or 0.0,
            mib(t["state_bytes_per_request"] or 0.0)))
        out.append("  (the turn log must cover the same requests as the engine log; bytes per request = parks' copied upper bound + derived checkpoint bytes)")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log", help="engine stderr log")
    ap.add_argument("--turns", help="the server's turn log (pancake-turns.jsonl) for the same requests")
    ap.add_argument("--json", help="write the summary here")
    a = ap.parse_args(argv)
    with open(a.log, encoding="utf-8", errors="replace") as f:
        ev = parse_log(f)
    s = summarize(ev, load_turns(a.turns) if a.turns else None)
    print(render(s))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
