#!/usr/bin/env python3
"""force-rescreen.py - owner-forced flags go into the RESCREEN (round 2) whatever their run-1 verdict (owner order 2026-11-20 ~12:45; TEST_LEDGER.md T-1006).

  force-rescreen.py --select SELECT.json --triage us-triage.v2.json --flags STRATA_KV_PREFETCH,STRATA_EXCHANGE_ROTATE,... --slice-prefix usS1-1of5

Adds the flags' run-1 records of the triage (any verdict) to the select file written by `sweep.py select` (the HIGH + HURTS flags), marks them "forced" and records the
prior evidence from the old design (healthy starts only) in "prior_evidence"; sweep.py spec copies both fields into the rescreen rows.  Prints one `SELECT n ...` line like
sweep.py select (the queue reads the count from it) and a FORCED line.  Nothing is deleted."""
import argparse
import json

EVIDENCE = {
    "STRATA_KV_PREFETCH": "prior evidence (old design, healthy starts only): owner 2026-11-20: 'is ok, goes to round 2'",
    "STRATA_EXCHANGE_ROTATE": "prior evidence (old design, healthy starts only): long64k decode +5.5 % (b01) and +3.1 % (b01r), the arm ran in the fast state both times; owner 2026-11-20: round 2",
    "STRATA_PIN_BY_NEED": "prior evidence (old design, healthy starts only): owner 2026-11-20: 'to be retested'",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", required=True)
    ap.add_argument("--triage", required=True)
    ap.add_argument("--flags", required=True)
    ap.add_argument("--slice-prefix", required=True)
    a = ap.parse_args()
    want = [x.strip() for x in a.flags.split(",") if x.strip()]
    sel = json.load(open(a.select, encoding="utf-8"))
    tri = json.load(open(a.triage, encoding="utf-8"))
    recs = {r["flag"]: r for r in tri["flags"].values() if r.get("arm") and str(r.get("slice") or "").startswith(a.slice_prefix)}
    missing = []
    for fl in want:
        r = recs.get(fl)
        if r is None:
            missing.append(fl)
            continue
        cur = sel["flags"].get(r["arm"], r)
        sel["flags"][r["arm"]] = {**cur, "forced": True, "prior_evidence": EVIDENCE.get(fl, "owner-forced into round 2")}
    json.dump(sel, open(a.select, "w"), indent=1)
    print("FORCED %d of %d owner-forced flag(s) into the rescreen%s" % (len(want) - len(missing), len(want), ("; not in the triage of %s (no row yet): %s" % (a.slice_prefix, ", ".join(missing))) if missing else ""))
    print("SELECT %d flag(s) with verdict %s + owner-forced: %s" % (len(sel["flags"]), "/".join(sel.get("verdicts", [])), ", ".join(r["flag"] for r in sel["flags"].values())))


if __name__ == "__main__":
    main()
