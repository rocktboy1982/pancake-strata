#!/usr/bin/env python3
"""gdn_chunk64_rule.py - MOD_PLAN A31 decision rule (GDN prefill chunk 64) from two logs.

QUESTION   Build the chunk-64 gfx11 GDN scan only if (a) the 16-token scan is latency/barrier-bound, not WMMA-bound
           (rec_out / WMMA floor >= 2 at 32K), AND (b) the recurrence is >= 6% of the cold-prefill GPU timeline at 16-32K.
INPUTS     --stage  output of `HIP_VISIBLE_DEVICES=0 STRATA_GDN_CHUNKED=1 build-hip/hip_gdn_stage_bench 4096 16384 32768`
                    (the "STAGE" and "A31" lines; per layer, one prompt chunk of T tokens)
           --timing engine log of one cold prompt with STRATA_PREFILL_TIMING=1 (the "strata prefill timing:" line)
           --prefill-tok-s  prompt speed of the same card and preset at that length, when there is no timing log: the share
                    is then estimated as rec_out_ms * --layers / (T / tok_s)
READ       (b) needs rec_out per layer >= 0.06 * (T / tok_s) / layers.  On the 7900 XT at 32K (2,580 tok/s measured, 36 GDN
           layers) that is 21 ms, 7.5 times the 2.84 ms WMMA floor, so (b) holding implies (a) holding; a stage bench far below
           ~21 ms closes A31 without a timing run.  A timing log from a CUDA card (the 3060 run, 2026-11-08) is not the HIP path
           and is reported as not applicable.
VERDICT    build / close / need the other input.  Exit code 0 always (a report, not a gate).
"""
import argparse
import re
import sys

RULE_FLOOR = 2.0     # (a) rec_out / floor at 32K
RULE_SHARE = 6.0     # (b) % of the cold-prefill GPU timeline


def parse_stage(text):
    """{T: {"rec_out_ms": float, "floor_ms": float|None, "over_floor": float|None}} from the stage bench output."""
    out = {}
    for m in re.finditer(r"^STAGE T=(\d+) .*?rec_out_ms=([\d.]+)", text, re.M):
        out.setdefault(int(m.group(1)), {"rec_out_ms": float(m.group(2)), "floor_ms": None, "over_floor": None})
    for m in re.finditer(r"^A31 T=(\d+) wmma_floor_ms=([\d.]+).*?rec_out_over_floor=([\d.]+)", text, re.M):
        t = int(m.group(1))
        if t in out:
            out[t]["floor_ms"], out[t]["over_floor"] = float(m.group(2)), float(m.group(3))
    return out


def parse_timing(text):
    """{"tokens", "gpu_ms", "rec_ms", "rec_pct", "gpu_name", "hip"} from an engine log, or None."""
    m = re.search(r"strata prefill timing: (\d+) tokens, GPU timeline (\d+) ms.*?gdn recurrence (\d+) \(([\d.]+)%\)", text)
    if not m:
        return None
    g = re.search(r"GPU \d+: ([^,\n]+)", text)
    name = g.group(1).strip() if g else "unknown"
    return {"tokens": int(m.group(1)), "gpu_ms": int(m.group(2)), "rec_ms": int(m.group(3)), "rec_pct": float(m.group(4)),
            "gpu_name": name, "hip": re.search(r"Radeon|AMD|gfx\d", name) is not None}


def share_from_stage(rec_out_ms, T, tok_s, layers):
    """Recurrence share (%) of a T-token cold prompt that reads at tok_s, from one layer's rec_out_ms."""
    return 100.0 * rec_out_ms * layers / (1000.0 * T / tok_s)


def decide(stage, timing, tok_s, layers, t_ref=32768):
    lines, a, b = [], None, None
    if stage and t_ref in stage:
        s = stage[t_ref]
        if s["over_floor"] is not None:
            a = s["over_floor"] >= RULE_FLOOR
            lines.append(f"(a) T={t_ref}: rec_out {s['rec_out_ms']:.2f} ms, WMMA floor {s['floor_ms']:.2f} ms, "
                         f"ratio {s['over_floor']:.2f} (rule >= {RULE_FLOOR:g}): {'holds' if a else 'fails'}")
        else:
            lines.append(f"(a) T={t_ref}: no A31 line (not gfx11, or STRATA_GDN_CHUNKED unset)")
        if tok_s:
            share = share_from_stage(s["rec_out_ms"], t_ref, tok_s, layers)
            b = share >= RULE_SHARE
            lines.append(f"(b) estimated from the stage bench: {share:.2f}% of a {t_ref}-token prompt at {tok_s:g} tok/s "
                         f"over {layers} layers (rule >= {RULE_SHARE:g}%): {'holds' if b else 'fails'}")
    elif stage:
        lines.append(f"(a) no T={t_ref} line in the stage bench output")
    if timing:
        if timing["hip"]:
            b = timing["rec_pct"] >= RULE_SHARE
            lines.append(f"(b) timing log ({timing['gpu_name']}, {timing['tokens']} tokens): gdn recurrence "
                         f"{timing['rec_pct']:.1f}% of {timing['gpu_ms']} ms (rule >= {RULE_SHARE:g}%): {'holds' if b else 'fails'}")
        else:
            lines.append(f"(b) timing log is from {timing['gpu_name']}, not a HIP card: gdn recurrence {timing['rec_pct']:.1f}% "
                         f"there; not the gfx11 chunked path, not applicable")
    if a is False:
        verdict = "close chunk 64 (the scan is WMMA-bound; chunk 64 adds ~12% WMMAs)" + (
            "; (b) holds, so only the column split (more blocks per head) is left to consider" if b else "")
    elif a and b:
        verdict = "build chunk 64 (both parts of the rule hold)"
    elif a and b is False:
        verdict = "close A31 (latency-bound, but the recurrence is under 6% of the prompt: halving it gains < 3%)"
    elif b is False:
        verdict = "close A31 (the recurrence is under 6% of the prompt)"
    else:
        verdict = "need both numbers: (a) from the stage bench with STRATA_GDN_CHUNKED=1 on gfx11, (b) from a timing log or --prefill-tok-s"
    return lines, verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", help="stage bench output file")
    ap.add_argument("--timing", help="engine log with a 'strata prefill timing:' line")
    ap.add_argument("--prefill-tok-s", type=float, default=None, help="prompt speed at 32K for the estimate of (b) from the stage bench")
    ap.add_argument("--layers", type=int, default=36, help="GDN layers of the model (Flash-Next: 36 of 48)")
    a = ap.parse_args()
    stage = parse_stage(open(a.stage).read()) if a.stage else {}
    timing = parse_timing(open(a.timing).read()) if a.timing else None
    lines, verdict = decide(stage, timing, a.prefill_tok_s, a.layers)
    for ln in lines:
        print(ln)
    print("verdict:", verdict)


if __name__ == "__main__":
    sys.exit(main())
