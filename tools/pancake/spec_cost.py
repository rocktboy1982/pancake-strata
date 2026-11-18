"""spec_cost.py - hours of an abtest spec on the box, from measured per-start and per-request seconds.

    python3 tools/pancake/spec_cost.py SPEC.json ... [--confirm-share 0.3] [--quick-idle]

Model (constants measured from the finished chains N, G, F, 6, 8a ... by tools/pancake/chain_time.py and the per-request engine
lines of 122 abtest result files; TEST_LEDGER.md):
  START_S   : seconds per engine start that are not a measured request (load, warm-up request, stop, quiet gate) = 70
  CASE_S    : seconds per request of a case (prompt / prompt_tps + gen / decode_tps)
  screen    : one start per arm (+ the drift-check start), 2 requests per short / edit / pep8 / longdec case, 1 per long* case;
              long* cases that are not longdec* are dropped unless the spec says "prefill": true (abtest.py)
  confirm   : --block 3 --warmup 1 --reps 5: 2 starts per arm, 4 passes (1 warm-up + 3 measured, rounded up) of every case, 1 request
              per case and pass; --confirm-share of the non-base arms get one (the rest are dropped at the screen)
  idle      : 3 minutes before every step (old) or 0.5 minute (--quick-idle, after the first one)
Nothing is measured here; the numbers are an estimate and say so.  Sizes 64k and 100k are the least certain (see CASE_S; the 64K figures differ per model: flashnext 27 s, Qwen3.6 101 s).
"""
import argparse
import json
import math
import sys

START_S = 70.0
# seconds per request.  Measured: short, edit, long1k, long3k, long, long32k, longdec10k, longdec32k (ledger result files).
# The 64K figures are PER MODEL (TEST_LEDGER.md T-943): the 27 s of long64k and the 101 s of longdec64k were never one case twice, they are two models.
# flashnext IQ2_XS (CASE_S, the daily preset): long64k 64000 tokens / 2405.7 tok/s = 26.6 s (T-822 trend point; T-614 1980 tok/s = 32 s);
# longdec64k = 26.6 + a 12.2 s reply (the q36 reply time stands in: decode at 64K on flashnext is NOT measured) = 38.8 s.
# Qwen3.6-35B-A3B Q4_K_XL (Q36_S, presets named qwen36*): 65149 prompt tokens at 643 tok/s = 101.3 s + 256 tokens at 21.0 tok/s = 113.5 s (spec 98, T-?; result file
# bench/results/2026-11-16-chain6-screens/win6-98-pass2-q36-xattn.jsonl), 32K: 31418 tokens at 1015 tok/s = 31.0 s + 256 at 41.2 = 37.2 s.
# long10k = measured longdec10k prefill; long100k / longdec100k: NOT measured, 101 s x (100 / 64)^1.4 = 190 s (extrapolation from the q36 figures, so an upper bound for flashnext).
CASE_S = {"short": 5.6, "edit": 9.0, "pep8": 9.0, "long": 11.5, "long1k": 1.9, "long3k": 3.1, "long10k": 5.9, "long32k": 25.2,
          "long64k": 26.6, "long100k": 190.0, "longdec10k": 8.1, "longdec32k": 17.9, "longdec64k": 38.8, "longdec100k": 202.0,
          "longdec128k": 330.0}
Q36_S = {"long32k": 31.0, "longdec32k": 37.2, "long64k": 101.3, "longdec64k": 113.5}


def case_s(spec, c):
    """Seconds of one request of case C for the model the spec's base preset runs."""
    if "qwen36" in str(spec.get("base", "")) and c in Q36_S:
        return Q36_S[c]
    return CASE_S.get(c, 10.0)


LONG_DEC = ("longdec10k", "longdec32k", "longdec64k", "longdec100k", "longdec128k")


def is_long(c):
    return c.startswith("long")


def screen_hours(spec, drift=None):
    arms = len(spec["arms"]) + (1 if (drift if drift is not None else spec.get("drift_check")) else 0)
    cases = spec.get("cases", ["short", "edit", "long"])
    if not spec.get("prefill"):
        cases = [c for c in cases if not is_long(c) or c in LONG_DEC] or ["short"]
    per = sum(case_s(spec, c) * (1 if is_long(c) else 2) for c in cases)
    return arms * (START_S + per + CASE_S["short"]) / 3600   # + the single warm-up request


def confirm_hours(spec, winners):
    cases = spec.get("cases", ["short", "edit", "long"])
    per = sum(case_s(spec, c) for c in cases) * 4
    return (1 + winners) * 2 * (START_S + per) / 3600   # base + the winners, 2 starts each


def spec_hours(spec, share=0.3, with_confirm=True):
    s = screen_hours(spec)
    n = max(1, len(spec["arms"]) - 1)
    c = confirm_hours(spec, max(1, math.ceil(n * share))) if with_confirm else 0.0
    return s, c


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--confirm-share", type=float, default=0.3)
    a = ap.parse_args(argv)
    tot = [0.0, 0.0]
    for p in a.specs:
        sp = json.load(open(p))
        s, c = spec_hours(sp, a.confirm_share)
        tot[0] += s
        tot[1] += c
        print(f"{p.split('/')[-1][:44]:44} arms {len(sp['arms']):2d} screen {s:5.2f} h  confirm(est) {c:5.2f} h")
    print(f"{'total':44}          screen {tot[0]:5.2f} h  confirm(est) {tot[1]:5.2f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
