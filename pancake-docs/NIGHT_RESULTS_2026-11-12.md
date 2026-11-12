# Night queue results (2026-11-11 22:55 to 2026-11-12 02:43) and owner decisions

Source: `/home/dan/night-queue-run/RESULTS.md` (43 of 43 jobs marked done; one job, ttft-avalon-iq2xs, has no usable long-prompt data).
Machine: not fully quiet for the TTFT group (every TTFT job logged a load warning); decode group guard-checked. Clock not pinned.

## 1. Expert deferral, IQ2_XS (17.1) - DECISION: off
Kept decode tok/s (median of kept passes): on 105.7 (01A), 104.7 (06A), 105.7 (07A); off 104.3 (02B), 101.4 (05B), 107.5 (08B); gate auto-closed 95.2 (03C), 89.8 (04C), 89.2 (09C).
Read: on and off within noise. Gate arm 10-15% lower. Decision taken by the owner: deferral off in `flashnext-iq2_xs-fast-hip.json` (key removed, the same as the night queue's "off" arm). Not supported by a measured gain; recorded as an owner decision.

## 2. Expert deferral, Q2_0 (17.1) - DECISION: off
Kept decode: on 91.2 (01A, one run only); off 112.7 (02B), 112.4 (05B), 112.8 (08B), 107.1 (06A); gate 101.5 (04C), 102.9 (03C), 100.8 (09C), 101.8 (07A).
Read: on is about 19% below off, consistent with the earlier finding (off = +7-8% reproduced, larger here). Decision: deferral off in `flashnext-q2_0-fast-hip-adopted.json`. The on arm has one run only; a replicate is still advisable.

## 3. TTFT baseline (17.0, priority 1) - DECISION: rerun tonight
Pancake: 8K 3.59 s, 32K 11.62 s, 64K 22.42 s (engine prompt 2297 / 2798 / 2921 tok/s). Agent session 10 turns: total TTFT 22.98 s, reuse 83.5% on follow-up turns. Load warning flagged. Rerun queued for after tonight's A/B chain (`~/night-queue-rerun-ttft`).

## 4. Agent lever runs (17.0) - DECISION: none adopted (data read)
Total session TTFT over 10 turns (median of complete sessions, 3/3 each):
- control 01A 25.08 s, control 07A 24.84 s (noise between controls about 0.2 s)
- 03C `--conversation-cache-mib 6144` 24.89 s
- 04D `STRATA_PARK_TRUNCATE=1` 24.93 s
- 06F `STRATA_PARK_REWIND_MAX=8192` 26.35 s
- 02B `--conversation-cache-min-tokens 8192` 27.27 s (worse)
- 05E all three combined 28.55 s (worst)
Read: no lever beats the control. Cache-admission levers are slower in this setup.

## 5. Prose gap to Avalon (17.2) - DECISION: improve the code; deep research started
Kept decode, prose: Pancake story 100.6 / tips 109.7 / moe 106.7; Avalon 104.2 / 114.6 / 115.7 (Avalon 3-8% faster on prose). Code: Pancake js 149.2, lru 149.0. Deep research (Avalon verify path, Pancake portability, A/B test design) is in progress: `/home/dan/research_notes/Avalon prose decode gap/`.

## 6. KV ring (17.4) - DECISION: pending (summary only)
Depth table, decode tok/s at 32K / 64K / 120K: static 49152 (02T) 94.6 / 91.7 / 101.0; adaptive (03R) 97.3 / 93.3 / 93.2; adaptive with max-context 65536 (04P) 91.8 / - / -; static preset (01S, 08S) 85.7-92.5 / 76.5-93.3 / 74.0-93.5. Prompt tok/s is flat across arms (about 2810-2920 at 32K). Read: no measurable effect of the adaptive ring at these depths.

## 7. Bundles (17.6)
Pending cells in `pancake-docs/PRESET_BUNDLES.md` updated after decisions 1-2 (fast bundle changes: deferral off).

## Also in progress tonight
A/B chain at 23:00 (`~/bin/tonight.sh`): recheck-kern (edit base and copy256), confirm-kern, confirm-a8, then rank-guard (HIP build, screen tier), then the TTFT baseline rerun.
