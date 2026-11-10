# dev/batch-1 tests (RX 7900 XT, fast-hip base, 2026-11-10 afternoon)

## Parity / unit tests on the full batch build (29 commits): all PASS

| Test | Item | Result |
| --- | --- | --- |
| q2_fast_parity_test | F7 Q2_FAST | 0 of 584,676 outputs differ (after the explicit-fma fix dd8bf619b; before: 2240 differed) |
| native_multi_parity | F9 ROUTE_FAST | OK |
| iq_multi_parity | F12 IQ_SIGN_FAST | 0 failures |
| hip_prompt_attn_wmma 32768 2048 3 | F22 WMMA_Q4 on gfx11 | 0 failures |
| conversation_disk_tier_test | disk key v2 + F5/F13 keys | 95 checks |
| conversation_cache_test | F1 PARK_TRUNCATE | 4,866 checks |
| verify_ecap_test | F5 | ok |
| route_prior_scope_test | F13 | ok |
| q2_rows_bench (CPU) | F7 | bit-exact, down rows x1.33, gate/up x1.23 |

## Speed: inconclusive (machine too noisy)

Screen 18 (1 pass, agents running) and confirm 23 (3 reps, quiet CPU) both gave intervals far wider than the effects
(base edit 98.2 +- 47.5 tok/s; the same binary ran 76-104 tok/s on edit between reps). Every run started with
< 2 GiB of free RAM in 2 MiB blocks (arena 33 GiB): memory fragmentation, not foreign CPU, dominates. No arm is
decided; nothing is promoted or rejected from these runs.

Confirm 23 means (n=3, % vs base, 95% interval): short / edit
- q2fast +0.2% / +6.3%; routefast -2.3% / +9.2%; signfast -11.3% / +5.3%; fusewait -2.8% / -16.7%;
  fusewait-copy256 -11.8% / +1.1%; top2 -4.8% / +10.2%; top2-fold2 +3.7% / +10.0% (all "no difference").

Also learned: the generated-text hash differs even for the identical daily binary (adaptive tier timing), so the text
check is not an exactness test on this engine; exactness comes from the parity tests above.

Next: rerun tonight after `sudo tools/pancake/os_tune.sh install --hugepages` + reboot (or drop caches + compact right
before the batch), specs 23, 15-kmiss, 12-spec-sampling, 13-lookup, 14-hip-env, 11-conv-cache via replay_turns.py.
