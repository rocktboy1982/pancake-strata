# Test phase results (RX 7900 XT, fast-hip preset, abtest.py 3 reps A B B A, 95% intervals) - 2026-11-09

| Group | Arm | Short chat | Code edit | 11.7K read | Quality | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| 01 deferral | base | 132.6 +-6.2 | 116.6 +-2.5 | ~1,920 | - | - |
| 01 deferral | STRATA_VERIFY_DEFER_CPU=1 | 130.0 (-1.9%, n.s.) | **135.0 (+15.8% +-3.9)** | n.s. | PASS | **into fast-hip** |
| 01 deferral | + _TOP=2 (rank guard) | 123.4 (**-6.9% +-3.5**) | 140.0 (+20.1% +-3.3) | n.s. | PASS | opt-in; short-chat loss to optimize (two pool passes when few misses) |
| 02 tier | base (blocking tier, deferral on) | 130.6 +-4.0 | 136.8 +-2.9 | 1,916 | - | - |
| 02 tier | --adapt-async 1 | -5.7% +-3.1 | -2.5% +-1.8 | n.s. | - | rejected (fixed cost of the async tier) |
| 02 tier | async + STRATA_ADAPT_KMISS=2,8 | -10.1% | -5.4% n.s. | n.s. | - | rejected on async |
| 02 tier | async + STRATA_ADAPT_SWAPS_AUTO=1 | -9.6% | -0.8% n.s. | n.s. | - | rejected on async |
| 02 tier | STRATA_IDLE_ADAPT=4 | n.s. | n.s. | n.s. | - | opt-in (benchmark has no idle gaps) |
| 02 tier | STRATA_ADAPT_PERSIST=0 (old thread) | n.s. | n.s. | n.s. | - | dan-9f's persistent default stays |
| 02b optimization | KMISS=2,8 on the blocking tier | -6.4% (acceptance 0.74 -> 0.69: different text) | **+2.0% +-1.6** | n.s. | - | opt-in; re-judge on real agent traces |
| 02b optimization | KMISS=3,8 on the blocking tier | n.s. | n.s. | n.s. | - | opt-in |

Profiling note (02): the base hit rate on this benchmark is already ~89% (the model assumed 81%: the benchmark repeats its
prompts, the cache stays warm), so admission policies have little room here; and any placement change alters the last bits
of expert arithmetic (GPU vs CPU kernels), so the generated text and the draft acceptance change - short-chat speed between
placement arms is confounded by different text.  Code edits (acceptance stable at 0.94-0.95) are the fair comparison.
| 03 CPU pool | STRATA_POOL_TAIL4 / MWAITX / LAZY_PARK / Q2_PREFETCH=1024 | n.s. | n.s. | n.s. | - | opt-in (no measurable gain) |
| 03 CPU pool | STRATA_POOL_FUSE_PHASES=1 (first run) | -8% | +23% | - | not checked | INVALID: stale mtasks_ bug (wrong down-row ranges); fixed |
| 03b after fix | FUSE_PHASES=1, no threshold | -1.3% n.s. | **+2.5% +-1.7** | - | PASS | **into fast-hip** |
| 03b after fix | FUSE_MIN=8 / 12, FUSE_MIN=8 + TAIL4 | n.s. | n.s. / n.s. / +1.6% n.s. | - | PASS | threshold kept in code, default 0 |

Note (03): two whole runs came out ~25% slow (base r2, q2-prefetch r1) with no foreign CPU but kswapd / kcompactd active:
memory compaction, not the arm.  The harness's foreign-CPU monitor does not see it; such runs show as large intervals.
| 04 GPU launches | COMBINE_POLL1 / EXPDEC_SWIGLU / ROUTER_FUSE / IC_WARM_MB=32 / EXPDEC_DOWN / DEVICE_SPIN | +0.4..+4.3% n.s. (base run hit by reclaim) | -10..+0.3% n.s. | 0 | PASS (all 6) | retest the best three |
| 04c retest (new harness, `2026-11-10-test-phase/`) | EXPDEC_DOWN=1 | **+1.7% +-0.5** | +5.4% +-7.3 n.s. | - | PASS (04) | **into fast-hip** |
| 04c retest | COMBINE_POLL1 / DEVICE_SPIN | -0.6% n.s. | +1.3% / +1.4% n.s. | - | PASS (04) | opt-in (no gain) |

Harness change before 04c: the 1-minute load-average wait is gone (it was the finished server's own threads decaying:
45 of 04's 70 min); runs during memory reclaim / compaction (kswapd, kcompactd > 1% of a core) wait or are repeated.
04c: 12 runs in 13.4 min (67 s per run, was ~200 s), no noisy run, intervals tighter (base short +-0.8 vs +-10.3).
| 04b / 04d / 04e presum (`2026-11-10-test-phase/`) | STRATA_CPU_PRESUM=1, + PRESUM_VRAM=1 | not decided (runs bimodal, then noisy) | not decided | not decided | PASS (both) | opt-in: no gain shown in three tries; not retried |

Note (04b-04e): the ~25% slow runs line up with the CPU clock in the new APU telemetry (slow 3.43-3.67 GHz, normal
3.9-4.1 GHz; GPU clocks equal).  04e also hit the oomd load kills (fixed in a595222ca: THP only when the free 2 MiB
blocks hold the whole arena; os_tune.sh compact_unevictable_allowed=0) and other sessions' CPU load (most runs repeated).
| 05s screen (`2026-11-10-test-phase/`, 1 pass, every attempt counted) | README small-flag stack | **-20%** | -1..+1% | - | - | drop (the stack is slower on the 7900 XT) |
| 05s screen | --lookup-max-t 12 | -1.5..+2% | **+10..+12%** | - | (text differs) | promote -> 05c confirm + quality gate |
| 05s screen | --spec-min-p 0.7 / LOOKAHEAD=0 / ROUTED_STREAM+COPY_WAIT_DEDUPE / MMVF_ROWS=1 / GDN_PP=2 | -2..-20% | -19..+4% | - | - | drop |

Note (05s): the harness's own verdict dropped every arm because the base had no clean run (caveman-proxy and another
session's python3 above the 25%-of-a-core foreign limit on all three attempts); noise did not move the base numbers
(41% vs 74% foreign: same tok/s).  Fix: when every attempt is noisy, abtest keeps the quietest one instead of none.
Arms with a single slow attempt at 3.2-3.6 GHz CPU (routed-dedupe, mmvf-rows) carry the CPU-clock effect, not the arm.
| 05c confirm (`2026-11-10-test-phase/`, 4 reps) | --lookup-max-t 12 | -3.5% +-3.6 n.s. | **+8.8% +-6.8** | - | **PASS** (events p 0.70) | **into fast-hip** (Avalon report; credited) |
| 06 screen (`2026-11-10-test-phase/`, base = fast-hip + lookup12) | turbo (STRATA_ROUTE_PRIOR=0.5) | -2.0% | **+32.4%** | - | (lossy) | promote -> 06c confirm + quality gate |
| 06 screen | turbo + ROUTE_PRIOR_DEPTH=0.75 | -0.3% | -1.3% | - | - | drop |
| 07 screen | --mtp-q4 head | - | - | - | - | **fails to start on HIP** ("mtp: writing the draft head": cudaMemcpy of the Q4_0 head fails, exit 1); bug, not tested |
| 07b screen | STRATA_MTP_EAGER=1 | -0.6% | +15.4% | - | - | promote -> 06c confirm |
| 07b screen | STRATA_CACHE_MESSAGE_BOUNDARY=1 + --prompt-cache-tail / STRATA_MTP_HIP_PRIO=1 | -2.3% / -17.7% | -11.0% / +14.5% | - | - | drop |
| 06c confirm (4 reps) | turbo (ROUTE_PRIOR=0.5) on fast-hip + lookup12 | +4.1% n.s. | +29.7% +-53.6 (3 of 4 runs +45%: 194-199 vs 130-140; one whole run slow) | - | **INCONCLUSIVE** (2.65% tokens differ vs stock 1.8%, diff +0.83 pts, CI90 upper 1.27 vs bound 1.25; 0 large-margin events) | stays the opt-in lossy `turbo` preset; turbo-hip synced with the adopted items |
| 06c confirm | STRATA_MTP_EAGER=1 | +8.6% n.s. | +1.6% +-5.2 n.s. | - | PASS | opt-in (the screen's +15% was noise) |
| 07c short check (2 reps) | STRATA_MTP_HIP_PRIO=1 | -11% n.s. (runs bimodal: 85 or 128) | -1.1% n.s. | - | - | drop: no edit gain to protect, so no acceptance-gated priority switch (the 07b screen gain was noise, like MTP_EAGER) |
