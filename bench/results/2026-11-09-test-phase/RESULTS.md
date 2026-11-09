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
