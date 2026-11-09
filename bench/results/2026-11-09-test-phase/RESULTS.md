# Test phase results (RX 7900 XT, fast-hip preset, abtest.py 3 reps A B B A, 95% intervals) - 2026-11-09

| Group | Arm | Short chat | Code edit | 11.7K read | Quality | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| 01 deferral | base | 132.6 +-6.2 | 116.6 +-2.5 | ~1,920 | - | - |
| 01 deferral | STRATA_VERIFY_DEFER_CPU=1 | 130.0 (-1.9%, n.s.) | **135.0 (+15.8% +-3.9)** | n.s. | PASS | **into fast-hip** |
| 01 deferral | + _TOP=2 (rank guard) | 123.4 (**-6.9% +-3.5**) | 140.0 (+20.1% +-3.3) | n.s. | PASS | opt-in; short-chat loss to optimize (two pool passes when few misses) |
