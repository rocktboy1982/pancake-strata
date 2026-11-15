# Retired presets (2026-11-15)

Moved here so nobody takes them for the daily preset. Nothing in `bench/specs`, `tests`, `tools`, `src` or `CMakeLists.txt` referenced them. Restore with `git mv`.

| file | why retired |
| --- | --- |
| `flashnext-iq2_xs-turbo-hip.json`, `flashnext-iq2_xs-turbo.json` | `fast` plus `STRATA_ROUTE_PRIOR=0.5` (the HIP file also has `STRATA_VERIFY_DEFER_CPU=1` and lacks `STRATA_AGENT_COUNTERS`, so the docs' "one line" was out of date). Measured -4..-8% on 2026-11-11, never a default, and its interaction with `STRATA_VERIFY_ECAP` (which caps the same draft-row expert reads) is untested. The daily lossy speed-up is now `STRATA_VERIFY_ECAP=1` in `flashnext-iq2_xs-fast-hip.json` (quality gate PASS, `bench/quality-hip/2026-11-15-ecap`). |
| `flashnext-iq2_xs-hip.json` | a copy of the exact (non-fast) HIP settings with no users; `flashnext-iq2_xs-fast-hip.json` is the preset every spec uses. |

The one daily preset is `configs/pancake/flashnext-iq2_xs-fast-hip.json`, served by `tools/pancake/pancake.service` (`run-pancake.sh iq2_xs fast`, port 11437).
