# Research index

Start here. Each topic folder has a `report.md` (the conclusions) and a `notes/` folder (the researchers' working files). Claims are labelled [code], [measured], [paper], [derived] or [unverified]; trust [unverified] less.

## Engine speed (how to make Pancake faster)
| Topic | Report | Notes |
|---|---|---|
| `engine-speed/moe-speedup` | `report.md` | `notes/` |
| `engine-speed/speed-ideas-all-forks` | `report.md` | `notes/` |
| `engine-speed/strata-forks-speed-gains` | `report.md` | `notes/` |
| `engine-speed/pain-points` | `report.md` | `notes/` |
| `engine-speed/code-deep-research` | `report.md` (code study, 2026-11-09) | `notes/` |
| `engine-speed/active-forks-nov` | `report.md` | `notes/` |
| `engine-speed/source-commits` | (notes only) | `notes/` |
| `engine-speed/projects-derived` | `report.md` | — |
| `engine-speed/fork-knowledge-base` | `report.md` | — |
| `engine-speed/grouped-technologies` | `report.md` | — |

## Window overhead (Phase 18, the speculative verify window)
| Topic | Report | Notes |
|---|---|---|
| `window-overhead` | `report.md` (start here for Phase 18) | `notes/` (audit, literature, untested switches, profile summary) |

## Avalon (the llama.cpp comparison engine)
| Topic | Report | Notes |
|---|---|---|
| `avalon/prose-gap` | `report.md` (why Avalon is faster on prose) | `notes/` |
| `avalon/upgrades-from-upstream` | `report.md` | `notes/` |

### Papers by region, for Avalon (`avalon/literature/`)
| Region | Report | Notes |
|---|---|---|
| `asia` | `report.md` | `notes/` |
| `canada` | `report.md` | `notes/` |
| `europe` | `report.md` | `notes/` |
| `mit-harvard` | `report.md` | `notes/` |
| `stanford` | `report.md` | `notes/` |
| `us-universities` | `report.md` | `notes/` |
| `pacific-africa-latin-america` | `report.md` | `notes/` |

## Hardware and platform
| Topic | Report | Notes |
|---|---|---|
| `hardware-platform/data-path` | `report.md` | `notes/` |
| `hardware-platform/consumer-platforms` | `report.md` | `notes/` |
| `hardware-platform/llm-runtimes-2026` | `report.md` | `notes/` |

## Plan pages
`plan/` has three rendered HTML copies of the development plan (open in a browser).

## Benchmark results and logs
`bench/` has every Pancake run from 2026-11-11 and 2026-11-12:
- `bench/night-queue-run/` the 43-job night queue (`RESULTS.md` is the report; raw logs in `raw/`).
- `bench/night-queue-controls/` the three prose controls.
- `bench/night-queue-rerun-avalon/` the Avalon TTFT rerun (8K only).
- `bench/abtest/` the A/B runs (kern-hip, a8, tonight's chain). `stopped-odd-partial/` holds an aborted run.
- `bench/abtest-specs/` the spec copies used for those runs.
- `bench/profile/` the short-case profile summary and CSV stats; `run_kernel_trace.csv.gz` is the 36 MB kernel trace, compressed (gunzip to read).
- `bench/logs/` the night log and the handoff note.

## Where the plan points
`MOD_PLAN.md` Phase 17 (night work) and Phase 18 (window overhead, worker split) link into these folders.
