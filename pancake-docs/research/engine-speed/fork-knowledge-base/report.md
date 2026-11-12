# Strata fork knowledge base

Date: 2026-11-10. Tool: `tools/forks/fork_kb.py`, database in the repo, in `data/fork-kb/`
(`forks.sqlite.gz`, an 18 MB compressed snapshot, rewritten on every run). How to run and update it: README, section 3A.

## 1. What it is

A knowledge base that updates incrementally: it holds all direct forks of `Niko1221/Strata`, their own commits
(the ones that are not in upstream) and their READMEs. Each new run queries GitHub only for
new forks or forks that received a push in the meantime, and writes a CSV with only the commits that appeared since the previous
run. For each commit we know at which run it first appeared (`first_seen_run`) and at which run it was last seen
(`last_seen_run`), so commits deleted by force-push are visible too.

The script starts from the CSV export received from the user, with these changes:

- a single SQLite database that grows from one run to the next, instead of overwritten CSVs;
- forks with the same `pushed_at` are skipped (an update takes under a minute instead of 70 minutes);
- READMEs are saved only once per distinct content (167 distinct READMEs across 4,550 forks);
- a fork without a README is no longer an error; there is no longer an extra request at the end of each list;
- a fork with an error is not marked as scanned and is retried on the next run;
- forks with no common ancestor with upstream are handled separately (section 2).

## 2. The problem found: upstream history was rewritten

On the first run, 995 forks returned 404 on `compare`. The reason, confirmed manually: GitHub answers "No common ancestor".
Upstream rewrote its history around the start of October (neither the old tags nor the `rc/0.1.30` and
`ci/checks-green` branches share history with the old forks any more), so 974 forks created before that can no longer be compared.

The solution: for them we take the commits on the fork's branch dated after the fork was created. Among them are also
old upstream commits that the fork synced; these are marked `upstream_history = 1` if they are
written by the upstream owner or appear in at least two such forks. Of 13,006 commits brought in this way,
1,672 are the own work of 66 forks; the rest is old upstream history. 21 forks are left with a plain 404
("Not Found", probably empty or blocked repos), retried on every run.

Limit: an old upstream commit synced by a single fork would be wrongly counted as own work. This is rare, because
upstream commits usually appear in many forks.

## 3. Figures

| Measure | Value |
| --- | ---: |
| Direct forks | 4,550 |
| Forks with their own commits | 407 (340 diverged, 66 with no common ancestor, 1 only ahead) |
| Own commits in total | 43,685 |
| Forks active since 2026-11-01 | 140 (2,957 commits) |
| Forks active since 2026-11-08 | 60 (666 commits) |
| Forks identical to upstream or only behind | 3,216 |

Activity per week (own commits / active forks): 1,682 / 29 (week 38), 20,882 / 71 (39), 6,964 / 124 (40),
4,767 / 132 (41), 3,794 / 131 (42), 3,025 / 115 (43), 2,216 / 110 (44), 355 / 46 (45, partial). The peak was in
the week after the launch; since then the number of active forks is stable (110-130), but each makes fewer commits.

The most active forks since 2026-11-01 (own commits, last commit): Zutfen-LLC/Strata (220, 11-07),
zapabob/Strata (219, 11-10), rocktboy1982/pancake-strata (176, 11-09; our fork), bsorensen110/Strata (173, 11-10),
DamirLukina88/D.L.-AI---Strata (112), eddoursul/Lamina (112), phoenixclyde/Strata (112), qwertie/Strataq (92),
svgop/Strata (92), serboox/strata (85), tyloryang/Strata (84), this-cute-world/Strata (78),
TheHubUser4860938/Strata-Mnemosyne (76), RandoTeam/4gbVramStrata (75), aktalascloud/strata-legacy-gpu (68),
lighttransport/Strata (65).

## 4. Themes since 2026-11-01 (from commit messages)

| Theme | Commits | Forks |
| --- | ---: | ---: |
| Server, API, agents | 953 | 75 |
| Expert cache, offload, routing | 548 | 71 |
| Quantization | 372 | 55 |
| CUDA / NVIDIA | 284 | 54 |
| KV, attention, context | 278 | 57 |
| GUI / app | 255 | 36 |
| HIP / ROCm / RDNA | 238 | 40 |
| Speculative decoding, MTP | 214 | 51 |
| Metal / Mac | 97 | 30 |
| Docker / packages | 95 | 35 |
| DeepSeek | 89 | 29 |
| SYCL / Intel | 73 | 25 |
| Multi-GPU | 16 | 15 |
| Vulkan | 14 | 3 |

The classification is by keywords, so it is approximate: one commit can fall under several themes.

## 5. Relevant for pancake (HIP / AMD, since 2026-11-01)

- **this-cute-world/Strata** (34 HIP commits): `STRATA_FUSE_SHGATE`, the shared expert's gate row moved into
  the router GEMV, with fused sigmoid and scaling. A candidate to read for the decode path.
- **dardeaw/strata-mxfp4-rdna**: exact wave32 QSA kernels for decode and MMVQ fragment reuse on gfx12.
- **storm-ace/Strata**: optional FP16 K/V cache for Qwen3.6 (`STRATA_Q36_KV16=1`).
- **rpcribari/Strata**, **Zutfen-LLC/Strata**: work on gfx906 (prefetch for down, occupancy, W4A4).
- **MyEcoria/Strata**: `strata-flint`, a new engine for the family (Linux, HIP).
- **bsorensen110/Strata** (52): mostly merges from upstream (0.2.13.x), worth following for what upstream brings.

Nothing is adopted without a measured A/B and, for anything lossy, without the quality gate; any adoption goes into
the credits (README §9 + RESEARCH_AND_FINDINGS §4) in the same commit.

## 6. Next steps

1. Periodic update run (for example weekly): `GITHUB_TOKEN=$(gh auth token) python3 tools/forks/fork_kb.py`.
2. Each run's CSV of new commits becomes the input for the next batch of fork notes.
3. To read in code: `STRATA_FUSE_SHGATE` (this-cute-world) and the wave32 QSA kernels (dardeaw), after the
   test phase benchmarks are done.
