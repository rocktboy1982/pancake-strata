# pancake-strata: status and plan after the reboot (written 2026-11-11 00:40)

## Why the reboot
The ntfs3 mount `/home/dan/clone-of-d` (nvme0n1p5) has had stuck writeback since 2026-11-10 16:26.
- `flush-259:0` and `inode_switch_wbs` are stuck in D state, and the `sync` processes started at 18:22 never return.
- 2.6 GB of dirty pages never drain. The NVMe is idle, so this is a kernel hang, not a failing disk.
- Effect: the engine stalls in balance_dirty_pages, decode runs ~25% slower, and the quality gate dies with "no progress for 60 s".
- Reboot after 08:00, when dan-2b's night GPU slot (02:00-08:00) ends.

## Backup (root disk, ext4, other physical disk): /home/dan/backup-pancake-2026-11-11/

| File | Contents |
|---|---|
| all-refs.bundle | every local branch (git bundle verify: okay) |
| wt1.tgz | untracked files of the main checkout (170), mostly bench/results |
| wt2.tgz | untracked files of Strata-batch1 (77) |
| scratchpad.tgz | dan-99's /tmp scratchpad (run scripts) |
| worktrees.txt, index.txt | lists |
| dan-2b/ | dan-2b was asked to copy its tmpfs results here before the reboot |

## State on GitHub (rocktboy1982/pancake-strata, branch port/rdna3-decode)
- HEAD d251be8f5: the --mtp-q4 fix. The pushes before it are 0765dc938 (os_tune cpu-install) and 3c4a57359 (07d).
- Adopted in fast-hip: VERIFY_DEFER_CPU, POOL_FUSE_PHASES, EXPDEC_DOWN, `--lookup-max-t 12`.

## Test phase status

| Test | Result |
|---|---|
| 05c lookup12 | adopted |
| 06c turbo | opt-in preset (quality inconclusive) |
| 06c mtp-eager | opt-in |
| 07 --mtp-q4 | was a crash: Q4_0 rows are larger than the IQ4_XS head. Fixed; 07e: server starts, text identical, -2%, no use on IQ2_XS |
| 07c/07d MTP_HIP_PRIO | dropped: its slow runs stay at full CPU clock |
| os_tune cpu | CPU at 4.0-4.1 GHz, no slow base runs |
| 08 native build | not done (user stopped the build); ask first |
| 09 desc cache | no gain; stays on (default) |
| 15 Q2_0 vs IQ2_XS | screen: short +8.6%, edit +4.5% (q2_0 with the adopted items); stock Q2_0 edit -13.8%, so the items transfer. 15c confirm and quality gate were RUINED by the disk hang: redo |

## After the reboot, in order
1. Check the mount: `ls /home/dan/clone-of-d/src/Strata`. If it does not mount: `sudo ntfsfix -d /dev/nvme0n1p5`.
2. Check the repo:
   - `cd /home/dan/clone-of-d/src/Strata && git fsck --no-dangling && git status -sb`
   - If it is broken: `git clone` from GitHub, or `git fetch /home/dan/backup-pancake-2026-11-11/all-refs.bundle`; restore results with `tar xzf wt1.tgz`.
3. OS settings (user, with sudo):
   - `sudo tools/pancake/os_tune.sh cpu-install` (CPU clocks, permanent)
   - `tools/pancake/os_tune.sh` check: fresh RAM has large free blocks
4. Rebuild the binary: `cmake --build build-hip -j16 --target strata` (the `all` target includes strata-poly, which does not compile on HIP).
5. Redo 15c, ~20 min:
   - abtest `bench/specs/15c-q2_0.json`, then `quality_gate.py` on it.
   - If it passes: add the 4 adopted items to `configs/pancake/flashnext-q2_0-fast-hip.json`; the owner decides which model is the default.
6. Start the pancake session for the owner's test:
   - `./run-pancake.sh`, port 8080
   - Copilot: OpenAI endpoint `http://localhost:8080/v1`, model `qwen3.8-flash-next-iq2_xs` (or q2_0)
   - GUI: chat `/`, status page `/status` or `/pancake`, plus `serve/web/monitor.html`. Check each one in the browser first.
7. Remaining tests, short ones only:
   - Phase 3 bundle A: global-vs-flat hint, gfx11 barrier, DPP.
   - Phase 3 bundle B: ACTQ_SKIP, FUSE_DECODE on HIP, PARK_REWIND_MAX, POOL_DAG=2, PREFILL_LAYER_MAJOR.
   - 08 native build: only with the owner's OK.
8. Closing:
   - restore checks and `ctest -L gpu`
   - fast-forward main
   - delete only the branches whose code was integrated
   - credits in README §9 and RESEARCH_AND_FINDINGS §4
   - tell dan-9f "done"

## Rules
- Commit by path, never `git add -a`.
- Tell dan-2b before using the GPU.
- Short tests (screen tier).
- Reports in English, chat in Romanian.
