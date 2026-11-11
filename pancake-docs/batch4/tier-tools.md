# Batch 4, tier-tools: capped.sh tier line, queue_run.sh night window and summary, drop_cache.py

Branch `dev/p16b/tier-tools` (from `dev/phase16-academic` 0dd7ef2e8). Tools only: no engine code, no switch, no GPU, no
engine or abtest run. Files owned: `tools/pancake/capped.sh`, `tools/pancake/queue_run.sh`, `tools/pancake/drop_cache.py` (new),
plus the new tests `tools/pancake/test_drop_cache.py` and `tools/pancake/test_queue_run.sh`.

## What changed

- **drop_cache.py (new).** Non-root. Asks the kernel to drop the page cache (posix_fadvise DONTNEED) of the regular files you
  own, at least 4 MiB, under the given directories (default: `~/.cache`, `~/clone-of-d/src`, `~/backup-pancake-*`). Symlinks
  are not followed and files are opened with O_NOATIME. It prints MemFree and the free blocks of buddy order >= 9 (2 MiB
  blocks) before and after. `--dry-run` sends no hint. It never deletes or writes anything.
- **capped.sh.** Prints one stderr line before the start, `capped: tier=arena|mmap keep_gib=N big_free_gib=M`, and exports
  and passes `STRATA_TIER_GUESS`. Tier is `arena` only if keep <= 12 and at least 33 GiB is free in 2 MiB blocks, else `mmap`.
  New `-n` prints the command it would run and waits for nothing. The wait loop, CAP, WAIT_GIB and the scope are unchanged.
- **queue_run.sh.**
  - `--night-only HH-HH`: outside the window it exits 5 at once (no waiting). The window is checked before each spec and again
    before the launch. `23-07` wraps midnight; equal hours mean all day.
  - `--drop-cache`: runs drop_cache.py once, before the first spec that would run.
  - `--reclaim-max X`: passed to every abtest call that does not set its own; default 3.0.
  - `OUTDIR/queue.summary.txt`: one line per spec that ran, `<spec> exit=<rc> runs=<n> minutes=<m>`.
  - Overrides for tests and other tools: `PYTHON`, `ABTEST`, `DROP_CACHE`, `BUSY_PATTERN`.
  - Exit codes: 0 done, 2 usage, 3 another test or engine running, 4 RAM short after the wait, 5 outside the night window.
  - Fixed: usage text is read from the script's own header (it used the path after `cd`, which broke for relative calls).

## Source check

- Sources: MOD_PLAN.md on `dev/phase16-academic`, "Testing: why 45-minute windows" (`--reclaim-max 3.0` for new runs, the
  queue, the drop_cache fallback) and the tier item 4 (record the tier in every result). These are the owner's notes from
  2026-11-11, not papers. Checked against them: the defaults and the tier rule match the text.
- The free-RAM figures (19 -> 32 GiB, 10 -> 13 GiB in 2 MiB blocks) are quoted from MOD_PLAN. They were not measured again here.
- The 3.0 default sits at the edge of the data: abtest.py's own help text says edits were ~8% slower above 3 (n=4). Owner
  decision; flagged, not changed.
- Not checked: whether the engine itself picks the arena by the same 33 GiB / keep 12 rule. capped.sh only guesses it.

## Tests run (CPU only, 2026-11-11)

- `python3 -m pytest tools/pancake/test_drop_cache.py`: 10 passed (temp dirs; owner filter, size boundary, symlinks, no
  modification, dry run, /proc parsers on fixtures, CLI).
- `bash tools/pancake/test_queue_run.sh`: all passed, 55 checks. Covers the dry run, argument errors (exit 2), the real queue with
  a fake abtest (summary lines, resume, a failing spec, the drop-cache step once and first, the night window both ways and
  mid-queue, the busy refusal with the default pattern) and capped.sh with `-n`. The fake `date` gives the hour.
- `bash -n` on the three scripts: ok. shellcheck: not installed on this machine, not run.
- `queue_run.sh -n` on the real `bench/specs` files, and `drop_cache.py --dry-run` on the default dirs (1066 files, 33.4 GiB
  would be hinted; nothing hinted). `capped.sh -n` read the real /proc/buddyinfo: 17 GiB in 2 MiB blocks, so `mmap`.
- Observed, not a failure: while the lead's GPU run was live (`abtest.py bench/specs/25-a8-confirm.json`), the first
  real-run tests stopped at the busy check (exit 3) before anything started. That is why the tests use `BUSY_PATTERN`; the busy
  test still uses the default pattern.
- Not run: any abtest, engine or server; capped.sh without `-n`; drop_cache.py on the real directories.

## Switch and defaults

No engine switch. Behaviour changes for the callers that exist today:
- queue_run.sh now gives every abtest call `--reclaim-max 3.0` (abtest's own default stays 1.0 for direct calls).
- queue_run.sh writes `queue.summary.txt`; the dry-run lines show the full command (`.venv/bin/python tools/pancake/abtest.py ...`).
- capped.sh prints one more stderr line (abtest.py sends that stream into the engine log) and sets STRATA_TIER_GUESS. Nothing
  else changes.

## Risk and limits

- drop_cache.py only hints. Mapped and dirty pages stay. The effect is whatever the kernel does with the hint; it is a
  fallback for `sudo os_tune.sh compact`, not a cure.
- STRATA_TIER_GUESS is set in the capped.sh environment, but abtest.py starts capped.sh with its own env, so abtest cannot read
  the variable. To record the tier in results, abtest would have to read the capped line from the engine log. That file is
  harness-core's (abtest.py), so I did not change it.
- The night window stops a queue between specs and before a launch. A spec that is already running is not stopped at the end
  of the window.
- The busy pattern is the same three patterns as before (abtest.py, restore_check.py, `bin/strata `).

## Next

- abtest.py: record the tier from the capped line in each result (harness-core, after this merges).
- Owner: decide the cron line (`queue_run.sh --night-only 23-07 --drop-cache ...`) and whether 3.0 stays the default.
- Push: not done (the brief says never push). Commit is local only.
