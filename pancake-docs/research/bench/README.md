# Pancake benchmark results, 2026-11-11 to 2026-11-12

- `night-queue-run/` the 43-job night queue: RESULTS.md (the report), state.json, queue.log, per-job configs and raw pass logs.
- `night-queue-controls/` the three prose controls (Avalon tail-skip off, Avalon IQ3_KS main, Pancake repeat).
- `night-queue-rerun-avalon/` the Avalon TTFT rerun (only the 8K prompt measured; long prompts failed with ROCm out of memory).
- `abtest/` A/B runs: kern-hip (first and pinned retests), a8 swaps, tonight chain outputs. Stopped partial runs are in `stopped-odd-partial/`.
- `abtest-specs/` the spec copies used for those runs, with expanded config paths.
- `profile/` the short-case Pancake profile summaries and CSV stats (the 35 MB kernel trace is not included).
- `pancake-night-log-2026-11-12.md` timestamped notes for the night.
- `pancake-handoff-2026-11-12.md` the handoff written for a continuation agent.
Plan HTML copies (rendered Pancake development plan, 2026-11-10/11): open in a browser.
Kernel trace: profile/run_kernel_trace.csv.gz (gunzip to read; 121,265 lines, rocprofv3 kernel dispatches for the short-case profile).
