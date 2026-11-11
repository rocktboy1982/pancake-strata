# Batch 3, `datapath`: loader, PCIe, NVMe (16.12), cold reads (16.6), miss-path quantization (16.5)

Branch `dev/p16/datapath` (from dev/phase16-academic 3f4ba97cf), 2026-11-11. No GPU, no engine, no model loaded: model files were read for
about 670 MiB in total (read-only probes, `posix_fadvise DONTNEED` before each) and 24 MiB (entropy windows). Hardware facts below are sysfs reads.

## Verdicts

| Item | Source check | Verdict | Code / tool | Modeled gain |
| --- | --- | --- | --- | --- |
| A52 / item 38 O_DIRECT arena loader | ServerlessLLM OSDI'24: **mismatch of setting** (see 1) | code sound and compiles; **default-on here**, gain on this disk unproven | `direct_read_probe.py` (new) | start 30.6 s -> >= 16.9 s floor, probe shows no gain yet |
| A53 IOMMU guardrail | SOSP'24 paper concerns NIC/NVMe; no GPU number | ok as information only | existing `serve/iommu_check.py`, verified on this box | 0 |
| ReBAR (items 50-52), pinned memory, huge pages (85), oomd / fragmentation | kernel docs and systemd source: **ok** (see 3); fork "+5-7% decode" **does not fit this box's model** | nothing to build offline; item 85 has an ordering problem | none | <= 1.3% short, <= 2.9% edit even at -6% CPU miss time |
| 4.0.1 conversation cache flags | replay arithmetic reproduced; **`min-tokens 8192` carries the gain; 2048 MiB is only a floor guard (needs >= ~1.8 GiB)** | tool + spec change | `conv_cache_sim.py` (new), `11-conv-cache.json` arms | 158 s of 162.7 s cold re-read time on the 11-09 session (97%), upper bound |
| A32 stream experts for large prefill ubatches | paper is not what the plan says (see 4) | **already in the engine**; closed | code read only | PCIe floor table in 4 |
| A28 ISTA Coder variant | GSQ/RCO abstracts: no decode-speed claim, the -10..+20% is a model-card figure | not built (needs 29.6 GB download + D space; ask first) | none | unknown |
| A29 entropy check | reproduced on both files | **closed**: best saving 12.0% < 20% | existing `expert_entropy.py` | free-decode upper bound 2.6% short / 5.8% edit, decode cost makes it negative |

## 1. A52 / item 38: direct loader (5c071d364, already an ancestor)

**Source.** ServerlessLLM (OSDI'24, arXiv 2401.14351), read from the PDF: loading-optimized format + chunk pool; testbeds A5000 / A40, PCIe 4.0 NVMe
(RAID0 12 GB/s), FP16 safetensors/PyTorch baselines; 6x / 8.2x faster than PyTorch, 3.6x / 4.7x than Safetensors; breakdown over a read-by-tensor
baseline: bulk reads, direct I/O 2.1x, multi-thread 2.3x, pinned memory 1.4x, pipeline 1.5x (16 MB chunks, 4 cores saturate). Only direct I/O and
threads can transfer: our baseline already reads in bulk with 6 threads, and our destination is the host arena, not GPU memory (no pinned-chunk pool, no
pipeline). fastsafetensors (2505.23072): copies to the device, P2P DMA; not applicable (hipFile/GDS rejected, misses come from RAM).

**Hardware floor.** The drive is a Kingston SNVS2000G (NV1 2 TB, DRAM-less, PCIe 3.0 x4; sysfs: 8.0 GT/s x4). Kingston's datasheet: up to 2,100 MB/s
sequential read (https://kingston.com/datasheets/SNVS_us.pdf); one review measured 1,613 MB/s (https://ssd-tester.com/kingston_nv1_2tb.html). The
arena is 33.02 GiB = 35.45 GB: **16.9 s at the datasheet figure, 22 s at the review's**. The plan's "28-65 s -> 10-14 s" needs 2.5-3.5 GB/s, which this
drive cannot deliver. The engine's own buffered loads (11-09 log, 44 starts): 1.06-1.14 GiB/s = 29-31 s, one outlier 0.43 GiB/s (77 s). So the most a
perfect loader saves is ~13 s at best (30.6 -> 17), ~8 s at the review figure.

**Probe** (`tools/pancake/direct_read_probe.py`, tested): 3 runs on the IQ2_XS and Q2_0 shards (ntfs3, cold, `resident 0%`), 64-128 MiB windows:
buffered 6 threads 0.66-0.85 GiB/s; direct 16 threads 1.07-1.27 GiB/s (O_DIRECT is accepted on ntfs3: `ran direct`); direct 32 threads 0.65 on a
64 MiB window (8 pieces for 32 threads: window too small). Projected arena read 26-31 s. The windows are too short to prove a steady state (the cap is
512 MiB per run), and the machine was compiling at the time; run it with `--window-mib 256 --chunk-mib 2` in a quiet moment, and `load_time.py` in the GPU
window for the real number.

**Code review** (`load_experts_gguf_direct_linux`, expert_source.cpp:5900): per-thread fd, 4 KiB-aligned bounce buffer, window rounded to sectors,
EINTR retry, EINVAL on a direct fd -> `leave_direct` and retry, short read -> whole direct load refused and the buffered path rereads (no half arena),
16 experts (~8 MB) per read, `loader_pressure_check()` per chunk. I found no defect. `STRATA_DIRECT_LOAD_FAULT` test hook exists.

**Compile.** HIP configure (gfx1100, `-DCMAKE_HIP_COMPILER=/opt/rocm/core-10.0/lib/llvm/bin/clang++`), objects only: `src/core/expert_source.cpp.o`,
`src/core/pinned.cu.o`, `tests/core/expert_layout_test.cpp.o` build with 0 errors (warnings: ignored hipError_t, pre-existing). **Not linked or run**: the test
needs the whole `strata_engine` (hundreds of HIP objects). Run `cmake --build build-hip --target expert_layout_test && ctest -R expert_layout` in the GPU window.

**Finding: it is the default on this box, not an opt-in.** `experts_unbuffered` (pinned.cu:1221) says "unbuffered unless the file cache can keep the
files beside the arena": MemAvailable ~49 GiB - 33 arena - 4 margin = 12 GiB < the shards' bytes -> `direct`, no env needed. So the first daily-preset start
after this branch is the first start through the new loader. `STRATA_UNBUFFERED_LOAD=0` restores the buffered path. The value is memory pressure (no
33 GiB page-cache copy charged to the capped scope), not speed; the oomd evidence below is unmeasured for the new loader (needs `load_time.py`: PSI full,
`cached_peak_gib`, `STRATA_VERIFY_ARENA` checksum equal in both modes).

## 2. A53 IOMMU guardrail

`serve/iommu_check.py` and `os_tune.sh check` run here: GPU 0000:12:00.0 is in an `identity` group (passthrough), no `iommu=` on the command line, the
note does not fire (`state()` -> mode passthrough). The SOSP'24 paper is about strict-mode translation cost for NICs / NVMe at 100+ Gbps; nothing measured on
a GPU at 12 GB/s, and the code says only "can cost throughput". Fine as information. 10 tests pass.

## 3. ReBAR, pinned memory, huge pages, oomd and fragmentation

Facts (sysfs, this box): Ryzen 7 5700G (PCIe 3.0, DDR4, 16 MiB L3); root port 00:01.1 links 8.0 GT/s x16, the card's internal ports 16 GT/s (the
endpoint's own `current_link_speed` says 16 and is not the host link); BAR0 = 32768 MiB (ReBAR on); THP `enabled=madvise`, `defrag=madvise`; `nr_hugepages` 0;
`compact_unevictable_allowed` 0, `compaction_proactiveness` 50 (os_tune applied); 14 GiB free in >= 2 MiB blocks right now (it drifts with page cache).
Link efficiency: `time_ceiling.py` prices a 1.4 MB promotion at 117 us = 12.0 GB/s of 15.75 GB/s theoretical (76%).

- Kernel doc (docs.kernel.org vm.html): `compact_unevictable_allowed` default 1 lets compaction examine mlocked pages; 0 keeps them in place;
  `compaction_proactiveness` 0-100, default 20, background. **os_tune.sh header: ok.** transhuge doc: `defrag=madvise` makes a MADV_HUGEPAGE region
  stall at fault time in direct reclaim + compaction: **matches the engine's "transparent huge pages requested (MADV_HUGEPAGE)" log line and item 85's reasoning.**
- systemd source (oomd-util.c): memory pressure is PSI **full**, `avg10 > limit`; `oomctl` here: user@1000.service, limit 50%, 20 s. **Header ok;
  `load_time.py`'s psi_full is the right gauge.**
- Item 85 (collapse THP after the load): `MADV_COLLAPSE` exists (ignores `enabled`, rejected under MADV_NOHUGEPAGE) but it migrates pages. The header's
  own finding is that migrating registered pages storms KFD (evict/restore queues). So collapse must run BEFORE `hipHostRegister`, while pin-by-need registers
  all 11,620 ranges before the read. That is a pipeline reorder, not a flag: not built.
- Fork claim "hugetlb +5-7% decode": the CPU miss path is 5.2 ms of a 24.1 ms short window and 17.4 of 36.3 ms on an edit (time_ceiling calibration, alpha 0.25).
  A 6% faster miss path (TLB walks are the only mechanism) is 0.31 ms = **1.3% short, 1.0 ms = 2.9% edit**; +5-7% total decode needs the CPU path ~29%
  faster. The claim came from forks whose decode is CPU-bound; it does not transfer to this model. `os_tune.sh install --hugepages` stays an optional, root, boot-time
  step; it was not run.
- ReBAR push (51/52): the combine pull is est. 0.4-0.8 ms per 37.4 ms edit window; removing it entirely is <= 2.1%, below the 2% gate once any cost is paid. Needs the
  item-50 measurement (GPU). Not built.

## 4. 4.0.1 conversation cache and A32

**Method check.** `tools/pancake/conv_cache_sim.py` (new, 12 tests) re-implements the cache from `conversation_cache.hpp` (`make_room`: evict oldest while
`entries >= slots` or bytes do not fit; incoming copy held) and `generate.cpp` (park at switch, `wants(tokens >= min)`), driven by the turn log with
`replay_turns.plan_from_rows`. **Validation:** on the 11-09 session the legacy 1280 MiB baseline skips parks of **1856, 1573, 1285, 1302, 1313, 1320, 1341 MiB; the engine
log says 1857, 1574, 1286, 1303, 1314, 1321, 1342** (within 1 MiB; the log's other 3 skips are from other sessions), and exactly the 7 recorded cold re-reads come out
cold (169.0 s modeled vs 162.7 s recorded). `replay_turns.py --dry-run` reproduces "7 cold re-reads (162.7 s), first reads 2 (56.4 s), recorded prompt 303.0 s".
One difference from `park_replay.py` (offline-tier0): it counts the incoming copy against the slot limit; the engine does not (it was `take()`n). Modeled results (s of prompt time
vs the 1280 MiB baseline; restore 0.9 s per 1257 MiB, 0.414 ms/token, fits above):

| scenario | prompt s | cold | saved |
| --- | --- | --- | --- |
| S0 1280 MiB (the 11-09 run) | 241.1 | 7 | - |
| S1 `--conversation-cache-mib 2048` | 172.8 | 4 | 68.3 |
| S2 min-tokens 8192 at 1280 | 220.4 | 6 | 20.7 |
| S3 the 4.0.1 flags (2048 + 8192) | 82.7 | 0 | **158.4** |
| S4 item-86 RAM rule (3584 MiB at 16 GiB available), no flags | 149.4 | 3 | 91.7 |
| S5 RAM rule 3584 + min-tokens 8192 | 82.7 | 0 | **158.4** |
| S7 RAM rule 3584, 8 slots, no min-tokens | 102.6 | 1 | 138.4 |

**Findings.** (a) The 97% is an upper bound by construction (every cold read becomes a 0.9 s restore); with restores at 2 s / 4 s the saving is 149 / 131 s. Parking
55,424 tokens (1258 MiB) took 706 ms in the log (1.74 GiB/s; the PCIe floor is 0.11 s), so 0.75-0.9 s is realistic. (b) **The 2048 MiB flag is not what saves the time; `min-tokens 8192` is.** On this branch the budget is sized by item 86 (`run-pancake.sh` sets `STRATA_RAM_KEEP_GIB=12`): available after the load is ~49 - 33 = 16 GiB, so 16 - 12 - 0.5 = **3.5 GiB** (14 GiB available: 1.5 GiB, 15: 2.5, 17: 4.5). Sweep with min-tokens 8192 on the 11-09 session (saved s): 1024 MiB -1.4, 1280 20.7, 1536 114.7, **1792 and above 158.4**; without min-tokens: 1536 45.6, 2048 68.3, 2560 and above 91.7 (the 4-slot limit, not the bytes, then binds: 41 side requests). So 2048 MiB is a floor guard when the RAM rule prints less than ~1.8 GiB (the 11-09 failure, 0.5-4.8% over a 1280 MiB budget) and a cut when it prints more; the first spec arm to read is the `conversation cache: N MiB` start line of `base`. With a main agent and a sub-agent both over 8K tokens taking turns, 2048 MiB holds one of the two and every switch is cold (test `test_two_large_conversations_need_room_for_both`). The replay's arms are now base, min8192, cache2048-min8192 (spec edit). (c) The interleaved side requests in that
session (55-75 token decodes, 3-4K prefill probes) look like the benchmark harness, not opencode (offline-tier0 RESULTS); the cold re-reads are the harness's doing plus the 1280 MiB budget miss
(0.5-4.8% over). The first 25 turns of that agent conversation, before the interleaving, were warm. The saving is real but conditional on that traffic shape. (d) min-tokens costs the small side
conversations their reuse: +1.3 s over the whole session here (3-4K probe repeats).

**Floors for reading (turn log, 878 turns, `prompt_ms` by tokens read):** 2,048-3,072 tokens: median 2.47 s (n=210); 4,088: 2.60 s; 11,978: 6.25 s; 59,052: 23.3 s (2.5K tok/s).
Cold reads are compute-bound (~2.5K tok/s). The 2-4K delta reads are almost flat in size: the model is the PCIe pass. The pinned non-resident experts are 18.45 GiB (engine log, 100% of
the non-resident ones; 10,879 of 24,576 pairs on the GPU): **19.8 GB per chunk / 12 GB/s = 1.65 s**, so chunk C is capped at C / 1.65 s tokens/s: 1,024 -> 620, 2,048 -> 1,241, 4,096 -> 2,481,
8,192 -> 4,962 (uniform routing: 99.4% of the 512 experts are touched by 256 tokens). A 2-3K delta (the 90.8% case) therefore costs ~2.5 s however short it is; the levers are the CPU share
(A30, `agentloop`) and more resident experts during prefill, not the cache. **A32 is already implemented**: `prefill.cpp` streams every non-resident expert for chunks >= `stream_all_min()`
(1024, measured on the 5070: 1,500-token prompts 621 -> 785 tok/s), `STREAM_AHEAD` keeps the copy engine busy through the attention halves, the preset sets `STRATA_PREFILL_AUTO_MAX=32768`
(a 32K cold read is one chunk: PCIe 1.65 s of ~12.7 s) and layer-major exists. The plan's source arXiv 2606.10493 is "Achieving Cloud-Grade SLOs for Local MoE Inference through CPU-GPU Hybrid Design"
(OSDI'26; stream-loading prefill 1,200 tok/s at 32K on dual-socket CPUs + consumer GPUs): our measured 2,580 tok/s at 32K is already above it. A32 closes.

## 5. A28 and A29 (16.5)

**A29** (`expert_entropy.py`, re-run today on both GGUFs, 12 tensors spread over layers 0-47, 12 MiB each): Q2_0 (18 B blocks): byte entropy 7.32 of 8, crumb 1.843 of 2, best real
compressor zstd -19 8.0%, per-position bound 11.7%; IQ2_XXS / IQ2_S / IQ1_M: 7.95-7.97 of 8, compressors 0%, bound 3.3% (the constant fp16 scale byte). Same numbers as batch 2.
Modeled with `time_ceiling.py` (trace bench/prefetch/routing.bin, 219 windows, 10,909 slots; it reproduces runs.txt: 346.9 CPU misses/window = 5.2 ms): a 12% cheaper CPU miss path with FREE
decode moves the window 24.08 -> 23.46 ms (-2.6%) short and 36.28 -> 34.19 ms (-5.8%) edit. The decode is not free: entropy decoders typically run ~1-2 GB/s per core (general figure, not measured here) against the ~6 GB/s per core the
DDR4 feed (51 GB/s peak, 8 cores) offers the expert dot, so the path would be slower; and the daily IQ2_XS file has 0-3%. **Closed** (also for wire compression of promotions: no HIP decoder).
**A28:** GSQ (2604.18556, Alistarh group) is scalar 2-3 bit PTQ on Llama-3.1 and Kimi-K2.5, no speed claim; RCO (2605.00649) is budget-constrained assignment, no speed claim. The Coder variant's
"256 of 512 experts, ~3.5 bpw, -10..+20% decode" is a model-card figure. Needs a 29.6 GB download onto a 94%-full volume (47 GB free): not done without the owner's say-so.

## Tests run (all `nice -n 19`, `systemd-run --scope MemoryMax`)

- `python3 -m pytest -q tools/pancake/test_direct_read_probe.py tools/pancake/test_conv_cache_sim.py tools/pancake/test_replay_turns.py tools/pancake/test_load_time.py
  tools/pancake/test_expert_entropy.py serve/test_iommu_check.py` -> **55 passed** (27 new).
- CPU configure + build: `conversation_memory_test` (67 checks), `conv_cache_test` PASS, `conversation_cache_test` (4,889 checks), `direct_file_async_test` PASS: built and run, all pass.
  `io_priority_test` skips (Windows only). The Python budget rules in `conv_cache_sim.py` use the same numbers as `conversation_memory_test.cpp`.
- HIP compile-only objects as in 1. Not run: expert_layout_test, load_time.py, the replay itself, anything with a GPU.
- `build-datapath` / `build-datapath-hip` are symlinks to `~/.cache/strata-p16-datapath-build*` (ext4): on purpose, so no build writes hit the ntfs3 mount after the 2026-11-10 stuck writeback. They are
  untracked and not in `.gitignore`: do not `git add -A`.

## Risk and next

1. Before the first engine start on this branch: decide whether the O_DIRECT loader stays the default (it is, by the room rule) or `STRATA_UNBUFFERED_LOAD=0` goes into the preset until `load_time.py` has
   shown PSI full and cached_peak_gib lower with the same checksum.
2. GPU window, in order: `direct_read_probe.py --window-mib 256 --chunk-mib 2` (quiet disk), `load_time.py --reps 3`, `replay_turns.py bench/specs/11-conv-cache.json` with the three arms.
3. A cheap lever outside my scope: park/restore moves 1.74 GiB/s against a 12 GB/s link (fresh 16 MiB segments, page-fault bound?): `STRATA_CKPT_POOL` / hugepages for the parked buffers.
