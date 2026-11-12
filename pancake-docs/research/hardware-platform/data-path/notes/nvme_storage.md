# NVMe / storage path for ML inference (Nov 2022 - Nov 2026), applied to pancake

Scope note. Sources are FAST/OSDI/ATC/EuroSys/SOSP/ASPLOS/VLDB/SIGCOMM papers plus arXiv, kernel docs and LWN. "Lead" = first author or senior group at the named institution; "co" = only among co-authors. Repo facts come from read-only greps of `/home/dan/clone-of-d/src/Strata` (file:line given). Host facts come from `/proc`, `/sys` and `mount` on this machine: no builds, runs or I/O benchmarks. Family tags (F2, F5, F10) follow `/home/dan/pancake-docs/research/engine-speed/grouped-technologies/report.md`. Items already covered there are not repeated: `MADV_SEQUENTIAL` PR #5561, FastStrata `--conversation-disk` 83 s -> 7.2 s, llama.cpp #31469 O_DIRECT pager, ServerlessLLM as a one-line "small: NVMe tier only" (EU report), FlexGen, PowerInfer. They are only cross-referenced.

Host baseline (local observation, not literature):
- Model, pack and PLE shard sit on `/dev/nvme0n1p5`, mounted `ntfs3 (rw,relatime,...,prealloc)` at `/home/dan/clone-of-d`. It is 758 G, 91% full, 74 G free (`mount`, `df`).
- The drive is a Kingston SNV2S2000G (NV2, 2 TB). The link runs at **8.0 GT/s x4 (PCIe 3.0)**, because the Ryzen 5700G only has Gen3 lanes. That caps throughput near ~3.9 GB/s raw. The drive's spec is 3,500 MB/s read ([Kingston datasheet](https://www.kingston.com/datasheets/SNV2S_us.pdf)).
- Block queue settings: scheduler `none`, `read_ahead_kb=128`, `max_sectors_kb=128`, `nr_requests=255`. THP is `madvise`.
- Kernel 7.0.0-43. `/proc/kallsyms` shows `ntfs_iomap_begin`, `ntfs_iomap_bio_read_ops` and `ntfs_readahead` in the ntfs3 module, so the running ntfs3 already has an iomap read path. There is **no `kfd_ais_rw_file` symbol**, which hipFile's fastpath needs (see Q4).
- Last log (pancake-flashnext.log, 2026-11-09 11:18): "expert arena read through the file cache (buffered (not Windows))", then "loaded 33.02 GiB at 0.49 GiB/s". The same log shows "PLE direct read probe: 33218 rows/s (fine)", "MAP_HUGETLB unavailable (needed 16920 2 MiB pages, vm.nr_hugepages=0); transparent huge pages requested (MADV_HUGEPAGE)", and the memory guard reporting "available 52.2 GiB".

## Q1. Why the 33 GiB arena loads at 0.5-1.2 GiB/s, and what the literature says about fast model loading

### Takeaway
The loader is the bottleneck, not ntfs3 or the drive. On Linux, pancake always reads the arena **buffered**: `experts_unbuffered()` returns false with the reason "buffered (not Windows)", and the O_DIRECT arena loader is `#ifdef _WIN32`. So 33 GiB passes through the page cache into a 33 GiB anonymous, mlocked arena on a 57 GB box. That explains the 2.3x variance and the oomd trips. The literature (ServerlessLLM, fastsafetensors, Haas & Leis) agrees on the fix: large aligned direct reads, several in flight, pipelined into the destination. On this Gen3 x4 link that bounds the load at roughly 10-14 s instead of 28-65 s (estimate).

### Cited Findings
- **Repo, arena path.** `src/core/pinned.cu:1160-1163`: on non-Windows, `experts_unbuffered` sets `why = "buffered (not Windows)"` and returns false. `src/core/expert_source.cpp:6288-6291`: `load_experts_direct(..., chunk=8 MiB)` runs only `if (unbuffered && !padded_)`; otherwise `load_experts_ranges` runs with 8 MiB `fread` chunks. `load_experts_direct` in `pinned.cu:~1028` is `#ifdef _WIN32`. The log confirms "loaded 33.02 GiB at 0.49 GiB/s" on the buffered path (local repo/log).
- **Repo, existing Linux O_DIRECT machinery.** `src/platform/direct_file.cpp:264-320` is a pool of `STRATA_IO_THREADS` (default 16) threads, each doing blocking `pread` on an `O_RDONLY|O_DIRECT` fd, with 4 KiB-aligned buffers via `posix_memalign`. It already works on this ntfs3 volume: the PLE probe reports 33K rows/s "fine" (local repo/log).
- **Repo, file tier.** `src/core/expert_source.cpp:915-985` already holds a Linux rule for the *file tier* (RAM_KEEP/mmap mode). It weighs MemAvailable against the read set and runs a direct-vs-cached drive probe (#3237): on a PCIe 5 drive the cached path decoded 3-4x slower, while on a 1.7-2.5 GB/s NVMe the cache was faster. The arena load does not use this logic (local repo).
- **ServerlessLLM** (Yao Fu, Leyang Xue, ... Luo Mai; **University of Edinburgh lead**; OSDI '24). It uses a loading-optimized checkpoint format of sequential chunks and a multi-tier loader with several I/O threads per tier, pipelined SSD -> DRAM (pinned) -> GPU. It reports 10-200x lower end-to-end latency than serverless baselines ([USENIX](https://www.usenix.org/conference/osdi24/presentation/fu); [arXiv 2401.14351](https://arxiv.org/abs/2401.14351)). Secondary summaries cite 3.6-8.2x faster checkpoint loading in microbenchmarks; I did not verify that against the paper's figures ([review](https://www.themoonlight.io/review/serverlessllm-low-latency-serverless-inference-for-large-language-models)).
- **fastsafetensors** (Yoshimura et al.; **IBM Research, industry**; IEEE CLOUD 2025, arXiv 2505.23072). It transfers files asynchronously in parallel to saturate the storage, then instantiates tensors lazily in device memory. Loading is 4.8-7.5x faster on Llama 7/13/70B, Falcon-40B and Bloom-176B. GDS is optional; `nogds=True` gives the fallback path ([PyPI](https://pypi.org/project/fastsafetensors/); [review](https://www.themoonlight.io/review/speeding-up-model-loading-with-fastsafetensors)). On ROCm it uses hipFile when `libhipfile.so` is present ([hipFile docs](https://rocm.docs.amd.com/projects/hipFile/en/develop/reference/hipFile-integrations.html)).
- **Haas & Leis, "What Modern NVMe Storage Can Do, And How To Exploit It"** (**TU Munich lead**; VLDB 2023, PVLDB 16). A few PCIe 4.0 lanes drive one SSD past 1M random IOPS and ~7 GB/s. The software, not the drive, is usually the bottleneck. The paper compares pread, libaio and io_uring ([PDF](https://www.vldb.org/pvldb/vol16/p2090-haas.pdf)).
- **Kernel facts.** Large folios for buffered I/O are only supported through iomap, and the buffer_head path will not get them ([kernel iomap porting guide, 7.0](https://docs.kernel.org/7.0/filesystems/iomap/porting.html)).

### Inferences
- **Mechanism (inference from the code plus memory math).** Buffered reads fill the page cache with up to 33 GiB of clean pages while the arena commits 33 GiB of anonymous mlocked memory. That is ~66 GiB of demand against 57 GB. Reclaim runs during the load, the speed then depends on what else is cached, and systemd-oomd sees pressure. This fits the 0.49-1.18 GiB/s spread across 13 starts and the oomd reports in the pain-points report. A second possible contributor: with `read_ahead_kb=128` and `max_sectors_kb=128`, each `fread` thread issues 128 KiB block I/Os with limited depth. The true depth is unmeasured; this is a hypothesis.
- **Technique A, Linux O_DIRECT arena load.** Port `load_experts_direct` to Linux on top of `platform::DirectFile` (it already exists), or simply drop the `#ifdef _WIN32` and back it with `pread(O_DIRECT)` workers. Use 8 MiB aligned chunks with 8-16 in flight, read straight into the THP-backed arena with no bounce buffer. Then make `experts_unbuffered()` return true on Linux when arena + read set exceeds MemAvailable, which is always true here.
  - Expected gain: 28-65 s -> ~10-14 s per start, bounded by Gen3 x4 and the NV2's ~3.5 GB/s. That is about 15-50 s saved per restart, plus fewer oomd trips and a more predictable start.
  - Effort: S-M (the Windows twin and the Linux DirectFile both exist).
  - Risk: low-medium. The `padded_` layout and any layer offset not 4 KiB-aligned fall back to buffered (the code already handles that fallback). O_DIRECT on ntfs3 needs sector-aligned offsets, which the PLE path shows are honored. `STRATA_VERIFY_ARENA` gives a checksum for the A/B.
  - Family: **F10** (operations/restart).
- **Technique B, the cheapest stop-gap.** Keep the buffered reader but call `posix_fadvise(fd, off, len, POSIX_FADV_DONTNEED)` behind each finished 8 MiB chunk, and `POSIX_FADV_SEQUENTIAL` at open (it doubles the readahead window). This removes the doubled memory pressure but not the 128 KiB I/O size. Effort: XS. Gain: less variance and oomd risk; speed gain uncertain. Family: F10.
- **ServerlessLLM's "format" lesson is already met.** The pack's `experts.bin` is one contiguous, layer-ordered file. What pancake lacks is the loader half: direct I/O plus a pipeline. A new format is not needed. The pinned/registered half is already overlapped: "pin by need: 11592 ranges registered" runs in a separate thread (`reg.join()`).
- **Hugepages (16,920 x 2 MiB) stay an owner decision.** THP via `MADV_HUGEPAGE` is already requested (`pinned.cu:506`). An O_DIRECT load into a THP region avoids the page-cache to anon copy entirely. Hugetlbfs would mainly cut the ~2.5 s of 4 KiB-page registration noted at `expert_source.cpp:6192`.

### Gaps
- No paper measures model loading on ntfs3, on PCIe 3.0 or on a DRAM-less NV2. All speed targets above are estimates from the link and drive specs.
- The actual depth and size of block I/O during the buffered load is unknown. `iostat -x` or `/proc/diskstats` during one start would settle it; not run, per the task.
- I did not verify ServerlessLLM's per-microbenchmark loading figures (3.6-8.2x) against the paper PDF.

## Q2. Random reads from the 28.8 GB PLE/n-gram shard: async I/O, queue depth, read size, layout

### Takeaway
PLE already uses the right primitive: aligned O_DIRECT preads from a thread pool whose size is the queue depth. The literature says a drop-in io_uring swap gives only ~6-10%; gains of ~2x need a batch-oriented redesign. That only matters if PLE reads turn out to be on the critical path. The cheaper levers are measuring rows/s against `STRATA_IO_THREADS` and a frequency-ordered layout of the table (LLM-in-a-flash and IMPRESS style), so that hot rows share 4 KiB blocks.

### Cited Findings
- **Repo.** `src/platform/direct_file.cpp:61-62`: "STRATA_IO_THREADS, else 4 (Windows: overlapped submits) / 16 (Linux: each thread does one blocking pread, so the thread count is the queue depth)". Requests must be aligned in offset, length and buffer (`direct_file.cpp:337`). The log shows an SSD keep-alive: one page read every 100 ms while active, until 60 s idle (local repo/log).
- **Jasny, El-Hindi, Ziegler, Leis, Binnig, "High-Performance DBMSs with io_uring: When and How to Use It"** (**TU Darmstadt / TU Munich lead**; PVLDB 19, arXiv Dec 2025). Dropping io_uring in gives 1.06x for a storage-bound buffer manager and 1.10x for a network shuffle. A redesign around batching and async completion gives 2.05x and 2.31x. Applying their guidelines to PostgreSQL's io_uring backend gave +14% ([arXiv 2512.04859](https://arxiv.org/abs/2512.04859)).
- **Didona, Pfefferle, Ioannou, Metzler, Trivedi, "Understanding modern storage APIs"** (**IBM Research Zurich lead, VU Amsterdam co**; SYSTOR 2022, just before the window). io_uring with kernel polling comes within ~10% of SPDK ([PDF](https://animeshtrivedi.github.io/files/papers/2022-systor-Understanding%20modern%20storage%20APIs:%20a%20systematic%20study%20of%20libaio,%20SPDK,%20and%20io%20uring.pdf)).
- **"LLM in a flash"** (Alizadeh et al.; **Apple, industry**; arXiv 2312.11514, ACL 2024). Its flash cost model favors fewer and larger contiguous reads. "Row-column bundling" stores data used together contiguously, and "windowing" reuses recently loaded neurons. It runs models up to 2x DRAM, 4-5x faster than naive loading on CPU and 20-25x on GPU ([arXiv](https://arxiv.org/abs/2312.11514)).
- **IMPRESS** (Weijian Chen, Shuibing He et al.; **Zhejiang University lead, Huawei Cloud co**; FAST '25). It reorders KV entries on disk so important ones are packed densely and reads only important prefix KVs. TTFT is up to 2.8x lower ([USENIX](https://www.usenix.org/conference/fast25/presentation/chen-weijian-impress); [PDF](https://shuibing9420.github.io/assets/pdf/impress-fast25.pdf)).

### Inferences
- **io_uring for PLE.** Verdict: low priority. With 16 blocking threads the drive already sees QD16. io_uring would mostly save the 16 threads' context switches, and the 8 physical cores are saturated by the expert pool. That CPU saving is the only real argument.
  - If done: one ring per process, registered buffers and fixed file, batched submits of all rows of one draft/verify window, no IOPOLL. IOPOLL support on ntfs3 is unknown, and SQPOLL burns a core the 5700G cannot spare.
  - Expected: <=5% of PLE-thread CPU, ~0 on tok/s unless profiling shows PLE waits on the critical path. The pain-points report lists "PLE prefetch" timing as unmeasured (Asia P7).
  - Effort: M. Risk: low. Family: F10 / F3 (overlap).
- **Queue-depth sweep (no code).** `STRATA_IO_THREADS` is already an env knob. Reading the startup probe's rows/s at 8/16/32/64 tells whether the NV2 is IOPS-bound at QD16. 33K rows/s at >=4 KiB per row is >=136 MB/s, far below what the drive can do, so either rows are small and IOPS-bound or the probe runs at low QD. Row size and probe QD were not checked (gap). Family: F10, measurement only.
- **Frequency-ordered PLE layout (LLM-in-a-flash bundling / IMPRESS reordering analogue).** A hashed n-gram table scatters hot rows, so each 4 KiB O_DIRECT read carries one useful row. Rewriting the shard once, ordered by observed hit frequency from a trace, with an index remap, would let the 1M-row cache and any 4 KiB+ read carry several hot rows.
  - Expected: fewer SSD reads per window; tok/s gain unknown until PLE stall time is measured.
  - Effort: M (offline repack plus an indirection table). Risk: an exact transform, but it doubles 28.8 GB on a 91%-full disk during the repack.
  - Family: F10 (or F2 if counted as placement).

### Gaps
- The PLE row size, the probe's QD and whether PLE reads ever stall the window are unknown from the greps. The literature cannot answer this; one profile run (I8 in the grouping report) can.
- No paper evaluates io_uring IOPOLL or registered buffers on ntfs3.

## Q3. Filesystem effects: ntfs3 vs ext4/xfs, and FUSE

### Takeaway
On this kernel (7.0), ntfs3 already has an iomap buffered-read path and accepts O_DIRECT. The new in-kernel "ntfs" driver (7.1) claims *identical* read throughput to ntfs3; its gains are in writes, mount time and correctness. Moving the 38-66 GB files to ext4/xfs would not speed up large O_DIRECT reads in any way the literature documents. It matters only for (a) hipFile's fastpath, which requires ext4 `data=ordered` or xfs, and (b) safety. Never use ntfs-3g (FUSE).

### Cited Findings
- **ntfs3 maintenance and iomap.** ntfs3 was merged in 5.15 with uneven upkeep (67 commits in a year, many tree-wide). It still used buffer_heads in Jan 2026, while Komarov's iomap patch sat in linux-next ([LWN, 2026-01-23](https://lwn.net/SubscriberLink/1055062/c5b3ee05ba8967c3/)). The running 7.0 kernel's ntfs3 exports `ntfs_iomap_*` read symbols, so that patch is in (local `/proc/kallsyms`).
- **New "ntfs" driver** (Namjae Jeon, ex-"ntfsplus"). It uses iomap, delayed allocation and optimized zeroing. Author benchmarks show 1-thread writes +3-5%, 4-thread writes +35-110%, read throughput identical to ntfs3, and mount 0.38-0.70 s vs 2.03-4.51 s. It passes more xfstests than ntfs3 (287 vs 218 in an early series) ([patch cover letter](https://patchew.org/linux/20251020020749.5522-1-linkinjeon@kernel.org)). It was merged for Linux 7.1 with ntfs3 kept in tree ([LKML pull](https://lkml.iu.edu/2604.2/00845.html); [OMG! Ubuntu](https://www.omgubuntu.co.uk/2026/06/linux-7-1-kernel-features/amp)); later counts are 326 vs 273 xfstests ([Linuxiac](https://linuxiac.com/linux-kernel-7-1-merges-new-ntfs-driver-with-full-write-support/)). These are author/press figures, not independent benchmarks.
- **FUSE overhead.** RFUSE (Kyu-Jin Cho, ..., Jin-Soo Kim; **Seoul National University lead**; FAST '24) replaces FUSE's request path with per-core ring buffers. Communication latency drops 53%, and random reads over ext4 get ~2.93x the throughput of stock FUSE, which shows how much stock FUSE costs at high IOPS ([USENIX](https://www.usenix.org/conference/fast24/presentation/cho); [code](https://github.com/snu-csl/rfuse)).
- **hipFile fastpath filesystems.** Only xfs, or ext4 with `data=ordered`, on direct local NVMe qualify, and an O_DIRECT open must succeed ([hipFile compatibility](https://rocm.docs.amd.com/projects/hipFile/en/latest/how-to/checking-system-compatibility.html)).

### Inferences
- **Verdict: keep ntfs3 for now.**
  - Fix the loader first (Q1, Technique A). O_DIRECT bypasses the buffered-read differences between filesystems; what remains is extent lookup.
  - Large files on a 91%-full NTFS volume may be fragmented. `filefrag` on `experts.bin` and the PLE shard is a free check that was not run.
  - The 7.1 "ntfs" driver is a robustness upgrade (the ntfs3 journal-replay doubts are noted in LWN), not a read-speed one. Pancake does not write to D in its hot path, except future disk sessions.
  - Family: F10.
- **ext4/xfs migration.** Worth it only when hipFile becomes usable (see Q4), or for writes (disk sessions). The root disk has ~7-12 GB free and D is 91% full, so there is no room to stage a 66 GB copy without deleting models. Effort: L (repartition or another disk). Risk: dual-boot data. Expected read gain: no documented gain for large O_DIRECT reads (gap).
- **ntfs-3g / FUSE.** Avoid. The FAST '24 numbers show stock FUSE losing a large factor on random reads, which is exactly the PLE pattern.

### Gaps
- I found no university or kernel benchmark that compares ntfs3 with ext4/xfs for large sequential or 4 KiB random O_DIRECT reads. The ntfsplus numbers compare the two NTFS drivers only.
- Whether ntfs3 on 7.0 serves O_DIRECT through `iomap_dio_rw` or its older direct-I/O path is not settled by the symbol grep.

## Q4. NVMe direct-to-GPU paths: GDS equivalents on AMD, BaM-style GPU-initiated I/O

### Takeaway
AMD's GDS equivalent, hipFile (AMD Infinity Storage), exists, but its fastpath is unavailable on this host. The loaded amdgpu lacks `kfd_ais_rw_file`, and the data sit on ntfs3. BaM is NVIDIA-only research code that does not work on 6.x+ kernels. More basically, pancake's experts go NVMe -> RAM arena, and VRAM is filled from RAM, so direct-to-GPU would at best speed up the one-time ~14 GiB VRAM cache fill over the same PCIe 3.0 link. Verdict: not worth pursuing.

### Cited Findings
- **hipFile.** Direct-to-GPU I/O without a host bounce buffer, with C/Python sync, async and batch APIs, and automatic POSIX fallback ([hipFile docs](https://rocm.docs.amd.com/projects/hipFile/en/latest)). Fastpath requires kernel P2PDMA, a HIP runtime exporting `hipAmdFileRead/Write`, an amdgpu exposing `kfd_ais_rw_file`, and xfs or ext4 (`data=ordered`) directly on local NVMe ([compatibility](https://rocm.docs.amd.com/projects/hipFile/en/latest/how-to/checking-system-compatibility.html)). Any block layer in between (LVM, dm-crypt, MD, loop) forces fallback ([limitations](https://rocm.docs.amd.com/projects/hipFile/en/latest/troubleshooting/limitations.html)). The docs I fetched do not name supported GPU families (Instinct vs Radeon). Press reports tie general availability to a 2026 ROCm release; unverified ([StorageReview](https://www.storagereview.com/news/amd-rocm-10-1-hipfile-fast-path-numa-memory-llvm-24)).
- **Local.** `grep -c kfd_ais_rw_file /proc/kallsyms` = 0 (local observation).
- **BaM** (Qureshi, Mailthody, Gelado, ..., Hwu; **UIUC lead, NVIDIA co**; ASPLOS '23). GPU-resident NVMe submission/completion queues plus a GPU software cache. It is 5.3x faster than CPU-initiated access on data analytics, and graph BFS/CC run 1.0x/1.49x at up to 21.7x lower hardware cost. The repo warns that 6.x kernels may not work ([arXiv 2203.04910](https://arxiv.org/abs/2203.04910); [code](https://github.com/ZaidQureshi/bam)).

### Inferences
- **hipFile for pancake.** Not applicable today. It needs a newer amdgpu/AIS driver, ext4/xfs and P2PDMA, and Radeon support is unconfirmed. Even if available, the arena (CPU experts) gains nothing. Only the VRAM expert cache fill (14.38 GiB) and the 835 MiB MTP head could skip RAM, saving perhaps a few seconds at startup over PCIe 3.0 (estimate).
  - Effort: L. Risk: high (driver stack). Family: F10.
  - Re-check `ais-check` when ROCm ships AIS for gfx1100.
- **BaM-style GPU-initiated expert fetch** (miss -> NVMe read from a kernel). Not applicable. NVIDIA-only, incompatible kernels, and pancake's misses are served from a RAM arena at ~27 GB/s DDR4, which is 7x faster than this NVMe link.

### Gaps
- I found no source stating hipFile fastpath support (or lack of it) for RDNA3 / gfx1100 consumer GPUs.

## Q5. Persistent conversation state on SSD for fast resume (parked conversations lost on restart)

### Takeaway
The research line (CachedAttention/AttentionStore, HCache, Bidaw, IMPRESS, CacheGen) shows tiered RAM+SSD KV stores cut TTFT by up to 87% (AttentionStore) or 5.7x vs recompute (HCache). The robust write path is already in pancake: `conversation_file.cpp` uses O_DIRECT 16 MiB blocks, fsync and an atomic rename, and is reachable through the manual `/slots/0?action=save|restore`. strata-poly has an automatic `--conversation-disk`, but run-pancake.sh refuses strata-poly on HIP. The missing piece is automatic write-behind of parked conversations plus restore-on-start in `strata --serve`. A 1.13 GB snapshot writes in about 0.5-1 s and reads in about 0.4-1 s, vs seconds to tens of seconds of re-prefill. Exact-output techniques only; the lossy ones (IMPRESS, CacheGen) are not needed at this size.

### Cited Findings
- **Repo.**
  - `src/core/conversation_file.cpp:79` reads "file I/O in aligned 16 MiB blocks. Linux: O_DIRECT when the filesystem takes it". It falls back to buffered on EINVAL, calls fsync at line 425, does an exclusive temp file + `renameat` + directory fsync at lines 462-499, and has fault injection for tests.
  - `src/poly/conversation_disk.hpp:1-17`: automatic `--conversation-disk` writes write-once blob files plus per-conversation manifests on a background thread. Blobs are shared between conversations. A key.txt invalidates files on a model/binary/option change. It enforces size limit and min_free settings.
  - MOD_PLAN.md:183-186 (status 2026-11-08): disk conversations exist "as session files (`slot_save_path`, `/slots/0?action=save|restore`), by hand rather than FastStrata's automatic `--conversation-disk`" (local repo).
- **CachedAttention / AttentionStore** (Bin Gao et al.; **NUS lead, Huawei co**; arXiv 2403.19708; reported at USENIX ATC '24, venue not verified in my fetch). It saves KV of idle sessions across HBM/DRAM/disk with scheduler-aware prefetch and eviction, overlapped with compute. It decouples positional encoding so truncated contexts stay reusable. TTFT is up to 87% lower and prefill throughput 7.8x higher on multi-turn ([arXiv](https://arxiv.org/abs/2403.19708v2)).
- **HCache** (Shiwei Gao, Youmin Chen, Jiwu Shu; **Tsinghua lead**; EuroSys '25). It stores per-layer hidden states (half the size of KV) and rebuilds KV with the K/V projections, using a bubble-free scheduler that overlaps I/O and compute and a chunked storage format. TTFT is up to 5.73x lower than recompute, with ~2x less I/O than KV offload ([arXiv 2410.05004](https://arxiv.org/abs/2410.05004v1); [PDF](https://chenyoumin1993.github.io/papers/eurosys25-hcache.pdf)).
- **Bidaw** (Shipeng Hu, Guangyan Zhang et al.; **Tsinghua lead, China Telecom co**; FAST '26). It finds that loading KV from a DRAM+SSD store raises latency up to 3.8x vs all-in-memory under existing designs. It adds tier-aware request ordering, *predictive eviction that uses the model's own response to guess whether the user will return*, and selective caching by storage efficiency. Latency is up to 3.58x lower ([USENIX](https://www.usenix.org/conference/fast26/presentation/hu-shipeng)).
- **CacheGen** (Yuhan Liu, ..., Junchen Jiang; **UChicago lead, Microsoft co**; SIGCOMM '24). A KV-specific codec makes KV 3.5-4.3x smaller and fetch+process delay 3.2-3.7x lower, with negligible quality loss (lossy) ([arXiv 2310.07240](https://arxiv.org/abs/2310.07240v6)).
- **IMPRESS** (ZJU lead, FAST '25) loads only important prefix KVs from disk (lossy); TTFT up to 2.8x lower ([USENIX](https://www.usenix.org/conference/fast25/presentation/chen-weijian-impress)).
- Already in the reports: FastStrata `--conversation-disk`, 70K tokens resumed after restart in 7.2 s vs 83 s (MOD_PLAN phase 7).

### Inferences
- **Technique C, automatic disk tier for parked conversations in `strata --serve`.** Hook the park/evict path to queue the snapshot to a write-behind thread using the existing `conversation_file.cpp` writer (O_DIRECT, fsync, rename). Port strata-poly's manifest/key scheme (`conversation_disk.cpp`, 559 lines) for invalidation and the size cap. On start, index the files and restore lazily on the first prefix match.
  - Expected: conversations survive the 28-65 s restarts and model swaps, and the RAM limit of 2-3 long parked conversations becomes a disk limit.
  - Resume cost: 1.13 GB at ~1.5-3 GB/s O_DIRECT on Gen3 x4 is about 0.4-0.8 s. Recomputing 50K tokens takes several seconds at the 11.7K tok/s rate the reports use, and more at 64-120K. These are estimates, consistent with FastStrata's 83 -> 7.2 s.
  - Effort: M. Risk: medium, around correctness of restored hybrid GDN+KV state. **Gate it on item 35 (UNSW restore-equivalence test)**.
  - Family: **F5** (also unlocks the F10 restart pain).
- **Bidaw's eviction idea fits cheaply.** The RAM-park vs disk-only vs drop decision can use a signal pancake already has, such as whether the last response ended in a tool call (the agent will be back within seconds) vs a final answer. Keep the former in RAM and demote the latter to disk first. Effort: S on top of C. Exact. Family: F5.
- **HCache for the hybrid model.** Only partly applicable. Hidden states can rebuild K/V for the full-attention (QSA) layers with one GEMM per layer, but the GDN recurrent state is fixed-size and cannot be rebuilt from hidden states without a sequential scan. Pancake's snapshot is already compact (KV int8 plus GDN state), so keep exact KV+state snapshots. Low priority.
- **CacheGen / IMPRESS (lossy).** Not needed. Pancake's bottleneck is restart loss, not snapshot bandwidth, and both change outputs.

### Gaps
- I did not find how the 1.13 GB per 50K-token snapshot splits between GDN state and KV rows, which would size an HCache-style variant.
- Real write/read throughput of `conversation_file.cpp` on this ntfs3 volume is unmeasured.

## Q6. Page cache behaviour and eviction under memory pressure; readahead and madvise hints

### Takeaway
Linux's default page cache is a single approximate-LRU. Research (cache_ext, SOSP '25) shows custom eBPF eviction policies win up to ~70% on some workloads, but they need a patched kernel. For pancake, the practical rule is to **keep the bulk loads out of the page cache** (O_DIRECT arena, Q1) and leave the page cache to the small, skewed file-tier set the repo already reasons about. `MADV_SEQUENTIAL`/`WILLNEED` only matter for the mmap/RAM_KEEP mode.

### Cited Findings
- **cache_ext** (Tal Zussman, ..., Asaf Cidon et al.; **Columbia lead**; SOSP '25). Eviction is replaced by eBPF struct_ops with hooks for init, eviction request, folio admission, access and removal, isolated per cgroup. Eight policies were implemented, with gains of up to 70% on some workloads (talk abstract). The artifact needs a modified Linux 6.6.8 ([project page](https://5ujinkang.github.io/cache_ext/); [LPC 2025](https://lpc.events/event/19/contributions/2165/)).
- **Repo.**
  - `src/platform/memory.cpp:203-231`: `advise_willneed` issues `MADV_WILLNEED` / `POSIX_FADV_WILLNEED` in 128 KiB steps (switch: `STRATA_READ_AHEAD`).
  - `src/core/expert_source.cpp:932-950` chooses between the file cache and unbuffered reads from MemAvailable and the "honest" clean-inactive figure. The comment there notes a skewed read set, where a cache holding 1/20 of it serves most repeats (#1194). It also notes a gfx906 ROCm issue where reclaim around an HMM-tracked arena stalled queues (#1752).
- Already in the reports: `MADV_SEQUENTIAL` PR #5561 (65-76 s -> 23-28 s follow-up turns on a RAM-starved RX 6900 XT box).

### Inferences
- **cache_ext.** Not applicable on Ubuntu's stock 7.0 kernel; idea only. If pancake ever runs RAM_KEEP (mmap file tier), a scan-resistant admission rule (second-touch, like F2's "admit on k-th miss") is the policy to want. The repo's existing choice between unbuffered and cached reads is the user-space equivalent.
- **Readahead.** `read_ahead_kb=128` is the distro default. Raising it (e.g. `blockdev --setra 4096`) is a system setting (owner-only, prohibited for me). It helps only buffered sequential loads, and is moot once Technique A lands.
- **`MADV_SEQUENTIAL` on the arena.** Not applicable: the arena is anonymous, filled by `fread`. It applies only to `STRATA_ARENA_MMAP=1` or RAM_KEEP. Family: F10.

### Gaps
- The exact cache_ext speedups by workload were not verified from the full paper.

## Q7. SSD offload of MoE experts (for the RAM_KEEP / file-tier mode)

### Takeaway
University and industry work on SSD-resident experts (FlashMoE, LLM in a flash, the KAIST-style energy study, llama.cpp #31469 already in the reports) targets machines where the model does not fit in RAM. Pancake's arena fits in RAM, and the NVMe link (~3.5 GB/s) is ~8x slower than the DDR4 pool (~27 GB/s), so SSD experts can only be a fallback tier. The reports already measured that mode: RAM_KEEP=20 decodes ~8% slower.

### Cited Findings
- **FlashMoE** (arXiv 2601.17063, Jan 2026; affiliation not checked). It offloads inactive experts to SSD with an ML-based recency+frequency cache. Hit rate is up to 51% higher than LRU/LFU, with up to 2.6x speedup over existing MoE offload systems on edge devices ([arXiv](https://arxiv.org/html/2601.17063v1)).
- **"SSD Offloading for LLM MoE Weights Considered Harmful in Energy Efficiency"** (Kyung et al., arXiv 2508.06978, IEEE CAL 2025; affiliation not checked). For DeepSeek-R1, SSD-resident experts cost ~4.9x more energy per token than HBM and 3.1x more than CPU DRAM ([arXiv](https://arxiv.org/abs/2508.06978)).

### Inferences
- **Verdict.** Do not move experts to SSD by design. If RAM_KEEP is chosen to free RAM for parked conversations (the RAM-policy conflict in the grouping report), the file tier should read O_DIRECT with a frequency-aware cache. The repo already probes direct vs cached and admits by skew (#1194/#3237). The trade is RAM for parked conversations vs ~8% decode. With Technique C (disk tier) those conversations move to SSD instead, which removes the pressure to pick RAM_KEEP. Family: F2/F10.

### Gaps
- Affiliations of FlashMoE and the energy paper were not verified (lead institution unknown).

## Q8. SSD wear for write-heavy caches (disk sessions)

### Takeaway
Wear is not a constraint for pancake's disk sessions. Even 30 parks of 1.13 GB per day (~34 GB/day, ~12 TB/year) uses about 2% per year of the NV2 2 TB's 640 TBW rating. The real risk is the 91%-full volume (less free space for the drive's garbage collection) and that NV2 batches vary in NAND. Write-once, large, sequential, shared-blob files, as `conversation_disk.cpp` already does, keep write amplification near 1.

### Cited Findings
- **Kingston NV2 2 TB (SNV2S/2000G).** 640 TBW (JEDEC client workload), 3-year warranty ([datasheet](https://www.kingston.com/datasheets/SNV2S_us.pdf)). TechPowerUp lists 0.3 DWPD ([TPU](https://techpowerup.com/ssd-specs/kingston-nv2-2-tb.d2100)). Kingston has shipped several controller/NAND combinations under this part number ([maxmybuild](https://www.maxmybuild.com/storage/nvme-ssds/kingston-nv2/2tb)).
- **HiFC** (**Seoul National University lead, SK hynix / NAVER Cloud co**; NeurIPS 2025 poster). It stores KV pages in pseudo-SLC regions of commodity NVMe and reports up to 8x better write endurance. It argues that frequent non-sequential writes raise write amplification ([NeurIPS](https://nips.cc/virtual/2025/poster/116008)).
- **HILOS** (arXiv 2502.09921) delays KV writeback to cut storage write amplification ([arXiv](https://arxiv.org/html/2502.09921v2)).
- **Repo.** `conversation_disk.hpp:6-11` writes once and never rewrites, shares blobs between conversations, renames into place, and caps size via `limit` and `min_free` (local).

### Inferences
- **For Technique C.** Write only on park/evict, not on every turn. Reuse shared-prefix blobs (the system prompt + tools prefix is already shared in RAM per MOD_PLAN). Set `limit` to ~20-40 GB and keep `min_free` well above it on a volume that is 91% full.
  - Expected wear: ~2% of rated TBW per year at heavy use (calculation from the datasheet figure).
  - Watch `nvme smart-log` "percentage_used" monthly (owner action).
  - Family: F5.

### Gaps
- Whether this NV2 unit is TLC or QLC, and its SLC cache size, is unknown; this affects sustained write speed for multi-GB snapshots, not lifetime at this volume.

## Summary table (for the report writer)

| # | Technique | Paper / lead | Pancake target | Expected gain | Effort | Risk | Exact? | Family |
|---|---|---|---|---|---|---|---|---|
| A | Linux O_DIRECT arena load (port `load_experts_direct`, use `DirectFile`) | ServerlessLLM (Edinburgh), Haas & Leis (TUM), fastsafetensors (IBM) | `pinned.cu:1160`, `expert_source.cpp:6288` | start 28-65 s -> ~10-14 s (est.), fewer oomd trips | S-M | low-med | yes | F10 |
| B | `FADV_DONTNEED` behind buffered chunks + `FADV_SEQUENTIAL` | kernel API | `load_experts_ranges` | less variance and memory pressure | XS | low | yes | F10 |
| C | Automatic disk tier for parked conversations in `strata --serve` | AttentionStore (NUS), HCache/Bidaw (Tsinghua), FastStrata | `conversation_file.cpp` + port `conversation_disk.cpp` | resume ~0.5-1 s vs re-prefill; survives restarts | M | med (gate on item 35) | yes | F5 |
| C' | Response-aware demotion (tool call -> keep in RAM; final answer -> disk) | Bidaw (Tsinghua, FAST '26) | park policy | better RAM hit for sub-agent switches | S | low | yes | F5 |
| D | `STRATA_IO_THREADS` sweep on the PLE probe | Haas & Leis | env only | find the QD knee | XS | none | yes | F10 |
| E | Frequency-ordered PLE shard layout | LLM in a flash (Apple), IMPRESS (ZJU) | offline repack + remap | fewer SSD reads per window (size unknown) | M | disk space | yes | F10/F2 |
| F | io_uring backend for DirectFile | Jasny et al. (TUDa/TUM) | `direct_file.cpp` | ~1.06x drop-in; CPU savings | M | low | yes | F10/F3 |
| G | ext4/xfs migration or 7.1 "ntfs" driver | LWN, ntfsplus | partition | no documented read gain; needed only for hipFile | L | high | yes | F10 |
| H | hipFile / GDS, BaM | AMD AIS; BaM (UIUC) | VRAM fill | few s at start at best; fastpath absent | L | high | yes | F10, drop |
| I | cache_ext eBPF eviction | Columbia (SOSP '25) | page cache | n/a on stock kernel | - | - | yes | drop |
