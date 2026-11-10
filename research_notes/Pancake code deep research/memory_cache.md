# pancake: MEMORY, LOADING and CACHE subsystems: code study (2026-11-09)

Repo: /home/dan/clone-of-d/src/Strata, branch port/rdna3-decode. Read-only study: nothing was edited, built or run.
Line numbers are 1-based and match the tree as of 2026-11-09.
Tags: **[meas]** = measured, from pancake-flashnext.log or MOD_PLAN; **[est]** = estimate; **[code]** = read directly from the code.

---

## 0. Headline findings

1. **Item 38 targets the wrong function for the daily preset.** MOD_PLAN 38 names `load_experts_direct` / `experts_unbuffered`.
   The daily IQ2_XS preset has **no experts.bin** in `packs/iq2_xs` (only dense.bin, index.txt, native_experts.txt) and runs
   with `STRATA_PIN_BY_NEED=1`. So the path is `from_gguf = true`, `padded_ = true` -> `load_experts_gguf(..., unbuffered, rp, &pad_)`
   (expert_source.cpp:6287). Its O_DIRECT branch is `#if defined(_WIN32)` only (5792-5876). On Linux the code hits
   `(void) unbuffered;` (5880) and the buffered `fread` workers run (5882-5937). Even with an experts.bin, `padded_` would skip
   `load_experts_direct` (6291). **The port belongs in the Linux branch of `load_experts_gguf`, plus `experts_unbuffered` on Linux.** [code]
2. **Likely mechanism of the oomd trips (supported by the code, not yet proven by a PSI trace).** The by-need arena calls
   `cudaHostRegister`/`hipHostRegister` on every range **before** the load (pinned.cu:858-947). On Linux that faults in and pins
   all ~33 GiB of anonymous pages up front (log: "11592 ranges registered (33838 MiB)"). Then 6 buffered readers pull ~33 GiB
   more through the page cache. Those cache pages are charged to the capped.sh scope (MemoryMax=54G), so the kernel reclaims
   inside the cgroup during the load. Memory PSI climbs, and systemd-oomd kills the scope once user@.service sits above 50 %
   pressure for 30 s. MADV_HUGEPAGE direct compaction during the registration faults also counts as a memory stall. [code + log; est.]
3. **Buffered load measured at 0.81-1.08 GiB/s for 33.02 GiB, i.e. 31-41 s** (log 2026-11-08/09). The buffered path uses only
   **6 threads** (generate.cpp:7712 `threads=6`); 16 readers are used only when `unbuffered` (expert_source.cpp:6233). [meas]
4. **Restore off-by-one (UNSW class) is guarded in the code.** `starts_with` accepts a cached prefix only when `L <= n-1`
   (generate.cpp:16161-16166). `consumed` records only the tokens whose state is in the session, and the last output is excluded
   (19419-19420, "the last output is not: it is next x"). End of request: `live.swap(consumed)` (19611). So a full-prompt hit
   always re-feeds the last token. Remaining risks are listed in section 3; item 35 is still needed. [code]
5. **Seeds (3-4) and k-th-miss (item 14 KMISS) do not compose.** Under `STRATA_ADAPT_KMISS` a candidate needs `m >= k` real misses
   (kmiss_tier.hpp select). The seeder only raises `usage`, which KMISS uses only as a tie-breaker. So seeding is close to a no-op
   when KMISS is on. Decide one of them, or add a "seeded" bit into the miss mask. [code]

---

## 1. Memory layout: start-up fill and runtime movement

### 1.1 Machine and budgets [meas, MEMORY.md / log]
- RAM 57.7 GiB (meminfo). At start, ~6.8-7.2 GiB is held by others, so 50.6-51.0 GiB is available (log 11:56 / 12:16 starts).
- capped.sh: `systemd-run --user --scope -p MemoryMax=54G -p MemorySwapMax=0`, `STRATA_RAM_KEEP_GIB=12` by default. It waits for 48 GiB available.
- VRAM: RX 7900 XT, 20 GB. ROCm 10.0.

### 1.2 Start-up sequence (daily preset flashnext-iq2_xs-fast-hip)
| Step | Where | Size / time |
|---|---|---|
| memory guard sample | generate.cpp:4018-4021, conversation_memory.cpp:205-300 | floor 2560 MiB |
| token embedding IQ4_XS, mapped host | native pack | 322 MiB, 0.4 s [meas] |
| pack dense weights | pack/dense.bin | 266 MiB, 0.3 s |
| native projection matrices (VRAM) | | 2119 MiB, 4.2-5.8 s [meas] |
| PLE table on direct reads (O_DIRECT probe) | ngram/ple_reader.cpp via DirectFile | 320M rows in shard 2; probe 16-33K rows/s [meas] |
| KV streaming pool, pinned RAM | | 1.55 GiB (32768 of 131072 cells per QSA layer in VRAM) [meas] |
| MTP draft layer (VRAM) | | 835 MiB, 1.04 s [meas] |
| ram-keep decision | expert_source.cpp:6030-6120, ple_residency.hpp:74-130 | arena only if 12 GiB stay free after it |
| arena reserve | pinned.cu `reserve` 141-513 | hugetlb fails (nr_hugepages=0), THP madvise (437-512) |
| by-need registration | pinned.cu:858-947 | 11592 ranges, 33838 MiB, all faulted + pinned before the load |
| arena fill | expert_source.cpp:6287 -> load_experts_gguf buffered (5882) | **33.02 GiB at 0.81-1.08 GiB/s = 31-41 s** [meas] |
| VRAM expert cache, auto | expert_cache.cpp | 10735 slots, 14.38 GiB; pre-filled from the profile in 2.0 s (7.6 GB/s) [meas] |
| conversation cache, auto | generate.cpp:12003-12030, conversation_memory.hpp:70 | 40 % of (available - min_free), cap 8 GiB: **4096-6144 MiB** across starts [meas] |
| VRAM free after load | | 854-908 MiB [meas] |

RAM at steady state [meas/est]: arena 33.0 + KV 1.55 + conversation cache 4-6 + engine ~3 (README of ple_residency: buffers, CUDA
context) + embedding 0.3, i.e. **~42-44 GiB** in the engine, ~7 GiB for others, the rest page cache. The conversation-cache size
varies by 2 GiB between starts because `available_host_bytes()` is sampled right after a buffered load. How much page cache is
left over (and charged to the scope) at that moment sets the cache size. An O_DIRECT load would make this sizing deterministic.

### 1.3 Arena (RAM tier for CPU experts)
- Layout: per layer, 512 blobs; pin-by-need pads every blob to 4 KiB strides (`PadMap`, pinned.hpp:154; +0.15 %). The file
  offsets stay packed (expert_source.cpp:6131-6150).
- Registration: by-need ranges merged by adjacency and sorted by profile rank (6151-6185); `PinnedArena(ByNeed)` registers
  inward-rounded ranges (pinned.cu:897-921) and mlocks the holes (936-946). On Linux there is no deferred register:
  `deferred = false` (expert_source.cpp:6203); the Windows-only #285 pipelined register is opt-in.
- THP: if THP is "always" the madvise is skipped. If there are fewer free 2 MiB blocks than half the arena, it is skipped
  (#2126). Otherwise MADV_HUGEPAGE is requested (pinned.cu:466-511). The log shows "transparent huge pages requested", so the
  faults during registration can direct-compact (with defrag=madvise).
- Fallbacks (#4846): if the RAM is short after the wait and after the PLE lock give-back, the engine falls back to
  `--mmap-experts` with a resident budget (`--resident-experts` form: hottest experts in RAM, the rest read unbuffered,
  generate.cpp:7720-7746). STRATA_RAM_KEEP_GIB=20 forces it: 69 vs 75 tok/s (-8 %) on the 3060 [meas, MOD_PLAN status].
- The shared arena file (/dev/shm, MAP_SHARED, pid-state CAS word) exists for replicas. Not used by pancake.

### 1.4 VRAM tier
- ExpertCache slots: 10735 x ~1.37 MiB = 14.38 GiB, PROFILE-ranked, no eviction in the static part (log).
- Adaptive tier: `--adapt-swaps 32 --adapt-every 1`. Async arena mode = copy-in only (generate.cpp:13395-13455). Rule:
  usage-gain trade `cand.u >= vict.u + adapt_min_gain`, candidates need u >= 2.0 (13194-13207), then decay x0.7 (13353).
  Measured round time 30-41 ms start to flip, about 60 swaps per round on the 3060-era log.
- KMISS (item 14, opt-in `STRATA_ADAPT_KMISS=k,W`): miss bit-history per pair, LRU victim (kmiss_tier.hpp).
- Seeder (items 3-4, opt-in `STRATA_DOMAIN_PROFILES`, `STRATA_CONV_WORKSET`): raises usage floors for R=16 rounds (expert_seed.hpp:134-150).
- ForesightSwap (`STRATA_FS_SLOTS`, off): a separate per-layer swap space, one filler thread, FIFO queue (expert_source.cpp:6539-6800).
  `STRATA_FS_ADMIT` already gives k-th-miss admission for this swap space.
- Prompt lending: a long prompt borrows up to 9412 slots (12.59 GiB) and refills after (log); item 2 is closed (no refill cost).

### 1.5 PLE / page cache / NVMe
- PLE table: O_DIRECT reads through `DirectFile` (16 pread threads, direct_file.cpp:240-371), row cache, SSD keep-alive.
  The residency lock (ple_residency.hpp) is given back when the arena or the conversation cache needs the RAM (#4846, #4518).
- The page cache holds only what the buffered load leaves behind, plus the embedding and other mapped files. The model is on
  the D ntfs3 partition (NVMe, PCIe 3.0 x4 on the 5700G platform).

### 1.6 Conversation cache (engine side)
- Image = live running state + chain checkpoints + per-QSA-layer authoritative K/V + draft K/V (conversation_cache.hpp:SavedConversation).
- **Measured park sizes**: 540 tok = 243,811,600 B in 105.7 ms; 2584 tok = 272,869,104 B in 124.9 ms. Fit: **~236 MB fixed
  (live GDN/PLE state ~118 MB + one checkpoint ~118 MB) + 14.2 KB per token**. At 70K tokens that is ~1.23 GB [est from meas;
  MOD_PLAN said 1.13 GB]. Restore of a 22-token checkpoint took 33.3 ms.
- Budget 4-6 GiB, 4 slots. Eviction is oldest-first, skipping pinned prefixes and the guarded donor (conversation_cache.hpp evict_oldest).
  `drop_superseded` (#342) removes the same conversation one turn back.
- Chain checkpoints: root pinned + LRU (conv_cache.hpp:eviction_victim); `--prompt-cache-tail` adds one tail checkpoint;
  `message_checkpoint_boundary` (message_boundary.hpp) places item 11's checkpoint at the last turn token, at >= 8192 prefix and within 1024 of the end.
- Disk: `/slots/0?action=save|restore` -> generate.cpp:15640-15835 -> conversation_file.cpp `session_file_write/read`.
  The file holds the live state + **only the deepest checkpoint** + K/V streamed from the pools (SessionKvSource). There is a
  16 MiB O_DIRECT block writer (try_direct via fcntl F_SETFL, buffered fallback on EINVAL: conversation_file.cpp:440-452),
  fsync, atomic rename, dir fsync, model/config fingerprints, RAM preflight, read limits, and a payload hash.
  A restore replaces the live session; it is not inserted into the parked cache.
- strata-poly `src/poly/conversation_disk.cpp`: a content-addressed blob store (b-<id>.blob, c-<id>.conv), key.txt
  invalidation, flock lock, a background writer thread, `commit(want)` that fits under a limit and the volume's free space,
  lazy checksum on first read.

---

## 2. Opportunities, ranked by expected gain

| # | Item | Gain | Exact? | Risk | Effort |
|---|---|---|---|---|---|
| 1 | **38 Linux O_DIRECT for `load_experts_gguf`** (+ the 3-line FADV stop-gap) | start 31-41 s -> ~12-15 s [est]; likely removes the oomd trips, so the arena (+8 % vs mmap fallback, meas. 3060) is kept | exact (same bytes; verify with STRATA_VERIFY_ARENA checksum) | ntfs3 O_DIRECT quirks -> fallback | S-M |
| 2 | **69 arbiter, first slice**: load with O_DIRECT, then size the conversation cache from a page-cache-free RAM figure; PSI watch during load | deterministic 6+ GiB parking; no oomd | exact | low | M |
| 3 | 39 disk tier for parked conversations | resume after restart/eviction ~1 s vs ~70-80 s re-read at 70K [est from FastStrata 83 -> 7.2 s] | exact | D 91 % full, restore bugs (gate on 35) | M |
| 4 | NEW-A THP policy for the arena: no MADV_HUGEPAGE at register time; `MADV_COLLAPSE` (kernel >= 6.1) on a background thread after the load, or rely on khugepaged | removes direct-compaction stalls (PSI) during the start; keeps the TLB gain | exact | MADV_COLLAPSE on a pinned range may fail (pages pinned) -> must collapse before registration or accept 4K | S |
| 5 | NEW-B pipelined registration on Linux (reuse `PinnedArena::Deferred` + `register_slices` already written for Windows), or register per range right after its layer is read | hides the register-fault time (unmeasured; log it first) and spreads the RAM ramp, which lowers PSI | exact | #253 says Linux keeps the whole pin; the Windows note says per-layer registration decoded ~10 % slower on the RTX 5070: measure on HIP | S |
| 6 | 35 restore test using existing hooks: `STRATA_CKPT_REREAD` (re-read instead of restore, compare token for token with `--adapt-swaps 0`) + `STRATA_SNAPSHOT_VERIFY` (draft K/V hash) | gate for 3, 11, 16, 31, 39 | n/a | none | S (a driver script plus a park/restore of a full and a partial hit) |
| 7 | 53 swap budget / stale drop: add a generation stamp to the ForesightSwap queue entries, and drop a queued slot whose `expert` changed or whose request is older than `depth` windows | 0-5 % (only if FS is turned on) | exact | low | S |
| 8 | NEW-C KMISS x seeder fix: treat a seeded pair as having `k` virtual misses for R rounds (set the low bits in `miss_` at `seeder.round_begin`) | lets 3-4 and 14 stack: +0.5-1.5 hit points at request start [est] | exact (placement) | low | S |
| 9 | 61 VRAM-aware defaults | 0 here | exact | low | S |
| 10 | 16/11/31 checkpoints: already coded (message boundary, tail); waiting on 35 and the server log | see MOD_PLAN S5 | exact | restore | - |

### 2.1 Item 38: implementation sketch (concrete; claimed by this session)

**Goal**: on Linux, fill the 33 GiB arena with O_DIRECT reads (no page-cache copy, no cgroup page-cache charge), 16 readers.
Fall back to buffered + `POSIX_FADV_DONTNEED` where O_DIRECT is refused.

**A. `experts_unbuffered` (pinned.cu:1098-1165) on Linux.** The current Linux branch returns `false` with "buffered (not Windows)" (1161-1163).
Replace it with:
```cpp
#elif defined(__linux__)
    // the RAM copy plus a cached copy of the experts cannot both fit (57 GB box: 33 + 33 GiB): read unbuffered
    detail::HostMemory hm;   // or strata::core::sample_host_memory()
    const uint64_t avail = host available (cgroup aware: sample_host_memory()->available);
    const uint64_t need  = read_bytes == kAllFileBytes ? sum(file sizes via stat) : read_bytes;
    const bool keepable  = strata::platform::file_cache_keeps(avail, arena_bytes, need);
    why = "...available, arena, experts: the file cache can/cannot keep them";
    return !keepable;
```
`STRATA_UNBUFFERED_LOAD=0/1` keeps overriding it (1100-1104, already the first lines). `file_cache_keeps` is the same helper the
Windows branch uses. On this box: avail ~50 GiB, arena 33 + 33 needed, so the result is `unbuffered = true`.

**B. `load_experts_gguf` Linux branch (expert_source.cpp:5775).** Turn `#if defined(_WIN32) if (unbuffered) {...} #else (void) unbuffered; #endif`
into a shared structure with a platform read primitive:
```cpp
#if defined(__linux__)
    if (unbuffered) {
        // max_chunk exactly as in the Windows branch (5793-5797): 16 experts of the largest role
        auto worker = [&]() {
            constexpr uint64_t kSector = 4096;   // DirectFile::alignment()
            const uint64_t cap = round_up(max_chunk + 2 * kSector, kSector);
            uint8_t* buf = (uint8_t*) strata::platform::DirectFile::alloc_aligned(cap);   // posix_memalign 4096
            int fd = -1; std::string open_name; bool direct = true;
            for (;;) {
                const int64_t l = next.fetch_add(1);
                if (l >= lay.n_layers || bad) break;
                wait_ready(l);
                ... per role r (same per[]/at[]/src/total/chunk as 5810-5825) ...
                if (name != open_name) {
                    if (fd >= 0) ::close(fd);
                    fd = ::open(name.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
                    if (fd < 0 && errno == EINVAL) { fd = ::open(name.c_str(), O_RDONLY | O_CLOEXEC); direct = false; }
                    if (fd < 0) { fail("cannot open " + name + ": " + strerror(errno)); break; }
                    if (!direct) posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
                    open_name = name;
                }
                for (done ...) {
                    const uint64_t a0 = floor4k(src+done), a1 = ceil4k(src+done+n);
                    // pread loop: EINTR retry; short read allowed only at EOF past the tensor's own bytes
                    uint64_t got = 0;
                    while (got < a1 - a0) {
                        ssize_t r = ::pread(fd, buf + got, a1 - a0 - got, (off_t)(a0 + got));
                        if (r < 0 && errno == EINTR) continue;
                        if (r < 0 && errno == EINVAL && direct) {   // ntfs3/other fs refusing this transfer
                            int fl = fcntl(fd, F_GETFL); fcntl(fd, F_SETFL, fl & ~O_DIRECT); direct = false; continue;
                        }
                        if (r <= 0) break;   // EOF or error
                        got += (uint64_t) r;
                    }
                    if (got < (src + done - a0) + n) { fail("short unbuffered read of layer ..."); break; }
                    if (!direct) posix_fadvise(fd, (off_t)a0, (off_t)(a1-a0), POSIX_FADV_DONTNEED);   // stop-gap path
                    scatter exactly as 5859-5863 using dst_blob(l, e) (pad-aware)
                }
            }
            if (fd >= 0) ::close(fd);
            DirectFile::free_aligned(buf);
        };
        threads = std::max(threads, 16) ... (the caller already passes `readers`)
        return st (bytes = lay.total, seconds), same error handling as 5868-5875
    }
#endif
```
Notes:
- Factor the Windows/Linux bodies into one worker plus `read_window(handle, a0, len, buf) -> bytes`, so the scatter, `max_chunk`
  and error text are shared. The `fcntl(F_SETFL, ~O_DIRECT)` degrade is the pattern already in conversation_file.cpp:440-452
  (`try_direct`/`leave_direct`); reuse it as a small helper in src/platform (e.g. `platform/direct_io.hpp`), so direct_file.cpp,
  conversation_file.cpp and the loader share one implementation.
- DirectFile itself is not ideal here: its `open` fails hard if O_DIRECT is refused (direct_file.cpp:330-336, no fallback), and
  its queue is one-shot reads with completions. Per-thread blocking `pread` is simpler for a sequential 8 MiB stream. If DirectFile
  is reused, add the same EINVAL fallback to its `open`. The PLE reader would benefit from it too, since tmpfs/overlay test runs fail now.
- The scatter memcpy is at most ~1 copy of 33 GiB across 16 threads, i.e. under 1 s at DDR4 rates [est]. The buffered path does 2 copies (kernel to user, then the scatter).
- The registration has already faulted the destination pages; the bounce buffer is per thread (~8-16 MiB x 16 = 256 MiB peak, freed after).
- Also change the buffered branch (5882-5937): add `posix_fadvise(fileno(f), src+done, n, POSIX_FADV_DONTNEED)` after each fread.
  That is the stop-gap behind `STRATA_LOAD_DONTNEED=1` (default on when `!unbuffered` and the file cache cannot keep the experts).
  Add the same in `load_experts_ranges` (pinned.cu:1249 after the memcpy) for experts.bin packs.
- Thread count: when `unbuffered`, the caller already sets `readers = max(threads, 16)` (expert_source.cpp:6233). When buffered,
  consider raising generate.cpp:7712's 6 to 12; this is a separate A/B.
- `load_experts_direct` (pinned.cu:1023) should get the same Linux body for experts.bin packs. It cannot serve `padded_`
  (it reads straight into dst): either add the PadMap scatter, or leave padded packs to the ranges path.
- Logging: keep "expert arena read unbuffered (...)" (6295). Add read_s / copy_s like LoadStats, and the time of the
  by-need registration (time `new PinnedArena(...)` at 6206), so 5 (NEW-B) can be judged.
- Verification: `STRATA_VERIFY_ARENA=1` prints the fnv1a64 of the arena (6313). Run once buffered and once unbuffered; the checksums
  must be equal. Then `cat /proc/pressure/memory` and `memory.pressure` of the scope during load (expect some avg10 near 0),
  plus `memory.stat file` of the scope staying flat.
- Error handling / fallbacks:
  (i) open with O_DIRECT gives EINVAL -> buffered fd + FADV_DONTNEED;
  (ii) pread EINVAL mid-file -> clear O_DIRECT on that fd and continue;
  (iii) any short read -> `bad`, the arena is refused (as now), and the caller's #4846 loop takes the mmap fallback;
  (iv) `STRATA_UNBUFFERED_LOAD=0` forces the old path.
- ntfs3 specifics: O_DIRECT needs the offset/length/buffer aligned to the logical block size; 4096 covers 512e and 4Kn. Compressed or
  sparse NTFS files refuse direct I/O (EINVAL), which is covered by the fallback. The PLE reader already uses O_DIRECT on the same
  partition with no problem (log probe "fine").

### 2.2 Item 39: implementation sketch (disk tier for parked conversations)

- **Where to hook**: `ConversationCache::evict_oldest()` (conversation_cache.hpp ~407) and `make_room`. Add an optional
  `std::function<void(SavedConversation&&, uint64_t serial)> on_evict` that **moves** the image out instead of destroying it.
  Generate.cpp then hands it to a `ConversationSpill` thread.
- **Writer**: reuse `session_file_write(path, image, id, bytes, err, opts, &st)` (conversation_file.cpp:1168; the full-image
  overload), with `durable=true` and `min_free_bytes` from a new `--conversation-disk-min-free-gib`. File name =
  `<dir>/c-<hash(live.ids prefix)>-<serial>.strs`. The identity is `session_identity()` (generate.cpp:12045-...), which is already
  computed once and lazily. Problem: `session_file_write` stores **only the checkpoints the caller passes**. Pass the deepest one
  plus the root, using `session_checkpoints_to_save` (1120); the root is the shared system prompt, keep it.
- **Index**: an in-memory vector of {path, tokens prefix (ids), imgs, cvec, bytes, stamp}, rebuilt at start from a small sidecar
  `<file>.idx` (ids + imgs + cvec, ~4 B/token). Without it, matching would need to parse 1 GB files. Write the sidecar after the
  main file and rename both; on start, drop any file without a valid sidecar.
- **Read path**: in the request path where `conversations.best(...)` is consulted (generate.cpp ~16381), also query the disk
  index. If a disk entry beats `max(resume, slot_tokens, parked.tokens)` by more than, say, 2048 tokens:
  `session_file_read(path, id, image, bytes, err, limits)` with `conversation_session_read_limits` (as in /slots restore,
  15775-15791) -> `conversation_snapshot_validate` -> treat it like `incoming` (the parked restore at 16660-16705). Insert it as
  the live session and delete or keep the file (keep it: it stays valid until the conversation moves on; `drop_superseded`
  logic applies on disk too).
- **Policy**: demote final-answer conversations first, keep tool-call-ended ones in RAM (Bidaw). The server knows the finish
  reason, so pass `park_hint=tool|final` in the GEN line. Cap the tier with `--conversation-disk-gib` (D is 91 % full: start at 8 GiB, i.e. ~6 x 70K images).
- **Restart**: optional prewarm of the newest N into RAM at start, only up to the parking budget.
- **Error handling**: ENOSPC/EDQUOT -> `SessionError::storage`: disable the tier for the session, log once. Identity mismatch
  (new engine version or config) -> delete the file. Read failure -> drop the entry and fall back to a re-read. All of this is
  already typed by `SessionStatus`.
- **What to take from strata-poly's conversation_disk**: the key.txt invalidation, the flock lock, the background writer with
  `put/flush`, `commit` fitting under the limit + free space, and deletion of *.tmp at open. Its blob sharing (dedup of checkpoint
  blobs between conversations) is worth it later for the 236 MB fixed part shared by siblings; v1 can skip it.
- Expected [est]: 70K image 1.23 GB, O_DIRECT write at ~1.5-2 GB/s = 0.6-0.8 s on the writer thread (off the critical path).
  Read ~0.5-0.7 s + validate + H2D ~0.3 s, so **~1-1.5 s vs ~70-80 s** re-read at 70K.

### 2.3 Item 69 (memory budget), first slice after 38
1. After the O_DIRECT load, `available_host_bytes()` stops depending on leftover page cache. Size the conversation cache from
   `RAM - arena - keep - engine_reserve` rather than from 40 % of whatever is available (conversation_memory.hpp:70). The 8 GiB cap and the 40 % rule were set for unknown hosts.
2. PSI guard during the load: read the scope's `memory.pressure` (the `/proc/self/cgroup` path is already resolved in
   pinned.cu:71-105 `free_ram_room`). If `some avg10 > 20 %`, pause the readers briefly (they are already a work queue).
   This guards against oomd's 50 %/30 s rule. No PSI awareness exists in the code today (grep: no "pressure" outside capped.sh comments).
3. RAM_KEEP default: keep 12 with the arena once 38 lands. 20 (mmap tier, -8 %) is no longer needed if the trips go away;
   measure 5 starts in a row.

---

## 3. Correctness risks (as risks, not proven bugs)

1. **Restore / off-by-one**: guarded as described in finding 4. Residual risks:
   (a) `mtp.kv_restore(resume)` and the DFlash2 log rebuild (16820, 16832) must agree with the main session's `resume`. A
   mismatch costs acceptance only, not output, because the drafts are verified.
   (b) A checkpoint taken inside a prefill-pipe segment must resume with the same chunk plan (16798-16812), or the result is not
   bit-exact. This applies only to layer splits (not pancake).
   (c) "exact" claims vs. GPU vs. CPU rounding: the adaptive tier makes the decode not bit-exact run to run (comment at 13395ff).
   So item 35's token-for-token test must run with `--adapt-swaps 0` (as STRATA_CKPT_REREAD's comment says).
2. **By-need registration before the load (Linux)**: all pages are pinned before any byte is read. If the load then fails, the
   destructor unregisters and frees (pinned.cu:990-1001). Fine. But oomd can kill mid-load with 33 GiB pinned; there is no cleanup
   issue (the process dies), only the lost time.
3. **ForesightSwap**: (a) if `cudaMemcpyAsync` fails in `fill_loop`, the slot stays `loading && !issued` forever and its
   `where[]` entry keeps pointing at it. `failed` stops new requests, so the impact is bounded. (b) The FIFO queue has no staleness
   check (MOD_PLAN 53): a slot re-targeted while still queued cannot happen (`!issued` slots are skipped as victims), but an old
   request still costs a copy. (c) `take()` calls `cudaEventQuery` under `smu_` on the decode path; contention with
   `request_locked` from the lookahead thread is possible. FS is off by default.
4. **KMISS in async mode** observes `host_res` live at round start, while candidates use the snapshot `a_res`. A pair swapped in
   by the previous round's late landing may be counted as a miss once. This affects placement only.
5. **Seeder vs KMISS**: see finding 5 (silent no-op, not a bug).
6. **Conversation cache auto size depends on page-cache state at sampling time** (2 GiB spread measured). Not a bug, but it
   makes park/evict behaviour vary between starts, which matters for benchmarks.
7. **DirectFile (POSIX)**: `open` with O_DIRECT has no fallback (fails on tmpfs/overlayfs). `pread` short reads are reported as
   `bytes < length` with ok=true, so callers must check `bytes`. `Completion.bytes` is `uint32_t` (requests are capped at 4 GiB, fine).
8. **Session file save** writes only the deepest checkpoint (conversation_file.hpp:189). A restored session loses the pinned
   root checkpoint, so siblings re-read the shared prefix after a restore. Relevant for 39: also save the root.
9. **`load_experts_gguf` buffered path** reports a generic "short read or unreadable shard" with no layer or offset; the Windows
   direct branch has precise errors. Port the precise messages with 38.

---

## 4. Measured vs estimated summary
- Measured: arena 33.02 GiB at 0.81-1.08 GiB/s; 11592 ranges / 33838 MiB registered; VRAM cache 10735 slots / 14.38 GiB; prefill
  2.0 s; conv cache 4096-6144 MiB; park 540 tok 243.8 MB / 105.7 ms, 2584 tok 272.9 MB / 124.9 ms; checkpoint restore 33 ms;
  KV pinned 1.55 GiB; MTP 835 MiB; free VRAM 854-908 MiB; mmap tier 69 vs arena 75 tok/s (3060).
- Estimated: O_DIRECT load 12-15 s; per-token image 14.2 KB (fit of two points); 70K image ~1.23 GB; disk-tier resume ~1-1.5 s;
  the oomd mechanism (needs a PSI trace during one buffered and one direct start).
