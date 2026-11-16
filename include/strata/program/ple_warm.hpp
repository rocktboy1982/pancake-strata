#pragma once
// MOD_PLAN.md section 6 item 50 (LEADS.md I7), STRATA_PLE_WARM=1 (default off, exact: it only reads a file).  BigMoeOnEdge warms its
// weights with one sequential sweep at load so the first tokens do not fault the pages in 4 KiB at a time (their 20x
// on a phone).  The only SSD-read part of ours is the PLE shard (28.8 GB).  The sweep helps only when the table is MAPPED
// and not locked (`--ple-io mmap`): the default `direct` reads bypass the page cache (a sweep fills a cache nothing reads)
// and a locked table (`--ple-io ram`, the residency policy) is resident already.  So the plan declines those, and it never
// asks for more than the RAM that is free after the expert arena and the headroom (page cache of the sweep counts against
// MemoryMax).  The read is a plain sequential read on a background thread; nothing it reads is used.
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace strata::program {

struct PleWarmPlan {
    bool run = false;
    uint64_t bytes = 0;      ///< how much of the file to read from its start
    const char* why = "";    ///< a declined plan says why (logged)
};

inline PleWarmPlan ple_warm_plan(bool env_on, bool mapped, bool locked, uint64_t file_bytes, uint64_t available,
                                 uint64_t arena_bytes, uint64_t headroom) {
    PleWarmPlan p;
    if (!env_on) { p.why = "STRATA_PLE_WARM is off"; return p; }
    if (!mapped) { p.why = "--ple-io direct reads bypass the page cache: a sweep would warm nothing"; return p; }
    if (locked) { p.why = "the table is locked in RAM already"; return p; }
    if (file_bytes == 0) { p.why = "no file bytes"; return p; }
    const uint64_t reserve = arena_bytes + headroom;
    if (available <= reserve) { p.why = "no RAM left after the expert arena and the headroom"; return p; }
    p.bytes = std::min(file_bytes, available - reserve);
    p.run = true;
    p.why = p.bytes < file_bytes ? "partial: the RAM left is smaller than the file" : "whole file";
    return p;
}

/// Reads the first `bytes` of `path` in `block`-byte reads and drops them; returns the bytes read (a stop flag or a short
/// file ends it early).  `stop` may be null.
inline uint64_t ple_warm_read(const std::string& path, uint64_t bytes, size_t block = (size_t) 8 << 20,
                              const std::atomic<bool>* stop = nullptr) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return 0;
    std::vector<char> buf(block);
    uint64_t done = 0;
    while (done < bytes && f && !(stop != nullptr && stop->load())) {
        const size_t want = (size_t) std::min<uint64_t>(block, bytes - done);
        f.read(buf.data(), (std::streamsize) want);
        const auto got = (uint64_t) f.gcount();
        done += got;
        if (got < want) break;
    }
    return done;
}

}  // namespace strata::program
