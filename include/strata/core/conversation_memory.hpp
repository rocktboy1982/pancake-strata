#pragma once

#include <algorithm>
#include <cstdint>
#include <istream>
#include <optional>
#include <string>

namespace strata::core {

// The RAM a runtime allocation (parking a conversation, saving or restoring a session file) may still take.
//
// available_host_bytes() is the smallest of
//   * MemAvailable (Windows: the available physical memory);
//   * the room under every memory limit of this process's cgroup and of each ancestor visible under
//     /sys/fs/cgroup (cgroup v2: memory.max and memory.high, each minus usage; cgroup v1:
//     memory.limit_in_bytes minus memory.usage_in_bytes);
//   * the operator's own cap, when one is set (--memory-limit-mib / STRATA_MEMORY_LIMIT_MIB): that many MiB minus
//     what the container uses now - the top visible cgroup's memory.current, or MemTotal - MemAvailable when the
//     cgroup files cannot be read.  For a container whose real limit sits on a parent cgroup it cannot see.
//
// Usage is memory.current as the kernel reports it, less one reclaimable kind: the clean inactive file cache of the
// same group (memory.stat inactive_file, v1 total_inactive_file, less dirty and writeback pages, never more than
// memory.current), which the kernel gives back at memory.high or the limit before it kills anything.  Nothing else is
// credited: active_file holds mlocked pages, and shmem is the pinned complements and the KV pool.  A memory.stat that
// is missing or unparsable gives no credit (it does not make the sample unknown).
// Pinned, locked and shared memory stays charged in full: a hard-limit kill is worse than a skipped park.
//
// Unknown telemetry is deliberately distinct from a measured zero.  A limit file that exists but cannot be parsed
// makes the sample unknown (fail closed); a file that is absent (no such controller, not mounted) is ignored.
// Not a reservation: other writers can take the room after the sample.

enum class MemorySource { meminfo, cgroup, flag };

const char* memory_source_name(MemorySource source);

/// One sample.  `available` is the smallest term; `source`, `limit` and `current` describe the term that was the
/// smallest (meminfo: MemTotal and MemTotal - MemAvailable; cgroup: that group's limit and memory.current; flag: the
/// operator's cap and the usage it was measured against).  `mem_available` is MemAvailable by itself (0 if unknown).
struct HostMemoryReading {
    uint64_t available = 0;
    MemorySource source = MemorySource::meminfo;
    uint64_t limit = 0;
    uint64_t current = 0;        ///< memory.current (meminfo: MemTotal - MemAvailable), before the credit
    uint64_t credit = 0;         ///< the clean inactive file cache counted as reclaimable (0 for meminfo)
    uint64_t mem_available = 0;
};

/// The files the sampler reads; the defaults are the real ones (tests point them at a fixture directory).
struct MemoryProbePaths {
    std::string meminfo = "/proc/meminfo";
    std::string self_cgroup = "/proc/self/cgroup";
    std::string cgroup_root = "/sys/fs/cgroup";
};

/// The operator's cap in bytes for the container/cgroup this engine runs in (0 = none).  Set once at start.
void set_memory_limit_bytes(uint64_t bytes);
uint64_t memory_limit_bytes();

/// Linux: reads `paths`.  `explicit_limit` 0 = no operator cap.  nullopt when the RAM cannot be determined.
std::optional<HostMemoryReading> sample_host_memory(const MemoryProbePaths& paths, uint64_t explicit_limit);

/// The real files (Windows: GlobalMemoryStatusEx) and the cap set by set_memory_limit_bytes().
std::optional<HostMemoryReading> sample_host_memory();

std::optional<uint64_t> available_host_bytes();

/// #3493: the conversation cache budget (MiB) --serve uses when --conversation-cache-mib was not given, from the RAM that is available
/// once the model is loaded: 40% of what is left above the parking floor, at most 8192 (four 100K-token agent conversations were
/// parked in 8 GiB in the report), in steps of 256, and 0 (off) when less than 1024 would be left to give.  Unknown telemetry: 0.
inline int64_t auto_conversation_cache_mib(std::optional<uint64_t> available, int64_t min_free_mib) {
    if (!available) return 0;
    const int64_t usable = (int64_t) (*available >> 20) - min_free_mib;
    if (usable <= 0) return 0;
    int64_t mib = usable * 2 / 5;
    if (mib > 8192) mib = 8192;
    mib -= mib % 256;
    return mib >= 1024 ? mib : 0;
}

/// MOD_PLAN 86 (first slice of 69): the conversation cache from a fixed RAM budget instead of 40% of what is free.  After the
/// arena is loaded (item 38 reads it with O_DIRECT, so no leftover page cache moves the figure between starts), the cache may
/// take what is available less the RAM to keep and a margin: available - max(keep, min_free) - margin, at most `cap_mib`
/// (0: no cap), in steps of 256, and 0 (off) under 1024 - the same steps and the same off rule as auto_conversation_cache_mib.
/// `keep` is --ram-keep-gib / STRATA_RAM_KEEP_GIB (or STRATA_CONV_KEEP_GIB for this rule alone); the parking floor
/// (--conversation-cache-min-free-mib) is the least that is kept.  Unknown telemetry: 0.
inline int64_t ram_budget_conversation_cache_mib(std::optional<uint64_t> available, uint64_t keep, int64_t min_free_mib,
                                                 int64_t margin_mib, int64_t cap_mib) {
    if (!available) return 0;
    const int64_t keep_mib = std::max<int64_t>((int64_t) (keep >> 20), std::max<int64_t>(min_free_mib, 0));
    int64_t mib = (int64_t) (*available >> 20) - keep_mib - std::max<int64_t>(margin_mib, 0);
    if (mib <= 0) return 0;
    if (cap_mib > 0 && mib > cap_mib) mib = cap_mib;
    mib -= mib % 256;
    return mib >= 1024 ? mib : 0;
}

/// Which rule sizes the automatic conversation cache, and its inputs.
struct ConversationBudgetPolicy {
    bool ram = false;            ///< ram_budget_conversation_cache_mib; false: auto_conversation_cache_mib (#3493)
    uint64_t keep = 0;           ///< bytes
    bool keep_from_env = false;  ///< STRATA_CONV_KEEP_GIB gave `keep` (else --ram-keep-gib)
    int64_t min_free_mib = 2560;
    int64_t margin_mib = 512;
    int64_t cap_mib = 8192;
};

/// From the switches (null = unset; a value that is not a number keeps the default):
///   STRATA_CONV_BUDGET    "ram" | "legacy"; unset/"auto": the RAM rule on Linux when a RAM to keep is set (--ram-keep-gib,
///                         STRATA_RAM_KEEP_GIB or STRATA_CONV_KEEP_GIB), else the 40% rule
///   STRATA_CONV_KEEP_GIB  the RAM to keep for this rule (replaces --ram-keep-gib here only)
///   STRATA_CONV_BUDGET_MARGIN_MIB (512), STRATA_CONV_BUDGET_CAP_MIB (8192; 0 = no cap)
ConversationBudgetPolicy conversation_budget_policy(const char* mode, const char* keep_gib, const char* margin_mib,
                                                    const char* cap_mib, uint64_t ram_keep, int64_t min_free_mib,
                                                    bool linux_host);

inline int64_t conversation_cache_budget_mib(std::optional<uint64_t> available, const ConversationBudgetPolicy& p) {
    return p.ram ? ram_budget_conversation_cache_mib(available, p.keep, p.min_free_mib, p.margin_mib, p.cap_mib)
                 : auto_conversation_cache_mib(available, p.min_free_mib);
}

/// #4518 under either rule: would `extra` more bytes available (the PLE table's lock given back) make the cache bigger?
inline bool conversation_budget_grows(std::optional<uint64_t> available, uint64_t extra, const ConversationBudgetPolicy& p) {
    if (!available || extra == 0 || *available > UINT64_MAX - extra) return false;
    return conversation_cache_budget_mib(*available + extra, p) > conversation_cache_budget_mib(available, p);
}

/// The parked-image size model for the log (measured 2026-11-09 on the daily preset: 540 tokens = 243.8 MB, 2584 tokens =
/// 272.9 MB): about 236 MB fixed (the live state and one checkpoint) plus 14.2 KB per token.  The tokens one image of
/// `bytes` holds, 0 below the fixed part.
inline uint64_t parked_tokens_in(uint64_t bytes) {
    constexpr uint64_t kFixed = 236000000ull, kPerToken = 14200ull;
    return bytes > kFixed ? (bytes - kFixed) / kPerToken : 0;
}

std::optional<uint64_t> conversation_mem_available(std::istream& meminfo);

inline bool conversation_memory_admit(std::optional<uint64_t> available,
                                      uint64_t allocation, uint64_t floor) {
    return available && *available >= floor && allocation <= *available - floor;
}

} // namespace strata::core
