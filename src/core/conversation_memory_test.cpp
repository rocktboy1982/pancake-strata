#include "strata/core/conversation_memory.hpp"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <sstream>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using namespace strata::core;
namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
}

int main() {
    for (const char* text : {"", "MemFree: 100 kB\n", "MemAvailable: -1 kB\n",
                            "MemAvailable: +1 kB\n", "MemAvailable: 1 MB\n",
                            "MemAvailable: 1\n", "MemAvailable: 1x kB\n",
                            "MemAvailable: 18446744073709551615 kB\n",
                            "MemAvailable: 18446744073709551616 kB\n",
                            "MemAvailable: 1 kB trailing\n",
                            "MemAvailable: 1 kB\nMemAvailable: 2 kB\n"}) {
        std::istringstream input(text);
        check(!conversation_mem_available(input), "missing/malformed telemetry is unknown");
    }
    std::istringstream normal("MemTotal: 999999 kB\nMemAvailable:    12345 kB\nSwapFree: 777 kB\n");
    check(conversation_mem_available(normal) == 12345ULL * 1024, "only MemAvailable is counted");
    std::istringstream zero("MemAvailable: 0 kB");
    check(conversation_mem_available(zero) == 0, "zero available memory is known");
    std::istringstream broken("MemAvailable: 123 kB\n");
    broken.setstate(std::ios::badbit);
    check(!conversation_mem_available(broken), "I/O failure declines admission");
    check(!conversation_memory_admit({}, 0, 0), "unknown fails closed even with zero floor");
    check(conversation_memory_admit(100, 40, 60), "exact allocation plus floor fits");
    check(!conversation_memory_admit(99, 40, 60), "one byte below required memory rejected");
    check(!conversation_memory_admit(59, 0, 60), "floor subtraction cannot underflow");
    check(conversation_memory_admit(60, 0, 60), "post-capture floor check");
    check(!conversation_memory_admit(100, std::numeric_limits<uint64_t>::max(), 1), "allocation arithmetic cannot overflow");
    check(conversation_memory_admit(std::numeric_limits<uint64_t>::max(),
                                   std::numeric_limits<uint64_t>::max(), 0), "maximal exact bound");
    check(!conversation_memory_admit(std::numeric_limits<uint64_t>::max(),
                                    std::numeric_limits<uint64_t>::max(), 1), "maximal sum overflow rejected");
    // #3493: the budget used when --conversation-cache-mib is not given
    const auto GiB = [](uint64_t n) { return n << 30; };
    check(auto_conversation_cache_mib({}, 2560) == 0, "unknown telemetry: no cache");
    check(auto_conversation_cache_mib(GiB(2), 2560) == 0, "below the parking floor: off");
    check(auto_conversation_cache_mib(GiB(4), 2560) == 0, "4 GiB free: 40% of 1.5 GiB is under 1 GiB: off");
    check(auto_conversation_cache_mib(GiB(8), 2560) == 2048, "8 GiB free: 40% of 5.5 GiB, in steps of 256");
    check(auto_conversation_cache_mib(GiB(20), 2560) == 7168, "20 GiB free: 40% of 17.5 GiB");
    check(auto_conversation_cache_mib(GiB(40), 2560) == 8192, "capped at 8 GiB");
    check(auto_conversation_cache_mib(GiB(1000), 0) == 8192, "a big server: still the cap");
    check(auto_conversation_cache_mib(GiB(40), 1000000) == 0, "a floor above the RAM: off, no underflow");
    // MOD_PLAN 86: the RAM-budget rule: available - max(keep, floor) - margin, cap, steps of 256, off under 1024
    check(ram_budget_conversation_cache_mib({}, GiB(12), 2560, 512, 8192) == 0, "ram rule: unknown telemetry: no cache");
    check(ram_budget_conversation_cache_mib(GiB(16), GiB(12), 2560, 512, 8192) == 3584, "ram rule: 16 - 12 keep - 0.5 margin");
    check(ram_budget_conversation_cache_mib(GiB(16) + (100ull << 20), GiB(12), 2560, 512, 8192) == 3584,
          "ram rule: rounded down to 256");
    check(ram_budget_conversation_cache_mib(GiB(13), GiB(12), 2560, 512, 8192) == 0, "ram rule: 512 MiB left: off");
    check(ram_budget_conversation_cache_mib(GiB(12), GiB(12), 2560, 512, 8192) == 0, "ram rule: below keep: off, no underflow");
    check(ram_budget_conversation_cache_mib(GiB(8), 0, 2560, 512, 8192) == 5120, "ram rule: no keep: the floor is kept");
    check(ram_budget_conversation_cache_mib(GiB(8), GiB(1), 2560, 512, 8192) == 5120, "ram rule: keep under the floor: the floor");
    check(ram_budget_conversation_cache_mib(GiB(40), GiB(12), 2560, 512, 8192) == 8192, "ram rule: capped");
    check(ram_budget_conversation_cache_mib(GiB(40), GiB(12), 2560, 512, 0) == 28160, "ram rule: cap 0 = none");
    check(ram_budget_conversation_cache_mib(GiB(16), GiB(12), 2560, -5, 8192) == 4096, "ram rule: a negative margin is 0");
    {
        // the switches
        auto p = conversation_budget_policy(nullptr, nullptr, nullptr, nullptr, GiB(12), 2560, true);
        check(!p.ram && p.keep == GiB(12) && !p.keep_from_env && p.margin_mib == 512 && p.cap_mib == 8192,
              "policy: auto on Linux with only --ram-keep-gib: still the 40% rule (RAM rule is opt-in)");
        p = conversation_budget_policy(nullptr, nullptr, nullptr, nullptr, 0, 2560, true);
        check(!p.ram, "policy: auto without a RAM to keep: the 40% rule");
        p = conversation_budget_policy("", nullptr, nullptr, nullptr, GiB(12), 2560, false);
        check(!p.ram, "policy: auto off Linux: the 40% rule");
        p = conversation_budget_policy("auto", "8", nullptr, nullptr, 0, 2560, true);
        check(p.ram && p.keep == GiB(8) && p.keep_from_env, "policy: STRATA_CONV_KEEP_GIB alone selects the RAM rule");
        p = conversation_budget_policy("legacy", nullptr, nullptr, nullptr, GiB(12), 2560, true);
        check(!p.ram, "policy: legacy forced");
        p = conversation_budget_policy("ram", nullptr, "1024", "0", 0, 2560, false);
        check(p.ram && p.margin_mib == 1024 && p.cap_mib == 0, "policy: ram forced anywhere, margin and cap");
        p = conversation_budget_policy("ram", "x", "-3", "abc", GiB(2), 2560, true);
        check(p.keep == GiB(2) && !p.keep_from_env && p.margin_mib == 512 && p.cap_mib == 8192, "policy: junk values keep defaults");
        // the legacy policy is exactly the #3493 rule, and the #4518 check is ple_lock_blocks_cache's
        const auto legacy = conversation_budget_policy("legacy", nullptr, nullptr, nullptr, 0, 2560, true);
        for (uint64_t g : {0ull, 2ull, 4ull, 8ull, 20ull, 40ull, 1000ull}) {
            check(conversation_cache_budget_mib(GiB(g), legacy) == auto_conversation_cache_mib(GiB(g), 2560), "legacy = #3493");
            check(conversation_budget_grows(GiB(g), GiB(3), legacy) ==
                  (auto_conversation_cache_mib(GiB(g) + GiB(3), 2560) > auto_conversation_cache_mib(GiB(g), 2560)), "grows = #4518");
        }
        check(!conversation_budget_grows({}, GiB(3), legacy), "grows: unknown telemetry: no");
        const auto ram = conversation_budget_policy("ram", nullptr, nullptr, nullptr, GiB(12), 2560, true);
        check(conversation_cache_budget_mib(GiB(16), ram) == 3584, "ram policy sizes with the RAM rule");
        check(conversation_budget_grows(GiB(13), GiB(3), ram), "ram rule: giving back 3 GiB turns the cache on");
    }
    // the parked-image model (236 MB + 14.2 KB/token)
    check(parked_tokens_in(0) == 0 && parked_tokens_in(236000000ull) == 0, "parked model: nothing under the fixed part");
    check(parked_tokens_in(236000000ull + 14200ull * 70000ull) == 70000, "parked model: 70K tokens");
    // Test the real provider without assuming any particular amount of free RAM.
    const auto available = available_host_bytes();
    check(!available || conversation_memory_admit(available, 0, 0), "provider returns bytes or unknown");
#if defined(_WIN32)
    // Unknown must fail closed in production, but must not let a broken Windows provider pass this test.
    // Compare with total physical memory, not a second available reading: other processes can allocate
    // between calls. No pressure allocation or fixed free-memory assumption is needed.
    MEMORYSTATUSEX physical{};
    physical.dwLength = sizeof physical;
    check(GlobalMemoryStatusEx(&physical) != 0, "Windows physical-memory API is available");
    check(available.has_value() && *available <= physical.ullTotalPhys,
          "Windows provider returns a known, physically bounded sample");
#endif
    std::printf("conversation_memory_test: %d checks passed\n", checks);
}
