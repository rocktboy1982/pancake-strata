// MOD_PLAN.md section 6 item 50: STRATA_PLE_WARM's plan and sweep (include/strata/program/ple_warm.hpp).
#include "strata/program/ple_warm.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace strata::program;

static int checks = 0;
static void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

int main() {
    const uint64_t GiB = (uint64_t) 1 << 30;
    check(!ple_warm_plan(false, true, false, 29 * GiB, 50 * GiB, 10 * GiB, 8 * GiB).run, "off: no run");
    check(!ple_warm_plan(true, false, false, 29 * GiB, 50 * GiB, 10 * GiB, 8 * GiB).run, "direct reads: declined");
    check(!ple_warm_plan(true, true, true, 29 * GiB, 50 * GiB, 10 * GiB, 8 * GiB).run, "locked table: declined");
    check(!ple_warm_plan(true, true, false, 0, 50 * GiB, 10 * GiB, 8 * GiB).run, "no file: declined");
    check(!ple_warm_plan(true, true, false, 29 * GiB, 18 * GiB, 10 * GiB, 8 * GiB).run, "available == arena + headroom: declined");
    const auto whole = ple_warm_plan(true, true, false, 29 * GiB, 50 * GiB, 10 * GiB, 8 * GiB);
    check(whole.run && whole.bytes == 29 * GiB, "room for all: the whole file");
    const auto part = ple_warm_plan(true, true, false, 29 * GiB, 30 * GiB, 10 * GiB, 8 * GiB);
    check(part.run && part.bytes == 12 * GiB, "partial: only what is left after arena + headroom");
    // the sweep reads exactly `bytes` of a real file, stops at the end of a short one and at a stop flag
    const std::string path = std::string(std::getenv("TEMP") ? std::getenv("TEMP") : ".") + "/ple_warm_test.bin";
    {
        FILE* f = std::fopen(path.c_str(), "wb");
        check(f != nullptr, "temp file");
        std::string blk(1 << 20, 'x');
        for (int i = 0; i < 5; ++i) std::fwrite(blk.data(), 1, blk.size(), f);
        std::fclose(f);
    }
    check(ple_warm_read(path, 3 << 20, 1 << 20) == (3u << 20), "reads exactly the asked bytes");
    check(ple_warm_read(path, 100u << 20, 1 << 20) == (5u << 20), "a short file ends the sweep at its size");
    check(ple_warm_read(path, 5u << 20, 3 << 20) == (5u << 20), "a block that does not divide the size");
    std::atomic<bool> stop{true};
    check(ple_warm_read(path, 5u << 20, 1 << 20, &stop) == 0, "a stop flag set before the sweep reads nothing");
    check(ple_warm_read(path + ".missing", 1 << 20) == 0, "a missing file reads nothing");
    std::remove(path.c_str());
    std::printf("ple_warm_test: %d checks passed\n", checks);
    return 0;
}
