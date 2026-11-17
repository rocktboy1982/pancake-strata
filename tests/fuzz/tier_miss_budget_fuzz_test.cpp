// Fuzz of the STRATA_TIER_MISS_FILE text table loader (include/strata/program/tier_miss_budget.hpp TierMissBudget::load_text,
// "strata-gate-importance 1 <layers> <experts>"): prefixes, byte flips and random blobs of a valid table, plus headers that
// claim billions of layers and experts. Asserts: no sanitizer report, no exception, no allocation over 256 MiB, a true
// return only for a consistent object.
// Run (host-only, no engine, no GPU):
//   clang++ -std=c++20 -fsanitize=address,undefined -fno-sanitize-recover=undefined -g -O1 -Iinclude -Itests/fuzz tests/fuzz/tier_miss_budget_fuzz_test.cpp -o tier_miss_budget_fuzz_test && ./tier_miss_budget_fuzz_test
#include "fuzz_common.hpp"
#include "strata/program/tier_miss_budget.hpp"

using namespace strata::program;

static bool one(const std::string& text, size_t& ok_count) {
    TierMissBudget t;
    bool ok = false;
    try { ok = t.load_text(text); } catch (const std::exception& e) { fuzz::fail("exception", e.what()); }
    if (!ok) { if (t.layers() != 0 || t.experts() != 0 || t.on()) fuzz::fail("refused table left state", ""); return false; }
    ++ok_count;
    if (t.layers() <= 0 || t.experts() <= 0) fuzz::fail("accepted empty shape", "");
    t.build(0.5);
    for (int64_t l = -1; l <= t.layers(); ++l) for (int32_t e = -1; e <= (int32_t) t.experts(); ++e) (void) t.may_stay_out(l, e);
    return true;
}

int main() {
    const std::string sample = "strata-gate-importance 1 3 4\n0.1 0.2 0.3 0.4\n1 0 0 0\n0.25 0.25 0.25 0.25\n";
    const std::vector<uint8_t> valid(sample.begin(), sample.end());
    size_t ok_count = 0;
    if (!one(sample, ok_count)) fuzz::fail("valid sample refused", "");
    // regression inputs: sizes whose product overflows, or that claim billions of values in a tiny file
    const char* huge[] = {
        "strata-gate-importance 1 4000000000 4000000000\n1 2 3",
        "strata-gate-importance 1 9223372036854775807 2\n1 2",
        "strata-gate-importance 1 2 9223372036854775807\n1 2",
        "strata-gate-importance 1 3037000500 3037000500\n1",
        "strata-gate-importance 1 1 4000000000\n1",
    };
    for (const char* h : huge) {
        TierMissBudget t; bool ok = true;
        try { ok = t.load_text(h); } catch (const std::exception&) { fuzz::fail("exception on a huge header", h); }
        if (ok) fuzz::fail("huge header accepted", h);
    }
    const size_t cases = fuzz::run_all(valid, [&](const uint8_t* p, size_t n) { one(std::string((const char*) p, n), ok_count); });
    if (fuzz::g_big_allocs != 0) fuzz::fail("allocation over the cap", "refused allocations seen");
    std::printf("tier_miss_budget_fuzz_test: PASS, %zu cases + %zu huge headers (%zu accepted), %zu over-cap allocations\n", cases, sizeof huge / sizeof *huge, ok_count, fuzz::g_big_allocs);
    return 0;
}
