// Fuzz of the env-value parsers added with the 2026-11 switches: STRATA_MISS_BUDGET (miss_budget_parse), STRATA_VERIFY_ECAP /
// _ROWS / _MINP (verify_ecap_parse, verify_ecap_rows_parse, verify_ecap_minp_parse), STRATA_CKPT stride (ckpt_stride_from_env),
// STRATA_WINDOW_POLICY (window_policy_parse), STRATA_ROUTE_RANGE (route_range_parse), STRATA_PREFETCH_PREV (prefetch_prev_parse)
// and the LKW1 header check of STRATA_LOOKAHEAD_WEIGHTS (lkw_header_ok; the weights loader itself sizes every read from the
// model's shape, never from the file, and lives in src/core/expert_source.cpp). Inputs: every prefix, byte flips and 2,000
// random blobs of valid samples (cut at the first NUL like an environment string). Asserts: no sanitizer report, results in
// their documented ranges, ROWS output within kEcapRowsMax values.
// Run (host-only, no engine, no GPU):
//   clang++ -std=c++20 -fsanitize=address,undefined -fno-sanitize-recover=undefined -g -O1 -Iinclude -Itests/fuzz tests/fuzz/env_parsers_fuzz_test.cpp -o env_parsers_fuzz_test && ./env_parsers_fuzz_test
#include "fuzz_common.hpp"
#include "strata/core/miss_budget.hpp"
#include "strata/core/verify_ecap.hpp"
#include "strata/program/ckpt_stride.hpp"
#include "strata/program/lookahead_weights.hpp"
#include "strata/program/prefetch_prev.hpp"
#include "strata/program/route_range.hpp"
#include "strata/spec/window_policy.hpp"

#include <cmath>
#include <cstring>

#define REQ(c) do { if (!(c)) fuzz::fail("range", #c); } while (0)

static size_t g_lkw_ok = 0;

static void parse_text(const std::string& s) {
    const char* v = s.c_str();   // a NUL inside the blob ends the string, as in an environment
    const float mb = strata::core::miss_budget_parse(v);
    REQ(mb >= 0.0f && mb < 1.0f);
    const int ec = strata::core::verify_ecap_parse(v);
    REQ(ec == -1 || ec == -2 || (ec >= 0 && ec < 64));
    int8_t rows[strata::core::kEcapRowsMax + 4];
    std::memset(rows, 0x5a, sizeof rows);
    const int nr = strata::core::verify_ecap_rows_parse(v, rows);
    REQ(nr == -2 || (nr >= 0 && nr <= strata::core::kEcapRowsMax));
    for (int i = 0; i < nr; ++i) REQ(rows[i] >= -1 && rows[i] < 64);
    for (int i = strata::core::kEcapRowsMax; i < (int) sizeof rows; ++i) REQ(rows[i] == 0x5a);   // nothing written past the list cap
    if (nr > 0) for (int64_t w = -1; w < 40; ++w) (void) strata::core::verify_ecap_row_cap(rows, nr, 3, w);
    const float mp = strata::core::verify_ecap_minp_parse(v);
    REQ(mp == -1.0f || mp == -2.0f || (mp > 0.0f && mp <= 1.0f));
    const auto cs = strata::program::ckpt_stride_from_env(v, v);
    REQ(cs.stride >= 0 && cs.max_count >= 0);
    (void) strata::program::ckpt_next_check(cs, 0, 1000, 512);
    const auto wp = strata::spec::window_policy_parse(v);
    REQ(wp.n >= -1 && wp.n <= 64 && wp.margin >= 0 && wp.margin <= 16 && wp.cap >= 0 && wp.cap <= 63);
    const float rr = strata::program::route_range_parse(v);
    REQ(rr >= 0.0f && rr <= 4.0f);
    const int pp = strata::program::prefetch_prev_parse(v);
    REQ(pp >= 0 && pp <= strata::program::kPrefetchPrevMaxK);
}

int main() {
    const char* samples[] = {"0.25", "7", "0,3,63,-1,5", "8,2,4", "1.5", "4", "-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17", "  12abc", "1e400", "nan", "0x1f", "9999999999999999999999"};
    size_t cases = 0;
    for (const char* s : samples) {
        const std::vector<uint8_t> valid(s, s + std::strlen(s));
        cases += fuzz::run_all(valid, [](const uint8_t* p, size_t n) { parse_text(std::string((const char*) p, n)); });
    }
    // LKW1 header: valid, then flips and blobs of 16 bytes
    {
        std::vector<uint8_t> valid(16);
        const uint32_t h[4] = {strata::program::kLkwMagic, 40, 256, 2048};
        std::memcpy(valid.data(), h, 16);
        REQ(strata::program::lkw_header_ok(h, 40, 256, 2048));
        cases += fuzz::run_all(valid, [](const uint8_t* p, size_t n) {
            uint32_t head[4] = {0, 0, 0, 0};
            std::memcpy(head, p, n < 16 ? n : 16);
            if (strata::program::lkw_header_ok(head, 40, 256, 2048)) {
                ++g_lkw_ok;
                REQ(head[0] == strata::program::kLkwMagic && head[1] == 40 && head[2] == 256 && head[3] == 2048);
            }
            (void) strata::program::lkw_file_bytes(head[1], head[2], head[3]);   // 64-bit math: no UB for any header
        });
    }
    if (fuzz::g_big_allocs != 0) fuzz::fail("allocation over the cap", "refused allocations seen");
    std::printf("env_parsers_fuzz_test: PASS, %zu cases over %zu text samples + the LKW1 header (%zu header hits), %zu over-cap allocations\n",
                cases, sizeof samples / sizeof *samples, g_lkw_ok, fuzz::g_big_allocs);
    return 0;
}
