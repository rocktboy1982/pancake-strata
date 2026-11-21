// window_canary_test: the fill-and-check helpers of STRATA_WINDOW_CANARY (host only; LEADS.md Part U U9), with a deliberate overrun.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "strata/core/window_canary.hpp"

using namespace strata::core::window_canary;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    // default off: the env is unset in this process
    CHECK(!enabled());
    // window sizes
    CHECK(window_fits(1, 16) && window_fits(16, 16) && !window_fits(17, 16) && !window_fits(0, 16));
    // an honest window: fills rows [live, cap), writes only rows < live, the guard stays intact
    const size_t cap = 16 + 4, live = 12;   // 16 rows + 16 bytes of slack, like h_out_
    std::vector<int32_t> buf(cap, 7);
    CHECK(fill(buf.data(), live, cap) == cap - live);
    for (size_t i = 0; i < live; ++i) buf[i] = (int32_t) i;
    CHECK(first_bad(buf.data(), live, cap) == -1);
    CHECK(describe("h_out_", 12, live, cap, -1).empty());
    // a deliberate overrun by one row: the first changed word is named
    buf[live] = 0;
    CHECK(first_bad(buf.data(), live, cap) == (long) live);
    const std::string d = describe("h_out_", 12, live, cap, first_bad(buf.data(), live, cap));
    CHECK(d.find("h_out_") != std::string::npos && d.find("first changed word 12") != std::string::npos && d.find("12-token") != std::string::npos);
    // an overrun in the slack past the last row is found too
    fill(buf.data(), live, cap);
    buf[cap - 1] = 5;
    CHECK(first_bad(buf.data(), live, cap) == (long) (cap - 1));
    // a write that equals the pattern word cannot hide: only the pattern value itself is invisible (one value in 2^32)
    CHECK(kPattern != 0 && kPattern != -1);
    // degenerate extents: nothing to fill, nothing to check, null buffer safe
    CHECK(fill(buf.data(), cap, cap) == 0 && first_bad(buf.data(), cap, cap) == -1);
    CHECK(fill(nullptr, 0, 4) == 0 && first_bad(nullptr, 0, 4) == -1);
    // the env switch
#ifdef _WIN32
    _putenv_s("STRATA_WINDOW_CANARY", "1");
#else
    setenv("STRATA_WINDOW_CANARY", "1", 1);
#endif
    CHECK(enabled());
#ifdef _WIN32
    _putenv_s("STRATA_WINDOW_CANARY", "0");
#else
    setenv("STRATA_WINDOW_CANARY", "0", 1);
#endif
    CHECK(!enabled());
    if (fails == 0) std::printf("window_canary_test: all ok\n");
    return fails == 0 ? 0 : 1;
}
