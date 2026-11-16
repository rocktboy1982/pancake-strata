// STRATA_STOP_LAYERS: where a prompt read polls for a Stop (header-only, no GPU).
#include <cstdio>

#include "strata/prefill/stop_rule.hpp"

using namespace strata::prefill::stop_rule;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    // the switch: unset, empty, 0 and text are off; 1..9 on
    CHECK(!parse_enabled(nullptr));
    CHECK(!parse_enabled(""));
    CHECK(!parse_enabled("0"));
    CHECK(!parse_enabled("off"));
    CHECK(parse_enabled("1"));
    CHECK(parse_enabled("2"));
    // off: no layer polls (the byte-identical default)
    for (int64_t l = 0; l < 64; ++l) CHECK(!poll_before_layer(false, true, l, 0));
    // on: every layer after the first polls; the first layer of the chunk never does (the chunk start just polled)
    CHECK(!poll_before_layer(true, true, 0, 0));
    for (int64_t l = 1; l < 64; ++l) CHECK(poll_before_layer(true, true, l, 0));
    // a stage that starts at layer 20 skips only layer 20
    CHECK(!poll_before_layer(true, true, 20, 20));
    CHECK(poll_before_layer(true, true, 21, 20));
    // no hook: nothing to ask
    CHECK(!poll_before_layer(true, false, 5, 0));
    // a Stop raised at layer k ends the loop at layer k, not at the chunk's end
    int64_t reached = -1;
    const int64_t stop_at = 7, LB = 0, LE = 48;
    for (int64_t l = LB; l < LE; ++l) {
        if (poll_before_layer(true, true, l, LB) && l >= stop_at) break;
        reached = l;
    }
    CHECK(reached == stop_at - 1);
    static_assert(!poll_before_layer(false, true, 3, 0), "constexpr");
    if (failed == 0) std::puts("stop_rule_test OK");
    return failed == 0 ? 0 : 1;
}
