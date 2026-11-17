// tests/prefill/combine_write_rule_test.cpp - host check of include/strata/prefill/combine_write_rule.hpp (T-630): the fused
// moe_combine_write_norm_rs is refused when xn16 shares the MoE set's memory (STRATA_PF_HC_REGION), allowed otherwise, and the
// image of a 64..2304-token chunk reaches the MoE inputs there.
#include "strata/prefill/combine_write_rule.hpp"

#include <cstdio>
#include <initializer_list>

namespace cw = strata::prefill::combine_write;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    CHECK(cw::allowed(false) && !cw::allowed(true));
    for (long long T : {7LL, 64LL, 71LL, 2245LL, 2304LL})
        for (long long pad : {0LL, 64LL}) CHECK(cw::hc_image_reaches_moe_inputs(T, 256, 10240, pad));
    std::printf("combine_write_rule_test: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
