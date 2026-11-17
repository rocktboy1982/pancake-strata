// raw_buffer_test.cpp - host test of the descriptor packing in include/strata/kernels/raw_buffer.hpp (LEADS.md Part L L11).
// The device loads are the probe's job (tools/pancake/probes/raw_buffer.hip.cpp, box run); this checks the four words.
#include <cstdio>

#include "strata/kernels/raw_buffer.hpp"

int main() {
    using namespace strata::kernels::rawbuf;
    int fail = 0;
    auto check = [&](bool c, const char* what) { if (!c) { std::printf("FAIL %s\n", what); ++fail; } };
    const Rsrc r = make_rsrc(0x00007f12'3456'7890ull, 1234567u);
    check(r.w[0] == 0x34567890u, "base low word");
    check(r.w[1] == 0x00007f12u, "base high word, stride 0 in bits 31:16");
    check(r.w[2] == 1234567u, "size in bytes");
    check(r.w[3] == kGfx11Word3, "flags word");
    const Rsrc big = make_rsrc(0xffff'ffff'ffff'ffffull, 4u, kGfx10Word3);   // bits above 47 never leak into the stride field
    check(big.w[1] == 0xffffu && big.w[0] == 0xffffffffu && big.w[3] == kGfx10Word3, "base bits above 47 are dropped");
    check(kAuxSlc == 2 && kAuxGlc == 1 && kAuxDlc == 4, "cache policy bits");
    if (fail) return 1;
    std::printf("raw_buffer_test: PASS\n");
    return 0;
}
