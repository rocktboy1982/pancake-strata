// LEADS.md Part U U7 (STRATA_VERIFY_GEMV_CHAINS=2|4): host test of the integer re-association the chain kernels of
// src/kernels/cuda/rdna_mmvq.cu rely on.  No GPU, no model.
//  1. dot8<1> (the existing dependent chain) against dot8<2> / dot8<4> (two independent sudot4 joined by an i32 add):
//     identical for random operands, for every corner byte pattern (-128, -1, 0, 1, 127 in each of the 8 positions)
//     and for the activation-sum dot (weights 0x01010101).
//  2. the whole lane stage of a Q4_K-shaped row (superblocks, columns, f32 fmaf fold, the final lane sum) computed in
//     the original order and in the phase-separated order of CH 4: bitwise equal f32 results.
//  3. the switch parser: unset / "" / "1" / "3" / "22" = off, "2" and "4" select.
#include "strata/kernels/rdna_chain_split.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace rc = strata::kernels::rdna_chain;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// host model of v_dot4_i32_iu8 with both sign flags set and no clamp: signed bytes, products added modulo 2^32
struct HostDot {
    static int dot4(int w, int q, int acc) {
        uint32_t sum = (uint32_t) acc;
        for (int i = 0; i < 4; ++i) {
            const int a = (int8_t) (((uint32_t) w >> (8 * i)) & 0xffu), b = (int8_t) (((uint32_t) q >> (8 * i)) & 0xffu);
            sum += (uint32_t) (a * b);
        }
        return (int) sum;
    }
};

uint32_t pack(const int8_t* v) {
    return (uint32_t) (uint8_t) v[0] | ((uint32_t) (uint8_t) v[1] << 8) | ((uint32_t) (uint8_t) v[2] << 16) | ((uint32_t) (uint8_t) v[3] << 24);
}

void same(int w0, int w1, int q0, int q1, long& n, long& bad) {
    const int r1 = rc::dot8<1, HostDot>(w0, w1, q0, q1);
    const int r2 = rc::dot8<2, HostDot>(w0, w1, q0, q1);
    const int r4 = rc::dot8<4, HostDot>(w0, w1, q0, q1);
    // independent oracle: a plain 8-term loop in 64-bit, truncated
    long long o = 0;
    const int8_t* pw[2] = {nullptr, nullptr};
    (void) pw;
    for (int h = 0; h < 2; ++h) {
        const uint32_t w = (uint32_t) (h ? w1 : w0), q = (uint32_t) (h ? q1 : q0);
        for (int i = 0; i < 4; ++i) o += (long long) (int8_t) ((w >> (8 * i)) & 0xffu) * (int8_t) ((q >> (8 * i)) & 0xffu);
    }
    ++n;
    if (r1 != r2 || r1 != r4 || r1 != (int) (uint32_t) (uint64_t) o) ++bad;
}
}  // namespace

int main() {
    std::mt19937_64 rng(7);
    long n = 0, bad = 0;
    for (int i = 0; i < 2000000; ++i) {
        const uint64_t a = rng(), b = rng();
        same((int) (uint32_t) a, (int) (uint32_t) (a >> 32), (int) (uint32_t) b, (int) (uint32_t) (b >> 32), n, bad);
    }
    check(bad == 0, "random operands: chain 1 == chain 2 == chain 4 == 64-bit oracle");

    // corner bytes: 5 values in each of the 8 positions of w and q (5^8 w patterns x 64 q patterns)
    const int8_t cv[5] = {-128, -1, 0, 1, 127};
    long cn = 0, cbad = 0;
    for (int wi = 0; wi < 390625; ++wi) {
        int8_t wb[8];
        int t = wi;
        for (int k = 0; k < 8; ++k) { wb[k] = cv[t % 5]; t /= 5; }
        for (int qi = 0; qi < 64; ++qi) {
            int8_t qb[8];
            for (int k = 0; k < 8; ++k) qb[k] = cv[(qi * 7 + k * 3 + (qi >> 2) * k) % 5];
            same((int) pack(wb), (int) pack(wb + 4), (int) pack(qb), (int) pack(qb + 4), cn, cbad);
        }
    }
    check(cbad == 0, "corner bytes (-128 .. 127 in all 8 positions)");
    // the activation-sum dot: weights all ones
    long on = 0, obad = 0;
    for (int i = 0; i < 500000; ++i) {
        const uint64_t b = rng();
        same(0x01010101, 0x01010101, (int) (uint32_t) b, (int) (uint32_t) (b >> 32), on, obad);
    }
    check(obad == 0, "activation-sum dot (weights 0x01010101)");

    // 2. the lane stage: NC columns, NSB superblocks; per sb: d1 = dot(w, q), d2 = dot(1, q); acc[c] = fmaf(d8, f1*d1 - f2*d2, acc[c])
    long lanes = 0, lbad = 0;
    for (int trial = 0; trial < 3000; ++trial) {
        const int NC = 1 + (int) (rng() % 16), NSB = 1 + (int) (rng() % 12);
        std::vector<float> a1(NC, 0.f), a4(NC, 0.f);
        for (int sb = 0; sb < NSB; ++sb) {
            const int w0 = (int) (uint32_t) rng(), w1 = (int) (uint32_t) rng();
            const float f1 = (float) ((int) (rng() % 2000) - 1000) * 0.001f, f2 = (float) ((int) (rng() % 2000) - 1000) * 0.001f;
            std::vector<int> q0(NC), q1(NC);
            std::vector<float> d8(NC);
            for (int c = 0; c < NC; ++c) { q0[c] = (int) (uint32_t) rng(); q1[c] = (int) (uint32_t) rng(); d8[c] = (float) (rng() % 4096) * 1e-4f; }
            for (int c = 0; c < NC; ++c) {   // original: per column, chain dots
                const int s1 = rc::dot8<1, HostDot>(w0, w1, q0[c], q1[c]), s2 = rc::dot8<1, HostDot>(0x01010101, 0x01010101, q0[c], q1[c]);
                a1[c] = std::fmaf(d8[c], f1 * (float) s1 - f2 * (float) s2, a1[c]);
            }
            std::vector<int> D1(NC), D2(NC);   // CH 4: every dot first, then the folds in the same column order
            for (int c = 0; c < NC; ++c) {
                D1[c] = rc::dot8<4, HostDot>(w0, w1, q0[c], q1[c]);
                D2[c] = rc::dot8<4, HostDot>(0x01010101, 0x01010101, q0[c], q1[c]);
            }
            for (int c = 0; c < NC; ++c) a4[c] = std::fmaf(d8[c], f1 * (float) D1[c] - f2 * (float) D2[c], a4[c]);
        }
        ++lanes;
        if (std::memcmp(a1.data(), a4.data(), (size_t) NC * 4) != 0) ++lbad;
    }
    check(lbad == 0, "lane stage (f32 fold) bitwise equal between the original order and the phase-separated order");

    check(rc::parse_chains(nullptr) == 0, "unset = off");
    check(rc::parse_chains("") == 0, "empty = off");
    check(rc::parse_chains("1") == 0, "1 = off");
    check(rc::parse_chains("3") == 0, "3 = off");
    check(rc::parse_chains("22") == 0, "22 = off");
    check(rc::parse_chains("2") == 2, "2");
    check(rc::parse_chains("4") == 4, "4");

    std::printf("%s: %ld dot cases (random + corners + sums), %ld lane-stage trials\n", g_fail ? "FAIL" : "PASS", n + cn + on, lanes);
    return g_fail ? 1 : 0;
}
