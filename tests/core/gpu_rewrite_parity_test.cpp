// MOD_PLAN Phase 15 (F14): host proof that the Q2_0 down row on 4 lanes keeps its bits.  No GPU, no model, no engine.
//
// native_down_q2_sub4_kernel (STRATA_EXP_MODE=9 + STRATA_EXP9_DOWN=42:10): the Q2_0 down row at n_ff 640 sums 20 calls.
// Mode 0 (row_dot_20_sub16) puts call t on lane t of 16 (call t + 16 for t < 4), joins the two with a plain add, then
// runs the xor 8, 4, 2, 1 butterfly.  The 4-lane kernel gives lane u of 4 the calls u, u + 4, ... u + 16 and adds them
// in the butterfly's own first two levels, then shuffles xor 2 and xor 1.  Both trees are modelled with the lanes as
// arrays and compared BITWISE on random call values (zeros, negative zeros, subnormals, cancellations, wide scales).
// (F12's sign form has its own host test, iq_sign_fast_test, on the shared header the kernels include.)
//
// Not a compile of the kernel: the host model transcribes the arithmetic the kernel spells out.  Two things keep that
// honest: the transcription is checked against the kernel SOURCE (the lines that carry the arithmetic must still be
// there, or the test fails and the model has to be re-derived), and the kernel's own parity run on the GPU
// (native_expert_bench) stays the proof for the compiled code (contraction choices, the HIP emulation).
// Exit 0 = PASS, 1 = FAIL.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#ifndef STRATA_SRC_DIR
#define STRATA_SRC_DIR "."
#endif

namespace {

int failed = 0;
#define CHECK(x, ...)                                                                           \
    do {                                                                                        \
        if (!(x)) {                                                                             \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x);                  \
            std::fprintf(stderr, "  " __VA_ARGS__);                                             \
            std::fprintf(stderr, "\n");                                                         \
            ++failed;                                                                           \
        }                                                                                       \
    } while (0)

struct Rng {
    uint64_t s = 0x9E3779B97F4A7C15ull;
    uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
};

// ---------------------------------------------------------------------------------------------------------- F14
float fadd(float a, float b) {   // one rounding per add; the volatile keeps the compiler from regrouping a chain of them
    volatile float r = a + b;
    return r;
}

// mode 0: row_dot_20_sub16.  Lane t of 16: s = 0 + call t; s16 = 0 + call t + 16 for t < 4 (else 0); s = s + s16; then the
// butterfly s += shfl_xor(s, o), o = 8, 4, 2, 1.  Returns the value every lane ends with (checked to be the same).
float tree16(const float (&c)[20]) {
    float s[16], n[16];
    for (int t = 0; t < 16; ++t) {
        const float s16 = t < 4 ? fadd(0.0f, c[t + 16]) : 0.0f;
        s[t] = fadd(fadd(0.0f, c[t]), s16);
    }
    for (int o = 8; o > 0; o >>= 1) {
        for (int t = 0; t < 16; ++t) n[t] = fadd(s[t], s[t ^ o]);
        std::memcpy(s, n, sizeof s);
    }
    for (int t = 1; t < 16; ++t) CHECK(std::memcmp(&s[t], &s[0], 4) == 0, "the 16-lane butterfly left lane %d unlike lane 0", t);
    return s[0];
}

// mode 10: native_down_q2_sub4_kernel.  Lane u of 4: v[i] = 0 + call u + 4 i (i = 0..4); a0 = v0 + v4, a4 = v1 + 0, a8 = v2 + 0,
// a12 = v3 + 0; b0 = a0 + a8; b4 = a4 + a12; x = b0 + b4; then shfl xor 2 and xor 1 inside the 4-lane group.
float tree4(const float (&c)[20]) {
    float x[4], n[4];
    for (int u = 0; u < 4; ++u) {
        float v[5];
        for (int i = 0; i < 5; ++i) v[i] = fadd(0.0f, c[u + 4 * i]);
        const float a0 = fadd(v[0], v[4]), a4 = fadd(v[1], 0.0f), a8 = fadd(v[2], 0.0f), a12 = fadd(v[3], 0.0f);
        const float b0 = fadd(a0, a8), b4 = fadd(a4, a12);
        x[u] = fadd(b0, b4);
    }
    for (int o = 2; o > 0; o >>= 1) {
        for (int u = 0; u < 4; ++u) n[u] = fadd(x[u], x[u ^ o]);
        std::memcpy(x, n, sizeof x);
    }
    for (int u = 1; u < 4; ++u) CHECK(std::memcmp(&x[u], &x[0], 4) == 0, "the 4-lane shuffles left lane %d unlike lane 0", u);
    return x[0];
}

float random_call(Rng& rng) {
    const uint64_t r = rng.next();
    switch (r % 10) {
        case 0: return 0.0f;
        case 1: return -0.0f;
        case 2: return std::ldexp(1.0f, -140) * (float) (int) ((r >> 8) & 0xFF);   // subnormals (and a few zeros)
        default: {
            const float m = (float) (int) ((r >> 8) & 0xFFFF) / 65536.0f - 0.5f;
            const int e = (int) ((r >> 32) % 30) - 15;   // 2^-15 .. 2^14: sums that round differently under another order
            return std::ldexp(m, e);
        }
    }
}

void test_down_fold() {
    Rng rng;
    long cases = 0, signed_zero_cases = 0;
    for (long i = 0; i < 4000000; ++i) {
        float c[20];
        for (float& v : c) v = random_call(rng);
        if (i % 7 == 0) {   // cancellation: pairs of opposite calls on different lanes
            c[(i / 7) % 20] = -c[(i / 7 + 3) % 20];
        }
        const float a = tree16(c), b = tree4(c);
        ++cases;
        CHECK(std::memcmp(&a, &b, 4) == 0, "case %ld: 16-lane %a vs 4-lane %a", i, (double) a, (double) b);
        if (failed) return;
        signed_zero_cases += a == 0.0f && std::signbit(a);
    }
    // the all-zero row: -0 calls everywhere must end in the same signed zero (the 0.0f joins turn -0 into +0 in both)
    float z[20];
    for (float& v : z) v = -0.0f;
    const float za = tree16(z), zb = tree4(z);
    CHECK(std::memcmp(&za, &zb, 4) == 0, "all -0: %a vs %a", (double) za, (double) zb);
    // a different order DOES change bits: the same calls summed straight (0..19) differ from the tree on some inputs,
    // so the comparison above is able to fail (it is not vacuous)
    long differs = 0;
    Rng r2;
    for (long i = 0; i < 100000; ++i) {
        float c[20];
        for (float& v : c) v = random_call(r2);
        float straight = 0.0f;
        for (float v : c) straight = fadd(straight, v);
        const float a = tree16(c);
        differs += std::memcmp(&a, &straight, 4) != 0;
    }
    CHECK(differs > 1000, "a straight sum must differ from the tree on many inputs (got %ld)", differs);
    std::printf("F14 4-lane down fold: %ld random rows, bitwise equal to the 16-lane tree (%ld rows ended in -0; a straight sum "
                "differs on %ld of 100000, so the check can fail)\n",
                cases, signed_zero_cases, differs);
}

// ------------------------------------------------------------------------------------------ the kernel source guard
std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::string();
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
void need(const std::string& src, const char* what, const char* line) {
    CHECK(src.find(line) != std::string::npos, "the kernel source no longer has %s: `%s` - re-derive the host model above", what, line);
}

void test_source_guard() {
    const std::string path = std::string(STRATA_SRC_DIR) + "/src/kernels/cuda/iq_kernels.cu";
    const std::string src = slurp(path);
    if (src.empty()) {
        std::printf("source guard: %s not readable here, skipped\n", path.c_str());
        return;
    }
    // mode 0's tree (row_dot_20_sub16)
    need(src, "mode 0 join", "s[c] = __fadd_rn(s[c], s16[c]);");
    need(src, "mode 0 butterfly", "for (int o = 8; o > 0; o >>= 1) s[c] += __shfl_xor_sync(0xffffffffu, s[c], o);");
    need(src, "mode 0 second call", "const int k = t + 16, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);");
    // mode 10's tree (native_down_q2_sub4_kernel)
    need(src, "mode 10 call map", "const int k = u + 4 * i;");
    need(src, "mode 10 a0", "const float a0 = __fadd_rn(v[0][c], v[4][c]);");
    need(src, "mode 10 a4", "const float a4 = __fadd_rn(v[1][c], 0.0f);");
    need(src, "mode 10 a8", "const float a8 = __fadd_rn(v[2][c], 0.0f);");
    need(src, "mode 10 a12", "const float a12 = __fadd_rn(v[3][c], 0.0f);");
    need(src, "mode 10 b0", "const float b0 = a0 + a8;");
    need(src, "mode 10 b4", "const float b4 = a4 + a12;");
    need(src, "mode 10 x", "float x = b0 + b4;");
    need(src, "mode 10 xor 2", "x += __shfl_xor_sync(0xffffffffu, x, 2);");
    need(src, "mode 10 xor 1", "x += __shfl_xor_sync(0xffffffffu, x, 1);");
    std::printf("source guard: the arithmetic the models transcribe is still in %s\n", path.c_str());
}

}  // namespace

int main() {
    test_down_fold();
    test_source_guard();
    if (failed) {
        std::fprintf(stderr, "gpu_rewrite_parity_test: FAIL (%d)\n", failed);
        return 1;
    }
    std::printf("gpu_rewrite_parity_test: PASS\n");
    return 0;
}
