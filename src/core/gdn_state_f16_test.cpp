// STRATA_SNAP_STATE_F16 (LEADS.md S6): the FP32 <-> FP16 conversion of include/strata/core/gdn_state_f16.hpp, the error bound of
// the round trip, the compact blob, and a host model of the snapshot save / restore path (device buffer -> saved host bytes ->
// device buffer through restore_source, whole and in aligned blocks as a disk stream applies them).  No GPU, no model.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "strata/core/gdn_state_f16.hpp"

using namespace strata::core::gdn_f16;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    // the switch
    CHECK(!snap_f16_on_from(nullptr));
    CHECK(!snap_f16_on_from("0"));
    CHECK(!snap_f16_on_from(""));
    CHECK(snap_f16_on_from("1"));

    // all 65,536 halves: widen then narrow is the identity (NaN stays NaN), and widening agrees with the formula
    for (uint32_t h = 0; h < 65536; ++h) {
        const float f = f16_to_f32((uint16_t) h);
        const uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ff;
        if (e == 0x1f && m != 0) { CHECK(std::isnan(f)); CHECK(std::isnan(f16_to_f32(f32_to_f16(f)))); continue; }
        bool sat = false;
        if (e == 0x1f) { CHECK(std::isinf(f)); continue; }   // infinity saturates by design (checked below)
        CHECK(f32_to_f16(f, &sat) == (uint16_t) h);
        CHECK(!sat);
        const double want = (e == 0 ? (double) m * std::ldexp(1.0, -24) : (1.0 + m / 1024.0) * std::ldexp(1.0, (int) e - 15)) * ((h & 0x8000) ? -1 : 1);
        CHECK((double) f == want);
    }
    // landmarks
    CHECK(f32_to_f16(0.0f) == 0x0000 && f32_to_f16(-0.0f) == 0x8000);
    CHECK(f32_to_f16(1.0f) == 0x3c00 && f32_to_f16(-2.0f) == 0xc000);
    CHECK(f32_to_f16(65504.0f) == 0x7bff);
    {
        bool sat = false;
        CHECK(f32_to_f16(70000.0f, &sat) == 0x7bff && sat);                  // saturates, never infinity
        sat = false;
        CHECK(f32_to_f16(65520.0f, &sat) == 0x7bff && sat);                  // rounds up past the largest finite: saturates
        sat = false;
        CHECK(f32_to_f16(-1e30f, &sat) == 0xfbff && sat);
        CHECK(std::isnan(f16_to_f32(f32_to_f16(std::nanf("")))));
    }
    CHECK(f32_to_f16(std::ldexp(1.0f, -24)) == 0x0001);                      // the smallest subnormal
    CHECK(f32_to_f16(std::ldexp(1.0f, -25)) == 0x0000);                      // a tie between 0 and 2^-24 goes to even (0)
    CHECK(f32_to_f16(std::ldexp(1.5f, -25)) == 0x0001);                      // above the tie
    CHECK(f32_to_f16(std::ldexp(1.0f, -26)) == 0x0000);
    CHECK(f32_to_f16(1.0f + std::ldexp(1.0f, -11)) == 0x3c00);               // tie to even (down)
    CHECK(f32_to_f16(1.0f + 3.0f * std::ldexp(1.0f, -11)) == 0x3c02);        // tie to even (up)

    // round to nearest: for random floats the result is no farther than either neighbour, and the bound holds
    {
        std::mt19937_64 rng(99);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        double worst_rel = 0;
        long checked = 0, saturated = 0;
        for (long i = 0; i < 4000000; ++i) {
            float x = nd(rng) * std::ldexp(1.0f, (int) (rng() % 40) - 28);   // 2^-28 .. 2^11 scales
            if (i % 7 == 0) x = std::ldexp(1.0f, (int) (rng() % 41) - 26) * (1.0f + (float) (rng() % 4096) / 4096.0f);
            bool sat = false;
            const uint16_t h = f32_to_f16(x, &sat);
            const float r = f16_to_f32(h);
            if (sat) { ++saturated; CHECK(std::fabs(x) > 65504.0f); continue; }
            ++checked;
            CHECK(std::fabs((double) r - (double) x) <= roundtrip_bound(x) * (1.0 + 1e-12));
            if (std::fabs(x) >= kF16MinNormal) worst_rel = std::max(worst_rel, std::fabs((double) r - x) / std::fabs((double) x));
            const uint16_t lo = (uint16_t) (h & 0x7fff) > 0 ? (uint16_t) (h - 1) : h, hi = (uint16_t) (h + 1);
            const double d0 = std::fabs((double) r - x);
            if ((h & 0x7fff) > 0) CHECK(d0 <= std::fabs((double) f16_to_f32(lo) - x) + 1e-30);
            if ((h & 0x7fff) < 0x7bff) CHECK(d0 <= std::fabs((double) f16_to_f32(hi) - x) + 1e-30);
            if (failed) return 1;
        }
        std::printf("round trip: %ld values within the bound (worst relative error %.3e = 2^%.2f), %ld saturated\n", checked,
                    worst_rel, std::log2(worst_rel), saturated);
        CHECK(worst_rel <= 1.0 / 2048.0 * (1 + 1e-12));
    }

    // a GDN-like state (small values, a few large) through the in-place rounding: relative L2 error about 2^-12
    {
        std::mt19937_64 rng(5);
        std::normal_distribution<float> nd(0.0f, 0.05f);
        std::vector<float> st(1 << 20);
        for (float& v : st) v = nd(rng);
        for (size_t i = 0; i < st.size(); i += 4099) st[i] *= 40.0f;
        std::vector<float> ref = st;
        const size_t sat = round_inplace(st.data(), st.size());
        double num = 0, den = 0, maxabs = 0;
        for (size_t i = 0; i < st.size(); ++i) {
            const double e = (double) st[i] - (double) ref[i];
            num += e * e; den += (double) ref[i] * ref[i];
            maxabs = std::max(maxabs, std::fabs(e));
            CHECK(std::fabs(e) <= roundtrip_bound(ref[i]) * (1 + 1e-12));
        }
        const double rel = std::sqrt(num / den);
        std::printf("state-like 2^20 values: relative L2 error %.3e, max abs error %.3e, %zu saturated\n", rel, maxabs, sat);
        CHECK(sat == 0);
        CHECK(rel < 3.0e-4);   // 2^-11 / sqrt(3) = 2.8e-4 is the uniform-rounding ceiling
    }

    // the blob: half the bytes plus a header, round trips to the rounded values, bad blobs refused
    {
        std::mt19937_64 rng(11);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> a(10007);
        for (float& v : a) v = nd(rng);
        std::vector<uint8_t> blob, back;
        size_t sat = 7;
        CHECK(encode(a.data(), a.size() * 4, blob, &sat) && sat == 0);
        CHECK(blob.size() == blob_bytes(a.size()) && blob.size() == 16 + a.size() * 2);
        CHECK(blob.size() * 2 < a.size() * 4 + 64);
        CHECK(decode(blob.data(), blob.size(), back) && back.size() == a.size() * 4);
        std::vector<float> r = a;
        round_inplace(r.data(), r.size());
        CHECK(std::memcmp(back.data(), r.data(), back.size()) == 0);
        std::vector<uint8_t> bad = blob;
        bad[0] ^= 1;
        std::vector<uint8_t> keep = {1, 2, 3};
        CHECK(!decode(bad.data(), bad.size(), keep) && keep.size() == 3);
        CHECK(!decode(blob.data(), blob.size() - 1, keep));
        CHECK(!decode(blob.data(), 8, keep));
        CHECK(!encode(a.data(), 7, blob));
        std::printf("blob: %zu FP32 bytes -> %zu bytes (%.3f)\n", a.size() * 4, blob.size(), (double) blob.size() / (double) (a.size() * 4));
    }

    // the save / restore path on the host: a "device" buffer is saved to host bytes, restored with the switch on and off
    {
        std::mt19937_64 rng(2);
        std::normal_distribution<float> nd(0.0f, 0.1f);
        const size_t n = 3 * 1024 * 1024 / 4 + 5;   // not a round size
        std::vector<float> device(n);
        for (float& v : device) v = nd(rng);
        std::vector<uint8_t> saved(n * 4);
        std::memcpy(saved.data(), device.data(), saved.size());   // checkpoint_save: the exact bytes
        std::vector<uint8_t> scratch;
        bool rounded = true;
        // off: the default path is the saved bytes themselves, nothing copied
        const void* off = restore_source(false, saved.data(), saved.size(), scratch, &rounded);
        CHECK(off == saved.data() && !rounded && scratch.empty());
        // on: whole restore
        std::vector<float> restored(n);
        const void* on = restore_source(true, saved.data(), saved.size(), scratch, &rounded);
        CHECK(on == scratch.data() && rounded);
        std::memcpy(restored.data(), on, saved.size());
        double worst = 0;
        for (size_t i = 0; i < n; ++i) {
            worst = std::max(worst, std::fabs((double) restored[i] - device[i]) / std::max(std::fabs((double) device[i]), 1e-30));
            CHECK(std::fabs((double) restored[i] - device[i]) <= roundtrip_bound(device[i]) * (1 + 1e-12));
        }
        CHECK(std::memcmp(saved.data(), device.data(), saved.size()) == 0);   // the saved image itself is untouched
        // on: a stream applies aligned blocks (the disk path): the result equals the whole restore
        std::vector<uint8_t> applied(saved.size());
        for (size_t off2 = 0; off2 < saved.size();) {
            const size_t len = std::min<size_t>(saved.size() - off2, 4 * (1 + rng() % 40000));
            std::vector<uint8_t> sc;
            const void* src = restore_source(true, saved.data() + off2, len, sc);
            std::memcpy(applied.data() + off2, src, len);
            off2 += len;
        }
        CHECK(std::memcmp(applied.data(), restored.data(), saved.size()) == 0);
        // a size that is not a multiple of 4 is passed through untouched
        std::vector<uint8_t> sc2;
        CHECK(restore_source(true, saved.data(), 6, sc2, &rounded) == saved.data() && !rounded);
        // saved size halves in a compact store
        std::printf("snapshot model: %zu FP32 bytes restore through FP16 rounding, worst relative error %.3e; a compact store takes %zu bytes\n",
                    saved.size(), worst, blob_bytes(n));
    }

    std::printf("%s\n", failed ? "FAILED" : "gdn_state_f16_test: ok");
    return failed ? 1 : 0;
}
