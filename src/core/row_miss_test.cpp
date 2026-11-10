// src/core/row_miss_test.cpp - the per-row new CPU misses (core/row_miss.hpp) and the window trace's record 5 (A22), no GPU.
//
// 1. RowMissCounter: distinct per layer, only CPU entries, a row adds only what earlier rows did not wake, summed over layers,
//    null placement = all CPU, reset;
// 2. WindowTrace: a window written with STRATA_WINDOW_TRACE carries record 5 right after its window record, with the
//    counts of (1); the file is parsed back here byte by byte (the format tools/pancake/trace/wtrace.py reads).
#include "strata/core/row_miss.hpp"
#include "strata/core/window_trace.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using strata::core::RowMissCounter;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++g_fail; }
}
template <class T> T rd(const std::vector<uint8_t>& b, size_t& off) {
    T v{};
    if (off + sizeof(T) <= b.size()) std::memcpy(&v, b.data() + off, sizeof(T));
    off += sizeof(T);
    return v;
}
}  // namespace

int main() {
    {   // 1. the counter
        RowMissCounter c;
        check(c.rows() == 0 && c.layers() == 0 && c.at(0) == 0, "empty");
        // layer A: 3 rows, k = 3.  kind: -1 CPU, 0 VRAM, 1 PCIe
        const int32_t ids_a[9] = {5, 7, 9,    5, 8, 9,    1, 8, 2};
        const int32_t kin_a[9] = {-1, -1, 0,  -1, -1, 0,  1, -1, -1};
        c.layer(3, 3, ids_a, kin_a);
        // row 0: 5, 7 on the CPU -> 2; row 1: 5 (seen), 8 new -> 1; row 2: 8 (seen), 2 new (1 is PCIe) -> 1
        check(c.rows() == 3 && c.at(0) == 2 && c.at(1) == 1 && c.at(2) == 1, "one layer: new CPU experts per row");
        // layer B: an expert id seen in layer A counts again (distinct per layer); kind null = all CPU
        const int32_t ids_b[4] = {5, 6,  6, 4};
        c.layer(2, 2, ids_b, nullptr);
        check(c.layers() == 2 && c.at(0) == 4 && c.at(1) == 2 && c.at(2) == 1, "summed over layers; null kind = every entry CPU");
        // skipped tail (-2) and negative ids are not misses
        const int32_t ids_c[4] = {3, -1,  3, 11};
        const int32_t kin_c[4] = {-2, -1,  -1, -2};
        c.layer(2, 2, ids_c, kin_c);
        check(c.at(0) == 4 && c.at(1) == 3, "-2 and negative ids skipped; an expert skipped in row 0 is new on the CPU in row 1");
        c.reset();
        check(c.rows() == 0 && c.layers() == 0 && c.at(0) == 0 && c.at(1) == 0, "reset");
        c.layer(3, 3, ids_a, kin_a);
        check(c.at(0) == 2 && c.at(1) == 1 && c.at(2) == 1, "after reset: the same window counts the same");
        // a wide id grows the stamp table
        const int32_t ids_w[2] = {4000, 4000};
        c.reset();
        c.layer(2, 1, ids_w, nullptr);
        check(c.at(0) == 1 && c.at(1) == 0, "large expert id");
        // many layers: the epoch stamps never confuse layers
        c.reset();
        const int32_t ids_r[2] = {9, 9};
        for (int l = 0; l < 1000; ++l) c.layer(2, 1, ids_r, nullptr);
        check(c.at(0) == 1000 && c.at(1) == 0, "1000 layers: one new expert each in row 0");
        check(!strata::core::row_misses().on(), "the process-wide counter is off by default");
    }
    {   // 2. the window trace's record 5
        const std::string path = "row_miss_test.stwt";
        std::remove(path.c_str());
#if defined(_WIN32)
        _putenv_s("STRATA_WINDOW_TRACE", path.c_str());
#else
        setenv("STRATA_WINDOW_TRACE", path.c_str(), 1);
#endif
        {
            strata::core::WindowTrace wt;
            wt.open(2, 16, 3);
            check(wt.on(), "trace open");
            const float prob[2] = {0.9f, 0.6f};
            const uint8_t src[2] = {0, 0};
            wt.drafts(prob, src, 2);
            const int32_t ids_a[9] = {5, 7, 9,    5, 8, 9,    1, 8, 2};
            const int32_t kin_a[9] = {-1, -1, 0,  -1, -1, 0,  1, -1, -1};
            wt.layer(0, 3, ids_a, kin_a, nullptr);
            const int32_t ids_b[9] = {1, 2, 3,  1, 2, 4,  6, 2, 3};
            const int32_t kin_b[9] = {-1, 0, 0,  -1, 0, -1,  -1, 0, 0};
            wt.layer(1, 3, ids_b, kin_b, nullptr);
            wt.commit(3, 1, 100);
            // a dropped window leaves nothing behind, the next one starts from zero
            wt.layer(0, 3, ids_a, kin_a, nullptr);
            wt.drop();
            wt.layer(0, 2, ids_b, kin_b, nullptr);   // rows 0-1 of layer B only: row 0 -> 1 (expert 1), row 1 -> 1 (expert 4)
            wt.commit(2, 0, 103);
        }
#if defined(_WIN32)
        _putenv_s("STRATA_WINDOW_TRACE", "");
#else
        unsetenv("STRATA_WINDOW_TRACE");
#endif
        std::vector<uint8_t> b;
        if (std::FILE* f = std::fopen(path.c_str(), "rb")) {
            uint8_t buf[4096];
            size_t n;
            while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) b.insert(b.end(), buf, buf + n);
            std::fclose(f);
        }
        check(b.size() > 20 && std::memcmp(b.data(), "STWT", 4) == 0, "magic");
        size_t off = 4;
        const uint32_t ver = rd<uint32_t>(b, off), nl = rd<uint32_t>(b, off), ne = rd<uint32_t>(b, off), k = rd<uint32_t>(b, off);
        check(ver == 1 && nl == 2 && ne == 16 && k == 3, "header");
        // window 1
        check(rd<uint8_t>(b, off) == 2, "window record");
        const uint8_t rows = rd<uint8_t>(b, off), acc = rd<uint8_t>(b, off);
        (void) rd<uint8_t>(b, off);
        const int64_t pos = rd<int64_t>(b, off);
        (void) rd<float>(b, off);
        check(rows == 3 && acc == 1 && pos == 100, "window fields");
        check(rd<uint8_t>(b, off) == 5, "record 5 follows the window record");
        const uint8_t n5 = rd<uint8_t>(b, off);
        (void) rd<uint16_t>(b, off);
        check(n5 == 3, "record 5: one count per row");
        uint16_t rm[3] = {rd<uint16_t>(b, off), rd<uint16_t>(b, off), rd<uint16_t>(b, off)};
        // layer A: 2, 1, 1 (as in 1.); layer B: row 0 -> {1}, row 1 -> {4} (1 seen), row 2 -> {6} -> 1, 1, 1
        check(rm[0] == 3 && rm[1] == 2 && rm[2] == 2, "record 5 counts = the counter's, summed over both layers");
        check(rd<uint8_t>(b, off) == 4, "then the drafts record");
        const uint8_t nd = rd<uint8_t>(b, off);
        (void) rd<uint16_t>(b, off);
        check(nd == 2, "two drafts");
        off += (size_t) nd * 5;
        for (int l = 0; l < 2; ++l) {   // the two layer records
            check(rd<uint8_t>(b, off) == 1, "layer record");
            (void) rd<uint8_t>(b, off);
            const uint8_t nt = rd<uint8_t>(b, off), flags = rd<uint8_t>(b, off);
            check(nt == 3 && flags == 0, "layer fields");
            off += (size_t) nt * k * 3;
        }
        // window 2: the dropped layers are gone
        check(rd<uint8_t>(b, off) == 2, "second window record");
        off += 15;
        check(rd<uint8_t>(b, off) == 5, "second record 5");
        const uint8_t n52 = rd<uint8_t>(b, off);
        (void) rd<uint16_t>(b, off);
        const uint16_t r0 = rd<uint16_t>(b, off), r1 = rd<uint16_t>(b, off);
        check(n52 == 2 && r0 == 1 && r1 == 1, "second window: only its own layer counted (the dropped one is not)");
        check(rd<uint8_t>(b, off) == 1, "its layer record");
        off += 3 + 2 * k * 3;
        check(off == b.size(), "nothing else in the file");
        std::remove(path.c_str());
    }
    if (g_fail == 0) std::printf("row_miss_test: all passed\n");
    return g_fail == 0 ? 0 : 1;
}
