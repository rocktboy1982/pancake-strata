// q2_rows_bench.cpp - probe 4.2.4 (bench/TEST_PLAN.md): the CPU Q2_0 row kernel, legacy vs the fast one
// (STRATA_Q2_FAST=1, imanu86 c3fc8f34d3, F7), single thread, random blocks, no engine, no model file.
//
// QUESTION   How many Q2_0 rows per second does one thread compute with the legacy AVX-2 kernel and with the fast one
//            (8-op code unpack, 4 rows at a time on short rows), on the expert shapes (down: 640 values = 10 blocks, row
//            of 180 B; gate/up: 2560 values = 40 blocks, row of 720 B), with the rows in L2 (compute-bound) and with
//            them in DRAM (bandwidth-bound)?  Are the two outputs bitwise equal?
// BUILD      cmake -DSTRATA_BUILD_PROBES=ON ... && cmake --build <dir> --target q2_rows_bench     (guarded, off by default)
//            or without CMake, from the repo root (three small files; AVX2/FMA/F16C only on the kernel file):
//              g++ -O2 -std=c++17 -Iinclude -c tools/pancake/probes/q2_rows_bench.cpp -o /tmp/q2b_main.o
//              g++ -O2 -std=c++17 -Iinclude -mavx2 -mfma -mf16c -DSTRATA_AVXVNNI=0 -c src/kernels/cpu/q2_avx2.cpp -o /tmp/q2b_k.o
//              g++ /tmp/q2b_main.o /tmp/q2b_k.o -o /tmp/q2_rows_bench
//            (no expert_layout.cpp needed: the probe checks AVX2 itself).  With -DSTRATA_AVXVNNI=1 on BOTH the main and
//            the kernel file, on a CPU with AVX-VNNI, the kernels use vpdpbusd; the CMake target does this as the engine does.
// RUN        /tmp/q2_rows_bench [--seconds 0.3] [--reps 5] [--nt 1] [--chunk 64] [--threads N] [--bw 33]
//            (about 20 s; pin it: taskset -c 3 ...).  --threads N (A48) adds the DRAM rows on N threads, the pool's
//            shape: each thread a contiguous slice, `chunk` rows per call; pin N cores (taskset -c 0-7 for the 5700G's 8).
//            --bw is the DDR4 read bandwidth the floor is computed from (default 33 GB/s, the 5700G's measured figure).
//            CPU benchmarks disturb the GPU decode benchmarks on the same box: run it when nothing else is measuring.
// READ       per shape and working set: ns per row and Mrows/s for legacy, fast, the dispatcher (what the engine calls:
//            fast only with STRATA_Q2_FAST=1, STRATA_Q2_LEGACY=1 forces legacy) and a plain read of the same bytes
//            (the floor of that working set).  "fast/legacy" > 1 is the speed-up.  Verdict for F7: the fast kernel is
//            worth keeping on if it is clearly faster in the L2 rows and the DRAM rows are near the read floor in both
//            (then CPU expert time is bandwidth-bound and the kernel choice cannot help); "bitwise equal: yes" must hold.
//            Rows/s here x the engine's rows per token gives the single-thread CPU time that the pool divides by its threads.
//            "floor" = row bytes / --bw: no kernel can stream a row from DRAM faster; "x floor" is how many times slower
//            an arm is.  bw / (one thread's GB/s) = the threads a kernel needs to reach the floor; past that count the
//            kernel choice cannot change the pool's time (the N-thread lines show where N threads land).
#include "strata/kernels/cpu/expert.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

namespace c = strata::kernels::cpu;

// the VNNI rows exist only if q2_avx2.cpp was compiled with -DSTRATA_AVXVNNI=1 (the CMake build says so); the probe only
// reports it, the dispatcher uses it by itself
#ifndef STRATA_AVXVNNI
#define STRATA_AVXVNNI 0
#endif

namespace {

struct Rng {
    uint64_t s = 0x9E3779B97F4A7C15ull;
    uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
    float uni() { return (float) ((next() >> 40) & 0xFFFFFF) / 16777216.f; }
    float gauss() {
        float a = 0.f;
        for (int i = 0; i < 6; ++i) a += uni();
        return a - 3.f;
    }
};

uint16_t half_of(float f) {   // normal, positive-and-negative fp16 scales around 2^-6 .. 2^-3 (a weight block's scale)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint16_t sign = (uint16_t) ((u >> 16) & 0x8000);
    const int e = (int) ((u >> 23) & 0xFF) - 127 + 15;
    return (uint16_t) (sign | ((e & 0x1F) << 10) | ((u >> 13) & 0x3FF));
}

void fill_rows(Rng& g, uint8_t* w, size_t rows, int nblocks) {   // random Q2_0 blocks: fp16 scale + 16 random code bytes
    for (size_t r = 0; r < rows; ++r)
        for (int b = 0; b < nblocks; ++b) {
            uint8_t* blk = w + (r * nblocks + b) * c::BB;
            const uint16_t d = half_of(std::ldexp(0.5f + g.uni(), -6 + (int) (g.next() % 4)) * (g.next() & 1 ? 1.f : -1.f));
            std::memcpy(blk, &d, 2);
            for (int i = 0; i < 16; ++i) blk[2 + i] = (uint8_t) g.next();
        }
}

using Clock = std::chrono::steady_clock;
volatile uint64_t g_sink;

using RowsFn = void (*)(const uint8_t*, size_t, int, const c::ActQ* const*, int, float* const*, int, int);

// plain read of the same bytes: the floor of the working set (8 bytes per step, enough accumulators to vectorise)
uint64_t read_rows(const uint8_t* w, size_t bytes) {
    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    const uint64_t* p = (const uint64_t*) w;
    const size_t n = bytes / 8;
    for (size_t i = 0; i + 4 <= n; i += 4) {
        a0 += p[i];
        a1 += p[i + 1];
        a2 += p[i + 2];
        a3 += p[i + 3];
    }
    return a0 + a1 + a2 + a3;
}

struct Arm {
    const char* name;
    RowsFn fn;       // null = the plain read
    double best_ns = 1e30;
};

}  // namespace

int main(int argc, char** argv) {
    double seconds = 0.3, bw = 33.0;
    int reps = 5, nt = 1, chunk = 64, threads = 1;
    for (int i = 1; i < argc; ++i) {
        auto arg = [&](const char* k) { return !std::strcmp(argv[i], k) && i + 1 < argc; };
        if (arg("--seconds")) seconds = std::atof(argv[++i]);
        else if (arg("--reps")) reps = std::atoi(argv[++i]);
        else if (arg("--nt")) nt = std::max(1, std::min(4, std::atoi(argv[++i])));
        else if (arg("--chunk")) chunk = std::max(1, std::atoi(argv[++i]));
        else if (arg("--threads")) threads = std::max(1, std::min(64, std::atoi(argv[++i])));
        else if (arg("--bw")) bw = std::max(1.0, std::atof(argv[++i]));
        else {
            std::fprintf(stderr, "usage: %s [--seconds S] [--reps N] [--nt 1..4] [--chunk ROWS] [--threads N] [--bw GB/s]\n",
                         argv[0]);
            return 2;
        }
    }
    if (!__builtin_cpu_supports("avx2")) {
        std::printf("q2_rows_bench: no AVX2 on this CPU - skipped\n");
        return 77;
    }
    const bool vnni = STRATA_AVXVNNI && __builtin_cpu_supports("avxvnni");
    std::printf("Q2_0 row kernels, one thread, nt=%d, chunk=%d rows, AVX-VNNI %s; dispatcher picks %s (STRATA_Q2_FAST / STRATA_Q2_LEGACY)\n",
                nt, chunk, vnni ? "yes" : "no", c::q2_fast_enabled() ? "FAST" : "LEGACY");

    Rng g;
    // activations, quantized as the engine does (act_quant_q8_1_avx2), 2560 values (gate/up) and 640 (down)
    std::vector<std::unique_ptr<c::ActQ>> act2560, act640;
    std::vector<float> x(2560);
    for (int t = 0; t < 4; ++t) {
        for (auto& v : x) v = g.gauss();
        act2560.emplace_back(new c::ActQ());
        c::act_quant_q8_1_avx2(x.data(), 2560, *act2560.back());
        act640.emplace_back(new c::ActQ());
        c::act_quant_q8_1_avx2(x.data(), 640, *act640.back());
    }

    struct Shape {
        const char* name;
        int nblocks;
        std::vector<std::unique_ptr<c::ActQ>>* act;
    } shapes[] = {{"down  (640 values, 10 blocks, 180 B/row)", c::SC_D, &act640},
                  {"gate/up (2560 values, 40 blocks, 720 B/row)", c::SC_GU, &act2560}};
    struct WS {
        const char* name;
        size_t bytes;
    } sets[] = {{"L2-hot, 256 KB", 256u << 10}, {"DRAM, 256 MB", 256u << 20}};

    bool all_equal = true;
    for (const auto& sh : shapes) {
        const size_t row_bytes = (size_t) sh.nblocks * c::BB;
        for (const auto& ws : sets) {
            const size_t rows = ws.bytes / row_bytes;
            std::vector<uint8_t> w(rows * row_bytes + 64);
            fill_rows(g, w.data(), rows, sh.nblocks);
            const int n = sh.nblocks * c::QK;
            std::vector<const c::ActQ*> ap;
            for (int t = 0; t < nt; ++t) ap.push_back(sh.act->at(t).get());
            std::vector<std::vector<float>> out_a(nt, std::vector<float>(rows)), out_b(nt, std::vector<float>(rows)),
                out_c(nt, std::vector<float>(rows));
            auto run_rows = [&](RowsFn fn, std::vector<std::vector<float>>& out, size_t a0, size_t a1) {
                std::vector<float*> op(nt);
                for (int t = 0; t < nt; ++t) op[t] = out[t].data();
                for (size_t r0 = a0; r0 < a1; r0 += chunk) {
                    const int r1 = (int) std::min(a1, r0 + (size_t) chunk);
                    fn(w.data(), row_bytes, sh.nblocks, ap.data(), nt, op.data(), (int) r0, r1);
                }
            };
            auto run = [&](RowsFn fn, std::vector<std::vector<float>>& out) { run_rows(fn, out, 0, rows); };
            // one pass over all rows on `nth` threads, each a contiguous slice (disjoint outputs); fn null = the plain read
            auto pass = [&](RowsFn fn, int nth) {
                if (nth == 1) {
                    if (fn) run(fn, out_a);
                    else g_sink += read_rows(w.data(), rows * row_bytes);
                    return;
                }
                std::vector<std::thread> th;
                for (int k = 0; k < nth; ++k)
                    th.emplace_back([&, k] {
                        const size_t a0 = rows * k / nth, a1 = rows * (k + 1) / nth;
                        if (fn) run_rows(fn, out_a, a0, a1);
                        else g_sink += read_rows(w.data() + a0 * row_bytes, (a1 - a0) * row_bytes);
                    });
                for (auto& t : th) t.join();
            };
            // ns per row: repeat the pass until `seconds` is spent; the best of `reps` timed blocks
            auto best_ns = [&](RowsFn fn, int nth) {
                double best = 1e30;
                for (int rep = 0; rep < reps; ++rep) {
                    long long passes = 0;
                    const auto t0 = Clock::now();
                    double el = 0;
                    do {
                        pass(fn, nth);
                        ++passes;
                        el = std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
                    } while (el < seconds * 1e9 / reps);
                    best = std::min(best, el / ((double) passes * rows));
                }
                return best;
            };
            Arm arms[] = {{"legacy", c::q2_0_gguf_rows_multi_avx2_legacy},
                          {"fast", c::q2_0_gguf_rows_multi_avx2_fast},
                          {"dispatch", c::q2_0_gguf_rows_multi_avx2},
                          {"read only", nullptr}};
            run(arms[0].fn, out_a);
            run(arms[1].fn, out_b);
            run(arms[2].fn, out_c);
            bool equal = true;
            for (int t = 0; t < nt; ++t) equal &= std::memcmp(out_a[t].data(), out_b[t].data(), rows * 4) == 0;
            all_equal &= equal;
            std::printf("\n%s, %s: %zu rows, fast vs legacy bitwise equal: %s\n", sh.name, ws.name, rows, equal ? "yes" : "NO");
            (void) n;
            const double floor_ns = row_bytes / bw;
            auto report = [&](Arm* as, int na, int nth) {
                for (int k = 0; k < na; ++k) as[k].best_ns = best_ns(as[k].fn, nth);
                for (int k = 0; k < na; ++k) {
                    char vs[48] = "";
                    if (as[k].fn && k > 0) std::snprintf(vs, sizeof vs, "   x%.2f vs legacy", as[0].best_ns / as[k].best_ns);
                    std::printf("  %-10s %7.2f ns/row  %7.2f Mrows/s  %6.2f GB/s of weights  %5.2f x floor%s\n", as[k].name,
                                as[k].best_ns, 1e3 / as[k].best_ns, row_bytes / as[k].best_ns, as[k].best_ns / floor_ns, vs);
                }
            };
            std::printf("  floor      %7.2f ns/row at %.1f GB/s (row bytes / DDR4 bandwidth)\n", floor_ns, bw);
            report(arms, 4, 1);
            if (threads > 1 && ws.bytes > (64u << 20)) {   // the pool's shape: DRAM rows only
                std::printf("  -- %d threads, %d rows per call:\n", threads, chunk);
                Arm mt[] = {{"legacy", c::q2_0_gguf_rows_multi_avx2_legacy},
                            {"fast", c::q2_0_gguf_rows_multi_avx2_fast},
                            {"read only", nullptr}};
                report(mt, 3, threads);
            }
        }
    }
    std::printf("\nbitwise equal on all shapes: %s\n", all_equal ? "yes" : "NO (fast kernel differs: do not enable)");
    return all_equal ? 0 : 1;
}
