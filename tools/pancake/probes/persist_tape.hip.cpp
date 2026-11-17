// persist_tape.hip.cpp - probe (LEADS.md Part O O7, MOD_PLAN.md section 6): what does ONE persistent launch cost per dependent stage on the 7900 XT,
// against one launch per stage?  This is the headline technique of the caminus engine (one GPU launch per token; its authors measure 3-5 us per
// dependent stage, about half of a 4 ms token), implemented here from the description of its protocol (AGPL: the design, not the code): a tape of
// segments, ONE monotonic claim counter, a task -> segment map by prefix sums, bounded relaxed polling on the dependency's done counter followed by
// one acquire fence per wave, a release fence before the done counter is bumped, bases carried between launches (nothing is ever reset).  The host
// rules are include/strata/program/persist_tape.hpp (tested by persist_tape_test on a CPU reference).
//
// QUESTION   For a chain of dependent stages that each stream some weight bytes and write a small result, how many microseconds per stage does a
//            persistent launch need, against N separate kernel launches on one stream and against a captured graph of the same N launches?
//            Our decode has ~470 kernels per token and launch gaps worth 19.7% of the wall of a window (LEADS.md line 1013): if the persistent
//            form is not clearly cheaper per stage at OUR stage sizes, a persistent decode is not worth building (close O7 for the GPU half).
// BUILD      hipcc -O2 --offload-arch=gfx1100 tools/pancake/probes/persist_tape.hip.cpp -Iinclude -o /tmp/persist_tape
//            (or cmake -DSTRATA_BUILD_PROBES=ON ... --target persist_tape_probe)
// RUN        HIP_VISIBLE_DEVICES=0 /tmp/persist_tape [--segments 432] [--tasks 192] [--wbytes 16384] [--grids 96,192,288,384] [--reps 20] [--device 0]
//            The GPU must be idle (stop the engine); a run takes seconds.  --tasks is the tasks per stage (one task = 256 threads), --wbytes the weight
//            bytes each task streams (16384 = a 3.1 MB stage at 192 tasks), --grids the persistent workgroup counts tried.
// READ       one line per arm: us per launch, us per stage, and whether the final result is bit-identical to the one-launch-per-stage arm (it must be:
//            the arithmetic is integer).  Verdict lines: persistent minus per-launch stage cost in us at each grid, the best grid, and "err" (must be 0:
//            the bounded wait never gave up).  Decision rule: a saving below ~1.5 us per stage at --wbytes of our real stages means a persistent token
//            cannot beat the launch gaps we have; above ~3 us (the authors' number) the wiring (STRATA_PERSIST_LAYER) is worth writing.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "strata/program/persist_tape.hpp"

#define CK(x)                                                                                      \
    do {                                                                                           \
        hipError_t e_ = (x);                                                                       \
        if (e_ != hipSuccess) {                                                                    \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

constexpr int kThreads = 256;

// the task's work, shared by both arms (same statements, so the bits agree): 256 outputs, each mixes two neighbouring inputs of the previous stage with
// a few words of the "weights" this task streams (integer arithmetic: exact, order free)
__device__ __forceinline__ void do_task(uint32_t* bufs, size_t stride, const uint32_t* __restrict__ w, size_t wwords, int seg, int task, int ntask,
                                        int wper) {
    const uint32_t* in = bufs + size_t(seg) * stride;
    uint32_t* out = bufs + size_t(seg + 1) * stride;
    const int e = task * kThreads + threadIdx.x;
    const int n = ntask * kThreads;
    uint32_t acc = in[e] * 1664525u + in[(e + 1) % n] + uint32_t(seg);
    const size_t base = (size_t(seg) * ntask + task) * size_t(wper) * kThreads;
    for (int j = 0; j < wper; ++j) acc ^= w[(base + size_t(j) * kThreads + threadIdx.x) % wwords];
    out[e] = acc;
}

// arm 1 / 3: one kernel per stage; blockIdx.x is the task
__global__ void __launch_bounds__(kThreads) stage_k(uint32_t* bufs, size_t stride, const uint32_t* w, size_t wwords, int seg, int ntask, int wper) {
    do_task(bufs, stride, w, wwords, seg, blockIdx.x, ntask, wper);
}

// arm 2: the persistent launch.  prefix: nseg + 1 task numbers; done: one 64-bit counter per segment on its own 128 B line; bases as the host holds them.
__global__ void __launch_bounds__(kThreads) persist_k(uint32_t* bufs, size_t stride, const uint32_t* w, size_t wwords, const int64_t* prefix, int nseg,
                                                       int ntask, int wper, unsigned long long* claim, unsigned long long claim_base,
                                                       unsigned long long* done, const unsigned long long* done_base, int* err, long long limit_ns) {
    __shared__ int s_task, s_seg;
    const int64_t total = prefix[nseg];
    for (;;) {
        if (threadIdx.x == 0) {
            const int64_t idx = int64_t(__hip_atomic_fetch_add(claim, 1ull, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) - claim_base);
            int lo = 0, hi = nseg;   // the segment: last prefix[s] <= idx
            while (hi - lo > 1) {
                const int mid = (lo + hi) >> 1;
                if (prefix[mid] <= idx) lo = mid; else hi = mid;
            }
            s_seg = idx >= total ? -1 : lo;
            s_task = int(idx - prefix[lo]);
        }
        __syncthreads();
        const int seg = s_seg, task = s_task;
        if (seg < 0) return;
        if (seg > 0 && threadIdx.x == 0) {   // the tape here is a chain: segment s depends on s - 1
            const unsigned long long want = done_base[seg - 1] + (unsigned long long) (prefix[seg] - prefix[seg - 1]);
            const long long t0 = wall_clock64();
            unsigned long long* d = done + size_t(seg - 1) * 16;
            while (__hip_atomic_load(d, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < want) {
                __builtin_amdgcn_s_sleep(1);
                if ((wall_clock64() - t0) * 10ll > limit_ns) { atomicExch(err, 1); break; }   // wall_clock64 ticks at 100 MHz on gfx11: x10 = ns
            }
        }
        __syncthreads();
        __atomic_thread_fence(__ATOMIC_ACQUIRE);   // every wave: the polls were relaxed, so one acquire after the wait
        do_task(bufs, stride, w, wwords, seg, task, ntask, wper);
        __atomic_thread_fence(__ATOMIC_RELEASE);   // every wave publishes its stores before the counter moves
        __syncthreads();
        if (threadIdx.x == 0) __hip_atomic_fetch_add(done + size_t(seg) * 16, 1ull, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
}

static uint64_t fnv(const std::vector<uint32_t>& v) {
    uint64_t h = 1469598103934665603ull;
    for (uint32_t x : v) { h ^= x; h *= 1099511628211ull; }
    return h;
}

static std::vector<int> parse_list(const std::string& s) {
    std::vector<int> o;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        o.push_back(std::atoi(s.substr(i, j - i).c_str()));
        i = j + 1;
    }
    return o;
}

int main(int argc, char** argv) {
    int nseg = 432, ntask = 192, reps = 20, dev = 0;
    long wbytes = 16384;
    std::vector<int> grids = {96, 192, 288, 384};
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--segments") nseg = std::atoi(argv[i + 1]);
        else if (a == "--tasks") ntask = std::atoi(argv[i + 1]);
        else if (a == "--wbytes") wbytes = std::atol(argv[i + 1]);
        else if (a == "--grids") grids = parse_list(argv[i + 1]);
        else if (a == "--reps") reps = std::atoi(argv[i + 1]);
        else if (a == "--device") dev = std::atoi(argv[i + 1]);
    }
    CK(hipSetDevice(dev));
    const int wper = int(std::max(0l, wbytes / (4 * kThreads)));   // 32-bit words per thread per task
    const size_t stride = size_t(ntask) * kThreads, wwords = size_t(1) << 24;   // 64 MiB of "weights", read with wrap-around
    std::vector<uint32_t> h0(stride), hw(wwords);
    for (size_t i = 0; i < stride; ++i) h0[i] = uint32_t(i * 2654435761u + 12345u);
    for (size_t i = 0; i < wwords; ++i) hw[i] = uint32_t(i * 40503u + 7u);
    uint32_t *bufs, *w;
    CK(hipMalloc(&bufs, size_t(nseg + 1) * stride * 4));
    CK(hipMalloc(&w, wwords * 4));
    CK(hipMemcpy(w, hw.data(), wwords * 4, hipMemcpyHostToDevice));
    hipStream_t st;
    CK(hipStreamCreate(&st));
    hipEvent_t a, b;
    CK(hipEventCreate(&a));
    CK(hipEventCreate(&b));
    auto reset = [&]() { CK(hipMemcpy(bufs, h0.data(), stride * 4, hipMemcpyHostToDevice)); };
    auto final_hash = [&]() {
        std::vector<uint32_t> o(stride);
        CK(hipMemcpy(o.data(), bufs + size_t(nseg) * stride, stride * 4, hipMemcpyDeviceToHost));
        return fnv(o);
    };
    auto median = [](std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };

    // arm 1: one launch per stage on one stream
    reset();
    std::vector<float> t1;
    for (int r = 0; r < reps; ++r) {
        CK(hipEventRecord(a, st));
        for (int s = 0; s < nseg; ++s) stage_k<<<ntask, kThreads, 0, st>>>(bufs, stride, w, wwords, s, ntask, wper);
        CK(hipEventRecord(b, st));
        CK(hipEventSynchronize(b));
        float ms;
        CK(hipEventElapsedTime(&ms, a, b));
        t1.push_back(ms);
    }
    const uint64_t ref = final_hash();
    const float m1 = median(t1);
    std::printf("per-launch  : %8.1f us/launch  %6.2f us/stage  hash %016llx (reference)\n", m1 * 1000, m1 * 1000 / nseg, (unsigned long long) ref);

    // arm 2: the same chain captured in a graph
    {
        hipGraph_t g;
        hipGraphExec_t ge;
        CK(hipStreamBeginCapture(st, hipStreamCaptureModeGlobal));
        for (int s = 0; s < nseg; ++s) stage_k<<<ntask, kThreads, 0, st>>>(bufs, stride, w, wwords, s, ntask, wper);
        CK(hipStreamEndCapture(st, &g));
        CK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
        reset();
        std::vector<float> tg;
        for (int r = 0; r < reps; ++r) {
            CK(hipEventRecord(a, st));
            CK(hipGraphLaunch(ge, st));
            CK(hipEventRecord(b, st));
            CK(hipEventSynchronize(b));
            float ms;
            CK(hipEventElapsedTime(&ms, a, b));
            tg.push_back(ms);
        }
        const float mg = median(tg);
        std::printf("graph       : %8.1f us/launch  %6.2f us/stage  hash %s\n", mg * 1000, mg * 1000 / nseg, final_hash() == ref ? "same" : "DIFFERENT");
    }

    // arm 3: one persistent launch, bases carried across the reps (never reset)
    std::vector<strata::program::TapeSegment> tape(size_t(nseg), {ntask, -1});
    for (int i = 1; i < nseg; ++i) tape[size_t(i)].dep = i - 1;
    const auto t = strata::program::PersistTape::build(tape);
    int64_t* d_prefix;
    CK(hipMalloc(&d_prefix, t.prefix.size() * sizeof(int64_t)));
    CK(hipMemcpy(d_prefix, t.prefix.data(), t.prefix.size() * sizeof(int64_t), hipMemcpyHostToDevice));
    int* d_err;
    CK(hipMalloc(&d_err, sizeof(int)));
    float best = 1e30f;
    int best_grid = 0, errs = 0;
    for (const int grid : grids) {
        unsigned long long *d_claim, *d_done, *d_base;
        CK(hipMalloc(&d_claim, 8));
        CK(hipMemset(d_claim, 0, 8));
        CK(hipMalloc(&d_done, size_t(nseg) * 128));
        CK(hipMemset(d_done, 0, size_t(nseg) * 128));
        CK(hipMalloc(&d_base, size_t(nseg) * 8));
        CK(hipMemset(d_err, 0, sizeof(int)));
        strata::program::PersistBases bases{size_t(nseg)};
        // done counters sit one per 128 B line: the kernel indexes done + seg * 16 (u64 words); the bases are a dense array uploaded each launch
        reset();
        std::vector<float> tp;
        bool same = true;
        for (int r = 0; r < reps; ++r) {
            std::vector<unsigned long long> hb{std::vector<unsigned long long>(size_t(nseg))};
            for (int s = 0; s < nseg; ++s) hb[size_t(s)] = bases.done[size_t(s)];
            CK(hipMemcpy(d_base, hb.data(), hb.size() * 8, hipMemcpyHostToDevice));
            CK(hipEventRecord(a, st));
            persist_k<<<grid, kThreads, 0, st>>>(bufs, stride, w, wwords, d_prefix, nseg, ntask, wper, d_claim, bases.claim, d_done, d_base, d_err,
                                                 200000000ll);
            CK(hipEventRecord(b, st));
            CK(hipEventSynchronize(b));
            float ms;
            CK(hipEventElapsedTime(&ms, a, b));
            tp.push_back(ms);
            bases.advance(t, grid);
            if (r == 0 || r == reps - 1) same = same && final_hash() == ref;
        }
        int e = 0;
        CK(hipMemcpy(&e, d_err, sizeof(int), hipMemcpyDeviceToHost));
        errs += e;
        const float mp = median(tp);
        std::printf("persist g%-4d: %8.1f us/launch  %6.2f us/stage  hash %s  err %d  saving vs per-launch %+6.2f us/stage\n", grid, mp * 1000,
                    mp * 1000 / nseg, same ? "same" : "DIFFERENT", e, (m1 - mp) * 1000 / nseg);
        if (mp < best) { best = mp; best_grid = grid; }
        CK(hipFree(d_claim));
        CK(hipFree(d_done));
        CK(hipFree(d_base));
    }
    std::printf("verdict: best persistent grid %d, %.2f us/stage against %.2f us/stage per-launch (saving %.2f us/stage = %.0f us per %d-stage token); err total %d\n",
                best_grid, best * 1000 / nseg, m1 * 1000 / nseg, (m1 - best) * 1000 / nseg, (m1 - best) * 1000, nseg, errs);
    return errs ? 2 : 0;
}
