// empty_grid.hip.cpp - probe 4.2.1 (bench/TEST_PLAN.md): what does an EMPTY kernel launch cost on the 7900 XT as a
// function of its block count?  Decides F20 (resize the decode graph's launch shapes; unimatrix099 d0c8d6c3e2 found
// that 384 x 256 on a 48-WGP 7900 XTX cost 14-16 us even when empty, while 352 or 320 blocks cost 2.7 us).
//
// QUESTION   Does any block count (1..512 x 256 threads, step 1) cost much more than the ~2.7-3 us floor of a dependent
//            kernel in a HIP graph on gfx1100 (42 WGPs)?  Is it a multiple of 42 (a "whole card" fill stall) like on
//            the XTX?  And do the block counts our decode graph launches (launch_shapes.py) land on a slow value?
// BUILD      hipcc -O2 --offload-arch=gfx1100 tools/pancake/probes/empty_grid.hip.cpp -o /tmp/empty_grid
//            (hipcc: /opt/rocm/core-10.0/bin/hipcc; or cmake -DSTRATA_BUILD_PROBES=ON ... and
//             cmake --build build-hip --target empty_grid_probe)
// RUN        HIP_VISIBLE_DEVICES=0 /tmp/empty_grid [--csv sweep.csv] [--max-blocks 512] [--step 1] [--threads 256]
//                                                   [--n 200] [--reps 5] [--list 384,96,48] [--device 0]
//            Needs the GPU idle (stop the engine first); about 1-3 minutes for the full sweep (two kernels x 512 graphs).
//            Quick look: --step 8 takes seconds; --list 336,384,96,42,84,1 times only those counts.
// READ       stdout: one line per block count (empty kernel / unimatrix's fetch_like with count 0 / direct stream
//            launch), then a summary: the floor (median), every block count more than 2 us above it, and the mean cost
//            by (blocks mod 42).  Verdict for F20: if no count in the decode-graph list (launch_shapes.py --sweep
//            sweep.csv) sits above floor + 2 us, there is nothing to gain by padding or shrinking grids on this card
//            (close F20); if some do, the "excess us per token" line of launch_shapes.py is the ceiling of the port.
//            The CSV (blocks,empty_us,fetch_like_us,stream_us) is what launch_shapes.py --sweep reads.
//            "graph" times are per node of a captured chain of N dependent launches (as in the decode graph);
//            "stream" times are per launch of N plain stream launches (host launch rate may dominate: a sanity column).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#define CK(x)                                                                                        \
    do {                                                                                             \
        hipError_t e_ = (x);                                                                         \
        if (e_ != hipSuccess) {                                                                      \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_));   \
            std::exit(1);                                                                            \
        }                                                                                            \
    } while (0)

// the truly empty kernel
__global__ void __launch_bounds__(256) empty_k() {}

// unimatrix099's fetch_like: a copy job whose count is 0 on the device (fetch_blobs launches 384 x 256 per layer and the
// PCIe share is nearly always empty); the loop body never runs but the kernel reads *n and has the same arguments
__global__ void __launch_bounds__(256) fetch_like(const unsigned long long* __restrict__ src, const int* __restrict__ n,
                                                   uint4* __restrict__ dst, long long per) {
    const long long total = (long long) *n * per;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long long) gridDim.x * blockDim.x) {
        const long long k = i / per, off = i - k * per;
        dst[i] = ((const uint4*) src[k])[off];
    }
}

struct Ctx {
    hipStream_t s;
    hipEvent_t a, b;
    const unsigned long long* src;
    const int* n;
    uint4* dst;
    int threads, N, reps;
};

template <class Launch>
static double graph_us(Ctx& c, Launch launch) {   // best of reps, per node, in a captured chain of N dependent launches
    hipGraph_t g;
    hipGraphExec_t ge;
    CK(hipStreamBeginCapture(c.s, hipStreamCaptureModeThreadLocal));
    for (int i = 0; i < c.N; ++i) launch();
    CK(hipStreamEndCapture(c.s, &g));
    CK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    CK(hipGraphLaunch(ge, c.s));
    CK(hipStreamSynchronize(c.s));
    float best = 1e9f;
    for (int r = 0; r < c.reps; ++r) {
        CK(hipEventRecord(c.a, c.s));
        CK(hipGraphLaunch(ge, c.s));
        CK(hipEventRecord(c.b, c.s));
        CK(hipEventSynchronize(c.b));
        float ms;
        CK(hipEventElapsedTime(&ms, c.a, c.b));
        best = std::min(best, ms);
    }
    CK(hipGraphExecDestroy(ge));
    CK(hipGraphDestroy(g));
    return best * 1000.0 / c.N;
}

static double stream_us(Ctx& c, int blocks) {
    float best = 1e9f;
    for (int r = 0; r < c.reps; ++r) {
        CK(hipEventRecord(c.a, c.s));
        for (int i = 0; i < c.N; ++i) empty_k<<<blocks, c.threads, 0, c.s>>>();
        CK(hipEventRecord(c.b, c.s));
        CK(hipEventSynchronize(c.b));
        float ms;
        CK(hipEventElapsedTime(&ms, c.a, c.b));
        best = std::min(best, ms);
    }
    return best * 1000.0 / c.N;
}

int main(int argc, char** argv) {
    int device = 0, max_blocks = 512, step = 1, threads = 256, N = 200, reps = 5;
    std::string csv, list;
    for (int i = 1; i < argc; ++i) {
        auto arg = [&](const char* k) { return !std::strcmp(argv[i], k) && i + 1 < argc; };
        if (arg("--device")) device = std::atoi(argv[++i]);
        else if (arg("--max-blocks")) max_blocks = std::atoi(argv[++i]);
        else if (arg("--step")) step = std::max(1, std::atoi(argv[++i]));
        else if (arg("--threads")) threads = std::atoi(argv[++i]);
        else if (arg("--n")) N = std::atoi(argv[++i]);
        else if (arg("--reps")) reps = std::atoi(argv[++i]);
        else if (arg("--csv")) csv = argv[++i];
        else if (arg("--list")) list = argv[++i];
        else if (std::isdigit((unsigned char) argv[i][0])) device = std::atoi(argv[i]);   // unimatrix's positional form
        else {
            std::fprintf(stderr, "unknown argument %s\n", argv[i]);
            return 2;
        }
    }
    CK(hipSetDevice(device));
    hipDeviceProp_t p;
    CK(hipGetDeviceProperties(&p, device));
    std::printf("device %d: %s (%s), %d CUs (%d WGPs on RDNA), %d threads/block, chain of %d, best of %d\n", device, p.name,
                p.gcnArchName, p.multiProcessorCount, p.multiProcessorCount / 2, threads, N, reps);

    Ctx c{};
    c.threads = threads;
    c.N = N;
    c.reps = reps;
    int* n;
    unsigned long long* src;
    uint4* dst;
    CK(hipMalloc(&n, 4));
    CK(hipMemset(n, 0, 4));
    CK(hipMalloc(&src, 64));
    CK(hipMalloc(&dst, 1 << 20));
    c.n = n;
    c.src = src;
    c.dst = dst;
    CK(hipStreamCreate(&c.s));
    CK(hipEventCreate(&c.a));
    CK(hipEventCreate(&c.b));

    std::vector<int> counts;
    if (!list.empty()) {
        for (size_t pos = 0; pos < list.size();) {
            size_t q = list.find(',', pos);
            counts.push_back(std::atoi(list.substr(pos, q == std::string::npos ? q : q - pos).c_str()));
            if (q == std::string::npos) break;
            pos = q + 1;
        }
    } else {
        counts.push_back(1);
        for (int b = step; b <= max_blocks; b += step)
            if (b > 1) counts.push_back(b);
    }

    FILE* f = csv.empty() ? nullptr : std::fopen(csv.c_str(), "w");
    if (f) std::fprintf(f, "blocks,empty_us,fetch_like_us,stream_us\n");
    struct Row {
        int b;
        double e, fl, st;
    };
    std::vector<Row> rows;
    // warm-up (clock ramp)
    for (int i = 0; i < 2000; ++i) empty_k<<<64, threads, 0, c.s>>>();
    CK(hipStreamSynchronize(c.s));
    for (int b : counts) {
        if (b < 1) continue;
        Row r{b, 0, 0, 0};
        r.e = graph_us(c, [&] { empty_k<<<b, threads, 0, c.s>>>(); });
        r.fl = graph_us(c, [&] { fetch_like<<<b, threads, 0, c.s>>>(c.src, c.n, c.dst, 147456); });
        r.st = stream_us(c, b);
        rows.push_back(r);
        std::printf("  %4d blocks x %d: empty %6.2f us  fetch_like(count 0) %6.2f us  stream %6.2f us\n", b, threads, r.e,
                    r.fl, r.st);
        if (f) std::fprintf(f, "%d,%.3f,%.3f,%.3f\n", b, r.e, r.fl, r.st);
    }
    if (f) std::fclose(f);

    if (rows.empty()) return 0;
    std::vector<double> v;
    for (auto& r : rows) v.push_back(r.e);
    std::sort(v.begin(), v.end());
    const double floor_us = v[v.size() / 2];
    std::printf("\nsummary (graph chain, empty kernel): floor = median = %.2f us\n", floor_us);
    std::printf("block counts more than 2 us above the floor:\n");
    int slow = 0;
    for (auto& r : rows)
        if (r.e > floor_us + 2.0) {
            std::printf("  %4d blocks: %.2f us (+%.2f)  [%d mod 42, %d mod 21, %d mod 84]\n", r.b, r.e, r.e - floor_us,
                        r.b % 42, r.b % 21, r.b % 84);
            ++slow;
        }
    if (!slow) std::printf("  none -> launch cost does not depend on the block count here\n");
    std::map<int, std::pair<double, int>> by;   // (blocks mod 42) -> sum, n
    for (auto& r : rows) {
        auto& e = by[r.b % 42];
        e.first += r.e;
        ++e.second;
    }
    std::printf("mean cost by (blocks mod 42), 0 = a whole number of WGP sets:\n  ");
    for (auto& kv : by) std::printf("%d:%.2f ", kv.first, kv.second.first / kv.second.second);
    std::printf("\n");
    return 0;
}
