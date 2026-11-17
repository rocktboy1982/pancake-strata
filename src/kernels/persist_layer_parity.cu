// src/kernels/persist_layer_parity.cu - on a gfx11 card: the persistent-launch kernel template (include/strata/kernels/persist_dev.hpp) with the layer-shaped tape
// of include/strata/program/persist_launch.hpp (LEADS.md Part O O7, STRATA_PERSIST_LAYER), against a serial host reference.
//
// Each task of segment s computes 256 integer outputs from the previous segment's buffer (a chain, as layer_tape's dependencies are), so a task that ran
// before its dependency finished would read stale words and the final buffer would differ.  Three launches on carried bases (the counters are never reset)
// at 1, 7 and 96 workgroups, each checked bit for bit against the reference run, with the error word required to stay 0.  Exit 0 ok, 1 a difference, an
// error word or a CUDA error.  Run: build/persist_layer_parity (the card alone, a second).
#include "strata/kernels/persist_dev.hpp"
#include "strata/program/persist_launch.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace strata;

#define CK(x)                                                                                      \
    do {                                                                                           \
        hipError_t e_ = (x);                                                                       \
        if (e_ != hipSuccess) {                                                                    \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

constexpr int kThreads = 256;

__host__ __device__ inline uint32_t mix(uint32_t in0, uint32_t in1, int seg, int e) { return in0 * 1664525u + in1 + uint32_t(seg) * 977u + uint32_t(e); }

struct LayerBody {
    uint32_t* bufs;     // (nseg + 1) rows of `stride` words; row s + 1 is segment s's output
    size_t stride;
    const int64_t* prefix;
    __device__ void run(int seg, int task) {
        const int n = int(prefix[seg + 1] - prefix[seg]) * kThreads;   // this segment's output width
        const int pn = seg == 0 ? n : int(prefix[seg] - prefix[seg - 1]) * kThreads;   // the previous segment's width (segment 0 reads the seed row, width n)
        const uint32_t* in = bufs + size_t(seg) * stride;
        uint32_t* out = bufs + size_t(seg + 1) * stride;
        const int e = task * kThreads + threadIdx.x;
        out[e] = mix(in[e % pn], in[(e + 1) % pn], seg, e);
    }
};

int main() {
    const program::PersistTape t = program::layer_tape(16, 4, 8, 24, 48, 6);
    const int nseg = int(t.seg.size());
    size_t maxw = 0;
    for (int s = 0; s < nseg; ++s) maxw = std::max<size_t>(maxw, size_t(t.count(s)) * kThreads);
    const size_t stride = maxw;
    std::vector<uint32_t> seed(stride);
    for (size_t i = 0; i < stride; ++i) seed[i] = uint32_t(i * 2654435761u + 99u);
    // serial reference
    std::vector<uint32_t> ref((size_t(nseg) + 1) * stride, 0);
    std::copy(seed.begin(), seed.end(), ref.begin());
    for (int s = 0; s < nseg; ++s) {
        const int n = t.count(s) * kThreads, pn = s == 0 ? n : t.count(s - 1) * kThreads;
        for (int e = 0; e < n; ++e) ref[size_t(s + 1) * stride + size_t(e)] = mix(ref[size_t(s) * stride + size_t(e % pn)], ref[size_t(s) * stride + size_t((e + 1) % pn)], s, e);
    }
    int bad = 0;
    uint32_t* dbufs;
    int64_t* dprefix;
    int32_t* ddep;
    unsigned long long *dtarget, *dclaim, *ddone;
    int* derr;
    CK(hipMalloc(&dbufs, (size_t(nseg) + 1) * stride * 4));
    CK(hipMalloc(&dprefix, (size_t(nseg) + 1) * 8));
    CK(hipMalloc(&ddep, size_t(nseg) * 4));
    CK(hipMalloc(&dtarget, size_t(nseg) * 8));
    CK(hipMalloc(&dclaim, 8));
    CK(hipMalloc(&ddone, size_t(nseg) * 16 * 8));
    CK(hipMalloc(&derr, 4));
    CK(hipMemset(dclaim, 0, 8));
    CK(hipMemset(ddone, 0, size_t(nseg) * 16 * 8));
    CK(hipMemset(derr, 0, 4));
    CK(hipMemcpy(dprefix, t.prefix.data(), (size_t(nseg) + 1) * 8, hipMemcpyHostToDevice));
    program::PersistLauncher L(t);
    int wall_khz = 0;
    CK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, 0));
    for (int workers : {1, 7, 96, 7, 1}) {   // carried bases across five launches of different worker counts
        const program::PersistImage im = L.image();
        CK(hipMemset(dbufs, 0, (size_t(nseg) + 1) * stride * 4));
        CK(hipMemcpy(dbufs, seed.data(), stride * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(ddep, im.dep.data(), size_t(nseg) * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dtarget, im.wait_target.data(), size_t(nseg) * 8, hipMemcpyHostToDevice));
        kernels::PersistDevArgs a{dprefix, ddep, dtarget, dclaim, im.claim_base, ddone, 16, nseg, derr, (long long) wall_khz * 2000};   // 2 s
        hipLaunchKernelGGL(kernels::persist_kernel<LayerBody>, dim3(workers), dim3(kThreads), 0, nullptr, a, LayerBody{dbufs, stride, dprefix});
        CK(hipDeviceSynchronize());
        L.advance(workers);
        std::vector<uint32_t> got((size_t(nseg) + 1) * stride);
        CK(hipMemcpy(got.data(), dbufs, got.size() * 4, hipMemcpyDeviceToHost));
        int err = 0;
        CK(hipMemcpy(&err, derr, 4, hipMemcpyDeviceToHost));
        bool same = err == 0;
        for (int s = 0; s < nseg && same; ++s)
            for (int e = 0; e < t.count(s) * kThreads; ++e)
                if (got[size_t(s + 1) * stride + size_t(e)] != ref[size_t(s + 1) * stride + size_t(e)]) { same = false; break; }
        std::printf("workers %3d: %s (error word %d)\n", workers, same ? "identical" : "DIFFERENT", err);
        if (!same) ++bad;
    }
    std::printf("persist_layer_parity: %s\n", bad ? "FAILED" : "ok");
    return bad ? 1 : 0;
}
