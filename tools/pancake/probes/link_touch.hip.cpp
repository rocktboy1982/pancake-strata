// link_touch.hip.cpp - probe (LEADS.md Part O O5, MOD_PLAN.md section 6 item 13): does a PCIe link that is allowed to sleep (ASPM L1) delay the completion of a
// ~4 ms GPU pass on this host, and does a tiny read / write of mapped host memory keep it awake?
//
// The caminus authors' note ("the PCIe link stays awake", AGPL, idea only): with ASPM L1 on the card's path the link sleeps during a ~4 ms launch and the
// completion reaches the host ~0.2 ms late on every token; one extra one-task segment that touches mapped host memory keeps the link awake (233 -> 246 tok/s
// on their XTX, nothing on their XT whose path has no L1).  Our decode window ends with a host hand-off through mapped host memory too (the doorbell /
// wait_flag path), at ~24.8 ms a window: 0.2 ms would be 0.8%.  First `python3 tools/pancake/pcie_link.py` says whether L1 is on for the path at all.
//
// QUESTION   Host-observed "pass end" latency = (time the host sees the flag the pass writes last) - (time the pass body ended), for a spin pass of --us
//            microseconds, in three arms run in alternation:
//              plain    the pass writes the flag at its end (what a window does)
//              end      the pass reads and writes one mapped host word just BEFORE the flag (one touch at the end: the cheap form of the idea)
//              keep     a second kernel on another stream touches a mapped host word every --touch-us microseconds while the pass runs (the link never idles
//                       longer than that): the persistent-engine form of the idea
//            The pass body end is stamped by the GPU's own clock into the mapped flag word pair (device ns at the end, then the flag); the host stamps its
//            clock when it sees the flag; both clocks are different, so the probe reports the DIFFERENCE between arms (the host cost of seeing the flag), not
//            an absolute latency: median and p90 per arm over --reps passes, and the median saving of each arm against plain.
// BUILD      hipcc -O2 --offload-arch=gfx1100 tools/pancake/probes/link_touch.hip.cpp -o /tmp/link_touch
// RUN        HIP_VISIBLE_DEVICES=0 /tmp/link_touch [--us 4000] [--reps 400] [--touch-us 5] [--device 0]       (engine stopped; seconds)
// READ       the three lines "plain / end / keep: median_us p90_us" and the two savings.  Decision rule (MOD_PLAN.md section 6): a median saving of keep or end
//            against plain below ~20 us means the link is not sleeping here: close O5 (STRATA_LINK_TOUCH would measure noise); 100 us or more (a ~0.4% of a
//            window) is worth the one-thread keep-alive kernel in the window; run it with `host_tune.sh aspm` on and off to separate the two causes.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                                      \
    do {                                                                                           \
        hipError_t e_ = (x);                                                                       \
        if (e_ != hipSuccess) {                                                                    \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

// mapped host words: [0] flag (the pass number, written last), [1] touch word (read and written by the touches), [2] stop word for the keep kernel
__global__ void pass_kernel(volatile unsigned* host, unsigned pass, long long ticks, int mode) {
    const long long t0 = wall_clock64();
    while (wall_clock64() - t0 < ticks) {
    }
    if (mode == 1) host[1] = host[1] + 1u;   // the end touch: a read and a write over the link
    __threadfence_system();
    host[0] = pass;                          // the completion the host polls
    host[2] = pass;                          // tells the keep kernel the pass is over
    __threadfence_system();
}
__global__ void keep_kernel(volatile unsigned* host, unsigned pass, long long every_ticks) {
    long long next = wall_clock64() + every_ticks;
    while (host[2] != pass) {                // one read per round; the pass kernel writes it at its end
        if (wall_clock64() >= next) {
            host[1] = host[1] + 1u;          // a read and a write over the link
            next += every_ticks;
        }
    }
}

static double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    double us = 4000, touch_us = 5;
    int reps = 400, dev = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--us") us = std::atof(argv[i + 1]);
        else if (a == "--reps") reps = std::atoi(argv[i + 1]);
        else if (a == "--touch-us") touch_us = std::atof(argv[i + 1]);
        else if (a == "--device") dev = std::atoi(argv[i + 1]);
    }
    CK(hipSetDevice(dev));
    int clock_khz = 0;
    CK(hipDeviceGetAttribute(&clock_khz, hipDeviceAttributeWallClockRate, dev));
    const double ticks_per_us = clock_khz / 1000.0;
    volatile unsigned* host = nullptr;
    CK(hipHostMalloc((void**) &host, 64, hipHostMallocMapped | hipHostMallocCoherent));
    unsigned* dhost = nullptr;
    CK(hipHostGetDevicePointer((void**) &dhost, (void*) host, 0));
    hipStream_t s_pass, s_keep;
    CK(hipStreamCreate(&s_pass));
    CK(hipStreamCreate(&s_keep));
    std::vector<double> lat[3];
    unsigned pass = 0;
    host[0] = host[1] = host[2] = 0;
    const char* names[3] = {"plain", "end", "keep"};
    for (int r = 0; r < reps + 20; ++r) {
        for (int arm = 0; arm < 3; ++arm) {
            ++pass;
            if (arm == 2) keep_kernel<<<1, 1, 0, s_keep>>>(dhost, pass, (long long) (touch_us * ticks_per_us));
            const double t0 = now_us();
            pass_kernel<<<1, 1, 0, s_pass>>>(dhost, pass, (long long) (us * ticks_per_us), arm == 1 ? 1 : 0);
            while (host[0] != pass) {
            }
            const double t1 = now_us();
            CK(hipStreamSynchronize(s_pass));
            CK(hipStreamSynchronize(s_keep));
            if (r >= 20) lat[arm].push_back(t1 - t0 - us);   // host-seen pass time minus the spin: launch + completion path
        }
    }
    double med[3] = {};
    for (int arm = 0; arm < 3; ++arm) {
        std::sort(lat[arm].begin(), lat[arm].end());
        med[arm] = lat[arm][lat[arm].size() / 2];
        std::printf("%-6s median %8.1f us  p90 %8.1f us  (pass %.0f us, %zu reps)\n", names[arm], med[arm], lat[arm][lat[arm].size() * 9 / 10], us, lat[arm].size());
    }
    std::printf("saving of end against plain: %.1f us; of keep against plain: %.1f us (positive = the touch makes the pass end sooner)\n", med[0] - med[1], med[0] - med[2]);
    CK(hipHostFree((void*) host));
    return 0;
}
