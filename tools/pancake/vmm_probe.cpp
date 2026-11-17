// tools/pancake/vmm_probe.cpp - does HIP virtual memory management work on this GPU, and what does it cost?
//
// Standalone, HIP only, nothing of the engine is linked.  Written for the Linux box (RX 7900 XT, gfx1100) because the
// engine's elastic K/V (--kv-grow) and its expert cache (ExpertCache::set_vmm) depend on hipMemAddressReserve /
// hipMemCreate / hipMemMap / hipMemSetAccess (src/core/vmm.cpp) and no ledger row shows them running on gfx1100
// (LEADS.md, the kvcached part).  The probe answers, in one run:
//   1. the device reports hipDeviceAttributeVirtualMemoryManagementSupported;
//   2. the minimum and the recommended allocation granularity (the engine uses the minimum);
//   3. reserve / create / map / set-access / unmap latency per chunk, set-access done per chunk and once per run
//      (vmm.hpp relies on the per-run form being as cheap as a map);
//   4. data written through a mapped range reads back, and a chunk unmapped and mapped at another address keeps its bytes;
//   5. one physical chunk mapped at two addresses, the second read-only (the "zero page" trick, STRATA_VMM_ZERO_BACK), reads
//      the same through both;
//   6. a device-to-device copy over a mapped range runs as fast as over a hipMalloc range (the question kvcached's
//      layout study raised: a mapped range of 2 MiB chunks must not cost attention reads).
// The idea list comes from ovg-project/kvcached (Apache-2.0, LEADS.md Part L); this file is our own code from the API.
//
// Build:  hipcc -O2 -std=c++17 tools/pancake/vmm_probe.cpp -o vmm_probe          (any ROCm 6+; add --offload-arch=gfx1100)
// Run:    ./vmm_probe [chunks=64] [copy_mib=512]
// Output: one "key value" line per fact on stdout, "PASS" or "FAIL <why>" last; exit code 0 only on PASS.
// Nothing is written to disk; the program frees everything it maps.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using clk = std::chrono::steady_clock;
double us_since(clk::time_point t0) { return std::chrono::duration<double, std::micro>(clk::now() - t0).count(); }

int g_fail = 0;
void bad(const char* what, hipError_t e) {
    std::printf("FAIL %s: %s\n", what, hipGetErrorString(e));
    ++g_fail;
    (void) hipGetLastError();
}
#define TRY(call, what)                                  \
    do {                                                 \
        const hipError_t e_ = (call);                    \
        if (e_ != hipSuccess) { bad(what, e_); return 1; } \
    } while (0)

hipMemAccessDesc rw(int dev) {
    hipMemAccessDesc d{};
    d.location.type = hipMemLocationTypeDevice;
    d.location.id = dev;
    d.flags = hipMemAccessFlagsProtReadWrite;
    return d;
}

double copy_gbps(void* dst, void* src, size_t bytes) {
    hipEvent_t a, b;
    (void) hipEventCreate(&a);
    (void) hipEventCreate(&b);
    for (int i = 0; i < 2; ++i) (void) hipMemcpyDtoDAsync(dst, src, bytes, nullptr);   // warm-up
    (void) hipStreamSynchronize(nullptr);
    const int reps = 8;
    (void) hipEventRecord(a, nullptr);
    for (int i = 0; i < reps; ++i) (void) hipMemcpyDtoDAsync(dst, src, bytes, nullptr);
    (void) hipEventRecord(b, nullptr);
    (void) hipEventSynchronize(b);
    float ms = 0;
    (void) hipEventElapsedTime(&ms, a, b);
    (void) hipEventDestroy(a);
    (void) hipEventDestroy(b);
    return ms > 0 ? (double) bytes * reps / (ms * 1e-3) / 1e9 : 0.0;
}

}  // namespace

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::max(2, std::atoi(argv[1])) : 64;
    const size_t copy_bytes = (size_t) (argc > 2 ? std::max(2, std::atoi(argv[2])) : 512) << 20;
    int dev = 0;
    TRY(hipGetDevice(&dev), "hipGetDevice");
    hipDeviceProp_t prop{};
    TRY(hipGetDeviceProperties(&prop, dev), "hipGetDeviceProperties");
    std::printf("device %d %s %s\n", dev, prop.name, prop.gcnArchName);
    std::printf("hip_runtime %d\n", HIP_VERSION);

    int vmm = 0;
    TRY(hipDeviceGetAttribute(&vmm, hipDeviceAttributeVirtualMemoryManagementSupported, dev), "VMM attribute");
    std::printf("vmm_supported %d\n", vmm);
    if (!vmm) { std::printf("FAIL the device reports no virtual memory management\n"); return 1; }

    hipMemAllocationProp ap{};
    ap.type = hipMemAllocationTypePinned;
    ap.requestedHandleType = hipMemHandleTypeNone;
    ap.location.type = hipMemLocationTypeDevice;
    ap.location.id = dev;
    size_t gmin = 0, grec = 0;
    TRY(hipMemGetAllocationGranularity(&gmin, &ap, hipMemAllocationGranularityMinimum), "granularity minimum");
    TRY(hipMemGetAllocationGranularity(&grec, &ap, hipMemAllocationGranularityRecommended), "granularity recommended");
    std::printf("granularity_min_bytes %zu\ngranularity_recommended_bytes %zu\n", gmin, grec);
    const size_t G = gmin;

    size_t free0 = 0, total = 0;
    (void) hipMemGetInfo(&free0, &total);
    std::printf("vram_free_mib %zu of %zu\n", free0 >> 20, total >> 20);

    // 3. latency
    void* va = nullptr;
    auto t0 = clk::now();
    TRY(hipMemAddressReserve(&va, (size_t) n * G, 0, nullptr, 0), "hipMemAddressReserve");
    std::printf("reserve_us %.1f (%d chunks, %zu MiB of addresses)\n", us_since(t0), n, ((size_t) n * G) >> 20);

    std::vector<hipMemGenericAllocationHandle_t> h((size_t) n);
    const hipMemAccessDesc acc = rw(dev);
    double t_create = 0, t_map = 0, t_acc = 0, t_unmap = 0;
    for (int i = 0; i < n; ++i) {
        t0 = clk::now();
        TRY(hipMemCreate(&h[(size_t) i], G, &ap, 0), "hipMemCreate");
        t_create += us_since(t0);
        t0 = clk::now();
        TRY(hipMemMap((char*) va + (size_t) i * G, G, 0, h[(size_t) i], 0), "hipMemMap");
        t_map += us_since(t0);
        t0 = clk::now();
        TRY(hipMemSetAccess((char*) va + (size_t) i * G, G, &acc, 1), "hipMemSetAccess (per chunk)");
        t_acc += us_since(t0);
    }
    std::printf("create_us_avg %.1f\nmap_us_avg %.1f\nset_access_per_chunk_us_avg %.1f\n", t_create / n, t_map / n, t_acc / n);

    // 4. data: write through the mapped range, read back, move one chunk to another address
    std::vector<uint8_t> pat((size_t) G), back((size_t) G);
    for (size_t i = 0; i < pat.size(); ++i) pat[i] = (uint8_t) (i * 131u + 7u);
    TRY(hipMemcpyHtoD((char*) va + (size_t) 3 * G, pat.data(), G), "write chunk 3");
    TRY(hipMemcpyDtoH(back.data(), (char*) va + (size_t) 3 * G, G), "read chunk 3");
    const bool same = back == pat;
    std::printf("readback_chunk %s\n", same ? "ok" : "MISMATCH");
    if (!same) ++g_fail;

    t0 = clk::now();
    for (int i = 0; i < n; ++i) TRY(hipMemUnmap((char*) va + (size_t) i * G, G), "hipMemUnmap");
    t_unmap = us_since(t0);
    std::printf("unmap_us_avg %.1f\n", t_unmap / n);

    // remap all chunks, set access ONCE for the whole run (what VmmRange::map_range does)
    t0 = clk::now();
    for (int i = 0; i < n; ++i) TRY(hipMemMap((char*) va + (size_t) i * G, G, 0, h[(size_t) i], 0), "hipMemMap (again)");
    const double t_remap = us_since(t0);
    t0 = clk::now();
    TRY(hipMemSetAccess(va, (size_t) n * G, &acc, 1), "hipMemSetAccess (one run)");
    std::printf("remap_us_avg %.1f\nset_access_per_run_us_total %.1f (per chunk %.2f)\n", t_remap / n, us_since(t0),
                us_since(t0) / n);
    TRY(hipMemcpyDtoH(back.data(), (char*) va + (size_t) 3 * G, G), "read chunk 3 after remap");
    const bool kept = back == pat;
    std::printf("bytes_kept_after_unmap_remap %s\n", kept ? "ok" : "MISMATCH");
    if (!kept) ++g_fail;

    // 5. one physical chunk at two addresses
    void* va2 = nullptr;
    TRY(hipMemAddressReserve(&va2, G, 0, nullptr, 0), "reserve second address");
    TRY(hipMemMap(va2, G, 0, h[3], 0), "map chunk 3 at a second address");
    hipMemAccessDesc ro = acc;   // read-only, as STRATA_VMM_ZERO_BACK maps its shared zero chunk
    ro.flags = hipMemAccessFlagsProtRead;
    TRY(hipMemSetAccess(va2, G, &ro, 1), "read-only access at the second address");
    TRY(hipMemcpyDtoH(back.data(), va2, G), "read through the second address");
    const bool alias = back == pat;
    std::printf("double_map_alias %s\n", alias ? "ok" : "MISMATCH");
    if (!alias) ++g_fail;
    TRY(hipMemUnmap(va2, G), "unmap second address");
    TRY(hipMemAddressFree(va2, G), "free second address");

    // 6. device copy bandwidth, mapped range vs hipMalloc range (only if the range is big enough)
    const size_t avail = (size_t) n * G / 2;
    const size_t bytes = std::min(copy_bytes, avail);
    if (bytes >= ((size_t) 8 << 20)) {
        void *plain_a = nullptr, *plain_b = nullptr;
        if (hipMalloc(&plain_a, bytes) == hipSuccess && hipMalloc(&plain_b, bytes) == hipSuccess) {
            const double gm = copy_gbps((char*) va + avail, va, bytes);
            const double gp = copy_gbps(plain_b, plain_a, bytes);
            std::printf("copy_gbps_mapped %.1f\ncopy_gbps_hipmalloc %.1f\ncopy_ratio %.3f (bytes %zu MiB)\n", gm, gp,
                        gp > 0 ? gm / gp : 0.0, bytes >> 20);
        } else {
            std::printf("copy_skipped hipMalloc failed\n");
            (void) hipGetLastError();
        }
        if (plain_a) (void) hipFree(plain_a);
        if (plain_b) (void) hipFree(plain_b);
    } else {
        std::printf("copy_skipped range too small (give more chunks)\n");
    }

    // release
    for (int i = 0; i < n; ++i) {
        (void) hipMemUnmap((char*) va + (size_t) i * G, G);
        (void) hipMemRelease(h[(size_t) i]);
    }
    (void) hipMemAddressFree(va, (size_t) n * G);
    size_t free1 = 0;
    (void) hipMemGetInfo(&free1, &total);
    std::printf("vram_free_mib_after %zu\n", free1 >> 20);
    std::printf(g_fail == 0 ? "PASS\n" : "FAIL see above\n");
    return g_fail == 0 ? 0 : 1;
}
