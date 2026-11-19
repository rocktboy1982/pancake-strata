#include "strata/core/vmm.hpp"

#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace strata::core {
namespace {

struct Api {
    bool ok = false;
    uint64_t gran = 0;
    int dev = -1;
    decltype(&cuMemAddressReserve) reserve = nullptr;
    decltype(&cuMemAddressFree) free_va = nullptr;
    decltype(&cuMemCreate) create = nullptr;
    decltype(&cuMemRelease) release = nullptr;
    decltype(&cuMemMap) map = nullptr;
    decltype(&cuMemUnmap) unmap = nullptr;
    decltype(&cuMemSetAccess) access = nullptr;
};

template <class F> bool resolve(const char* name, F& f) {
    cudaDriverEntryPointQueryResult q{};
    void* p = nullptr;
#if CUDART_VERSION >= 12050
    const cudaError_t e = cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &q);
#else   // the versioned query arrived in CUDA 12.5
    const cudaError_t e = cudaGetDriverEntryPoint(name, &p, cudaEnableDefault, &q);
#endif
    if (e != cudaSuccess || q != cudaDriverEntryPointSuccess || p == nullptr) {
        (void) cudaGetLastError();
        return false;
    }
    f = (F) p;
    return true;
}

// never destroyed: the K/V pools and the cache release their chunks from static destructors at exit
const Api& api() {
    static Api& a = *new Api;
    static std::once_flag once;
    std::call_once(once, [] {
        decltype(&cuDeviceGetAttribute) attr = nullptr;
        decltype(&cuMemGetAllocationGranularity) granularity = nullptr;
        if (cudaGetDevice(&a.dev) != cudaSuccess) return;
        if (!resolve("cuDeviceGetAttribute", attr) || !resolve("cuMemGetAllocationGranularity", granularity) ||
            !resolve("cuMemAddressReserve", a.reserve) || !resolve("cuMemAddressFree", a.free_va) ||
            !resolve("cuMemCreate", a.create) || !resolve("cuMemRelease", a.release) || !resolve("cuMemMap", a.map) ||
            !resolve("cuMemUnmap", a.unmap) || !resolve("cuMemSetAccess", a.access))
            return;
        int vmm = 0;
        if (attr(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, (CUdevice) a.dev) != CUDA_SUCCESS || !vmm)
            return;
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = a.dev;
        size_t g = 0;
        if (granularity(&g, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS || g == 0) return;
        a.gran = g;
        a.ok = true;
    });
    return a;
}

}  // namespace

bool vmm_available() { return api().ok; }
uint64_t vmm_granularity() { return api().ok ? api().gran : 0; }

VmmChunk vmm_chunk_new() {
    const Api& a = api();
    if (!a.ok) return 0;
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = a.dev;
    CUmemGenericAllocationHandle h = 0;
    if (a.create(&h, (size_t) a.gran, &prop, 0) != CUDA_SUCCESS) return 0;
    return (VmmChunk) h;
}

void vmm_chunk_free(VmmChunk h) {
    if (h != 0 && api().ok) api().release((CUmemGenericAllocationHandle) h);
}

bool VmmRange::reserve(uint64_t bytes) {
    release();
    const Api& a = api();
    if (!a.ok || bytes == 0) return false;
    const uint64_t n = (bytes + a.gran - 1) / a.gran;
    CUdeviceptr p = 0;
    if (a.reserve(&p, (size_t) (n * a.gran), 0, 0, 0) != CUDA_SUCCESS) return false;
    base_ = (unsigned long long) p;
    h_.assign((size_t) n, 0);
    // STRATA_VMM_ZERO_BACK=1 (default off): kvcached's zero-page backing, see VmmRange::zero_backed().  Any failure leaves
    // the range plain (unmapped), as without the switch.
    if (const char* zb = std::getenv("STRATA_VMM_ZERO_BACK"); zb != nullptr && zb[0] != '\0' && zb[0] != '0') {
        zero_ = vmm_chunk_new();
        bool ok = zero_ != 0;
        int64_t done = 0;
        for (; ok && done < (int64_t) n; ++done) ok = zero_map(done);
        if (ok) {   // the shared chunk is cleared through the first mapping, then every mapping is read-only
            CUmemAccessDesc rwd{};
            rwd.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            rwd.location.id = a.dev;
            rwd.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            ok = a.access(p, (size_t) a.gran, &rwd, 1) == CUDA_SUCCESS &&
                 cudaMemset((void*) (uintptr_t) p, 0, (size_t) a.gran) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
            if (ok) ok = set_access_ro(0, (int64_t) n);
        }
        if (!ok) {
            (void) cudaGetLastError();
            for (int64_t i = 0; i < done; ++i) zero_unmap(i);
            vmm_chunk_free(zero_);
            zero_ = 0;
            std::fprintf(stderr, "strata: STRATA_VMM_ZERO_BACK could not back the range with a zero chunk; the range stays unmapped\n");
        }
    }
    return true;
}

bool VmmRange::zero_map(int64_t i) {
    const Api& a = api();
    return a.map((CUdeviceptr) (base_ + (uint64_t) i * a.gran), (size_t) a.gran, 0, (CUmemGenericAllocationHandle) zero_, 0) ==
           CUDA_SUCCESS;
}

void VmmRange::zero_unmap(int64_t i) {
    const Api& a = api();
    (void) a.unmap((CUdeviceptr) (base_ + (uint64_t) i * a.gran), (size_t) a.gran);
}

void VmmRange::release() {
    if (base_ == 0) return;
    const Api& a = api();
    for (int64_t i = 0; i < chunks(); ++i) vmm_chunk_free(unmap(i));
    if (zero_ != 0) {
        for (int64_t i = 0; i < chunks(); ++i) zero_unmap(i);   // unmap() above put the zero chunk back on every one
        vmm_chunk_free(zero_);
        zero_ = 0;
    }
    a.free_va((CUdeviceptr) base_, (size_t) ((uint64_t) h_.size() * a.gran));
    base_ = 0;
    h_.clear();
}

int64_t VmmRange::mapped_count() const {
    int64_t n = 0;
    for (const VmmChunk h : h_) n += h != 0;
    return n;
}

bool VmmRange::map_one(int64_t i, VmmChunk h) {
    const Api& a = api();
    if (zero_ != 0) zero_unmap(i);   // the zero chunk steps aside for the real one
    if (a.map((CUdeviceptr) (base_ + (uint64_t) i * a.gran), (size_t) a.gran, 0, (CUmemGenericAllocationHandle) h, 0) !=
        CUDA_SUCCESS) {
        if (zero_ != 0 && zero_map(i)) (void) set_access_ro(i, i + 1);
        return false;
    }
    h_[(size_t) i] = h;
    return true;
}

bool VmmRange::set_access(int64_t lo, int64_t hi) {
    if (hi <= lo) return true;
    const Api& a = api();
    CUmemAccessDesc d{};
    d.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    d.location.id = a.dev;
    d.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    return a.access((CUdeviceptr) (base_ + (uint64_t) lo * a.gran), (size_t) ((uint64_t) (hi - lo) * a.gran), &d, 1) ==
           CUDA_SUCCESS;
}

bool VmmRange::set_access_ro(int64_t lo, int64_t hi) {
    if (hi <= lo) return true;
    const Api& a = api();
    CUmemAccessDesc d{};
    d.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    d.location.id = a.dev;
    d.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
    return a.access((CUdeviceptr) (base_ + (uint64_t) lo * a.gran), (size_t) ((uint64_t) (hi - lo) * a.gran), &d, 1) ==
           CUDA_SUCCESS;
}

bool VmmRange::commit_run(int64_t lo, int64_t hi) {
    if (set_access(lo, hi)) return true;
    for (int64_t c = lo; c < hi; ++c) vmm_chunk_free(unmap(c));
    return false;
}

VmmChunk VmmRange::unmap(int64_t i) {
    if (!mapped(i)) return 0;
    const Api& a = api();
    const VmmChunk h = h_[(size_t) i];
    if (a.unmap((CUdeviceptr) (base_ + (uint64_t) i * a.gran), (size_t) a.gran) != CUDA_SUCCESS) return 0;
    h_[(size_t) i] = 0;
    if (zero_ != 0 && zero_map(i)) (void) set_access_ro(i, i + 1);   // back to reading zeros
    return h;
}

// The zero-back start-up sequence and its read-back checks exist in the HIP branch only (T-950); CUDA keeps the simple version above.
bool VmmRange::zero_check(int64_t, int64_t, int64_t*, uint64_t*, int*) { return true; }
bool VmmRange::zero_setup(int64_t) { return false; }
void VmmRange::zero_disable(const char*) {}

}  // namespace strata::core

#else  // not CUDA
// HIP (ROCm 6+): the same elastic range through hipMemAddressReserve / hipMemCreate / hipMemMap / hipMemSetAccess.  Available
// only where the device reports virtual memory management support (STRATA_HIP_VMM=0 turns it off): --kv-grow stays off
// otherwise, as before.  Everything else (Metal, gfx906, ROCm older than 6): no virtual memory, the elastic K/V stays off.
#if defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
#include <hip/hip_runtime.h>
#include <hip/hip_version.h>
#include <cstring>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#endif

#if defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && defined(HIP_VERSION_MAJOR) && HIP_VERSION_MAJOR >= 6
namespace strata::core {
namespace {
struct Api {
    bool ok = false;
    uint64_t gran = 0;
    int dev = -1;
};
void fail(const char* what, hipError_t e) {
    static int n = 0;
    if (n++ < 8) std::fprintf(stderr, "strata: HIP virtual memory: %s failed: %s\n", what, hipGetErrorString(e));
    (void) hipGetLastError();
}
hipMemAllocationProp alloc_prop(int dev) {
    hipMemAllocationProp p{};
    p.type = hipMemAllocationTypePinned;
    p.requestedHandleType = hipMemHandleTypeNone;
    p.location.type = hipMemLocationTypeDevice;
    p.location.id = dev;
    return p;
}
const Api& api() {
    static Api& a = *new Api;   // never destroyed: the K/V pools and the cache release their chunks from static destructors
    static std::once_flag once;
    std::call_once(once, [] {
        const char* off = std::getenv("STRATA_HIP_VMM");
        if (off != nullptr && off[0] == '0') return;
        if (hipGetDevice(&a.dev) != hipSuccess) { (void) hipGetLastError(); return; }
        int vmm = 0;
        if (hipDeviceGetAttribute(&vmm, hipDeviceAttributeVirtualMemoryManagementSupported, a.dev) != hipSuccess || !vmm) {
            (void) hipGetLastError();
            return;
        }
        const hipMemAllocationProp prop = alloc_prop(a.dev);
        size_t g = 0;
        if (hipMemGetAllocationGranularity(&g, &prop, hipMemAllocationGranularityMinimum) != hipSuccess || g == 0) {
            (void) hipGetLastError();
            return;
        }
        // gfx1201 reports a 4 KiB minimum granularity; the elastic range works in 2 MiB chunks like CUDA's (any multiple of the minimum)
        const uint64_t want = 2ull << 20;
        a.gran = g >= want ? g : (want + g - 1) / g * g;
        a.ok = true;
    });
    return a;
}
}  // namespace

bool vmm_available() { return api().ok; }
uint64_t vmm_granularity() { return api().ok ? api().gran : 0; }

VmmChunk vmm_chunk_new() {
    const Api& a = api();
    if (!a.ok) return 0;
    const hipMemAllocationProp prop = alloc_prop(a.dev);
    hipMemGenericAllocationHandle_t h = nullptr;
    if (const hipError_t e = hipMemCreate(&h, (size_t) a.gran, &prop, 0); e != hipSuccess) { fail("hipMemCreate", e); return 0; }
    return (VmmChunk) (uintptr_t) h;
}

void vmm_chunk_free(VmmChunk h) {
    if (h != 0 && api().ok) (void) hipMemRelease((hipMemGenericAllocationHandle_t) (uintptr_t) h);
}

bool VmmRange::reserve(uint64_t bytes) {
    release();
    const Api& a = api();
    if (!a.ok || bytes == 0) return false;
    const uint64_t n = (bytes + a.gran - 1) / a.gran;
    void* p = nullptr;
    if (const hipError_t e = hipMemAddressReserve(&p, (size_t) (n * a.gran), 0, nullptr, 0); e != hipSuccess) { fail("hipMemAddressReserve", e); return false; }
    base_ = (unsigned long long) (uintptr_t) p;
    h_.assign((size_t) n, 0);
    // STRATA_VMM_ZERO_BACK=1 (default off): kvcached's zero-page backing, see VmmRange::zero_backed().  Any failure, or a chunk
    // that does not read zeros, undoes it and leaves the range plain (unmapped), as without the switch.
    zero_fell_back_ = false;
    if (const char* zb = std::getenv("STRATA_VMM_ZERO_BACK"); zb != nullptr && zb[0] != '\0' && zb[0] != '0') {
        std::fprintf(stderr, "strata: zero-back: granularity %zu B, %lld chunks (%zu MiB of addresses)\n", (size_t) a.gran, (long long) n,
                     (size_t) ((n * a.gran) >> 20));
        if (!zero_setup((int64_t) n)) zero_fell_back_ = true;
    }
    return true;
}

// Reads 16 bytes at the start, middle and end of every chunk in [lo, hi) through the mapping and compares them with zero; the first
// wrong byte (chunk, offset, value) is returned.  A failed copy counts as wrong (value -1).
bool VmmRange::zero_check(int64_t lo, int64_t hi, int64_t* bad_chunk, uint64_t* bad_off, int* bad_val) {
    const Api& a = api();
    const uint64_t offs[3] = {0, (uint64_t) a.gran / 2, (uint64_t) a.gran - 16};
    (void) hipDeviceSynchronize();
    for (int64_t i = lo; i < hi; ++i)
        for (const uint64_t o : offs) {
            unsigned char buf[16];
            std::memset(buf, 0xAB, sizeof buf);
            const hipError_t e = hipMemcpyDtoH(buf, (void*) (uintptr_t) (base_ + (uint64_t) i * a.gran + o), sizeof buf);
            int bad = -2;
            if (e != hipSuccess) { (void) hipGetLastError(); bad = -1; }
            else for (const unsigned char c : buf) if (c != 0) { bad = c; break; }
            if (bad != -2) { *bad_chunk = i; *bad_off = o; *bad_val = bad; return false; }
        }
    return true;
}

void VmmRange::zero_disable(const char* why) {
    for (int64_t i = 0; i < chunks(); ++i)
        if (!mapped(i)) zero_unmap(i);   // only the addresses that hold the shared zero chunk (a real chunk keeps its mapping)
    (void) hipGetLastError();
    vmm_chunk_free(zero_);
    zero_ = 0;
    zero_fell_back_ = true;
    std::fprintf(stderr, "strata: STRATA_VMM_ZERO_BACK undone (%s); the range works as a plain range, unmapped chunks stay unmapped\n", why);
}

// The start-up sequence, one stderr line per step.  Stage 1 clears the shared chunk through the mapping of chunk 0, stage 2 (only when
// stage 1 reads back wrong) through every mapping, each time followed by a device sync, read-only access for all, and a read-back
// of every chunk.  Nothing passing -> everything is unmapped again and the chunk is freed.
bool VmmRange::zero_setup(int64_t n) {
    const Api& a = api();
    zero_ = vmm_chunk_new();
    std::fprintf(stderr, "strata: zero-back: zero chunk create %s\n", zero_ != 0 ? "ok" : "FAILED");
    if (zero_ == 0) return false;
    int64_t maps = 0;
    for (; maps < n; ++maps)
        if (!zero_map(maps)) break;
    std::fprintf(stderr, "strata: zero-back: mapped the zero chunk at %lld of %lld chunk addresses\n", (long long) maps, (long long) n);
    bool ok = maps == n;
    const char* why = "mapping the zero chunk failed";
    for (int stage = 1; ok && stage <= 2; ++stage) {
        hipMemAccessDesc rwd{};
        rwd.location.type = hipMemLocationTypeDevice;
        rwd.location.id = a.dev;
        rwd.flags = hipMemAccessFlagsProtReadWrite;
        const int64_t upto = stage == 1 ? 1 : n;   // the addresses that get a write mapping and a clear
        const hipError_t e_acc = hipMemSetAccess((void*) (uintptr_t) base_, (size_t) ((uint64_t) upto * a.gran), &rwd, 1);
        hipError_t e_set = hipSuccess;
        for (int64_t i = 0; e_acc == hipSuccess && i < upto && e_set == hipSuccess; ++i)
            e_set = hipMemset((void*) (uintptr_t) (base_ + (uint64_t) i * a.gran), 0, (size_t) a.gran);
        const hipError_t e_syn = hipDeviceSynchronize();
        std::fprintf(stderr, "strata: zero-back: stage %d read-write access: %s, memset of %lld chunk(s): %s, sync: %s\n", stage, hipGetErrorString(e_acc),
                     (long long) upto, hipGetErrorString(e_set), hipGetErrorString(e_syn));
        if (e_acc != hipSuccess || e_set != hipSuccess || e_syn != hipSuccess) { why = "the clear failed"; ok = false; break; }
        const bool ro = set_access_ro(0, n);
        std::fprintf(stderr, "strata: zero-back: stage %d read-only access for all chunks: %s\n", stage, ro ? "ok" : "FAILED");
        if (!ro) { why = "read-only access failed"; ok = false; break; }
        int64_t bc = 0; uint64_t bo = 0; int bv = 0;
        if (zero_check(0, n, &bc, &bo, &bv)) {
            std::fprintf(stderr, "strata: zero-back: stage %d self-check ok (%lld chunks read zeros)\n", stage, (long long) n);
            return true;
        }
        std::fprintf(stderr, "strata: zero-back: stage %d self-check FAILED: chunk %lld offset %llu value %d (-1 = the copy failed)\n", stage, (long long) bc,
                     (unsigned long long) bo, bv);
        why = "the zero chunk does not read zeros through every address";
        ok = stage == 1;   // stage 2 (clear through every mapping) follows only a bad read-back
    }
    (void) hipGetLastError();
    for (int64_t i = 0; i < maps; ++i) zero_unmap(i);
    vmm_chunk_free(zero_);
    zero_ = 0;
    std::fprintf(stderr, "strata: STRATA_VMM_ZERO_BACK undone (%s); the range stays unmapped\n", why);
    return false;
}

bool VmmRange::zero_map(int64_t i) {
    const Api& a = api();
    const hipError_t e = hipMemMap((void*) (uintptr_t) (base_ + (uint64_t) i * a.gran), (size_t) a.gran, 0,
                                   (hipMemGenericAllocationHandle_t) (uintptr_t) zero_, 0);
    if (e != hipSuccess) fail("hipMemMap (zero chunk)", e);   // only reached with STRATA_VMM_ZERO_BACK=1; logs, never changes the result
    return e == hipSuccess;
}

void VmmRange::zero_unmap(int64_t i) {
    const Api& a = api();
    (void) hipMemUnmap((void*) (uintptr_t) (base_ + (uint64_t) i * a.gran), (size_t) a.gran);
}

void VmmRange::release() {
    if (base_ == 0) return;
    const Api& a = api();
    for (int64_t i = 0; i < chunks(); ++i) vmm_chunk_free(unmap(i));
    if (zero_ != 0) {
        for (int64_t i = 0; i < chunks(); ++i) zero_unmap(i);   // unmap() above put the zero chunk back on every one
        vmm_chunk_free(zero_);
        zero_ = 0;
    }
    (void) hipMemAddressFree((void*) (uintptr_t) base_, (size_t) ((uint64_t) h_.size() * a.gran));
    base_ = 0;
    h_.clear();
}

int64_t VmmRange::mapped_count() const {
    int64_t n = 0;
    for (const VmmChunk h : h_) n += h != 0;
    return n;
}

bool VmmRange::map_one(int64_t i, VmmChunk h) {
    const Api& a = api();
    if (zero_ != 0) zero_unmap(i);   // the zero chunk steps aside for the real one
    if (const hipError_t e = hipMemMap((void*) (uintptr_t) (base_ + (uint64_t) i * a.gran), (size_t) a.gran, 0,
                                      (hipMemGenericAllocationHandle_t) (uintptr_t) h, 0);
        e != hipSuccess) {
        fail("hipMemMap", e);
        if (zero_ != 0 && zero_map(i)) (void) set_access_ro(i, i + 1);
        return false;
    }
    h_[(size_t) i] = h;
    return true;
}

bool VmmRange::set_access(int64_t lo, int64_t hi) {
    if (hi <= lo) return true;
    const Api& a = api();
    hipMemAccessDesc d{};
    d.location.type = hipMemLocationTypeDevice;
    d.location.id = a.dev;
    d.flags = hipMemAccessFlagsProtReadWrite;
    const hipError_t e = hipMemSetAccess((void*) (uintptr_t) (base_ + (uint64_t) lo * a.gran), (size_t) ((uint64_t) (hi - lo) * a.gran),
                                         &d, 1);
    const bool ok = e == hipSuccess;
    if (!ok) fail("hipMemSetAccess", e);
    return ok;
}

bool VmmRange::set_access_ro(int64_t lo, int64_t hi) {
    if (hi <= lo) return true;
    const Api& a = api();
    hipMemAccessDesc d{};
    d.location.type = hipMemLocationTypeDevice;
    d.location.id = a.dev;
    d.flags = hipMemAccessFlagsProtRead;
    const hipError_t e = hipMemSetAccess((void*) (uintptr_t) (base_ + (uint64_t) lo * a.gran), (size_t) ((uint64_t) (hi - lo) * a.gran),
                                         &d, 1);
    if (e != hipSuccess) fail("hipMemSetAccess (read-only)", e);
    return e == hipSuccess;
}

bool VmmRange::commit_run(int64_t lo, int64_t hi) {
    if (set_access(lo, hi)) return true;
    for (int64_t c = lo; c < hi; ++c) vmm_chunk_free(unmap(c));
    return false;
}

VmmChunk VmmRange::unmap(int64_t i) {
    if (!mapped(i)) return 0;
    const Api& a = api();
    const VmmChunk h = h_[(size_t) i];
    if (zero_ != 0) (void) hipDeviceSynchronize();   // nothing may still use the chunk when its address is handed to the zero chunk
    if (hipMemUnmap((void*) (uintptr_t) (base_ + (uint64_t) i * a.gran), (size_t) a.gran) != hipSuccess) { (void) hipGetLastError(); return 0; }
    h_[(size_t) i] = 0;
    if (zero_ != 0) {   // back to reading zeros, and proved: a stale mapping would show the bytes the real chunk held
        int64_t bc = i; uint64_t bo = 0; int bv = 0;
        for (int attempt = 0; attempt < 2; ++attempt) {
            if (attempt > 0) { zero_unmap(i); (void) hipDeviceSynchronize(); }
            if (zero_map(i) && set_access_ro(i, i + 1) && zero_check(i, i + 1, &bc, &bo, &bv)) return h;
            std::fprintf(stderr, "strata: zero-back: chunk %lld read-back after unmap, attempt %d: wrong at offset %llu value %d (-1 = a call or the copy failed)\n",
                         (long long) i, attempt + 1, (unsigned long long) bo, bv);
        }
        zero_disable("an unmapped chunk did not read zeros again");
    }
    return h;
}
}  // namespace strata::core
#else


namespace strata::core {
bool vmm_available() { return false; }
uint64_t vmm_granularity() { return 0; }
VmmChunk vmm_chunk_new() { return 0; }
void vmm_chunk_free(VmmChunk) {}
bool VmmRange::reserve(uint64_t) { return false; }
void VmmRange::release() {}
int64_t VmmRange::mapped_count() const { return 0; }
bool VmmRange::map_one(int64_t, VmmChunk) { return false; }
bool VmmRange::set_access(int64_t, int64_t) { return false; }
bool VmmRange::commit_run(int64_t, int64_t) { return false; }
bool VmmRange::set_access_ro(int64_t, int64_t) { return false; }
bool VmmRange::zero_map(int64_t) { return false; }
bool VmmRange::zero_check(int64_t, int64_t, int64_t*, uint64_t*, int*) { return true; }
bool VmmRange::zero_setup(int64_t) { return false; }
void VmmRange::zero_disable(const char*) {}
void VmmRange::zero_unmap(int64_t) {}
VmmChunk VmmRange::unmap(int64_t) { return 0; }
}  // namespace strata::core

#endif
#endif
