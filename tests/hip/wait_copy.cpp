// F21 (STRATA_WAIT_COPY256): the verify plan copied after a host flag - wait_flag_copy_i32 (the STRATA_WAIT_FUSE path)
// and wait_copy_i32_from_mapped (the STRATA_FUSE_WAIT path) - while the host rewrites the SAME mapped words every round,
// after the GPU is already spinning: every copied word must be this round's (a read before the flag or a stale cache
// line fails).  CTest runs it twice: as is (the 128-thread 4-byte kernels) and with STRATA_WAIT_COPY256=1 (the 256-thread
// 16-byte kernel for plans of more than 512 words; a word tail and a misaligned source take its 4-byte loop).
#include <hip/hip_runtime.h>
#include "strata/kernels/verify_kernels.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#define CHECK(x) do { auto e = (x); if (e != hipSuccess) { std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e)); return 2; } } while (0)

int main() {
    constexpr int cap = 1328, rounds = 200;   // 1,320 words: a 16-row window's plan
    int32_t *host = nullptr, *mapped = nullptr, *dev = nullptr;
    uint32_t *flag = nullptr, *dflag = nullptr;
    CHECK(hipHostMalloc((void**) &host, cap * 4, hipHostMallocMapped));
    CHECK(hipHostMalloc((void**) &flag, 4, hipHostMallocMapped));
    CHECK(hipHostGetDevicePointer((void**) &mapped, host, 0));
    CHECK(hipHostGetDevicePointer((void**) &dflag, flag, 0));
    CHECK(hipMalloc((void**) &dev, cap * 4));
    hipStream_t stream;
    CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    struct Case { int off, n; bool fused; };
    // the engine's plan (aligned, 1,320 words) on both paths; a word tail; a misaligned source; a plan under 512 words
    const Case cases[] = {{0, 1320, true}, {0, 1320, false}, {0, 1323, true}, {1, 1320, true}, {0, 500, true}};
    std::vector<int32_t> got(cap);
    *flag = 0;
    uint32_t r = 0;
    long long checked = 0;
    for (const Case& c : cases) {
        for (int i = 0; i < rounds; ++i) {
            ++r;
            if (c.fused) strata::kernels::wait_flag_copy_i32(dflag, r, dev, mapped + c.off, c.n, (void*) stream);
            else strata::kernels::wait_copy_i32_from_mapped(dflag, r, dev, mapped + c.off, c.n, (void*) stream);
            std::this_thread::sleep_for(std::chrono::microseconds(20 + (i % 13) * 10));   // the GPU spins by now
            for (int j = 0; j < c.n; ++j) host[c.off + j] = (int32_t) (r * 4096u + (uint32_t) j);
            __atomic_store_n(flag, r, __ATOMIC_RELEASE);
            CHECK(hipStreamSynchronize(stream));
            CHECK(hipMemcpy(got.data(), dev, (size_t) c.n * 4, hipMemcpyDeviceToHost));
            for (int j = 0; j < c.n; ++j) {
                if (got[j] != (int32_t) (r * 4096u + (uint32_t) j)) {
                    std::fprintf(stderr, "stale plan word: %s n=%d off=%d round %u word %d: got %d\n",
                                 c.fused ? "wait_flag_copy_i32" : "wait_copy_i32_from_mapped", c.n, c.off, r, j, got[j]);
                    return 1;
                }
            }
            checked += c.n;
        }
    }
    CHECK(hipStreamDestroy(stream));
    CHECK(hipFree(dev));
    CHECK(hipHostFree(host));
    CHECK(hipHostFree(flag));
    const char* w = std::getenv("STRATA_WAIT_COPY256");
    std::printf("PASS plan copy after a host flag (%s): %u rounds, %lld words checked\n",
                w != nullptr && w[0] == '1' ? "STRATA_WAIT_COPY256=1" : "default kernels", r, checked);
    return 0;
}
