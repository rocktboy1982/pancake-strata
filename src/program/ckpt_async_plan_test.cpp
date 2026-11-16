// MOD_PLAN.md section 6 item 37: STRATA_CKPT_ASYNC on HIP, the host rules (include/strata/program/ckpt_async_plan.hpp).
#include "strata/program/ckpt_async_plan.hpp"

#include <cstdio>
#include <cstdlib>

using namespace strata::program;

static int checks = 0;
static void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

int main() {
    check(!ckpt_async_allowed(CkptBackend::Hip, nullptr), "unset: off on HIP");
    check(!ckpt_async_allowed(CkptBackend::Cuda, nullptr), "unset: off on CUDA");
    check(!ckpt_async_allowed(CkptBackend::Hip, "0"), "0: off");
    check(!ckpt_async_allowed(CkptBackend::Hip, ""), "empty: off");
    check(ckpt_async_allowed(CkptBackend::Hip, "1"), "1: on on HIP");
    check(ckpt_async_allowed(CkptBackend::Cuda, "1"), "1: on on CUDA");
    check(!ckpt_async_allowed(CkptBackend::HipGfx906, "1"), "gfx906 stays off");
    check(!ckpt_async_allowed(CkptBackend::Metal, "1"), "Metal stays off");
    const size_t mib = (size_t) 1 << 20, need = 118 * mib;
    check(ckpt_stage_room_ok(CkptBackend::Cuda, 0, need), "CUDA: no room check (old behaviour)");
    check(!ckpt_stage_room_ok(CkptBackend::Hip, need, need), "HIP: exactly the buffer: no");
    check(!ckpt_stage_room_ok(CkptBackend::Hip, need + kCkptStageReserve, need), "HIP: reserve exactly: no");
    check(ckpt_stage_room_ok(CkptBackend::Hip, need + kCkptStageReserve + 1, need), "HIP: reserve + 1: yes");
    check(!ckpt_stage_room_ok(CkptBackend::Hip, 10 * mib, need), "HIP: less free than the buffer: no, no underflow");
    std::printf("ckpt_async_plan_test: %d checks passed\n", checks);
    return 0;
}
