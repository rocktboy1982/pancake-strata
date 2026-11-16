// src/kernels/cuda/route_plan.cu - STRATA_ROUTE_PLAN=1: the routing top-10 and the all-resident plan in one launch.
// Ported from jstamagal/Strata-gfx1030 ef00764552 (LEADS.md "gfx1030 decode").
//
// One workgroup of 8 waves.  Wave t < n_tok routes token t with route_multi's code (native_router.cu: the softmax
// over 512 logits held 16 per lane, ten warp-argmax rounds, the renormalised weights) and also leaves its ten ids in
// shared memory.  After one barrier, thread i < n_tok * 10 is plan entry i with resident_plan_kernel's code
// (verify_kernels.cu: groups are the distinct experts in order of first occurrence, a prefix scan over the waves
// places them).  Both bodies are copies, so the ids, the weights and every word of the plan are the two kernels'.
// The plan is the all-resident one (no skip word): a routed expert that is not resident leaves an empty plan and
// raises *plan_err, as there.
#include "strata/kernels/route_plan.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int kEntries = 128;   // resident_plan's kResidentPlanMax: one thread per entry, at most 4 waves

__device__ __forceinline__ float rp_warp_sum(float value) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value += __shfl_xor_sync(0xffffffffu, value, mask, 32);
    return value;
}
__device__ __forceinline__ float rp_warp_max(float value) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask, 32));
    return value;
}

// route_multi for wave tk < n_tok, one barrier, resident_plan_kernel for thread tid < n_tok * 10.  `load(tk, i)` is
// logit i of token tk.  The caller's kernel declares the shared arrays (kEntries, kEntries, 4, 1).
template <typename Load>
__device__ __forceinline__ void route_plan_body(Load&& load, int32_t* __restrict__ ids_, float* __restrict__ weights_,
                                                int n_tok, const int32_t* __restrict__ res, int n_expert,
                                                const uint8_t* cache_base, const unsigned long long* slot_off,
                                                long long blob, int32_t* __restrict__ pl, long long capx,
                                                volatile uint32_t* plan_err, int32_t* s_ids, int32_t* s_excl,
                                                int32_t* s_wsum, int* s_bad_p) {
    int& s_bad = *s_bad_p;
    const int lane = threadIdx.x;
    const int tk = (int) threadIdx.y;
    const int tid = tk * 32 + lane;
    const int n = n_tok * 10, k = 10;
    if (tid == 0) s_bad = 0;
    if (tk < n_tok) {   // ---- route_multi, wave tk = token tk
        int32_t* ids = ids_ + (size_t) tk * 10;
        float* weights = weights_ + (size_t) tk * 10;
        float values[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) values[i] = load(tk, lane + i * 32);
        __syncwarp();
        float maximum = -INFINITY;
#pragma unroll
        for (int i = 0; i < 16; ++i) maximum = max(maximum, values[i]);
        maximum = rp_warp_max(maximum);
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            values[i] = expf(values[i] - maximum);
            sum += values[i];
        }
        const float reciprocal = 1.0f / rp_warp_sum(sum);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            values[i] *= reciprocal;
            if (__isnanf(values[i])) values[i] = -FLT_MAX;
        }
        float selected = 0.0f, selected_sum = 0.0f;
        for (int rank = 0; rank < 10; ++rank) {
            float best = values[0];
            int expert = lane;
#pragma unroll
            for (int i = 1; i < 16; ++i) {
                if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
            }
#pragma unroll
            for (int mask = 16; mask; mask >>= 1) {
                const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
                const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
                if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
            }
            if ((expert & 31) == lane) {
                values[expert / 32] = -INFINITY;
                ids[rank] = expert;
                s_ids[tk * 10 + rank] = expert;
                selected_sum += best;
            }
            if (rank == lane) selected = best;
        }
        selected_sum = max(rp_warp_sum(selected_sum), 6.103515625e-5f);
        const float inverse_selected_sum = 1.0f / selected_sum;
        if (lane < 10) weights[lane] = selected * inverse_selected_sum;
    }
    __syncthreads();

    // ---- resident_plan_kernel, thread tid = entry tid (waves 0..3)
    int32_t eid = -1;
    int32_t slot = -1;
    if (tid < n) {
        eid = s_ids[tid];
        slot = (eid >= 0 && eid < n_expert) ? res[eid] : -1;
        if (slot < 0) atomicOr(&s_bad, 1);
    }
    __syncthreads();
    if (s_bad) {
        if (tid == 0) {
            pl[0] = 0; pl[1] = 0; pl[2] = 0;
            if (plan_err != nullptr) { *plan_err = 1; __threadfence_system(); }
        }
        return;
    }
    int first_j = tid;
    int rank_in_group = 0;
    int count_same = 0;
    if (tid < n) {
        for (int j = 0; j < n; ++j) {
            if (s_ids[j] == eid) {
                if (j < first_j) first_j = j;
                if (j < tid) ++rank_in_group;
                ++count_same;
            }
        }
    }
    const bool is_first = (tid < n && first_j == tid && slot >= 0);
    const int my_cnt = is_first ? count_same : 0;
    const int my_pack = (my_cnt << 16) | (is_first ? 1 : 0);
    int pref = my_pack;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const int up = __shfl_up_sync(0xffffffffu, pref, d);
        if (lane >= d) pref += up;
    }
    if (tid < kEntries) {
        s_excl[tid] = pref - my_pack;
        if (lane == 31) s_wsum[tk] = pref;
    }
    __syncthreads();

    int32_t* counts = pl;
    int32_t* start = pl + 4;
    int32_t* dst = start + capx + 1;
    int32_t* tok = dst + capx;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    unsigned long long* ptr = (unsigned long long*) (pl + ptr_off);
    int32_t* start2 = pl + ptr_off + 4 * capx;

    if (is_first) {
        int tot = s_excl[tid];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            if (w < tk) tot += s_wsum[w];
        }
        const int grp_idx = tot & 0xffff;
        const int ent_start = tot >> 16;
        ptr[grp_idx] = (unsigned long long) (cache_base + (slot_off ? (size_t) slot_off[slot] : (size_t) slot * (size_t) blob));
        start[grp_idx] = ent_start;
    }
    if (tid < n) {
        const int fj_warp = first_j >> 5;
        int fj_tot = s_excl[first_j];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            if (w < fj_warp) fj_tot += s_wsum[w];
        }
        const int out_idx = (fj_tot >> 16) + rank_in_group;
        dst[out_idx] = tid;
        tok[out_idx] = tid / k;
    }
    if (tid == 0) {
        const int groups = (s_wsum[0] + s_wsum[1] + s_wsum[2] + s_wsum[3]) & 0xffff;
        start[groups] = n;
        start2[0] = n;
        counts[0] = groups;
        counts[1] = n;
        counts[2] = 0;
    }
}

__launch_bounds__(256, 1)
__global__ void route_plan_kernel(const float* __restrict__ logits_, int32_t* __restrict__ ids_,
                                  float* __restrict__ weights_, int n_tok, const int32_t* __restrict__ res,
                                  int n_expert, const uint8_t* cache_base, const unsigned long long* slot_off,
                                  long long blob, int32_t* __restrict__ pl, long long capx,
                                  volatile uint32_t* plan_err) {
    __shared__ int32_t s_ids[kEntries];
    __shared__ int32_t s_excl[kEntries];
    __shared__ int32_t s_wsum[4];
    __shared__ int s_bad;
    route_plan_body([&](int tk, int i) { return logits_[(size_t) tk * 512 + i]; }, ids_, weights_, n_tok, res, n_expert,
                    cache_base, slot_off, blob, pl, capx, plan_err, s_ids, s_excl, s_wsum, &s_bad);
}


}  // namespace

int route_plan_mode() {
    // STRATA_ROUTE_PLAN=1 only: the fork's =2 (the router's BF16 projection in the same launch) copies its
    // bf16_f32_mmvf_multi_kernel, which this tree has changed (block sizes, the fast / unrolled variants), so it is
    // not ported (LEADS.md "gfx1030 decode")
    static const int mode = [] {
#if defined(STRATA_HIP_GFX906)
        return 0;   // the kernel is written for 32-lane waves (the plan's prefix scan): wave64 keeps the two launches
#endif
        const char* v = std::getenv("STRATA_ROUTE_PLAN");
        return v != nullptr && v[0] == '1' && v[1] == '\0' ? 1 : 0;
    }();
    return mode;
}
bool route_plan_enabled() { return route_plan_mode() != 0; }

void native_route_plan(const float* logits, int32_t* ids, float* weights, int n_tok, const int32_t* res_layer,
                       int n_expert, const uint8_t* cache_base, const unsigned long long* slot_off, long long blob,
                       int32_t* plan, long long capx, uint32_t* plan_err, void* stream) {
    if (!stream || !logits || !ids || !weights || !res_layer || !plan || n_tok < 1 || n_tok > 8 || capx < (long long) n_tok * 10)
        throw std::invalid_argument("native_route_plan: needs a stream, device buffers and 1..8 tokens");
    route_plan_kernel<<<1, dim3(32, 8), 0, static_cast<cudaStream_t>(stream)>>>(
        logits, ids, weights, n_tok, res_layer, n_expert, cache_base, slot_off, blob, plan, capx, plan_err);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("native_route_plan: ") + cudaGetErrorString(e));
}

}  // namespace strata::kernels
