// src/kernels/cuda/route_row.cuh - the native router's top-10 row and STRATA_ROUTE_PUBLISH's publish, as device functions
// shared by native_router.cu (route_row / route_publish_res) and native_bf16.cu (the router GEMV with the publish as its
// last block's epilogue, MOD_PLAN item 48).  The code is route_row / route_publish_res's, with the lane and the block's
// thread index passed in instead of read from threadIdx: the same arithmetic, the same bits.  Internal (not a header of
// the public include tree); include inside namespace strata::kernels { namespace { ... } }.
#pragma once
__device__ __forceinline__ float rr_warp_sum(float value) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value += __shfl_xor_sync(0xffffffffu, value, mask, 32);
    return value;
}
__device__ __forceinline__ float rr_warp_max(float value) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask, 32));
    return value;
}
__device__ __forceinline__ void rr_route_row(const float* __restrict__ logits, int32_t* __restrict__ ids,
                                             float* __restrict__ weights, int tk, int lane) {
    logits += (size_t) tk * 512; ids += (size_t) tk * 10; weights += (size_t) tk * 10;
    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = logits[lane + i * 32];
    __syncwarp();
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) maximum = max(maximum, values[i]);
    maximum = rr_warp_max(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    const float reciprocal = 1.0f / rr_warp_sum(sum);
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
            selected_sum += best;
        }
        if (rank == lane) selected = best;
    }
    selected_sum = max(rr_warp_sum(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < 10) weights[lane] = selected * inverse_selected_sum;
}

// route_publish_res's body for a block of 256 threads, `tid` = the thread's index in it (warp tk = tid / 32 routes token tk)
__device__ __forceinline__ void rr_route_publish(const float* __restrict__ logits, int32_t* __restrict__ ids,
                                                 float* __restrict__ weights, int n_tok, const float* __restrict__ x,
                                                 const int32_t* __restrict__ d_res, int n_expert, int n, float* x_out,
                                                 int32_t* ids_out, uint32_t* seq, int tid) {
    const int tk = tid >> 5;
    if (tk < n_tok) rr_route_row(logits, ids, weights, tk, tid & 31);
    __syncthreads();
    const int t = tid, k = n_tok * 10;
    int any_miss = 0;
    for (int i = t; i < k; i += 256) {
        const int32_t id = ids[i];
        ids_out[i] = id;
        if (d_res == nullptr || id < 0 || id >= n_expert || d_res[id] < 0) any_miss = 1;
    }
    if (__syncthreads_or(any_miss)) {
        if ((n & 3) == 0 && ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(x_out)) & 15u) == 0)
            for (int i = t; i < n / 4; i += 256) reinterpret_cast<float4*>(x_out)[i] = reinterpret_cast<const float4*>(x)[i];
        else
            for (int i = t; i < n; i += 256) x_out[i] = x[i];
        __threadfence_system();
    } else if (t < k) {
        __threadfence_system();
    }
    __syncthreads();
    if (t == 0) {
        *(volatile uint32_t*) seq = *(volatile uint32_t*) seq + 1u;
#if defined(__HIPCC__)
        __threadfence_system();
#endif
    }
}
