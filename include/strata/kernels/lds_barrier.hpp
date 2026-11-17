// include/strata/kernels/lds_barrier.hpp - LEADS.md Part P lead L6, build macro STRATA_LDS_ONLY_BARRIER (default off).
//
// STRATA_BAR() is __syncthreads() unless the build defines STRATA_LDS_ONLY_BARRIER, HIP and a gfx11 target: then it is
// `s_waitcnt lgkmcnt(0); s_barrier` between two scheduling fences.  __syncthreads() on gfx11 also waits vmcnt(0), so every
// outstanding GLOBAL load of the wave (a register prefetch of the next tile) must land before the barrier; the LDS-only
// form lets that prefetch stay in flight (idea: hipfire attention_verify_wmma.gfx1100.hip VW_BAR, Apache-2.0; our own
// code, no text copied).
//
// VALID ONLY where the barrier orders LDS and no wave reads, after the barrier, a GLOBAL write that another wave made
// before it (the global-memory release the plain barrier gives is gone).  Audited kernels, and why (each: the data that
// crosses a barrier goes through __shared__, global stores are partial results that only a later launch reads):
//   qsa_decode_attn_wmma.cu: decode_attn_wmma_kernel, decode_attn_wmma_rows_kernel (every barrier);
//   qsa_prompt_attn.cu: prompt_attn_wmma_kernel and prompt_attn_wmma_q4_kernel chunk-loop barriers (the two inside `chunk`).
// Every other kernel keeps __syncthreads().  Exactness: scheduling only; the ISA check is the number of `s_waitcnt vmcnt`
// in the loop (tools/pancake/kernel_resources.py digest of the default build must NOT change: the macro is off by default).
#pragma once

#if defined(STRATA_LDS_ONLY_BARRIER) && defined(__HIPCC__) && \
    (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
     defined(__gfx1150__) || defined(__gfx1151__))
#define STRATA_BAR() \
    do { \
        __builtin_amdgcn_sched_barrier(0); \
        __asm__ volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); \
        __builtin_amdgcn_sched_barrier(0); \
    } while (0)
#else
#define STRATA_BAR() __syncthreads()
#endif
