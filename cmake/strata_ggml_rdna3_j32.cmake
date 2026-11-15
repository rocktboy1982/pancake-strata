# cmake/strata_ggml_rdna3_j32.cmake - opt-in RDNA3 MMQ register fix (llama.cpp PR #31751, "CUDA: avoid J=32 MMQ register
# spills on RDNA3", open upstream; LEADS.md section 0) applied to the pinned ggml for the HIP prompt MMQ (strata_mmq).
#
# On RDNA3 the fully unrolled k01 loop of ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma at J == 32 keeps the A fragments of every
# iteration live: the kernel needs 256 VGPRs and still spills to scratch.  The PR keeps that one loop rolled
# (#pragma unroll 1) for J == 32 only; the other J and every other architecture are unchanged.  The arithmetic and its
# order are the same, so the results are the same bits.  Author number: +36% at 17-32-token batches on gfx1100.  It
# touches the types that use this vec-dot (Q2_0, IQ2_XXS, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL, Q8_0, Q4_0, Q5_0, MXFP4), i.e. our
# down experts (Q2_0) and part of gate/up (IQ2_XXS) on the staged prompt chunks (< stream_all_min(), the agent deltas).
#
# This script writes, at configure time, an OVERLAY into the build tree: a copy of mmq-vec-dot.cuh with the change,
# a copy of mmq.cuh (the trellis overlay's when that one is on), and copies of the given MMQ template instances, which
# include "../mmq.cuh" and so pick up the overlay.  Only strata_mmq sees it (first on its include path).  Every edit is
# anchored on the pinned text; if an anchor is missing (another llama.cpp), it says so and stays off.
#
#   strata_ggml_rdna3_j32(<ggml-cuda dir> <dir holding the mmq.cuh to use> <out dir> <types...>)
#     -> STRATA_RDNA3_J32_OK (parent scope)

function(strata_ggml_rdna3_j32 CU MMQ_DIR OUT)
  set(STRATA_RDNA3_J32_OK OFF PARENT_SCOPE)
  foreach(_f "${CU}/mmq-vec-dot.cuh" "${MMQ_DIR}/mmq.cuh")
    if(NOT EXISTS "${_f}")
      message(STATUS "RDNA3 J=32 MMQ overlay: ${_f} is missing: off")
      return()
    endif()
  endforeach()
  file(READ "${CU}/mmq-vec-dot.cuh" _v)
  string(REPLACE "\r\n" "\n" _v "${_v}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CU}/mmq-vec-dot.cuh" "${MMQ_DIR}/mmq.cuh")

  # the AMD (MFMA / WMMA) branch of ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma: from its k01 loop to that branch's #else
  set(_fn "static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma(")
  set(_loop "    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {\n        const int k0 = k00 + k01;\n\n        tile_A A[ntx];")
  string(FIND "${_v}" "${_fn}" _p_fn)
  if(_p_fn LESS 0)
    message(STATUS "RDNA3 J=32 MMQ overlay: anchor not found (${_fn}): off")
    return()
  endif()
  string(SUBSTRING "${_v}" ${_p_fn} -1 _rest)
  string(FIND "${_rest}" "#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)\n" _p_amd)
  string(FIND "${_rest}" "${_loop}" _p_loop)
  string(FIND "${_rest}" "\n#else\n" _p_else)
  if(_p_amd LESS 0 OR _p_loop LESS 0 OR _p_else LESS 0 OR NOT (_p_amd LESS _p_loop AND _p_loop LESS _p_else))
    message(STATUS "RDNA3 J=32 MMQ overlay: the AMD branch of the q8_0 MMA vec-dot is not the pinned text: off")
    return()
  endif()
  math(EXPR _len "${_p_else} - ${_p_loop}")
  string(SUBSTRING "${_rest}" ${_p_loop} ${_len} _body)          # the whole k01 loop, ending with its closing brace
  set(_new "#if defined(RDNA3)
    // llama.cpp PR #31751: at J == 32 a fully unrolled k01 loop keeps every iteration's A fragments live (256 VGPRs and
    // still a spill); keep this loop rolled there.  Same arithmetic in the same order, so the same bits.
    if constexpr (J == 32) {
#pragma unroll 1
${_body}
    } else
#endif // defined(RDNA3)
    {
${_body}
    }")
  math(EXPR _abs_loop "${_p_fn} + ${_p_loop}")
  math(EXPR _abs_else "${_p_fn} + ${_p_else}")
  string(SUBSTRING "${_v}" 0 ${_abs_loop} _head)
  string(SUBSTRING "${_v}" ${_abs_else} -1 _tail)
  set(_v "${_head}${_new}${_tail}")

  file(MAKE_DIRECTORY "${OUT}/template-instances")
  # the whole header set of MMQ_DIR (the trellis overlay copies the full set too): a quoted include resolves next to the
  # including file first, so every header that pulls in mmq-vec-dot.cuh must sit beside the patched copy
  file(GLOB _all "${MMQ_DIR}/*.cuh" "${MMQ_DIR}/*.h")
  foreach(_f IN LISTS _all)
    get_filename_component(_n "${_f}" NAME)
    configure_file("${_f}" "${OUT}/${_n}" COPYONLY)
  endforeach()
  if(IS_DIRECTORY "${MMQ_DIR}/vendors")
    file(COPY "${MMQ_DIR}/vendors" DESTINATION "${OUT}")
  elseif(IS_DIRECTORY "${CU}/vendors")
    file(COPY "${CU}/vendors" DESTINATION "${OUT}")
  endif()
  file(WRITE "${OUT}/mmq-vec-dot.cuh" "${_v}")
  foreach(_t IN LISTS ARGN)
    if(EXISTS "${MMQ_DIR}/template-instances/mmq-instance-${_t}.cu")   # the trellis overlay's own instances first
      configure_file("${MMQ_DIR}/template-instances/mmq-instance-${_t}.cu" "${OUT}/template-instances/mmq-instance-${_t}.cu" COPYONLY)
    elseif(EXISTS "${CU}/template-instances/mmq-instance-${_t}.cu")
      configure_file("${CU}/template-instances/mmq-instance-${_t}.cu" "${OUT}/template-instances/mmq-instance-${_t}.cu" COPYONLY)
    else()
      message(STATUS "RDNA3 J=32 MMQ overlay: no template instance for ${_t}: off")
      return()
    endif()
  endforeach()
  set(STRATA_RDNA3_J32_OK ON PARENT_SCOPE)
endfunction()
