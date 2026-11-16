# cmake -DEXE=<hip_decode_attn_wmma_parity> -DOUT=<scratch dir> -P cmake/qsa_wave32_parity.cmake
# STRATA_QSA_WAVE32=1 (dardeaw 580e251e69; LEADS.md): the wave32 QSA decode set (PR #540's chunk kernel with LDS-only
# barriers, attn_merge_v2_kernel) must give the default decode attention's outputs bit for bit.  The parity binary dumps
# the default kernel's outputs of every case; it runs once without and once with the switch, and the dumps are compared.
# Exit code 77 of the binary (no gfx11 device) is passed through as a skip.
file(MAKE_DIRECTORY "${OUT}")
execute_process(COMMAND "${EXE}" --dump "${OUT}/plain.bin" RESULT_VARIABLE _r0 OUTPUT_QUIET)
if(_r0 EQUAL 77)
  message("SKIP: no gfx11 device")
  # CTest reads the skip from the output (SKIP_REGULAR_EXPRESSION)
  return()
endif()
if(NOT _r0 EQUAL 0)
  message(FATAL_ERROR "the plain run failed (${_r0})")
endif()
set(ENV{STRATA_QSA_WAVE32} 1)
execute_process(COMMAND "${EXE}" --dump "${OUT}/wave32.bin" RESULT_VARIABLE _r1 OUTPUT_QUIET)
unset(ENV{STRATA_QSA_WAVE32})
if(NOT _r1 EQUAL 0)
  message(FATAL_ERROR "the STRATA_QSA_WAVE32=1 run failed (${_r1})")
endif()
file(SHA256 "${OUT}/plain.bin" _a)
file(SHA256 "${OUT}/wave32.bin" _b)
if(NOT _a STREQUAL _b)
  message(FATAL_ERROR "STRATA_QSA_WAVE32=1 changed the decode attention outputs (not bit-identical)")
endif()
message("PASS: STRATA_QSA_WAVE32=1 outputs are bit-identical to the default kernel's")
