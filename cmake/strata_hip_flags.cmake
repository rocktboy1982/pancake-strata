# Build options for extra HIP compiler flags (LEADS.md Part S S4 and Part T T8).  Both are default-off: with the defaults
# no flag is added and the build is byte-identical to before.
#   STRATA_HIP_VOPD = default | off | on
#       default: no flag (the compiler's own choice)
#       off:     -mllvm -amdgpu-enable-vopd=false   (no VOPD dual-issue pairing on gfx11)
#       on:      -mllvm -amdgpu-enable-vopd=true
#   STRATA_HIP_EXTRA_FLAGS = "" (default) or a string of hipcc/amdclang flags, passed as given (split like a shell line).
#       Every candidate flag name is UNVERIFIED for LLVM 24; the option only passes strings.  Candidates and the check
#       script: LEADS.md Part T T8, tools/pancake/box_chains/hip-flag-check.v1.sh.
#   STRATA_HIP_EXTRA_FLAGS_SCOPE = mmq (default: the strata_mmq target only) | all (every HIP source)
# strata_hip_flag_items(<vopd> <extra> <out>) sets <out> to a list of "SHELL:..." items (one per flag group) and fails
# on a bad value.  Usable in cmake -P (cmake/strata_hip_flags_test.cmake).
function(strata_hip_flag_items vopd extra out)
  string(TOLOWER "${vopd}" _v)
  set(_items "")
  if(_v STREQUAL "" OR _v STREQUAL "default")
    # nothing
  elseif(_v STREQUAL "off")
    list(APPEND _items "SHELL:-mllvm -amdgpu-enable-vopd=false")
  elseif(_v STREQUAL "on")
    list(APPEND _items "SHELL:-mllvm -amdgpu-enable-vopd=true")
  else()
    message(FATAL_ERROR "STRATA_HIP_VOPD must be default, off or on (got '${vopd}')")
  endif()
  string(STRIP "${extra}" _e)
  if(NOT _e STREQUAL "")
    if(_e MATCHES "[;$<>`()]")
      message(FATAL_ERROR "STRATA_HIP_EXTRA_FLAGS must be plain compiler flags (no ; $ < > ( ) or backtick): '${extra}'")
    endif()
    list(APPEND _items "SHELL:${_e}")
  endif()
  set(${out} "${_items}" PARENT_SCOPE)
endfunction()
