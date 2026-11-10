// src/prefill/moe_ws_stub.cpp - ws:: (moe_ws.hpp) in a build without the weight-stationary experts (they are in the CUDA
// STRATA_EXPERIMENTAL_SM60 build only).  Moved out of prefill.cpp (MOD_PLAN 95b), bodies unchanged; empty when
// STRATA_PREFILL_WS is set.  Checked by tools/pancake/check_stubs.py.
#include "strata/prefill/moe_ws.hpp"

#include <cstddef>
#include <cstdint>

#ifndef STRATA_PREFILL_WS
// the weight-stationary experts (moe_ws.hpp) are in the CUDA STRATA_EXPERIMENTAL_SM60 build only
namespace strata::prefill::ws {
int mode() { return 0; }
bool supported(int, int) { return false; }
bool faster(int, double) { return false; }
size_t chunks(const int32_t*, size_t, const int32_t*, const int32_t*, Chunk*) { return 0; }
bool gate_up(const Layer&, void*) { return false; }
void down(const Layer&, void*) {}
}  // namespace strata::prefill::ws
#endif
