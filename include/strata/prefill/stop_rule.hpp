// include/strata/prefill/stop_rule.hpp - where a prompt read polls for a Stop (no GPU needed to check).
//
// Today a Stop (Prefill::should_stop) is honoured at a chunk's start only, and a chunk reaches 32768 tokens, so a
// cancelled long prompt keeps the GPU for the rest of that chunk (LEADS.md Part H, adambenhassen `ef156accf1`: next
// request after a Stop 17.9 s -> 1.1 s on 2 x P100).  STRATA_STOP_LAYERS=1 also polls between the layers of a chunk.
// Default off: the poll sites and the bytes of a finished read are the same as before.  A chunk stopped half way leaves
// the recurrent (GDN) state and the K/V past the last whole chunk, which the caller already treats as "a batched part is
// in flight" (generate.cpp pp_exact = false), so the session is dropped as for any cancel inside a batched read.
#pragma once

#include <cstdint>
#include <cstdlib>

namespace strata::prefill::stop_rule {

/// STRATA_STOP_LAYERS value: "1" (or any digit but 0) arms the poll between layers.
inline bool parse_enabled(const char* v) { return v != nullptr && v[0] >= '1' && v[0] <= '9'; }

inline bool enabled_from_env() {
    static const bool on = parse_enabled(std::getenv("STRATA_STOP_LAYERS"));
    return on;
}

/// True when layer `l` of a chunk whose layers start at `lb` polls the Stop hook before it runs.  The first layer never
/// polls: the chunk-start poll just ran.  Without a hook there is nothing to ask.
constexpr bool poll_before_layer(bool enabled, bool have_hook, int64_t l, int64_t lb) {
    return enabled && have_hook && l > lb;
}

}  // namespace strata::prefill::stop_rule
