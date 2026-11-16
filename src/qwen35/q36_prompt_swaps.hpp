// src/qwen35/q36_prompt_swaps.hpp - STRATA_Q36_PROMPT_SWAPS (opt-in; LEADS.md gputier/StrataGP 5f7c616102 / 819b1dba79, "swap cap"):
// the slot replacements of the hybrid rebalance after a prompt.  A pure rule with no GPU (tested by q36_prompt_swaps_test.cpp).
// Ported from gputier's q36_rebalance.hpp prompt_swaps_from_env; our default stays 2048 (gputier moved theirs to 4096 on
// 26 Oct 2026 after a measurement on Ornith-1.5-122B): unset is byte-identical to the tree before.
#pragma once

#include <climits>
#include <cstdlib>
#include <string>

namespace strata::q36 {

inline constexpr int kPromptSwapsDefault = 2048;        // STRATA_Q36_PROMPT_SWAPS unset (this tree's value before the switch)
inline constexpr int kPromptSwapsUnlimited = INT_MAX;   // the variable's 0: no cap

/// Decimal digits only (no sign, no space, nothing after them) for a value from 1 to INT_MAX; 0 gives no cap.  Unset or empty gives
/// the default.  Any other value gives the default as well, and `err` (when given) gets the line the engine prints for it; an
/// accepted value leaves `err` alone.
inline int prompt_swaps_from_env(const char* v, std::string* err) {
    if (v == nullptr || v[0] == '\0') return kPromptSwapsDefault;
    char* end = nullptr;
    const bool digit = v[0] >= '0' && v[0] <= '9';               // strtoll alone would also take a leading space or sign
    const long long n = digit ? std::strtoll(v, &end, 10) : 0;   // a value out of the long long range comes back above INT_MAX
    if (digit && *end == '\0' && n <= INT_MAX) return n == 0 ? kPromptSwapsUnlimited : (int) n;
    if (err != nullptr)
        *err = std::string("STRATA_Q36_PROMPT_SWAPS=") + v + " is not a whole number from 0 to " + std::to_string(INT_MAX) +
               " in decimal digits (0: no cap): using " + std::to_string(kPromptSwapsDefault);
    return kPromptSwapsDefault;
}

}  // namespace strata::q36
