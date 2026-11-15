#pragma once

// STRATA_Q36_EXPERTS_USED=K (opt-in, lossy; MOD_PLAN.md section 6 item 15, LEADS.md): route each token of the 35B
// (strata-q36) to K experts instead of the file's expert_used_count; the router renormalises the weights over the K kept.
// Ported from serboox/strata (author serboox): f8fb28ca1f08 (engine), 0f040ef5e8c8 (router / grouping kernels for 1..8).
// Author's numbers (RTX 3050 laptop, Qwen3.6 IQ4_XS): +14.4% decode at K=7/8, +18.7% at K=6/8, KLD 0.0254 at K=7.
// Unset or empty: the file's count, the same kernels and launches as before.

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace strata::core {

inline constexpr const char* kExpertsUsedEnv = "STRATA_Q36_EXPERTS_USED";

/// The routed experts per token a run uses: the GGUF's `expert_used_count` (`file_value`) when `text` is null or empty, else the
/// integer in `text`, which must lie in 1 .. file_value (a run never uses more experts than the file was trained with).
/// False with `err` for anything else, so a typo is never taken for the file's value.
inline bool resolve_experts_used(const char* text, int64_t file_value, int64_t& top_k, std::string& err) {
    top_k = file_value;
    if (text == nullptr || *text == '\0') return true;
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text, &end, 10);
    if (end == text || *end != '\0' || errno == ERANGE) {
        err = std::string(kExpertsUsedEnv) + "=" + text + " is not an integer";
        return false;
    }
    if (value < 1 || value > file_value) {
        err = std::string(kExpertsUsedEnv) + "=" + text + " is outside 1 .. " + std::to_string(file_value) +
              " (the model's expert_used_count)";
        return false;
    }
    top_k = value;
    return true;
}

}  // namespace strata::core
