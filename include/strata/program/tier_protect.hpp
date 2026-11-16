#pragma once
// MOD_PLAN.md section 6 items 19 and 26, STRATA_ADAPT_PROTECT=f (opt-in, 0 < f <= 1): the serve tier's adapt() never
// evicts a resident expert whose cache slot index is below ceil(f * slots).  The startup fill hands the slots out in
// profile rank order and a swap reuses the victim's slot, so the same slots (the profile's top f share) stay a static
// core while the rest trades by usage; a victim with count 0 is a free slot in effect.  Placement only: outputs unchanged.
// Pure decision logic here (no engine types) so a ctest covers it; tools/pancake/tier_laws.py --protect is the offline twin.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::program {

/// Protected slot count for fraction f of `slots`: ceil(f * slots), clamped to [0, slots]; 0 for f <= 0 or not finite.
inline int64_t tier_protect_slots(double f, int64_t slots) {
    if (!(f > 0.0) || !std::isfinite(f) || slots <= 0) return 0;
    if (f >= 1.0) return slots;
    const int64_t n = (int64_t) std::ceil(f * (double) slots - 1e-9);
    return n < 0 ? 0 : (n > slots ? slots : n);
}

/// A resident expert (host_res value `slot` >= 0) is protected when its slot index is below `n_protected`.
inline bool tier_protect_holds(int32_t slot, int64_t n_protected) { return slot >= 0 && (int64_t) slot < n_protected; }

/// STRATA_ADAPT_PROTECT read once: the fraction, 0 when unset or invalid (a stderr note for an invalid value).
inline double tier_protect_from_env() {
    const char* v = std::getenv("STRATA_ADAPT_PROTECT");
    if (v == nullptr || v[0] == '\0') return 0.0;
    char* end = nullptr;
    const double f = std::strtod(v, &end);
    if (end == v || !(f > 0.0) || f > 1.0 || !std::isfinite(f)) {
        std::fprintf(stderr, "strata serve: STRATA_ADAPT_PROTECT=%s is not a fraction in (0, 1]: off\n", v);
        return 0.0;
    }
    return f;
}

}  // namespace strata::program
