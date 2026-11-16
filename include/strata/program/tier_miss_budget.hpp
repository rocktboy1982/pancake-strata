#pragma once
// MOD_PLAN.md section 6 item 20 (miss budget), STRATA_TIER_MISS_BUDGET=b with STRATA_TIER_MISS_FILE=<table>
// (opt-in, 0 < b < 1).  The table is the per-expert gate importance written by
// `python tools/pancake/gate_importance.py ... --emit-importance FILE`: how much of a layer's gate weight each expert
// received on a recorded routing trace.  Per layer the experts are sorted by importance, lowest first, and the ones whose
// cumulative importance share stays within b may stay out of tier 0: the serve tier's adapt() never admits them and
// treats a resident one as the first victim.  The rest of the layer competes by usage as before.  Placement only: a
// missed expert is still computed on the CPU with the same math, so outputs are unchanged up to GPU / CPU rounding.
// Pure logic (no engine types) so a ctest covers it; tools/pancake/gate_importance.py --emit-importance writes the file.
//
// File format (text):  line 1  "strata-gate-importance 1 <layers> <experts>"
//                      then <layers> lines of <experts> floats (>= 0, any scale; a layer's row is normalised here).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace strata::program {

class TierMissBudget {
public:
    /// STRATA_TIER_MISS_BUDGET read once: the share in (0, 1), 0 when unset or invalid (a stderr note when invalid).
    static double budget_from_env() {
        const char* v = std::getenv("STRATA_TIER_MISS_BUDGET");
        if (v == nullptr || v[0] == '\0') return 0.0;
        char* end = nullptr;
        const double b = std::strtod(v, &end);
        if (end == v || !(b > 0.0) || !(b < 1.0) || !std::isfinite(b)) {
            std::fprintf(stderr, "strata serve: STRATA_TIER_MISS_BUDGET=%s is not a share in (0, 1): off\n", v);
            return 0.0;
        }
        return b;
    }

    /// Parse the table text.  False (and the object stays off) on a bad header, a short row or a negative / non-finite value.
    bool load_text(const std::string& text) {
        clear();
        std::istringstream in(text);
        std::string magic;
        int ver = 0;
        long long nl = 0, ne = 0;
        if (!(in >> magic >> ver >> nl >> ne) || magic != "strata-gate-importance" || ver != 1 || nl <= 0 || ne <= 0) return false;
        imp_.assign((size_t) (nl * ne), 0.0f);
        for (size_t i = 0; i < imp_.size(); ++i) {
            double x;
            if (!(in >> x) || !(x >= 0.0) || !std::isfinite(x)) { imp_.clear(); return false; }
            imp_[i] = (float) x;
        }
        layers_ = nl;
        experts_ = ne;
        return true;
    }

    bool load_file(const std::string& path) {
        std::ifstream f(path);
        if (!f) return false;
        std::ostringstream ss;
        ss << f.rdbuf();
        return load_text(ss.str());
    }

    /// Mark per layer the lowest-importance experts whose cumulative share of the layer's importance is <= b.
    /// Ties break by expert id (lower first).  A layer with no importance at all marks nothing (no evidence).
    void build(double b) {
        out_.assign(imp_.size(), 0);
        n_out_ = 0;
        if (!(b > 0.0) || imp_.empty()) return;
        std::vector<int32_t> ord((size_t) experts_);
        for (int64_t l = 0; l < layers_; ++l) {
            const float* row = imp_.data() + (size_t) (l * experts_);
            double tot = 0.0;
            for (int64_t e = 0; e < experts_; ++e) tot += (double) row[e];
            if (!(tot > 0.0)) continue;
            std::iota(ord.begin(), ord.end(), 0);
            std::stable_sort(ord.begin(), ord.end(), [&](int32_t a, int32_t c) { return row[a] < row[c]; });
            double cum = 0.0;
            for (int32_t e : ord) {
                cum += (double) row[e] / tot;
                if (cum > b + 1e-6) break;
                out_[(size_t) (l * experts_ + e)] = 1;
                ++n_out_;
            }
        }
    }

    bool on() const { return n_out_ > 0; }
    int64_t layers() const { return layers_; }
    int64_t experts() const { return experts_; }
    int64_t n_stay_out() const { return n_out_; }
    bool shape_matches(int64_t layers, int64_t experts) const { return layers == layers_ && experts == experts_; }

    /// True when (layer, expert) may stay out of tier 0.  Out of range or not built: false.
    bool may_stay_out(int64_t l, int32_t e) const {
        if (l < 0 || l >= layers_ || e < 0 || e >= experts_ || out_.empty()) return false;
        return out_[(size_t) (l * experts_ + e)] != 0;
    }

    void clear() { imp_.clear(); out_.clear(); layers_ = experts_ = n_out_ = 0; }

private:
    std::vector<float> imp_;
    std::vector<uint8_t> out_;
    int64_t layers_ = 0, experts_ = 0, n_out_ = 0;
};

}  // namespace strata::program
