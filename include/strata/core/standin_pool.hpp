// include/strata/core/standin_pool.hpp - LEADS.md "HotRod" stand-in learner (STRATA_STANDIN_LEARN / STRATA_STANDIN_APPLY, opt-in):
// the host-side estimates, no GPU.  Ported from sangharshadhyeta/Strata-HotRod include/strata/core/standin_pool.hpp (stand-in learn
// pooled estimates 4e0b5d18dd, 8 Nov; the scale's clip removed 7cf5e8d715, 9 Nov).
//
// A stand-in s serving the router's pick e at a layer gives one sample (y_s, y_e): the two experts' outputs for the same input.
// The best scale c of y_s toward y_e is the least-squares num / den (num = y_s . y_e, den = y_s . y_s); ee = y_e . y_e gives the
// residual error.  The sums decay by kDecay a sample.  They are kept per pair (layer, e, s), per (layer, e) group and per layer.  A
// pair's estimate adds a prior from its parent group: the expert group when it has >= kReady samples, else the layer group, else
// none.  The prior is the parent's mean sums times K * K / (K + n) pseudo-samples (n: the pair's own count).  An empty pair's
// estimate is therefore its parent's scale: the pooled (layer, e) scale this engine applies (one float per replaced expert).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace strata::standin {
constexpr int kReady = 32;          // samples a pair (or a group) needs before its estimate is used
constexpr double kShrinkK = 16.0;   // the prior's pseudo-samples at n = 0
constexpr double kDecay = 0.98;     // every sum decays by this per sample

struct Pair {
    double num = 0.0, den = 0.0, ee = 0.0;
    int n = 0;                       // raw samples
};

// A group's decayed sums.  w is the decayed sample weight (the mean is the sums over w); n the raw sample count.
struct Sums {
    double num = 0.0, den = 0.0, ee = 0.0, w = 0.0;
    int64_t n = 0;
};

// One sample into a pair (its sums decay as a group's do).
inline void add_sample(Pair& p, double sxy, double sxx, double syy) {
    p.num = kDecay * p.num + sxy;
    p.den = kDecay * p.den + sxx;
    p.ee = kDecay * p.ee + syy;
    ++p.n;
}
// One sample into a group.
inline void add_sample(Sums& g, double sxy, double sxx, double syy) {
    g.num = kDecay * g.num + sxy;
    g.den = kDecay * g.den + sxx;
    g.ee = kDecay * g.ee + syy;
    g.w = kDecay * g.w + 1.0;
    ++g.n;
}

struct Est {
    bool ready = false;    // usable: the parent group exists, or the pair's own count reached kReady (and ee > 0)
    bool pooled = false;   // the estimate took a parent's prior
    double c = 1.0;        // the least-squares scale num / den (>= 0: a mixture weight cannot be negative)
    double err = 0.0;      // sqrt((ee - 2 c num + c^2 den) / ee) over the shrunk sums
};

// The pair's estimate.  expert: the (layer, e) group, layer: the layer group (either may be null).
inline Est estimate(const Pair& p, const Sums* expert, const Sums* layer) {
    Est r;
    const Sums* par = nullptr;
    if (expert != nullptr && expert->n >= kReady) par = expert;
    else if (layer != nullptr && layer->n >= kReady) par = layer;
    double num = p.num, den = p.den, ee = p.ee;
    if (par != nullptr && par->w > 0.0) {
        const double prior = kShrinkK * kShrinkK / (kShrinkK + (double) p.n);
        num += prior * par->num / par->w;
        den += prior * par->den / par->w;
        ee += prior * par->ee / par->w;
        r.pooled = true;
    }
    r.ready = (r.pooled || p.n >= kReady) && ee > 0.0;
    r.c = std::max(0.0, num / std::max(den, 1e-30));
    if (ee > 0.0) {
        double e2 = (ee - 2.0 * r.c * num + r.c * r.c * den) / ee;
        if (!(e2 > 0.0)) e2 = 0.0;
        r.err = std::sqrt(e2);
    }
    return r;
}

// The sample's three dot products (double sums over n floats).
inline void sample_dots(const float* ys, const float* ye, int64_t n, double& sxy, double& sxx, double& syy) {
    sxy = sxx = syy = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        sxy += (double) ys[i] * (double) ye[i];
        sxx += (double) ys[i] * (double) ys[i];
        syy += (double) ye[i] * (double) ye[i];
    }
}

}  // namespace strata::standin
