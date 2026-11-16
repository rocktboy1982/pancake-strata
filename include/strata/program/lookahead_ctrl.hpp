#pragma once
// MOD_PLAN.md section 6 (the BigMoeOnEdge switches, LEADS.md Part I) (LEADS.md Part I row I2), STRATA_LOOKAHEAD_CTRL=1 (measurement only, extends
// STRATA_LOOKAHEAD_STATS): the predictor CONTROL and the per-layer table.  BigMoeOnEdge scores its stale-gate prediction
// (layer l's input through layer l+1's router) as slot overlap with the true routing of layer l+1 and runs a control
// (the same router on the TRUE input of its own layer, which must read 100.0%; below that the probe is wrong).  Our
// STRATA_LOOKAHEAD_STATS scores against the experts the next layer routes OUTSIDE the GPU cache only (19-29% recall,
// MOD_PLAN.md item 29) and never ran a control, so the gap to their 80-89% is unexplained.
//
// The counting rule is here, pure (no engine types), so a ctest covers it:
//   slot overlap of a token at a layer = (# of the token's true routed experts found in the predicted set) /
//   (# valid true routed experts); a prediction never contributes more than the true slots (a predicted set larger than
//   k raises the overlap but is reported with its size, so compare equal sizes only).
//   The tally keeps, per layer and per predictor: hits and slots.  Predictors:
//     stale_gpu  what the engine acts on (the GPU table or the CPU thread's own router GEMV, whichever fed the thread)
//     stale_cpu  layer l+1's BF16 router on layer l's input, computed on the host (the FATE definition); with the GPU
//                table on this is computed in addition, so stale_gpu vs stale_cpu checks the GPU side
//     control    layer l's router on layer l's own true input: 100% expected (not with ROUTE_PRIOR / BUDGET / the
//                residency-biased routing families on: those change the committed top-k on purpose)
//     prev       the previous token of the window as predictor of the next (tokens 1..n-1 of a window only; the
//                I1 base rate)
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace strata::program {

enum class CtrlKind { StaleGpu = 0, StaleCpu = 1, Control = 2, Prev = 3 };
constexpr int kCtrlKinds = 4;

/// Number of valid experts (0 <= e < n_expert) in `truth[0..nt)` that also appear in `pred[0..np)`; `*valid` gets the
/// number of valid truth entries.  Truth entries repeat-safe: each truth slot counts once.
inline int ctrl_overlap(const int32_t* pred, int np, const int32_t* truth, int nt, int n_expert, int* valid) {
    int hit = 0, v = 0;
    for (int i = 0; i < nt; ++i) {
        const int32_t e = truth[i];
        if (e < 0 || e >= n_expert) continue;
        ++v;
        for (int j = 0; j < np; ++j)
            if (pred[j] == e) { ++hit; break; }
    }
    if (valid != nullptr) *valid = v;
    return hit;
}

/// Top-k expert ids of `logits[0..n)`, best first; ties go to the LOWER id (a stable rule, so the control cannot read
/// under 100% because of an arbitrary tie break of its own).  `k` is clamped to n.  Returns the count written.
inline int ctrl_topk(const float* logits, int n, int k, int32_t* out) {
    if (k > n) k = n;
    int cnt = 0;
    for (int e = 0; e < n; ++e) {
        // insert e into the sorted list of the best k so far
        int pos = cnt;
        while (pos > 0 && logits[out[pos - 1]] < logits[e]) --pos;   // strictly better only: an equal earlier id stays ahead
        if (pos >= k) continue;
        if (cnt < k) ++cnt;
        for (int j = cnt - 1; j > pos; --j) out[j] = out[j - 1];
        out[pos] = e;
    }
    return cnt;
}

struct LookaheadCtrlTally {
    struct Cell { uint64_t hit = 0, slots = 0; };
    int n_layers = 0;
    uint64_t windows = 0;
    std::vector<Cell> cells;   // [layer][kind]

    void init(int layers) {
        n_layers = layers < 0 ? 0 : layers;
        windows = 0;
        cells.assign((size_t) n_layers * kCtrlKinds, Cell{});
    }
    Cell& at(int layer, CtrlKind k) { return cells[(size_t) layer * kCtrlKinds + (size_t) k]; }
    const Cell& at(int layer, CtrlKind k) const { return cells[(size_t) layer * kCtrlKinds + (size_t) k]; }

    /// One token at one layer: score `pred[0..np)` against the true routing `truth[0..nt)`.
    void add(int layer, CtrlKind kind, const int32_t* pred, int np, const int32_t* truth, int nt, int n_expert) {
        if (layer < 0 || layer >= n_layers) return;
        int valid = 0;
        const int hit = ctrl_overlap(pred, np, truth, nt, n_expert, &valid);
        Cell& c = at(layer, kind);
        c.hit += (uint64_t) hit;
        c.slots += (uint64_t) valid;
    }

    /// Overlap of one predictor over `[first, last)` layers, -1 when it scored no slot.
    double rate(CtrlKind kind, int first, int last) const {
        uint64_t h = 0, s = 0;
        for (int l = first < 0 ? 0 : first; l < last && l < n_layers; ++l) { h += at(l, kind).hit; s += at(l, kind).slots; }
        return s ? (double) h / (double) s : -1.0;
    }
    double rate(CtrlKind kind, int layer) const { return rate(kind, layer, layer + 1); }

    static std::string fmt(double r) {
        if (r < 0.0) return "n/a";
        char b[32];
        std::snprintf(b, sizeof b, "%.4f", r);
        return b;
    }

    /// `strata lookahead ctrl: windows=100 slots=8000 stale_gpu=0.4120 stale_cpu=0.4050 control=1.0000 prev=0.3100`
    /// (the layers 1..n-1 that have a stale prediction; slots = the stale_gpu denominator).
    std::string summary_line() const {
        uint64_t sl = 0;
        for (int l = 0; l < n_layers; ++l) sl += at(l, CtrlKind::StaleGpu).slots;
        char b[160];
        std::snprintf(b, sizeof b, "strata lookahead ctrl: windows=%llu slots=%llu", (unsigned long long) windows, (unsigned long long) sl);
        return std::string(b) + " stale_gpu=" + fmt(rate(CtrlKind::StaleGpu, 0, n_layers)) +
               " stale_cpu=" + fmt(rate(CtrlKind::StaleCpu, 0, n_layers)) +
               " control=" + fmt(rate(CtrlKind::Control, 0, n_layers)) + " prev=" + fmt(rate(CtrlKind::Prev, 0, n_layers));
    }

    /// `strata lookahead ctrl layer 7: slots=160 stale_gpu=... stale_cpu=... control=... prev=...`
    std::string layer_line(int layer) const {
        char b[96];
        std::snprintf(b, sizeof b, "strata lookahead ctrl layer %d: slots=%llu", layer,
                      (unsigned long long) (layer >= 0 && layer < n_layers ? at(layer, CtrlKind::StaleGpu).slots : 0));
        return std::string(b) + " stale_gpu=" + fmt(rate(CtrlKind::StaleGpu, layer)) +
               " stale_cpu=" + fmt(rate(CtrlKind::StaleCpu, layer)) + " control=" + fmt(rate(CtrlKind::Control, layer)) +
               " prev=" + fmt(rate(CtrlKind::Prev, layer));
    }
};

}  // namespace strata::program
