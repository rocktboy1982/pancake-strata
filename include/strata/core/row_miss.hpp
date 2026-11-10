#pragma once
// include/strata/core/row_miss.hpp - per verify window: the CPU-missed experts each row adds (MOD_PLAN A22 DISCO, A23).
//
// A verify window of T rows (row 0 = the committed row, row j = the j-th draft) pays, on a card with experts in RAM, for
// the distinct experts its rows route to that are not in VRAM: they run on the CPU.  Row j only adds the CPU experts
// that no earlier row of the window already routed to (per layer: the pool computes each distinct expert once).  This
// counts them: for every layer fed, row r's entries whose placement is -1 (CPU) and whose expert no earlier row put on
// the CPU in that layer, summed over the layers fed since reset().  It is the count tools/pancake/trace/wtrace.py
// new_cpu_miss() reads back from a window trace's layer records, and the per-position cost DISCO's stop rule learns
// (SpecFit, arXiv 2610.15417: a draft position costs the CPU time of the new missed experts it wakes).
//
// Host-only bookkeeping: it reads the routing ids and placements the verifier already produced, writes nothing the
// window reads, and is fed only when the window trace (STRATA_WINDOW_TRACE) or DISCO (STRATA_DISCO / _TRACE) is on.
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace strata::core {

class RowMissCounter {
public:
    static constexpr int kMaxRows = 64;   // rows past this (long prompt windows) are not counted
    static constexpr int kMaxLayers = 4096;   // far more layers than one window has: a counter nobody resets restarts

    /// A new window: every row back to 0.
    void reset() {
        std::fill(rows_.begin(), rows_.begin() + n_rows_, 0);
        n_rows_ = 0;
        layers_ = 0;
    }
    /// One layer of the window: `n_tok` rows of `k` entries (row-major, entry r * k + j), `kind[i]` the entry's placement
    /// (0 VRAM, 1 PCIe, 2 peer/remote, -1 CPU, -2 skipped).  `kind` null: every entry reads as CPU, as the window trace
    /// writes such a layer.  Negative ids are skipped.
    void layer(int n_tok, int k, const int32_t* ids, const int32_t* kind) {
        if (ids == nullptr || n_tok <= 0 || k <= 0) return;
        if (layers_ >= kMaxLayers) reset();   // never read (a path with no reader): start over rather than grow without end
        ++layers_;
        if (++epoch_ == 0) {   // the stamps wrapped: clear them once
            std::fill(stamp_.begin(), stamp_.end(), 0u);
            epoch_ = 1;
        }
        const int nr = std::min(n_tok, kMaxRows);
        n_rows_ = std::max(n_rows_, nr);
        for (int r = 0; r < nr; ++r) {
            for (int j = 0; j < k; ++j) {
                const int64_t i = (int64_t) r * k + j;
                if (kind != nullptr && kind[i] != -1) continue;
                const int32_t id = ids[i];
                if (id < 0) continue;
                if ((size_t) id >= stamp_.size()) stamp_.resize((size_t) id + 1, 0u);
                if (stamp_[(size_t) id] == epoch_) continue;   // an earlier row (or this one) already woke it
                stamp_[(size_t) id] = epoch_;
                ++rows_[(size_t) r];
            }
        }
    }
    int rows() const { return n_rows_; }            ///< rows seen since reset() (the widest layer fed)
    int layers() const { return layers_; }          ///< layers fed since reset()
    int at(int r) const { return r >= 0 && r < n_rows_ ? rows_[(size_t) r] : 0; }
    const int32_t* data() const { return rows_.data(); }   ///< rows() counts, row 0 first

    /// The process-wide counter DISCO reads (row_misses()): fed by the verify pool only once something enabled it.
    void enable() { on_ = true; }
    bool on() const { return on_; }

private:
    std::array<int32_t, kMaxRows> rows_{};
    int n_rows_ = 0, layers_ = 0;
    uint32_t epoch_ = 0;
    std::vector<uint32_t> stamp_;   // stamp_[expert] == epoch_: on the CPU in an earlier row of this layer
    bool on_ = false;
};

/// DISCO's counter (STRATA_DISCO / STRATA_DISCO_TRACE): the serve and CLI verify pools feed it while on(); the loop reads
/// it after the window's commit and resets it.  Off (the default): never fed, nothing changes.
inline RowMissCounter& row_misses() {
    static RowMissCounter c;
    return c;
}

}  // namespace strata::core
