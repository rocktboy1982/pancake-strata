#pragma once
// MOD_PLAN item 64: the shared window trace ("one trace run, ten answers").  STRATA_WINDOW_TRACE=<file> (opt-in) records,
// per verify window, every layer's routing (ids, weights as the doorbell published them, where each entry was served:
// VRAM / PCIe / CPU / skipped) and the window's outcome (rows, drafts accepted, position, wall time).  One format for
// the cache simulator, the seed learner, the DraftPolicy cost fitter, DISCO/SpecFit, AcceptMoE and tail-row replays
// (reader: tools/pancake/trace/wtrace.py).
//   STRATA_WINDOW_TRACE_EVERY=N   keep every N-th window (default 1); the others are dropped before they are written
//   STRATA_WINDOW_TRACE_MAX_MB=M  stop writing at M MiB (default 2048; D is nearly full)
// Layer records are buffered and written with their window's record at commit, so a window is never half in the file.
//
// File: "STWT" u32 version=1, u32 n_layers, u32 n_expert, u32 k.  Then records:
//   layer : u8 1, u8 layer, u8 n_tok, u8 flags (bit0 weights present), then n_tok*k x {i16 id, i8 kind}
//           (kind: 0 VRAM, 1 PCIe, 2 peer/remote, -1 CPU, -2 skipped tail), then n_tok*k f32 weights if bit0
//   window: u8 2, u8 rows (T), u8 accepted (255: a merged prompt window, no drafts), u8 0, i64 position, f32 ms since the previous window's commit;
//           a window record comes first, then its layer records
//   drafts: u8 4, u8 n, u16 0, n x f32 prob, n x u8 src.  Written when the window is built (before its layers), so it belongs to the
//           window record that follows; n = rows - 1, entry j-1 describes row j (the draft token at window[j]).  prob is the
//           probability the drafter gave that token (NaN for a lookup / chained row: no confidence); src: 0 MTP, 1 DFlash2,
//           2 suffix/n-gram lookup window, 3 lookup chained after the drafter's rows.  Whether row j was accepted is
//           j <= the window's `accepted`; the CPU-missed experts a row added are in its layer records (kind -1; wtrace.py new_cpu_miss).
//           Absent for merged prompt windows.  (Legacy record 3, u8 3, u8 n, u16 0, n x f32 probs, is still read; no longer written.)
// Only the serve verify windows (drive_pool_multi_impl + the commit in the serve loop) write it.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace strata::core {

class WindowTrace {
public:
    bool on() const { return f_ != nullptr; }

    /// Called once the model's shape is known.  Opens the file if STRATA_WINDOW_TRACE is set.
    void open(int n_layers, int n_expert, int k) {
        if (opened_) return;
        opened_ = true;
        const char* p = std::getenv("STRATA_WINDOW_TRACE");
        if (p == nullptr || p[0] == '\0') return;
        if (const char* e = std::getenv("STRATA_WINDOW_TRACE_EVERY")) every_ = std::max(1, std::atoi(e));
        if (const char* m = std::getenv("STRATA_WINDOW_TRACE_MAX_MB")) cap_ = (int64_t) std::max(1, std::atoi(m)) << 20;
        f_ = std::fopen(p, "wb");
        if (f_ == nullptr) { std::fprintf(stderr, "strata: STRATA_WINDOW_TRACE: cannot write %s\n", p); return; }
        const uint32_t hdr[4] = {1u, (uint32_t) n_layers, (uint32_t) n_expert, (uint32_t) k};
        std::fwrite("STWT", 1, 4, f_);
        std::fwrite(hdr, 4, 4, f_);
        bytes_ = 20;
        k_ = k;
        last_ = std::chrono::steady_clock::now();
        std::fprintf(stderr, "strata: STRATA_WINDOW_TRACE: verify windows -> %s (every %d, cap %lld MiB)\n", p, every_,
                     (long long) (cap_ >> 20));
    }

    /// One layer of the current window.  `kind` may be null (then every entry reads as CPU), `w` may be null.
    void layer(int layer, int n_tok, const int32_t* ids, const int32_t* kind, const float* w) {
        if (f_ == nullptr || n_tok <= 0 || n_tok > 255 || layer < 0 || layer > 255) return;
        const int n = n_tok * k_;
        const uint8_t h[4] = {1, (uint8_t) layer, (uint8_t) n_tok, (uint8_t) (w != nullptr ? 1 : 0)};
        put(h, 4);
        for (int i = 0; i < n; ++i) {
            const int16_t id = (int16_t) ids[i];
            const int8_t kd = kind != nullptr ? (int8_t) kind[i] : (int8_t) -1;
            put(&id, 2);
            put(&kd, 1);
        }
        if (w != nullptr) put(w, (size_t) n * 4);
    }

    /// The window committed: `rows` rows were verified, `accepted` drafts kept, at sequence position `pos`.
    void commit(int rows, int accepted, int64_t pos) {
        if (f_ == nullptr) return;
        const auto now = std::chrono::steady_clock::now();
        const float ms = (float) std::chrono::duration<double, std::milli>(now - last_).count();
        last_ = now;
        const bool keep = (windows_++ % every_) == 0;
        if (keep && !buf_.empty() && bytes_ + (int64_t) buf_.size() + 16 <= cap_) {
            // the window record first, then its layers (straight to the file: buf_ holds the layers)
            const uint8_t h[4] = {2, (uint8_t) std::min(rows, 255), (uint8_t) (accepted < 0 ? 255 : std::min(accepted, 254)), 0};
            std::fwrite(h, 1, 4, f_);
            std::fwrite(&pos, 8, 1, f_);
            std::fwrite(&ms, 4, 1, f_);
            std::fwrite(buf_.data(), 1, buf_.size(), f_);
            bytes_ += 16 + (int64_t) buf_.size();
            if ((windows_ & 63) == 0) std::fflush(f_);
        } else if (keep && bytes_ + (int64_t) buf_.size() + 16 > cap_ && !capped_) {
            capped_ = true;
            std::fprintf(stderr, "strata: STRATA_WINDOW_TRACE reached its cap (%lld MiB): no more windows written\n",
                         (long long) (cap_ >> 20));
        }
        buf_.clear();
    }

    /// The window's draft confidences (items 5, F8 STRATA_SPEC_CUM, A22 DISCO): call when the window is built, before it runs.
    /// `prob[j-1]` and `src[j-1]` describe draft row j = 1..n (see the file format).  Record 4: u8 4, u8 n, u16 0, n f32, n u8.
    void drafts(const float* prob, const uint8_t* src, int n) {
        if (f_ == nullptr || prob == nullptr || src == nullptr || n <= 0) return;
        n = std::min(n, 255);
        const uint8_t h[4] = {4, (uint8_t) n, 0, 0};
        put(h, 4);
        put(prob, (size_t) n * 4);
        put(src, (size_t) n);
    }

    /// A window that ended without a commit (an error, a cancelled request): its layers are dropped.
    void drop() { buf_.clear(); }

    ~WindowTrace() {
        if (f_ != nullptr) std::fclose(f_);
    }

private:
    void put(const void* p, size_t n) {
        const uint8_t* b = (const uint8_t*) p;
        buf_.insert(buf_.end(), b, b + n);
    }
    std::FILE* f_ = nullptr;
    bool opened_ = false, capped_ = false;
    int every_ = 1, k_ = 10;
    int64_t cap_ = (int64_t) 2048 << 20, bytes_ = 0, windows_ = 0;
    std::vector<uint8_t> buf_;
    std::chrono::steady_clock::time_point last_{};
};

inline WindowTrace& window_trace() {
    static WindowTrace t;
    return t;
}

}  // namespace strata::core
