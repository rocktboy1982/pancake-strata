// include/strata/core/expert_entropy_tier.hpp - LEADS.md A29, the compressed tier (STRATA_EXPERT_ENTROPY_TIER=1, opt-in).
//
// The first step (expert_entropy.hpp, STRATA_EXPERT_ENTROPY=N) measured how small the lossless order-0 coder makes the
// expert blobs.  This header holds what the tier itself needs, all of it CPU code that the tests run without a GPU:
//
//   * a SEGMENTED stream: one canonical-Huffman code table per blob (the same code_lengths / canonical as the first
//     step) and the blob cut into independent segments (64 KiB by default), each its own byte-aligned bit stream, so
//     several threads decode one blob at once;
//   * a table decoder (15-bit codes: one 32768-entry lookup per symbol, not the bit-by-bit loop of decode());
//   * SegDecodePool: helper threads that decode one blob's segments together with the caller;
//   * pick_slot: which staging slot a decoded blob may go to (the rule ArenaExpertSource's ring follows).
//
// Exact: decode(encode(x)) == x bit for bit, or the decoder says false (a malformed stream is never half-used).
// Credits: the lead is LEADS.md A29 (entropy coding of expert weights); the coder is the textbook canonical Huffman.
#pragma once

#include "strata/core/expert_entropy.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace strata::core::entropy {

constexpr uint32_t kSegMagic = 0x39324541u;   // "AE29", little endian
constexpr uint32_t kSegDefault = 64u * 1024u;
constexpr uint64_t kSegHead = 4 + 128 + 8 + 4 + 4;   // magic, code lengths, n, segment bytes, segment count

/// A parsed segmented stream (pointers into the caller's bytes).
struct SegHeader {
    std::array<uint8_t, 256> len{};
    uint64_t n = 0;              ///< decoded bytes
    uint32_t seg_bytes = 0;      ///< decoded bytes per segment (the last one may be shorter)
    uint32_t nseg = 0;
    const uint8_t* offs = nullptr;   ///< nseg u32: where segment k starts in `data`
    const uint8_t* data = nullptr;
    uint64_t data_bytes = 0;
    uint64_t seg_begin(uint32_t k) const { uint32_t o; std::memcpy(&o, offs + 4 * (size_t) k, 4); return o; }
    uint64_t seg_end(uint32_t k) const { return k + 1 < nseg ? seg_begin(k + 1) : data_bytes; }
    uint64_t out_begin(uint32_t k) const { return (uint64_t) k * seg_bytes; }
    uint64_t out_len(uint32_t k) const { return std::min<uint64_t>(seg_bytes, n - out_begin(k)); }
};

/// `src` (n bytes) -> header + segments.  `seg_bytes` >= 1.
inline std::vector<uint8_t> encode_segmented(const uint8_t* src, uint64_t n, uint32_t seg_bytes = kSegDefault) {
    if (seg_bytes == 0) seg_bytes = kSegDefault;
    std::array<uint64_t, 256> cnt{};
    for (uint64_t i = 0; i < n; ++i) ++cnt[src[i]];
    const auto len = code_lengths(cnt);
    const auto code = canonical(len);
    const uint32_t nseg = n == 0 ? 0u : (uint32_t) ((n + seg_bytes - 1) / seg_bytes);
    std::vector<uint8_t> out((size_t) (kSegHead + 4ull * nseg), 0);
    std::memcpy(out.data(), &kSegMagic, 4);
    for (int s = 0; s < 256; ++s) out[(size_t) (4 + s / 2)] |= (uint8_t) ((len[(size_t) s] & 15) << (4 * (s & 1)));
    std::memcpy(out.data() + 132, &n, 8);
    std::memcpy(out.data() + 140, &seg_bytes, 4);
    std::memcpy(out.data() + 144, &nseg, 4);
    const size_t data0 = out.size();
    std::vector<uint32_t> offs(nseg, 0);
    for (uint32_t k = 0; k < nseg; ++k) {
        offs[k] = (uint32_t) (out.size() - data0);
        const uint64_t b = (uint64_t) k * seg_bytes, e = std::min<uint64_t>(n, b + seg_bytes);
        uint64_t acc = 0;
        int bits = 0;
        for (uint64_t i = b; i < e; ++i) {
            const int l = len[src[i]];
            acc = (acc << l) | code[src[i]];
            bits += l;
            while (bits >= 8) { out.push_back((uint8_t) (acc >> (bits - 8))); bits -= 8; }
            acc &= (bits ? ((1ull << bits) - 1) : 0);
        }
        if (bits) out.push_back((uint8_t) (acc << (8 - bits)));
    }
    if (nseg) std::memcpy(out.data() + kSegHead, offs.data(), 4ull * nseg);
    return out;
}

/// false: not a segmented stream, or an inconsistent one.
inline bool parse_seg(const uint8_t* src, uint64_t n_src, SegHeader& h) {
    if (src == nullptr || n_src < kSegHead) return false;
    uint32_t magic = 0;
    std::memcpy(&magic, src, 4);
    if (magic != kSegMagic) return false;
    for (int s = 0; s < 256; ++s) h.len[(size_t) s] = (uint8_t) ((src[4 + s / 2] >> (4 * (s & 1))) & 15);
    std::memcpy(&h.n, src + 132, 8);
    std::memcpy(&h.seg_bytes, src + 140, 4);
    std::memcpy(&h.nseg, src + 144, 4);
    if (h.seg_bytes == 0) return false;
    const uint64_t want = h.n == 0 ? 0 : (h.n + h.seg_bytes - 1) / h.seg_bytes;
    if (want != h.nseg) return false;
    const uint64_t head = kSegHead + 4ull * h.nseg;
    if (n_src < head) return false;
    h.offs = src + kSegHead;
    h.data = src + head;
    h.data_bytes = n_src - head;
    uint64_t prev = 0;
    for (uint32_t k = 0; k < h.nseg; ++k) {
        const uint64_t o = h.seg_begin(k);
        if (o < prev || o > h.data_bytes) return false;
        prev = o;
    }
    return true;
}

/// The 32768-entry lookup: entry (top 15 bits of the stream) = (code length << 8) | symbol, 0 = no code.  false when the
/// lengths are not a prefix code (a malformed header).
inline bool build_table(const std::array<uint8_t, 256>& len, uint16_t* tab) {
    std::fill(tab, tab + 32768, (uint16_t) 0);
    const auto code = canonical(len);
    for (int s = 0; s < 256; ++s) {
        const int l = len[(size_t) s];
        if (l == 0) continue;
        const uint32_t lo = (uint32_t) code[(size_t) s] << (15 - l);
        const uint32_t hi = ((uint32_t) code[(size_t) s] + 1u) << (15 - l);
        if (hi > 32768u || lo >= hi) return false;
        for (uint32_t x = lo; x < hi; ++x) {
            if (tab[x] != 0) return false;   // two codes overlap: not a prefix code
            tab[x] = (uint16_t) ((l << 8) | s);
        }
    }
    return true;
}

/// One segment's bit stream [p, e) -> m bytes at dst.  false when a code is missing or the stream runs short.
inline bool decode_segment(const uint16_t* tab, const uint8_t* p, const uint8_t* e, uint8_t* dst, uint64_t m) {
    uint64_t buf = 0;   // the next bits, most significant first
    int cnt = 0;        // valid bits in buf
    uint64_t loaded = 0;
    const uint8_t* q = p;
    for (uint64_t i = 0; i < m; ++i) {
        while (cnt <= 56) {
            const uint64_t b = q < e ? (uint64_t) *q++ : 0ull;
            buf |= b << (56 - cnt);
            cnt += 8;
            loaded += 8;
        }
        const uint16_t t = tab[buf >> 49];
        const int l = t >> 8;
        if (l == 0) return false;
        dst[i] = (uint8_t) (t & 0xff);
        buf <<= l;
        cnt -= l;
    }
    return loaded - (uint64_t) cnt <= (uint64_t) (e - p) * 8ull;   // no bit past the segment's end was used
}

/// Segments [k0, k1) of `h` into dst (the whole blob's buffer).
inline bool decode_segments(const SegHeader& h, const uint16_t* tab, uint32_t k0, uint32_t k1, uint8_t* dst) {
    for (uint32_t k = k0; k < k1 && k < h.nseg; ++k) {
        const uint64_t b = h.seg_begin(k), e = h.seg_end(k);
        if (e < b) return false;
        if (!decode_segment(tab, h.data + b, h.data + e, dst + h.out_begin(k), h.out_len(k))) return false;
    }
    return true;
}

inline uint16_t* thread_table() {
    thread_local std::vector<uint16_t> tab(32768);
    return tab.data();
}

/// The whole stream on the calling thread.  `n_dst` must be the stream's decoded size.
inline bool decode_segmented(const uint8_t* src, uint64_t n_src, uint8_t* dst, uint64_t n_dst) {
    SegHeader h;
    if (!parse_seg(src, n_src, h) || h.n != n_dst) return false;
    if (h.nseg == 0) return true;
    uint16_t* tab = thread_table();
    if (!build_table(h.len, tab)) return false;
    return decode_segments(h, tab, 0, h.nseg, dst);
}

/// Decoded size of a segmented stream, 0 when it is not one.
inline uint64_t seg_decoded_bytes(const uint8_t* src, uint64_t n_src) {
    SegHeader h;
    return parse_seg(src, n_src, h) ? h.n : 0;
}

/// Helper threads that decode one blob's segments with the caller (the tier's on-demand decode: a CPU miss waits for
/// its blob, so the decode is split across `helpers` + 1 threads).  One decode at a time; decode() is thread-safe.
class SegDecodePool {
public:
    explicit SegDecodePool(int helpers) {
        for (int i = 0; i < helpers; ++i) threads_.emplace_back([this] { loop(); });
    }
    ~SegDecodePool() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : threads_) t.join();
    }
    SegDecodePool(const SegDecodePool&) = delete;
    SegDecodePool& operator=(const SegDecodePool&) = delete;
    int helpers() const { return (int) threads_.size(); }

    bool decode(const uint8_t* src, uint64_t n_src, uint8_t* dst, uint64_t n_dst) {
        std::lock_guard<std::mutex> call(call_mu_);
        SegHeader h;
        if (!parse_seg(src, n_src, h) || h.n != n_dst) return false;
        if (h.nseg <= 1 || threads_.empty()) return decode_segmented(src, n_src, dst, n_dst);
        {
            std::lock_guard<std::mutex> lk(mu_);
            job_h_ = &h;
            job_dst_ = dst;
            next_.store(0, std::memory_order_relaxed);
            ok_.store(true, std::memory_order_relaxed);
            pending_ = (int) threads_.size();
            ++gen_;
        }
        cv_.notify_all();
        work(h, dst);
        std::unique_lock<std::mutex> lk(mu_);
        done_cv_.wait(lk, [this] { return pending_ == 0; });
        job_h_ = nullptr;
        job_dst_ = nullptr;
        return ok_.load(std::memory_order_relaxed);
    }

private:
    void work(const SegHeader& h, uint8_t* dst) {
        if (next_.load(std::memory_order_relaxed) >= h.nseg) return;   // nothing left: skip the table
        uint16_t* tab = thread_table();
        if (!build_table(h.len, tab)) { ok_.store(false, std::memory_order_relaxed); return; }
        for (;;) {
            const uint32_t k = next_.fetch_add(1, std::memory_order_relaxed);
            if (k >= h.nseg) break;
            if (!decode_segments(h, tab, k, k + 1, dst)) ok_.store(false, std::memory_order_relaxed);
        }
    }
    void loop() {
        uint64_t seen = 0;
        for (;;) {
            const SegHeader* h = nullptr;
            uint8_t* dst = nullptr;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return stop_ || gen_ != seen; });
                if (stop_) return;
                seen = gen_;
                h = job_h_;
                dst = job_dst_;
            }
            if (h != nullptr) work(*h, dst);
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (--pending_ == 0) done_cv_.notify_all();
            }
        }
    }

    std::vector<std::thread> threads_;
    std::mutex call_mu_;
    std::mutex mu_;
    std::condition_variable cv_, done_cv_;
    bool stop_ = false;
    uint64_t gen_ = 0;
    int pending_ = 0;
    const SegHeader* job_h_ = nullptr;
    uint8_t* job_dst_ = nullptr;
    std::atomic<uint32_t> next_{0};
    std::atomic<bool> ok_{true};
};

/// One staging slot as the ring sees it: which blob it holds (-1: none), the layer sequence number of its last hand-out,
/// and whether an asynchronous copy may still read it.
struct SlotState {
    int64_t idx = -1;
    uint64_t seq = 0;
    bool busy = false;
};

/// The slot for blob `want` at layer sequence `now`: the slot that already holds it (`*hit` = 1: same bytes, no decode),
/// else the free slot, or the slot handed out longest ago that is at least `keep` layers old and not read by a copy
/// (`*hit` = 0).  -1: every slot may still be read - the caller must not overwrite any of them.
inline int pick_slot(const SlotState* s, int n, int64_t want, uint64_t now, uint64_t keep, int* hit) {
    *hit = 0;
    for (int i = 0; i < n; ++i)
        if (s[i].idx == want && want >= 0) { *hit = 1; return i; }
    int best = -1;
    for (int i = 0; i < n; ++i) {
        if (s[i].idx < 0) return i;
        if (s[i].busy || now < s[i].seq || now - s[i].seq < keep) continue;
        if (best < 0 || s[i].seq < s[best].seq) best = i;
    }
    return best;
}

}  // namespace strata::core::entropy
