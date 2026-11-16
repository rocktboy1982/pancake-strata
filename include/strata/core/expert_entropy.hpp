// include/strata/core/expert_entropy.hpp - LEADS.md A29 (entropy coding of expert bytes; the lead's bound is 11.7-12.0%
// on Q2_0, zstd 8.0%, ~0% on the IQ formats).  A lossless order-0 byte coder (canonical Huffman, code lengths <= 15) for one
// expert blob: encode() -> header (256 code lengths, 4 bits each = 128 bytes, then the u64 byte count) + the bit stream;
// decode() restores the blob bit for bit.  Used by STRATA_EXPERT_ENTROPY=1 (generate.cpp: per layer, the coded size of a
// sample of the arena's blobs against their raw size, the measured form of the bound) and by the compressed-tier work
// that follows it.  Header only, no dependency.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <queue>
#include <vector>

namespace strata::core::entropy {

/// Huffman code lengths (0 = absent) for the byte histogram, limited to 15 bits (by flattening the counts until they fit).
inline std::array<uint8_t, 256> code_lengths(std::array<uint64_t, 256> cnt) {
    std::array<uint8_t, 256> len{};
    for (;;) {
        struct Node { uint64_t w; int id; };
        auto cmp = [](const Node& a, const Node& b) { return a.w > b.w || (a.w == b.w && a.id > b.id); };
        std::priority_queue<Node, std::vector<Node>, decltype(cmp)> q(cmp);
        std::vector<int> parent(512, -1);
        int next = 256, used = 0;
        for (int s = 0; s < 256; ++s) if (cnt[(size_t) s]) { q.push({cnt[(size_t) s], s}); ++used; }
        len.fill(0);
        if (used == 0) return len;
        if (used == 1) { for (int s = 0; s < 256; ++s) if (cnt[(size_t) s]) len[(size_t) s] = 1; return len; }
        while (q.size() > 1) {
            const Node a = q.top(); q.pop();
            const Node b = q.top(); q.pop();
            parent[(size_t) a.id] = next;
            parent[(size_t) b.id] = next;
            q.push({a.w + b.w, next++});
        }
        int maxl = 0;
        for (int s = 0; s < 256; ++s) {
            if (!cnt[(size_t) s]) continue;
            int l = 0;
            for (int p = s; parent[(size_t) p] >= 0; p = parent[(size_t) p]) ++l;
            len[(size_t) s] = (uint8_t) std::min(l, 255);
            maxl = std::max(maxl, l);
        }
        if (maxl <= 15) return len;
        for (auto& c : cnt) if (c) c = (c >> 1) | 1;   // flatten and retry
    }
}

/// Canonical codes from the lengths (shorter first, then by symbol).
inline std::array<uint16_t, 256> canonical(const std::array<uint8_t, 256>& len) {
    std::array<uint16_t, 256> code{};
    uint16_t c = 0;
    for (int l = 1; l <= 15; ++l) {
        for (int s = 0; s < 256; ++s)
            if (len[(size_t) s] == l) code[(size_t) s] = c++;
        c = (uint16_t) (c << 1);
    }
    return code;
}

inline std::vector<uint8_t> encode(const uint8_t* src, uint64_t n) {
    std::array<uint64_t, 256> cnt{};
    for (uint64_t i = 0; i < n; ++i) ++cnt[src[i]];
    const auto len = code_lengths(cnt);
    const auto code = canonical(len);
    std::vector<uint8_t> out(128 + 8, 0);
    for (int s = 0; s < 256; ++s) out[(size_t) (s / 2)] |= (uint8_t) ((len[(size_t) s] & 15) << (4 * (s & 1)));
    std::memcpy(out.data() + 128, &n, 8);
    uint64_t acc = 0;
    int bits = 0;
    for (uint64_t i = 0; i < n; ++i) {
        const int l = len[src[i]];
        acc = (acc << l) | code[src[i]];
        bits += l;
        while (bits >= 8) { out.push_back((uint8_t) (acc >> (bits - 8))); bits -= 8; }
        acc &= (bits ? ((1ull << bits) - 1) : 0);
    }
    if (bits) out.push_back((uint8_t) (acc << (8 - bits)));
    return out;
}

/// false on a malformed stream.
inline bool decode(const uint8_t* src, uint64_t n_src, std::vector<uint8_t>& dst) {
    if (n_src < 136) return false;
    std::array<uint8_t, 256> len{};
    for (int s = 0; s < 256; ++s) len[(size_t) s] = (uint8_t) ((src[s / 2] >> (4 * (s & 1))) & 15);
    uint64_t n = 0;
    std::memcpy(&n, src + 128, 8);
    const auto code = canonical(len);
    // decode table by (length, code): first code and first index per length
    std::vector<int> sym;
    std::array<int, 16> first{}, count{}, base{};
    for (int l = 1; l <= 15; ++l) {
        base[(size_t) l] = (int) sym.size();
        for (int s = 0; s < 256; ++s) if (len[(size_t) s] == l) sym.push_back(s);
        count[(size_t) l] = (int) sym.size() - base[(size_t) l];
        first[(size_t) l] = count[(size_t) l] ? code[(size_t) sym[(size_t) base[(size_t) l]]] : 0;
    }
    dst.assign(n, 0);
    uint64_t pos = 136 * 8;
    const uint64_t end = n_src * 8;
    for (uint64_t i = 0; i < n; ++i) {
        int c = 0;
        int l = 0;
        for (;;) {
            if (pos >= end || l >= 15) return false;
            c = (c << 1) | ((src[pos >> 3] >> (7 - (pos & 7))) & 1);
            ++pos;
            ++l;
            if (count[(size_t) l] && c >= first[(size_t) l] && c < first[(size_t) l] + count[(size_t) l]) {
                dst[i] = (uint8_t) sym[(size_t) (base[(size_t) l] + c - first[(size_t) l])];
                break;
            }
        }
    }
    return true;
}

}  // namespace strata::core::entropy
