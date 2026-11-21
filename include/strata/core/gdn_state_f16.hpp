// include/strata/core/gdn_state_f16.hpp - the recurrent (GDN) state of a conversation snapshot held as FP16
// (STRATA_SNAP_STATE_F16=1, off by default; LEADS.md S6, radiance's RADIANCE_GDN_STATE_F16 reduced to the snapshot).
//
// The live state stays FP32 on the device; only what a snapshot / checkpoint keeps is meant to be FP16, which halves its
// bytes (z.gdn, MOD_PLAN.md section 6: about 111-118 MiB each).  This header is the pure part, no GPU: the FP32 <-> FP16
// conversion (round to nearest even, subnormals kept, overflow saturated to +-65504 and counted, NaN kept), the in-place
// "round through FP16" used to take the numbers of an FP16 store without changing any size, and a self-describing blob
// (magic, count, FP16 payload) for a compact store.  Error bound for a finite x with |x| <= 65504 (kF16MaxFinite):
//     |round(x) - x| <= 2^-11 * |x|  for |x| >= 2^-14 (normal range),  <= 2^-25  below it (subnormal grid 2^-24).
// What is wired (conversation_state.cpp, conversation_file.cpp): with the switch ON a checkpoint's GDN part is stored as the
// compact blob below (ConversationCheckpoint::gdn_f16 = true; RAM copy, disk image, streamed disk blocks): about half the bytes.
// The VRAM slots and the asynchronous device staging of a checkpoint are not used then (a slot holds exact FP32 and would not
// be compact; there is no device-side FP16 conversion kernel): the save takes the synchronous path, FP32 to the host, encode.
// A restore decodes the blob (or, for a checkpoint stored as FP32, reads it through the same rounding).  The disk header carries
// a format flag so an image written with the switch off loads with it on and the other way round (see conversation_file.cpp).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace strata::core::gdn_f16 {

inline constexpr float kF16MaxFinite = 65504.0f;
inline constexpr float kF16MinNormal = 6.103515625e-05f;   // 2^-14

/// STRATA_SNAP_STATE_F16 as set (null: unset): on only for a value that starts with '1' (as STRATA_PREFILL_EQUAL reads its switch).
inline bool snap_f16_on_from(const char* v) { return v != nullptr && v[0] == '1'; }

/// The bound the round trip keeps for a finite x within the FP16 range.
inline double roundtrip_bound(double x) {
    const double a = std::fabs(x);
    return a >= (double) kF16MinNormal ? a * (1.0 / 2048.0) : 1.0 / 33554432.0;
}

inline uint16_t f32_to_f16(float f, bool* saturated = nullptr) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    const uint32_t sign = (b >> 16) & 0x8000u;
    const uint32_t exp = (b >> 23) & 0xffu;
    uint32_t man = b & 0x7fffffu;
    if (exp == 0xffu) return (uint16_t) (sign | 0x7c00u | (man ? 0x200u | (man >> 13) : 0u));   // inf / NaN (NaN stays NaN)
    const int e = (int) exp - 127 + 15;                                                     // the half's biased exponent
    if (e >= 0x1f) {   // too large for a half: saturate to the largest finite
        if (saturated) *saturated = true;
        return (uint16_t) (sign | 0x7bffu);
    }
    if (e <= 0) {   // subnormal half, or zero
        if (e < -10) return (uint16_t) sign;   // below half the smallest subnormal (ties to even gives zero too)
        man |= 0x800000u;                      // the implicit bit
        const int shift = 14 - e;              // 14 .. 24
        uint32_t h = man >> shift;
        const uint32_t rem = man & ((1u << shift) - 1u), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1u))) ++h;   // round to nearest even (a carry into the exponent is correct)
        return (uint16_t) (sign | h);
    }
    uint32_t h = ((uint32_t) e << 10) | (man >> 13);
    const uint32_t rem = man & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;   // a carry may reach exponent 31
    if ((h & 0x7fffu) >= 0x7c00u) {   // rounded up past the largest finite half: saturate
        if (saturated) *saturated = true;
        return (uint16_t) (sign | 0x7bffu);
    }
    return (uint16_t) (sign | h);
}

inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = ((uint32_t) h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t man = h & 0x3ffu;
    uint32_t b;
    if (exp == 0) {
        if (man == 0) b = sign;
        else {   // subnormal: man * 2^-24, normalised
            int e = -1;
            do { ++e; man <<= 1; } while (!(man & 0x400u));
            b = sign | ((uint32_t) (127 - 15 - e) << 23) | ((man & 0x3ffu) << 13);
        }
    } else if (exp == 0x1fu) {
        b = sign | 0x7f800000u | (man << 13);
    } else {
        b = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}

/// FP32 -> FP16 for n values; returns how many saturated (|x| beyond the FP16 range, infinities included).
inline size_t pack(const float* src, uint16_t* dst, size_t n) {
    size_t sat = 0;
    for (size_t i = 0; i < n; ++i) {
        bool s = false;
        dst[i] = f32_to_f16(src[i], &s);
        sat += s;
    }
    return sat;
}

inline void unpack(const uint16_t* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = f16_to_f32(src[i]);
}

/// The FP32 values an FP16 store would hand back, in place.  Returns the number of saturated values.
inline size_t round_inplace(float* x, size_t n) {
    size_t sat = 0;
    for (size_t i = 0; i < n; ++i) {
        bool s = false;
        x[i] = f16_to_f32(f32_to_f16(x[i], &s));
        sat += s;
    }
    return sat;
}

/// The same on raw bytes (a state part as the snapshot code holds it); `bytes` must be a multiple of 4 and `p` 4-byte
/// aligned for float access, which a vector<uint8_t> from new[] is.  Returns saturated values, or SIZE_MAX if not a multiple of 4.
inline size_t round_bytes_inplace(void* p, size_t bytes) {
    if (bytes % sizeof(float) != 0) return SIZE_MAX;
    size_t sat = 0;
    uint8_t* b = static_cast<uint8_t*>(p);
    for (size_t i = 0; i < bytes; i += 4) {
        float f;
        std::memcpy(&f, b + i, 4);
        bool s = false;
        f = f16_to_f32(f32_to_f16(f, &s));
        sat += s;
        std::memcpy(b + i, &f, 4);
    }
    return sat;
}

/// The bytes a restore copies to the device: `saved` itself when the switch is off or the size is not a multiple of 4
/// (the default path, byte-identical), else a copy in `scratch` read through the FP16 rounding.  `rounded` (optional) says which.
inline const void* restore_source(bool on, const void* saved, size_t bytes, std::vector<uint8_t>& scratch, bool* rounded = nullptr) {
    if (rounded) *rounded = false;
    if (!on || bytes == 0 || bytes % sizeof(float) != 0) return saved;
    scratch.assign(static_cast<const uint8_t*>(saved), static_cast<const uint8_t*>(saved) + bytes);
    round_bytes_inplace(scratch.data(), bytes);
    if (rounded) *rounded = true;
    return scratch.data();
}

// ---- the compact blob: 16 bytes header (magic "GF16", u32 version 1, u64 count of values), then count FP16 ------------------
inline constexpr uint32_t kBlobMagic = 0x36314647u;   // "GF16" little-endian
inline constexpr size_t kBlobHeader = 16;

/// Bytes of the blob of `n_f32` values (about half of 4 * n_f32).
inline size_t blob_bytes(size_t n_f32) { return kBlobHeader + 2 * n_f32; }

/// Encode `bytes` of FP32 (a multiple of 4) into `out`; false when the size is not a multiple of 4.  `saturated` (optional) gets the count.
inline bool encode(const void* f32_bytes, size_t bytes, std::vector<uint8_t>& out, size_t* saturated = nullptr) {
    if (bytes % sizeof(float) != 0) return false;
    const size_t n = bytes / sizeof(float);
    out.assign(blob_bytes(n), 0);
    const uint32_t ver = 1;
    const uint64_t cnt = n;
    std::memcpy(out.data(), &kBlobMagic, 4);
    std::memcpy(out.data() + 4, &ver, 4);
    std::memcpy(out.data() + 8, &cnt, 8);
    size_t sat = 0;
    const uint8_t* src = static_cast<const uint8_t*>(f32_bytes);
    for (size_t i = 0; i < n; ++i) {
        float f;
        std::memcpy(&f, src + 4 * i, 4);
        bool s = false;
        const uint16_t h = f32_to_f16(f, &s);
        sat += s;
        std::memcpy(out.data() + kBlobHeader + 2 * i, &h, 2);
    }
    if (saturated) *saturated = sat;
    return true;
}

/// Decode a blob into FP32 bytes; false on a wrong magic / version / size (nothing is written to `out` then).
inline bool decode(const uint8_t* blob, size_t blob_len, std::vector<uint8_t>& out) {
    if (blob == nullptr || blob_len < kBlobHeader) return false;
    uint32_t magic, ver;
    uint64_t cnt;
    std::memcpy(&magic, blob, 4);
    std::memcpy(&ver, blob + 4, 4);
    std::memcpy(&cnt, blob + 8, 8);
    if (magic != kBlobMagic || ver != 1 || cnt > (SIZE_MAX - kBlobHeader) / 4 || blob_len != blob_bytes((size_t) cnt)) return false;
    out.assign((size_t) cnt * 4, 0);
    for (size_t i = 0; i < (size_t) cnt; ++i) {
        uint16_t h;
        std::memcpy(&h, blob + kBlobHeader + 2 * i, 2);
        const float f = f16_to_f32(h);
        std::memcpy(out.data() + 4 * i, &f, 4);
    }
    return true;
}

/// Bytes a stored GDN part takes: `raw_f32_bytes` (a multiple of 4) as FP32, or its compact blob.
inline size_t stored_bytes(bool f16, size_t raw_f32_bytes) { return f16 ? blob_bytes(raw_f32_bytes / sizeof(float)) : raw_f32_bytes; }

/// The FP32 bytes a blob of `blob_len` bytes holds; 0 when it cannot be one (shorter than the header, odd payload).
inline size_t blob_raw_bytes(size_t blob_len) {
    return blob_len < kBlobHeader || ((blob_len - kBlobHeader) & 1u) ? 0 : (blob_len - kBlobHeader) * 2;
}

/// Bytes [off, off + n) of the blob of an FP32 array of `n_values` values, without building the whole blob.  `read_f32(byte_offset,
/// dst, bytes)` reads FP32 bytes of the array (offsets and sizes are multiples of 4).  `off` and `n` are arbitrary (a streamed block
/// is any size, not a multiple of 4, and may start inside a value).  Returns false when the range is outside the blob or a read fails.
template <class Read>
inline bool blob_slice(const Read& read_f32, size_t n_values, size_t off, size_t n, uint8_t* dst, size_t* saturated = nullptr) {
    const size_t total = blob_bytes(n_values);
    if (off > total || n > total - off) return false;
    if (n == 0) return true;
    uint8_t head[kBlobHeader] = {};
    const uint32_t ver = 1;
    const uint64_t cnt = n_values;
    std::memcpy(head, &kBlobMagic, 4);
    std::memcpy(head + 4, &ver, 4);
    std::memcpy(head + 8, &cnt, 8);
    size_t done = 0;
    if (off < kBlobHeader) {
        const size_t c = std::min(n, kBlobHeader - off);
        std::memcpy(dst, head + off, c);
        done = c;
    }
    if (done == n) return true;
    const size_t pb0 = off + done - kBlobHeader, pb1 = off + n - kBlobHeader;   // payload byte range
    const size_t v0 = pb0 / 2, v1 = (pb1 + 1) / 2;                               // values touching it
    constexpr size_t kChunk = (size_t) 1 << 18;   // 1 MiB of FP32 per read: few device copies for a 16 MiB block
    std::vector<float> f(std::min(kChunk, v1 - v0));
    std::vector<uint16_t> h(f.size());
    for (size_t v = v0; v < v1; v += f.size()) {
        const size_t k = std::min(f.size(), v1 - v);
        if (!read_f32(v * sizeof(float), f.data(), k * sizeof(float))) return false;
        bool s;
        for (size_t i = 0; i < k; ++i) {
            s = false;
            h[i] = f32_to_f16(f[i], &s);
            if (s && saturated) ++*saturated;
        }
        // the payload bytes of this chunk that fall inside [pb0, pb1)
        const size_t c0 = std::max(pb0, 2 * v), c1 = std::min(pb1, 2 * (v + k));
        const uint8_t* hb = reinterpret_cast<const uint8_t*>(h.data());
        std::memcpy(dst + (kBlobHeader + c0 - off), hb + (c0 - 2 * v), c1 - c0);
    }
    return true;
}

/// A streamed restore of a blob: blocks of any size and offset arrive in order (a disk block is not a multiple of 4, and may start
/// inside a value), are staged here (about half the FP32 bytes), and `finish` decodes the whole blob once it is complete.
class BlobStage {
public:
    explicit BlobStage(size_t blob_len) : buf_(blob_len) {}
    size_t size() const { return buf_.size(); }
    /// The next block: `off` must continue exactly where the previous one ended and stay inside the blob.
    bool put(size_t off, const void* data, size_t n) {
        if (off != next_ || n > buf_.size() - next_) return false;
        if (n) std::memcpy(buf_.data() + off, data, n);
        next_ += n;
        return true;
    }
    bool complete() const { return next_ == buf_.size(); }
    /// Decodes into `f32` (FP32 bytes); false when the blob is incomplete, invalid, or not `expect_raw` FP32 bytes.
    bool finish(std::vector<uint8_t>& f32, size_t expect_raw) const {
        return complete() && decode(buf_.data(), buf_.size(), f32) && f32.size() == expect_raw;
    }
    void release() { std::vector<uint8_t>().swap(buf_); next_ = 0; }
private:
    std::vector<uint8_t> buf_;
    size_t next_ = 0;
};

}  // namespace strata::core::gdn_f16
