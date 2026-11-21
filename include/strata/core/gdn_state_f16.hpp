// include/strata/core/gdn_state_f16.hpp - the recurrent (GDN) state of a conversation snapshot held as FP16
// (STRATA_SNAP_STATE_F16=1, off by default; LEADS.md S6, radiance's RADIANCE_GDN_STATE_F16 reduced to the snapshot).
//
// The live state stays FP32 on the device; only what a snapshot / checkpoint keeps is meant to be FP16, which halves its
// bytes (z.gdn, MOD_PLAN.md section 6: about 111-118 MiB each).  This header is the pure part, no GPU: the FP32 <-> FP16
// conversion (round to nearest even, subnormals kept, overflow saturated to +-65504 and counted, NaN kept), the in-place
// "round through FP16" used to take the numbers of an FP16 store without changing any size, and a self-describing blob
// (magic, count, FP16 payload) for a compact store.  Error bound for a finite x with |x| <= 65504 (kF16MaxFinite):
//     |round(x) - x| <= 2^-11 * |x|  for |x| >= 2^-14 (normal range),  <= 2^-25  below it (subnormal grid 2^-24).
// What is wired today (conversation_state.cpp): the RESTORE of a checkpoint from its host bytes reads them through this
// rounding, which is numerically the same as having stored FP16 and widened it back; the stored size is NOT yet halved
// (every size check, the disk image and the VRAM slots still take z.gdn FP32 bytes): MOD_PLAN.md section 6.
#pragma once

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

}  // namespace strata::core::gdn_f16
