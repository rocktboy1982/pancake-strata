// src/kernels/cpu/q2_avx2_rows.inl - the Q2_0 row kernel of q2_avx2.cpp, included there twice like
// iq_avx2_rows.inl: the AVX2 form (STRATA_ROWS_VNNI 0, namespace plain) and the AVX-VNNI one (1, namespace vnni), every
// function under STRATA_ROWS_FN (GCC/Clang's target("avxvnni") in the VNNI copy).  Not a header.

// The 2-bit codes (0..3) times the int8 activation, four products per int32 lane.  vpdpbusd is one uop where maddubs
// + madd(ones) take two on the multiply ports; maddubs cannot saturate (a pair is at most 2 * 3 * 128), so the sums
// are the same.
STRATA_ROWS_FN inline __m256i dot4(__m256i codes, __m256i act) {
#if STRATA_ROWS_VNNI
    return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), codes, act);
#else
    return _mm256_madd_epi16(_mm256_maddubs_epi16(codes, act), _mm256_set1_epi16(1));
#endif
}

template <int NT> STRATA_ROWS_FN
inline void row_multi(const uint8_t* row, const ActQ* const* a, int nblocks, float* res) {
    __m256 acc[NT];
    float corr[NT];
    for (int t = 0; t < NT; ++t) { acc[t] = _mm256_setzero_ps(); corr[t] = 0.f; }
    for (int b = 0; b < nblocks; ++b) {
        const uint8_t* blk = row + (size_t) b * 18;
        const float d = h2f(blk);
        __m256i lo, hi;
        unpack64(blk + 2, lo, hi);
        for (int t = 0; t < NT; ++t) {
            const int8_t* q = a[t]->q + b * 64;
            const __m256i s0 = dot4(lo, _mm256_loadu_si256((const __m256i*) q));
            const __m256i s1 = dot4(hi, _mm256_loadu_si256((const __m256i*) (q + 32)));
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * a[t]->scale[2 * b]), _mm256_cvtepi32_ps(s0), acc[t]);
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * a[t]->scale[2 * b + 1]), _mm256_cvtepi32_ps(s1), acc[t]);
            corr[t] += d * (a[t]->hx[2 * b] + a[t]->hx[2 * b + 1]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        const __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc[t]), _mm256_extractf128_ps(acc[t], 1));
        const __m128 s = _mm_add_ps(h, _mm_movehl_ps(h, h));
        res[t] = _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s))) - corr[t];
    }
}

template <int NT> STRATA_ROWS_FN
void rows(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, float* const* out, int r0, int r1) {
    float res[NT];
    const int pf = q2_prefetch_bytes();   // item 40 (q2_avx2.cpp)
    for (int r = r0; r < r1; ++r) {
        if (pf > 0) {
            const char* p = (const char*) (w + (size_t) r * row_bytes) + pf;
            for (size_t o = 0; o < row_bytes; o += 64) _mm_prefetch(p + o, _MM_HINT_T0);
        }
        row_multi<NT>(w + (size_t) r * row_bytes, a, nblocks, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

STRATA_ROWS_FN void rows_nt(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                            float* const* out, int r0, int r1) {
    switch (nt) {
        case 1: rows<1>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 2: rows<2>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 3: rows<3>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 4: rows<4>(w, row_bytes, nblocks, a, out, r0, r1); break;
        default:
            for (int t0 = 0; t0 < nt; t0 += 4) {
                const int k = nt - t0 < 4 ? nt - t0 : 4;
                rows_nt(w, row_bytes, nblocks, a + t0, k, out + t0, r0, r1);
            }
    }
}

// ---- the default kernel (STRATA_Q2_LEGACY unset): a cheaper code unpack and R rows at a time.  Adapted from the
// Q2_0 AVX2 kernel of imanu86's fork (commit c3fc8f34d3, Niko1221/Strata upstream).  The arithmetic is the one above:
// unpack64_fast gives lo/hi byte for byte as unpack64 does, and every float operation (d times the chunk scale, the
// two FMAs per block in block order, the -1 correction, the reduction tree) is the same operation on the same
// operands in the same order.  A row's sequence of operations does not depend on how many rows share its group.

// 16 bytes of codes -> lo/hi exactly as unpack64: 8 vector ops instead of 14.  The 16 code bytes are broadcast to both
// 128-bit lanes and zero-extended to 16-bit words in the order (bytes 0-3, 8-11 | 4-7, 12-15); x = w | w << 6 holds
// the low byte of x holds code 0 (bits 0..1) and code 2 (bits 4..5) of each source byte, the high byte code 1 (bits
// 0..1 of the high byte) and code 3 (bits 4..5 of the high byte after the shift); so x & 0x0303 gives the codes of
// values 4i, 4i+1 (c01) and (x >> 4) & 0x0303 those of 4i+2, 4i+3 (c23).  The two 16-bit interleaves then give
// values 0..31 (lo) and 32..63 (hi), in order.
STRATA_ROWS_FN inline void unpack64_fast(const uint8_t* codes, __m256i& lo, __m256i& hi) {
    const __m256i idx = _mm256_setr_epi8(0, -128, 1, -128, 2, -128, 3, -128, 8, -128, 9, -128, 10, -128, 11, -128,
                                         4, -128, 5, -128, 6, -128, 7, -128, 12, -128, 13, -128, 14, -128, 15, -128);
    const __m256i m = _mm256_set1_epi16(0x0303);
    const __m256i x2 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) codes));
    const __m256i w = _mm256_shuffle_epi8(x2, idx);
    const __m256i x = _mm256_or_si256(w, _mm256_slli_epi16(w, 6));
    const __m256i c01 = _mm256_and_si256(x, m);
    const __m256i c23 = _mm256_and_si256(_mm256_srli_epi16(x, 4), m);
    lo = _mm256_unpacklo_epi16(c01, c23);
    hi = _mm256_unpackhi_epi16(c01, c23);
}

// R consecutive rows, one token group of NT: the block loop is outside, so the R rows' FMA chains are independent.
template <int NT, int R> STRATA_ROWS_FN
inline void group_multi(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, float* const* out, int r) {
    const int pf = q2_prefetch_bytes();   // item 40 (q2_avx2.cpp), as in rows()
    if (pf > 0)
        for (int k = 0; k < R; ++k) {
            const char* p = (const char*) (w + (size_t) (r + k) * row_bytes) + pf;
            for (size_t o = 0; o < row_bytes; o += 64) _mm_prefetch(p + o, _MM_HINT_T0);
        }
    __m256 acc[R][NT];
    float corr[R][NT];
    for (int k = 0; k < R; ++k)
        for (int t = 0; t < NT; ++t) { acc[k][t] = _mm256_setzero_ps(); corr[k][t] = 0.f; }
    for (int b = 0; b < nblocks; ++b) {
        for (int k = 0; k < R; ++k) {
            const uint8_t* blk = w + (size_t) (r + k) * row_bytes + (size_t) b * 18;
            const float d = h2f(blk);
            const __m256 dv = _mm256_set1_ps(d);
            __m256i lo, hi;
            unpack64_fast(blk + 2, lo, hi);
            for (int t = 0; t < NT; ++t) {
                const int8_t* q = a[t]->q + b * 64;
                const __m256i s0 = dot4(lo, _mm256_loadu_si256((const __m256i*) q));
                const __m256i s1 = dot4(hi, _mm256_loadu_si256((const __m256i*) (q + 32)));
                acc[k][t] = _mm256_fmadd_ps(_mm256_mul_ps(dv, _mm256_broadcast_ss(&a[t]->scale[2 * b])),
                                            _mm256_cvtepi32_ps(s0), acc[k][t]);
                acc[k][t] = _mm256_fmadd_ps(_mm256_mul_ps(dv, _mm256_broadcast_ss(&a[t]->scale[2 * b + 1])),
                                            _mm256_cvtepi32_ps(s1), acc[k][t]);
                corr[k][t] += d * (a[t]->hx[2 * b] + a[t]->hx[2 * b + 1]);
            }
        }
    }
    for (int k = 0; k < R; ++k)
        for (int t = 0; t < NT; ++t) {
            const __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc[k][t]), _mm256_extractf128_ps(acc[k][t], 1));
            const __m128 s = _mm_add_ps(h, _mm_movehl_ps(h, h));
            out[t][r + k] = _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s))) - corr[k][t];
        }
}

template <int NT, int R> STRATA_ROWS_FN
void rows_fast(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, float* const* out, int r0, int r1) {
    int r = r0;
    for (; r + R <= r1; r += R) group_multi<NT, R>(w, row_bytes, nblocks, a, out, r);
    for (; r < r1; ++r) group_multi<NT, 1>(w, row_bytes, nblocks, a, out, r);
}

// Short rows (a down projection, about 10 blocks) are worked in groups; long ones (gate/up, about 40 blocks) one at a
// time, where interleaved rows break the hardware prefetcher's single stream (measured 0.89x from RAM in the fork).
STRATA_ROWS_FN void rows_fast_nt(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                                 float* const* out, int r0, int r1) {
    switch (nt) {
        case 1:
            if (nblocks <= 16) rows_fast<1, 4>(w, row_bytes, nblocks, a, out, r0, r1);
            else rows_fast<1, 1>(w, row_bytes, nblocks, a, out, r0, r1);
            break;
        case 2:
            if (nblocks <= 16) rows_fast<2, 2>(w, row_bytes, nblocks, a, out, r0, r1);
            else rows_fast<2, 1>(w, row_bytes, nblocks, a, out, r0, r1);
            break;
        case 3: rows_fast<3, 1>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 4: rows_fast<4, 1>(w, row_bytes, nblocks, a, out, r0, r1); break;
        default:
            for (int t0 = 0; t0 < nt; t0 += 4) {
                const int k = nt - t0 < 4 ? nt - t0 : 4;
                rows_fast_nt(w, row_bytes, nblocks, a + t0, k, out + t0, r0, r1);
            }
    }
}

#undef STRATA_ROWS_VNNI
#undef STRATA_ROWS_FN
