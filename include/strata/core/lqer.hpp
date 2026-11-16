// include/strata/core/lqer.hpp - the host factorization behind STRATA_LQER (opt-in; LEADS.md "LQER/QERA low-rank error
// correction"): a rank-r approximation A B of a quantization error E = W - deq(Q(W)).
//
// LQER (Zhang et al., ICML 2024, arXiv 2402.02446) takes the SVD of the error (scaled by activation magnitudes when a
// calibration set is at hand); QERA (Zhang et al., arXiv 2410.06040) derives the scaling that minimises the output error.
// No calibration activations exist at load time here, so this is LQER with the identity scale (plain truncated SVD, the
// best rank-r approximation in Frobenius norm).  The SVD is a randomized subspace iteration (Halko, Martinsson, Tropp
// 2011): Y = E Omega (r + p columns), q power steps with re-orthonormalisation, Q = orth(Y), Bs = Q^T E, the top-r left
// singular vectors U of Bs from the eigenvectors of Bs Bs^T (cyclic Jacobi), A = Q U (m x r), B = U^T Bs (r x n).
// Deterministic (fixed seed).  Header only, CPU, no device types: native_dense.cpp calls it at load; lqer_test checks it.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace strata::core {

namespace lqer_detail {
/// Modified Gram-Schmidt on the k columns of Y (m x k, row-major); a column that collapses becomes 0.
inline void orthonormalize(std::vector<double>& Y, int64_t m, int k) {
    for (int j = 0; j < k; ++j) {
        for (int i = 0; i < j; ++i) {
            double d = 0.0;
            for (int64_t r = 0; r < m; ++r) d += Y[(size_t) r * k + i] * Y[(size_t) r * k + j];
            for (int64_t r = 0; r < m; ++r) Y[(size_t) r * k + j] -= d * Y[(size_t) r * k + i];
        }
        double n = 0.0;
        for (int64_t r = 0; r < m; ++r) n += Y[(size_t) r * k + j] * Y[(size_t) r * k + j];
        n = std::sqrt(n);
        const double inv = n > 1e-300 ? 1.0 / n : 0.0;
        for (int64_t r = 0; r < m; ++r) Y[(size_t) r * k + j] *= inv;
    }
}

/// Eigen-decomposition of the symmetric k x k matrix S (row-major) by cyclic Jacobi: S -> diag, V columns = eigenvectors.
inline void jacobi_eig(std::vector<double>& S, int k, std::vector<double>& V, std::vector<double>& w) {
    V.assign((size_t) k * k, 0.0);
    for (int i = 0; i < k; ++i) V[(size_t) i * k + i] = 1.0;
    for (int sweep = 0; sweep < 60; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < k; ++p)
            for (int q = p + 1; q < k; ++q) off += S[(size_t) p * k + q] * S[(size_t) p * k + q];
        if (off < 1e-30) break;
        for (int p = 0; p < k; ++p)
            for (int q = p + 1; q < k; ++q) {
                const double apq = S[(size_t) p * k + q];
                if (std::fabs(apq) < 1e-300) continue;
                const double app = S[(size_t) p * k + p], aqq = S[(size_t) q * k + q];
                const double tau = (aqq - app) / (2.0 * apq);
                const double t = (tau >= 0 ? 1.0 : -1.0) / (std::fabs(tau) + std::sqrt(1.0 + tau * tau));
                const double c = 1.0 / std::sqrt(1.0 + t * t), s = t * c;
                for (int r = 0; r < k; ++r) {   // S = J^T S J
                    const double srp = S[(size_t) r * k + p], srq = S[(size_t) r * k + q];
                    S[(size_t) r * k + p] = c * srp - s * srq;
                    S[(size_t) r * k + q] = s * srp + c * srq;
                }
                for (int r = 0; r < k; ++r) {
                    const double spr = S[(size_t) p * k + r], sqr = S[(size_t) q * k + r];
                    S[(size_t) p * k + r] = c * spr - s * sqr;
                    S[(size_t) q * k + r] = s * spr + c * sqr;
                }
                for (int r = 0; r < k; ++r) {
                    const double vrp = V[(size_t) r * k + p], vrq = V[(size_t) r * k + q];
                    V[(size_t) r * k + p] = c * vrp - s * vrq;
                    V[(size_t) r * k + q] = s * vrp + c * vrq;
                }
            }
    }
    w.resize((size_t) k);
    for (int i = 0; i < k; ++i) w[(size_t) i] = S[(size_t) i * k + i];
}
}  // namespace lqer_detail

/// E (m x n, row-major: m outputs, n inputs) -> A (m x r), B (r x n), both row-major f32, with A B ~= E.
/// r is clamped to min(m, n).  `power` subspace steps (2 is the usual choice), `oversample` extra columns.
inline void lqer_factor(const float* E, int64_t m, int64_t n, int r, std::vector<float>& A, std::vector<float>& B,
                        int power = 2, int oversample = 8, uint32_t seed = 1234567u) {
    r = (int) std::max<int64_t>(1, std::min<int64_t>(r, std::min(m, n)));
    const int k = (int) std::min<int64_t>(r + oversample, std::min(m, n));
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> Om((size_t) n * k), Y((size_t) m * k, 0.0), Z((size_t) n * k, 0.0);
    for (auto& v : Om) v = nd(rng);
    auto mul_E = [&](const std::vector<double>& X, std::vector<double>& out) {   // out (m x k) = E X, X (n x k)
        std::fill(out.begin(), out.end(), 0.0);
        for (int64_t i = 0; i < m; ++i)
            for (int64_t c = 0; c < n; ++c) {
                const double e = E[(size_t) i * n + c];
                if (e == 0.0) continue;
                for (int j = 0; j < k; ++j) out[(size_t) i * k + j] += e * X[(size_t) c * k + j];
            }
    };
    auto mul_Et = [&](const std::vector<double>& X, std::vector<double>& out) {   // out (n x k) = E^T X, X (m x k)
        std::fill(out.begin(), out.end(), 0.0);
        for (int64_t i = 0; i < m; ++i)
            for (int64_t c = 0; c < n; ++c) {
                const double e = E[(size_t) i * n + c];
                if (e == 0.0) continue;
                for (int j = 0; j < k; ++j) out[(size_t) c * k + j] += e * X[(size_t) i * k + j];
            }
    };
    mul_E(Om, Y);
    lqer_detail::orthonormalize(Y, m, k);
    for (int q = 0; q < power; ++q) {
        mul_Et(Y, Z);
        lqer_detail::orthonormalize(Z, n, k);
        mul_E(Z, Y);
        lqer_detail::orthonormalize(Y, m, k);
    }
    // Bs (k x n) = Q^T E, stored as its transpose Z (n x k)
    mul_Et(Y, Z);
    // S = Bs Bs^T (k x k)
    std::vector<double> S((size_t) k * k, 0.0);
    for (int a = 0; a < k; ++a)
        for (int b = a; b < k; ++b) {
            double s = 0.0;
            for (int64_t c = 0; c < n; ++c) s += Z[(size_t) c * k + a] * Z[(size_t) c * k + b];
            S[(size_t) a * k + b] = S[(size_t) b * k + a] = s;
        }
    std::vector<double> V, w;
    lqer_detail::jacobi_eig(S, k, V, w);
    std::vector<int> order((size_t) k);
    for (int i = 0; i < k; ++i) order[(size_t) i] = i;
    std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return w[(size_t) x] > w[(size_t) y]; });
    A.assign((size_t) m * r, 0.0f);
    B.assign((size_t) r * n, 0.0f);
    for (int j = 0; j < r; ++j) {
        const int e = order[(size_t) j];
        for (int64_t i = 0; i < m; ++i) {   // A[:, j] = Q u_e
            double s = 0.0;
            for (int a = 0; a < k; ++a) s += Y[(size_t) i * k + a] * V[(size_t) a * k + e];
            A[(size_t) i * r + j] = (float) s;
        }
        for (int64_t c = 0; c < n; ++c) {   // B[j, :] = u_e^T Bs
            double s = 0.0;
            for (int a = 0; a < k; ++a) s += V[(size_t) a * k + e] * Z[(size_t) c * k + a];
            B[(size_t) j * n + c] = (float) s;
        }
    }
}

/// ||E - A B||_F^2 (f64), for the test and the load-time log line.
inline double lqer_residual2(const float* E, int64_t m, int64_t n, const std::vector<float>& A, const std::vector<float>& B, int r) {
    double s = 0.0;
    for (int64_t i = 0; i < m; ++i)
        for (int64_t c = 0; c < n; ++c) {
            double ab = 0.0;
            for (int j = 0; j < r; ++j) ab += (double) A[(size_t) i * r + j] * B[(size_t) j * n + c];
            const double d = (double) E[(size_t) i * n + c] - ab;
            s += d * d;
        }
    return s;
}

}  // namespace strata::core
