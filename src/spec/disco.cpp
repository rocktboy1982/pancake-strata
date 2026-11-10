// src/spec/disco.cpp - see include/strata/spec/disco.hpp.
#include "strata/spec/disco.hpp"

#include "strata/spec/draft_policy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace strata::spec {

bool DiscoModel::parse(const std::string& text, std::string& err) {
    DiscoModel m;
    std::istringstream in(text);
    std::string line;
    bool head = false, have_w = false;
    m.scale.fill(1.0);
    while (std::getline(in, line)) {
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line.resize(hash);
        std::istringstream ls(line);
        std::string key;
        if (!(ls >> key)) continue;
        auto read_vec = [&](std::array<double, kF>& v) {
            for (int i = 0; i < kF; ++i)
                if (!(ls >> v[(size_t) i])) return false;
            return true;
        };
        if (key == "disco") {
            int ver = 0;
            if (!(ls >> ver) || ver != kVersion) {
                err = ver == 1 ? "disco 1 weights (feature 6 was a window-level miss rate): refit with tools/pancake/disco_fit.py"
                               : "unsupported disco weights version";
                return false;
            }
            head = true;
        } else if (key == "features") {
            int n = 0;
            if (!(ls >> n) || n != kF) { err = "expected " + std::to_string(kF) + " features"; return false; }
        } else if (key == "bias") {
            if (!(ls >> m.bias)) { err = "bad bias line"; return false; }
        } else if (key == "w") {
            if (!read_vec(m.w)) { err = "bad w line (need " + std::to_string(kF) + " numbers)"; return false; }
            have_w = true;
        } else if (key == "mean") {
            if (!read_vec(m.mean)) { err = "bad mean line"; return false; }
        } else if (key == "scale") {
            if (!read_vec(m.scale)) { err = "bad scale line"; return false; }
        } else if (key == "threshold") {
            if (!(ls >> m.threshold)) { err = "bad threshold line"; return false; }
        } else if (key == "thresholds") {
            int n = 0;
            if (!(ls >> n) || n < 1 || n > DiscoModel::kMaxPos) { err = "bad thresholds line (1..16 positions)"; return false; }
            for (int i = 0; i < n; ++i) {
                double t = -1.0;
                if (!(ls >> t) || !std::isfinite(t) || t < 0.0 || t > 1.0) { err = "bad thresholds line (each in 0..1)"; return false; }
                m.thr_pos[(size_t) i] = t;
            }
            m.n_thr_pos = n;
        }   // other keys (fit statistics) are ignored
    }
    if (!head) { err = "not a disco weights file (no 'disco 2' line)"; return false; }
    if (!have_w) { err = "no 'w' line"; return false; }
    for (int i = 0; i < kF; ++i) {
        if (!std::isfinite(m.w[(size_t) i]) || !std::isfinite(m.mean[(size_t) i]) || !std::isfinite(m.scale[(size_t) i]) ||
            m.scale[(size_t) i] <= 0.0) {
            err = "non-finite weight or non-positive scale";
            return false;
        }
    }
    if (!std::isfinite(m.bias) || !std::isfinite(m.threshold) || m.threshold < 0.0 || m.threshold > 1.0) {
        err = "bad bias or threshold (threshold in 0..1)";
        return false;
    }
    m.loaded = true;
    *this = m;
    return true;
}

bool DiscoModel::load(const std::string& path, std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "cannot read " + path; return false; }
    std::stringstream ss;
    ss << f.rdbuf();
    return parse(ss.str(), err);
}

double DiscoModel::score(const float* x) const {
    double z = bias;
    for (int i = 0; i < kF; ++i) z += w[(size_t) i] * (((double) x[i] - mean[(size_t) i]) / scale[(size_t) i]);
    return 1.0 / (1.0 + std::exp(-z));
}

void DiscoPolicy::init_from_env(const char* who) {
    if (const char* wf = std::getenv("STRATA_DISCO"); wf != nullptr && wf[0] != '\0' && std::string(wf) != "0") {
        std::string err;
        DiscoModel m;
        if (m.load(wf, err)) {
            model_ = m;
            std::fprintf(stderr, "%s: STRATA_DISCO: learned MTP window rule from %s (threshold %.3f%s); it replaces --spec-min-p "
                                 "and STRATA_SPEC_COST for MTP windows\n", who, wf, m.threshold,
                         m.n_thr_pos > 0 ? ", per-position table" : "");
        } else {
            std::fprintf(stderr, "%s: STRATA_DISCO: %s - keeping the current window rule\n", who, err.c_str());
        }
    }
    if (const char* tf = std::getenv("STRATA_DISCO_TRACE"); tf != nullptr && tf[0] != '\0') {
        if (open_trace(tf)) std::fprintf(stderr, "%s: STRATA_DISCO_TRACE: draft features -> %s\n", who, tf);
        else std::fprintf(stderr, "%s: STRATA_DISCO_TRACE: cannot write %s\n", who, tf);
    }
}

bool DiscoPolicy::open_trace(const std::string& path) {
    if (trace_ != nullptr) std::fclose(trace_);
    trace_ = std::fopen(path.c_str(), "a");
    if (trace_ == nullptr) return false;
    std::fseek(trace_, 0, SEEK_END);
    if (std::ftell(trace_) == 0) {
        std::fprintf(trace_, "round,idx,label,T,a,round_ms");
        for (int i = 0; i < kF; ++i) std::fprintf(trace_, ",f%d", i);
        std::fprintf(trace_, ",nmiss\n");   // nmiss: the new CPU misses the draft's row added (-1: unknown); the fitter's window cost
    }
    return true;
}

DiscoPolicy::~DiscoPolicy() {
    if (trace_ != nullptr) std::fclose(trace_);
}

double DiscoPolicy::row_miss_expect(int row) const {
    for (int r = std::clamp(row, 0, kMaxT); r >= 1; --r)
        if (nm_n_[(size_t) r] > 0) return nm_[(size_t) r];   // a row not measured yet: the nearest shallower one
    return 0.0;
}

void DiscoPolicy::features(int idx, float p, double chain, float* out) const {
    const double pp = std::clamp((double) p, 1e-6, 1.0);
    const int rows = idx + 2;   // the window if this draft is kept
    double dcost = 0.35;        // a prior: one more row adds about a third of a one-row round (DraftPolicy's shape)
    if (rows <= kMaxT && cost_[(size_t) rows - 1] > 0.0 && cost_[(size_t) rows] > 0.0)
        dcost = cost_[(size_t) rows] / cost_[(size_t) rows - 1] - 1.0;
    out[0] = (float) pp;
    out[1] = (float) (-std::log(pp) / 5.0);
    out[2] = (float) std::clamp(chain, 0.0, 1.0);
    out[3] = (float) idx / 8.0f;
    out[4] = (float) recent_;
    out[5] = (float) match_f_;
    out[6] = (float) (row_miss_expect(idx + 1) / 64.0);
    out[7] = (float) std::clamp(dcost, -1.0, 4.0);
    out[8] = (float) ctx_f_;
}

bool DiscoPolicy::keep_draft(int idx, float p, int cap) {
    cap = std::clamp(cap, 1, kMaxT);
    if (idx == 0) { kept_ = 0; chain_ = 1.0; }
    chain_ *= std::clamp((double) p, 0.0, 1.0);
    if (kept_ == idx && idx + 2 <= cap) {   // every draft before this one was kept
        float x[kF];
        features(idx, p, chain_, x);
        if (model_.score(x) >= model_.threshold_at(idx)) kept_ = idx + 1;
    }
    rows_ = kept_ + 1;
    return kept_ == idx + 1 && idx + 3 <= cap;   // kept, and a row left for the next draft
}

void DiscoPolicy::end_round(const Round& r) {
    if (!enabled()) return;
    // 1. the trace: features as the chain saw them (the state is still the one its drafting read)
    if (trace_ != nullptr && r.mtp_window && r.p != nullptr) {
        const int n = std::clamp(r.T - 1, 0, kMaxT - 1);
        float x[kF];
        if (n == 0) {
            std::fprintf(trace_, "%lld,-1,-1,1,0,%.3f", (long long) round_id_, r.round_ms);
            for (int i = 0; i < kF; ++i) std::fprintf(trace_, ",0");
            std::fprintf(trace_, ",-1\n");
        }
        double chain = 1.0;
        for (int j = 0; j < n; ++j) {
            chain *= std::clamp((double) r.p[j], 0.0, 1.0);
            features(j, r.p[j], chain, x);
            const int label = j < r.accepted ? 1 : j == r.accepted ? 0 : -1;
            std::fprintf(trace_, "%lld,%d,%d,%d,%d,%.3f", (long long) round_id_, j, label, r.T, r.accepted, r.round_ms);
            for (int i = 0; i < kF; ++i) std::fprintf(trace_, ",%.5g", (double) x[i]);
            const int nmiss = r.row_miss != nullptr && j + 1 < r.row_miss_n ? (int) r.row_miss[j + 1] : -1;
            std::fprintf(trace_, ",%d\n", nmiss);
        }
        ++logged_;
        if ((logged_ & 63) == 0) std::fflush(trace_);
    }
    ++round_id_;
    // 2. the state the next chain reads
    if (r.mtp_window && r.p != nullptr) {
        const int n = std::clamp(r.T - 1, 0, kMaxT - 1);
        for (int j = 0; j < n && j <= r.accepted; ++j) recent_ = 0.9 * recent_ + 0.1 * (j < r.accepted ? 1.0 : 0.0);
        // feature 6: every verified row's new CPU misses (paid whether the draft held or not)
        if (r.row_miss != nullptr)
            for (int row = 1; row <= n && row < r.row_miss_n && row <= kMaxT; ++row) {
                const double v = std::max(0, (int) r.row_miss[row]);
                nm_[(size_t) row] = nm_n_[(size_t) row] > 0 ? 0.9 * nm_[(size_t) row] + 0.1 * v : v;
                ++nm_n_[(size_t) row];
            }
    }
    if (r.misses_total >= 0) {
        if (misses_last_ >= 0 && r.misses_total >= misses_last_ && r.T > 0) {
            const double per_row = (double) (r.misses_total - misses_last_) / ((double) r.T * std::max(1, r.n_layers));
            miss_ = 0.9 * miss_ + 0.1 * std::min(per_row, 64.0);
        }
        misses_last_ = r.misses_total;
    }
    match_f_ = std::log1p((double) std::max(0, r.match)) / 3.0;
    ctx_f_ = std::log2(1.0 + (double) std::max(0LL, r.pos)) / 17.0;
    if (r.policy != nullptr)
        for (int t = 1; t <= kMaxT; ++t) cost_[(size_t) t] = r.policy->cost_ms(t);
}

}  // namespace strata::spec
