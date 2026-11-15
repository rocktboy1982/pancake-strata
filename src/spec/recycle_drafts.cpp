// src/spec/recycle_drafts.cpp - see include/strata/spec/recycle_drafts.hpp.
#include "strata/spec/recycle_drafts.hpp"

#include <algorithm>
#include <cstdlib>

namespace strata::spec {

namespace {
bool env_one(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] == '1';
}
}  // namespace

bool token_recycle_env() {
    static const bool on = env_one("STRATA_TOKEN_RECYCLE");
    return on;
}
bool ssd_lite_env() {
    static const bool on = env_one("STRATA_SSD_LITE");
    return on;
}
bool hydra_top2_env() {
    static const bool on = env_one("STRATA_HYDRA_TOP2");
    return on;
}

// ---- TokenRecycler

void TokenRecycler::push(int32_t tok, int32_t next) {
    if (tok < 0 || next < 0 || tok >= kMaxVocab || next >= kMaxVocab) return;
    if ((size_t) tok * K + K > succ_.size()) succ_.resize(((size_t) tok + 1) * K, -1);
    int32_t* r = succ_.data() + (size_t) tok * K;
    int at = K - 1;   // where `next` is now (an older copy), else the oldest slot
    for (int i = 0; i < K; ++i)
        if (r[i] == next) { at = i; break; }
    for (int i = at; i > 0; --i) r[i] = r[i - 1];
    r[0] = next;
}

void TokenRecycler::observe(const int32_t* window, const int32_t* out, int T) {
    for (int t = 0; t < T; ++t) push(window[t], out[t]);
}

int TokenRecycler::row(int32_t tok, int32_t* out) const {
    if (tok < 0 || (size_t) tok * K + K > succ_.size()) return 0;
    const int32_t* r = succ_.data() + (size_t) tok * K;
    int n = 0;
    for (int i = 0; i < K; ++i)
        if (r[i] >= 0) out[n++] = r[i];
    return n;
}

int32_t TokenRecycler::alt(int32_t tok, int32_t exclude) const {
    if (tok < 0 || (size_t) tok * K + K > succ_.size()) return -1;
    const int32_t* r = succ_.data() + (size_t) tok * K;
    for (int i = 0; i < K; ++i)
        if (r[i] >= 0 && r[i] != exclude) return r[i];
    return -1;
}

int TokenRecycler::propose(int32_t tok, int max_k, int32_t* out) const {
    int k = 0;
    for (int32_t cur = tok; k < max_k; ++k) {
        const int32_t nx = first(cur);
        if (nx < 0) break;
        out[k] = nx;
        cur = nx;
    }
    return k;
}

// ---- OutcomeDrafter

OutcomeDrafter::~OutcomeDrafter() {
    if (thr_.joinable()) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            quit_ = true;
        }
        cv_.notify_all();
        thr_.join();
    }
}

void OutcomeDrafter::run() {
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
        cv_.wait(lk, [&] { return job_ || quit_; });
        if (quit_) return;
        job_ = false;
        lk.unlock();
        compute(*sfx_, rec_, win_, T_, max_k_, top2_);
        lk.lock();
        busy_ = false;
        cv_.notify_all();
    }
}

void OutcomeDrafter::start(const SuffixDrafter& sfx, const TokenRecycler* rec, const int32_t* window, int T, int max_k,
                           bool top2) {
    wait();   // (a window retried after a failed capture: the earlier job is over first)
    n_ent_ = 0;
    fa_ = -1;
    started_ = false;
    if (T < 1 || T > kMaxT || max_k < 1) return;
    started_ = true;
    if (!thr_.joinable()) thr_ = std::thread([this] { run(); });
    std::lock_guard<std::mutex> lk(mu_);
    sfx_ = &sfx;
    rec_ = rec;
    std::copy(window, window + T, win_);
    T_ = T;
    max_k_ = std::min(max_k, kMaxK);
    top2_ = top2 && rec != nullptr;
    job_ = busy_ = true;
    cv_.notify_all();
}

void OutcomeDrafter::wait() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return !busy_; });
}

void OutcomeDrafter::compute(const SuffixDrafter& sfx, const TokenRecycler* rec, const int32_t* window, int T, int max_k,
                             bool top2) {
    if ((int) ent_.size() < kMaxT * kPerA) ent_.resize((size_t) kMaxT * kPerA);   // (once: the worker's first job)
    n_ent_ = 0;
    max_k = std::min(max_k, kMaxK);
    int32_t ext[kMaxT + 1];
    int32_t buf[kMaxK + 1];
    for (int a = 0; a < T && a < kMaxT; ++a) {
        // outcome a: window[1..a] accepted, then a bonus b the target picks (b != window[a + 1] when a < T - 1)
        for (int i = 0; i < a; ++i) ext[i] = window[i + 1];
        const int32_t rejected = a < T - 1 ? window[a + 1] : -1;
        int32_t cand[kPerA];
        int nc = 0;
        auto add = [&](int32_t b) {
            if (b < 0 || b == rejected || nc >= kPerA) return;
            for (int i = 0; i < nc; ++i)
                if (cand[i] == b) return;
            cand[nc++] = b;
        };
        {   // the lookup's own guess of b: the first token it proposes after window[1..a]
            int m = 0;
            if (sfx.propose_after_const(ext, a, 1, buf, &m) > 0) add(buf[0]);
        }
        if (rec != nullptr) {   // the matrix's best candidate, and with top2 its runner-up (both other than the rejected draft)
            int32_t cands[TokenRecycler::K];
            const int nr = rec->row(window[a], cands);
            if (!top2) {
                if (nr > 0) add(cands[0]);   // (add() drops it when it is the rejected draft)
            } else {
                for (int i = 0, took = 0; i < nr && took < 2; ++i) {
                    if (cands[i] == rejected) continue;
                    add(cands[i]);
                    ++took;
                }
            }
        }
        for (int c = 0; c < nc; ++c) {
            ext[a] = cand[c];
            int m = 0;
            int n = sfx.propose_after_const(ext, a + 1, max_k, buf, &m);
            if (n <= 0 && rec != nullptr) {
                n = rec->propose(cand[c], max_k, buf);
                m = 0;
            }
            if (n <= 0) continue;
            Entry& e = ent_[(size_t) n_ent_++];
            e.a = a;
            e.b = cand[c];
            e.n = n;
            e.match = m;
            std::copy(buf, buf + n, e.tok);
        }
    }
}

void OutcomeDrafter::finish(int a, int32_t b, int64_t next_pos) {
    // only the window start() prepared: a round that was not started leaves no outcome (its entries are an older window's)
    fa_ = started_ ? a : -1;
    started_ = false;
    fb_ = b;
    fpos_ = next_pos;
}

int OutcomeDrafter::take(int64_t pos, int max_k, int32_t* out, int* match) {
    *match = 0;
    if (fa_ < 0) return 0;
    ++asks_;
    const int a = fa_;
    fa_ = -1;
    if (pos != fpos_) return 0;
    for (int i = 0; i < n_ent_; ++i) {
        const Entry& e = ent_[(size_t) i];
        if (e.a != a || e.b != fb_) continue;
        const int n = std::min(e.n, max_k);
        std::copy(e.tok, e.tok + n, out);
        *match = e.match;
        ++hits_;
        return n;
    }
    return 0;
}

}  // namespace strata::spec
