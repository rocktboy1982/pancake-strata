// src/spec/disco_test.cpp - DISCO's learned MTP window rule (no model, no GPU).
//
// 1. the weights file: parse, reject, fall back when missing;
// 2. the chain rule: keeps confident drafts, stops at the first predicted rejection, honours the window cap;
// 3. the trace logger: one labelled line per drafted token of an MTP window;
// 4. exactness: a speculative greedy loop over a mock target and a mock MTP drafter (driven by the same calling
//    convention as MtpDrafter's chain: rule(j - 1, p_j) before each next step, once more after a full chain) commits
//    the same tokens as plain greedy decoding with the static --spec-min-p rule, with every draft verified, and with
//    the DISCO rule at several thresholds.  The window length changes the speed only, never the output.
#include "strata/spec/disco.hpp"
#include "strata/spec/draft_policy.hpp"
#include "strata/spec/window_rule.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

using namespace strata::spec;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++g_fail; }
}

uint64_t mix(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33;
    return x;
}
constexpr int kVocab = 64;
// the target: greedy next token from the last three tokens
int32_t target_next(const std::vector<int32_t>& ctx) {
    const size_t n = ctx.size();
    uint64_t h = 1469598103934665603ull;
    for (size_t i = n >= 3 ? n - 3 : 0; i < n; ++i) h = mix(h ^ (uint64_t) ctx[i]);
    return (int32_t) (h % kVocab);
}
// the drafter: right with a chance that falls with depth, and says how sure it is (correlated with being right)
struct Draft { int32_t tok; float p; };
Draft mock_draft(const std::vector<int32_t>& ctx, int depth, uint64_t salt) {
    const uint64_t h = mix(salt ^ mix(ctx.size() * 7919u + (uint64_t) depth));
    const double u = (double) (h % 10000) / 10000.0;
    const bool right = u < 0.85 - 0.1 * depth;
    const double conf = right ? 0.45 + 0.5 * ((double) ((h >> 20) % 1000) / 1000.0)
                              : 0.05 + 0.6 * ((double) ((h >> 20) % 1000) / 1000.0);
    const int32_t t = target_next(ctx);
    return {right ? t : (int32_t) ((t + 1 + (int) ((h >> 40) % (kVocab - 1))) % kVocab), (float) conf};
}

// One speculative greedy run.  rule: the chain rule (empty = the static min_p test, as MtpDrafter without a rule).
// window_of(probs, n): the verified window (rows) from the drafted chain.
struct RunStats { std::vector<int32_t> out; int64_t rounds = 0, drafts = 0, accepted = 0; };
RunStats run_spec(const std::vector<int32_t>& prompt, int n_new, int S, float min_p, uint64_t salt,
                  const std::function<bool(int, float)>& rule, const std::function<int(const float*, int)>& window_of,
                  DiscoPolicy* disco, const DraftPolicy* policy) {
    RunStats st;
    std::vector<int32_t> ctx = prompt;
    while ((int) st.out.size() < n_new) {
        // the chain, exactly as src/core/mtp.cpp drives it
        std::vector<int32_t> dctx = ctx, drafts;
        std::vector<float> probs;
        const int max_steps = S - 1;
        Draft d = mock_draft(dctx, 0, salt);
        drafts.push_back(d.tok); probs.push_back(d.p); dctx.push_back(d.tok);
        float pj = d.p;
        int n = 1;
        for (int j = 1; j < max_steps && (rule ? rule(j - 1, pj) : pj >= min_p); ++j) {
            d = mock_draft(dctx, j, salt);
            drafts.push_back(d.tok); probs.push_back(d.p); dctx.push_back(d.tok);
            pj = d.p;
            ++n;
        }
        if (rule && n >= max_steps) (void) rule(n - 1, pj);
        const int T = std::max(1, std::min(window_of(probs.data(), n), S));
        // verify: the target's output at every row; accept the drafts that equal it
        std::vector<int32_t> vctx = ctx, outv;
        for (int i = 0; i < T; ++i) {
            outv.push_back(target_next(vctx));
            if (i + 1 < T) vctx.push_back(drafts[(size_t) i]);
        }
        int a = 0;
        while (a < T - 1 && drafts[(size_t) a] == outv[(size_t) a]) ++a;
        for (int i = 0; i <= a && (int) st.out.size() < n_new; ++i) { st.out.push_back(outv[(size_t) i]); ctx.push_back(outv[(size_t) i]); }
        ++st.rounds; st.drafts += T - 1; st.accepted += a;
        if (disco != nullptr) {
            DiscoPolicy::Round r;
            r.p = probs.data(); r.T = T; r.accepted = a; r.mtp_window = true; r.round_ms = 10.0 + 3.0 * T;
            r.misses_total = st.rounds * 40; r.n_layers = 48; r.pos = (long long) ctx.size(); r.policy = policy;
            disco->end_round(r);
        }
    }
    return st;
}

std::vector<int32_t> greedy(const std::vector<int32_t>& prompt, int n_new) {
    std::vector<int32_t> ctx = prompt, out;
    for (int i = 0; i < n_new; ++i) { const int32_t t = target_next(ctx); out.push_back(t); ctx.push_back(t); }
    return out;
}

DiscoModel model_on_p(double threshold) {   // a hand-set model: the score follows the draft's probability
    DiscoModel m;
    std::string err;
    const std::string txt = "disco 1\nfeatures 9\nbias 0\nw 8 0 0 0 0 0 0 0 0\nmean 0.5 0 0 0 0 0 0 0 0\n"
                            "scale 1 1 1 1 1 1 1 1 1\nthreshold " + std::to_string(threshold) + "\n";
    check(m.parse(txt, err), "hand-set model parses");
    return m;
}
}  // namespace

int main() {
    {   // 1. weights file
        DiscoModel m;
        std::string err;
        check(!m.load("/nonexistent/disco.txt", err) && !m.loaded, "missing file: not loaded");
        check(!m.parse("bias 1\nw 1 2 3 4 5 6 7 8 9\n", err), "no header: rejected");
        check(!m.parse("disco 1\nw 1 2 3\n", err), "short w: rejected");
        check(!m.parse("disco 2\nw 1 2 3 4 5 6 7 8 9\n", err), "wrong version: rejected");
        check(!m.parse("disco 1\nw 1 2 3 4 5 6 7 8 9\nscale 1 1 1 0 1 1 1 1 1\n", err), "zero scale: rejected");
        check(!m.parse("disco 1\nw 1 2 3 4 5 6 7 8 9\nthreshold 1.5\n", err), "threshold > 1: rejected");
        check(m.parse("# fitted\ndisco 1\nfeatures 9\nbias -0.5\nw 1 0 0 0 0 0 0 0 0  # p only\nthreshold 0.4\nauc 0.8\n", err) && m.loaded,
              "valid file parses (comments, unknown keys)");
        float x[9] = {0.5f, 0, 0, 0, 0, 0, 0, 0, 0};
        check(std::fabs(m.score(x) - 1.0 / (1.0 + std::exp(0.0))) < 1e-9, "score = sigmoid(bias + w.x)");

        DiscoPolicy pol;   // no STRATA_DISCO: inactive, nothing to keep
        check(!pol.active() && !pol.enabled(), "fallback: a policy with no weights is inactive");
        DiscoPolicy::Round r; r.T = 3; r.accepted = 1; r.mtp_window = true;
        pol.end_round(r);   // a no-op
        check(pol.recent() == 0.7, "inactive policy keeps no state");
    }
    {   // 2. the chain rule
        DiscoPolicy pol;
        pol.set_model(model_on_p(0.5));
        check(pol.active(), "model set: active");
        check(pol.keep_draft(0, 0.9f, 6) && pol.rows() == 2, "confident first draft kept, chain goes on");
        check(pol.keep_draft(1, 0.8f, 6) && pol.rows() == 3, "second kept");
        check(!pol.keep_draft(2, 0.2f, 6) && pol.rows() == 3, "unsure third: chain stops, third left out");
        check(!pol.keep_draft(0, 0.1f, 6) && pol.rows() == 1, "unsure first: one-row window");
        check(pol.keep_draft(0, 0.9f, 3) && pol.rows() == 2, "cap 3: first kept, room for one more");
        check(!pol.keep_draft(1, 0.9f, 3) && pol.rows() == 3, "cap 3: second kept, no room for a third");
        check(!pol.keep_draft(0, 0.99f, 2) && pol.rows() == 2, "cap 2: one draft at most");
        // the window rule takes the chain's rows exactly as it takes STRATA_SPEC_COST's
        WindowRuleIn in;
        in.use_mtp = true; in.draft_on = true; in.spec_cost = true; in.chain_rows = pol.rows(); in.cap = 4; in.start = 4;
        check(choose_window(in) == 2, "choose_window uses the DISCO rows");
    }
    {   // 3. the trace
        const std::string path = "disco_test_trace.csv";
        std::remove(path.c_str());
        {   // the file closes with the policy
        DiscoPolicy pol;
        check(pol.open_trace(path) && pol.tracing() && !pol.active(), "trace only: logging, rule inactive");
        const float p[3] = {0.9f, 0.7f, 0.3f};
        DiscoPolicy::Round r; r.p = p; r.T = 4; r.accepted = 1; r.mtp_window = true; r.round_ms = 20.0; r.pos = 1000;
        pol.end_round(r);
        DiscoPolicy::Round r1; r1.p = p; r1.T = 1; r1.mtp_window = true; r1.round_ms = 12.0;
        pol.end_round(r1);
        DiscoPolicy::Round rl; rl.p = p; rl.T = 5; rl.mtp_window = false;   // a lookup window: not logged
        pol.end_round(rl);
        }
        std::ifstream f(path);
        std::vector<std::string> lines;
        for (std::string l; std::getline(f, l);) lines.push_back(l);
        check(lines.size() == 5, "header + 3 draft lines + 1 empty-window line");
        if (lines.size() == 5) {
            check(lines[0].rfind("round,idx,label,T,a,round_ms,f0", 0) == 0, "header");
            check(lines[1].rfind("0,0,1,4,1,", 0) == 0, "draft 0 accepted");
            check(lines[2].rfind("0,1,0,4,1,", 0) == 0, "draft 1 rejected");
            check(lines[3].rfind("0,2,-1,4,1,", 0) == 0, "draft 2 not reached");
            check(lines[4].rfind("1,-1,-1,1,0,", 0) == 0, "one-row window marker");
        }
        std::remove(path.c_str());
    }
    {   // 4. exactness on the mock target
        const int S = 6, n_new = 600;
        int64_t runs = 0;
        for (uint64_t salt = 1; salt <= 6; ++salt) {
            std::vector<int32_t> prompt;
            for (int i = 0; i < 12; ++i) prompt.push_back((int32_t) (mix(salt * 31 + (uint64_t) i) % kVocab));
            const std::vector<int32_t> ref = greedy(prompt, n_new);
            auto by_min_p = [&](float mp) {
                return [mp, S](const float* pr, int n) { int T = 1; while (T < S && T - 1 < n && pr[T - 1] >= mp) ++T; return T; };
            };
            // the static rule (preset: --spec-min-p 0.5), and every draft verified
            const RunStats st = run_spec(prompt, n_new, S, 0.5f, salt, nullptr, by_min_p(0.5f), nullptr, nullptr);
            check(st.out == ref, "static --spec-min-p 0.5 = greedy");
            const RunStats all = run_spec(prompt, n_new, S, 0.0f, salt, nullptr, by_min_p(0.0f), nullptr, nullptr);
            check(all.out == ref, "every draft verified = greedy");
            for (double thr : {0.05, 0.3, 0.5, 0.7, 0.95}) {
                DiscoPolicy pol;
                pol.set_model(model_on_p(thr));
                DraftPolicy dp(S);
                const RunStats ds = run_spec(prompt, n_new, S, 0.5f, salt,
                                             [&](int idx, float pj) { return pol.keep_draft(idx, pj, S); },
                                             [&](const float*, int) { return pol.rows(); }, &pol, &dp);
                check(ds.out == ref, "DISCO rule = greedy");
                ++runs;
                if (salt == 1)
                    std::printf("threshold %.2f: %lld rounds, %lld drafts verified, %lld accepted (static 0.5: %lld rounds)\n", thr,
                                (long long) ds.rounds, (long long) ds.drafts, (long long) ds.accepted, (long long) st.rounds);
            }
            {   // a model on every feature (the trained shape): still exact
                DiscoPolicy pol;
                DiscoModel m;
                std::string err;
                check(m.parse("disco 1\nbias 0.3\nw 2 -1 1.5 -0.4 1.2 0.5 -0.8 -0.6 0.1\nmean 0.6 0.1 0.4 0.2 0.7 0.1 0.5 0.3 0.5\n"
                              "scale 0.2 0.1 0.3 0.15 0.1 0.2 0.5 0.2 0.1\nthreshold 0.55\n", err), "full model parses");
                pol.set_model(m);
                DraftPolicy dp(S);
                const RunStats ds = run_spec(prompt, n_new, S, 0.5f, salt,
                                             [&](int idx, float pj) { return pol.keep_draft(idx, pj, S); },
                                             [&](const float*, int) { return pol.rows(); }, &pol, &dp);
                check(ds.out == ref, "DISCO full-feature model = greedy");
                check(pol.miss_rate() > 0.0, "miss feature tracks the pool's misses");
                ++runs;
            }
        }
        std::printf("exactness: %lld DISCO runs of %d tokens equal plain greedy decoding\n", (long long) runs, n_new);
    }
    if (g_fail == 0) std::printf("disco_test: all passed\n");
    return g_fail == 0 ? 0 : 1;
}
