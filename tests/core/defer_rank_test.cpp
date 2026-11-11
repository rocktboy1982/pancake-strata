// MOD_PLAN items 28 / A4: the deferral's heavy rule and rank guard (STRATA_VERIFY_DEFER_CPU_BELOW / _TOP) and the pool's
// heavy/light split with STRATA_VERIFY_DEFER_CPU_FOLD (core/defer_rank.hpp).  No GPU, no model.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "strata/core/defer_rank.hpp"

using namespace strata::core;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

namespace {

struct Job {
    int expert = -1;
};

// one random window: n_tok rows of k distinct experts out of n_expert, weights normalised per row and sorted descending
// (the router's order), each expert CPU-missed with probability p_cpu (every entry of an expert has the same placement)
struct Window {
    int n_tok = 0, k = 0;
    std::vector<int> ids;
    std::vector<float> w;
    std::vector<char> cpu;
};

Window random_window(std::mt19937& rng, int n_expert, int k, double p_cpu) {
    Window win;
    win.n_tok = 1 + (int) (rng() % 8);
    win.k = k;
    std::vector<char> cpu_expert((size_t) n_expert);
    std::bernoulli_distribution miss(p_cpu);
    for (char& c : cpu_expert) c = miss(rng) ? 1 : 0;
    std::lognormal_distribution<float> logit(0.0f, 1.0f);
    std::vector<int> all((size_t) n_expert);
    for (int e = 0; e < n_expert; ++e) all[(size_t) e] = e;
    for (int t = 0; t < win.n_tok; ++t) {
        std::shuffle(all.begin(), all.end(), rng);
        std::vector<float> row((size_t) k);
        float sum = 0.0f;
        for (float& v : row) sum += (v = logit(rng));
        for (float& v : row) v /= sum;
        std::sort(row.begin(), row.end(), [](float a, float b) { return a > b; });
        for (int j = 0; j < k; ++j) {
            win.ids.push_back(all[(size_t) j]);
            win.w.push_back(row[(size_t) j]);
            win.cpu.push_back(cpu_expert[(size_t) all[(size_t) j]]);
        }
    }
    return win;
}

}  // namespace

int main() {
    // the switch is read once: set before the first call
    setenv("STRATA_VERIFY_DEFER_CPU_FOLD", "2", 1);
    CHECK(defer_fold() == 2);

    // the weight rule and the rank guard on one window of 2 tokens x 4 entries
    {
        const float w[8] = {0.50f, 0.20f, 0.10f, 0.09f,     // token 0: three at or above 0.1
                            0.09f, 0.08f, 0.08f, 0.05f};    // token 1: none above 0.1, a tie at ranks 1-2
        CHECK(defer_heavy_entry(w, 0, 4, 0.1f, 0) && defer_heavy_entry(w, 2, 4, 0.1f, 0));
        CHECK(!defer_heavy_entry(w, 3, 4, 0.1f, 0));
        CHECK(!defer_heavy_entry(w, 3, 4, 0.1f, 3));   // rank 3 of its token
        CHECK(defer_heavy_entry(w, 3, 4, 0.1f, 4));
        for (int i = 4; i < 8; ++i) CHECK(!defer_heavy_entry(w, i, 4, 0.1f, 0));
        CHECK(defer_heavy_entry(w, 4, 4, 0.1f, 1));    // ranks count within the token, not the window
        CHECK(!defer_heavy_entry(w, 5, 4, 0.1f, 1));
        CHECK(defer_heavy_entry(w, 5, 4, 0.1f, 2));    // the tie goes to the lower index
        CHECK(!defer_heavy_entry(w, 6, 4, 0.1f, 2));
        CHECK(defer_heavy_entry(w, 6, 4, 0.1f, 3));
        CHECK(!defer_heavy_entry(w, 7, 4, 0.1f, 3));
    }

    // with k routing weights normalised to 1 the largest is >= 1/k: at the default _BELOW 0.1 and k = 10 the token's top
    // entry is always heavy, so _TOP=1 changes nothing and _TOP=2 adds at most its rank-1 entry (when that is below 0.1)
    {
        std::mt19937 rng(7);
        int added_by_top2 = 0, rows = 0;
        for (int it = 0; it < 2000; ++it) {
            const Window win = random_window(rng, 512, 10, 0.3);
            for (int t = 0; t < win.n_tok; ++t, ++rows) {
                int added = 0;
                for (int j = 0; j < 10; ++j) {
                    const int64_t i = (int64_t) t * 10 + j;
                    const bool base = defer_heavy_entry(win.w.data(), i, 10, 0.1f, 0);
                    CHECK(base == defer_heavy_entry(win.w.data(), i, 10, 0.1f, 1));
                    if (!base && defer_heavy_entry(win.w.data(), i, 10, 0.1f, 2)) {
                        CHECK(j == 1);
                        ++added;
                    }
                }
                CHECK(added <= 1);
                added_by_top2 += added;
            }
        }
        CHECK(added_by_top2 > 0 && added_by_top2 < rows);
    }

    // the fold rule
    CHECK(!defer_fold_light(3, 1, 0));   // off
    CHECK(!defer_fold_light(0, 1, 2));   // no heavy job: the split stays (the partial flag first)
    CHECK(!defer_fold_light(3, 0, 2));   // nothing light to fold
    CHECK(defer_fold_light(3, 1, 2) && defer_fold_light(3, 2, 2));
    CHECK(!defer_fold_light(3, 3, 2));

    // the pool's split on random windows (64 experts so rows share experts): heavy jobs first in their order, every heavy
    // entry's expert runs before the partial flag whatever the fold, and the fold only moves light jobs before it
    {
        std::mt19937 rng(11);
        std::vector<Job> scratch;
        int folded = 0, split = 0;
        for (int it = 0; it < 4000; ++it) {
            const Window win = random_window(rng, 64, 10, 0.4);
            const int n = win.n_tok * win.k;
            const float heavy = 0.1f;
            const int top = (int) (rng() % 3);
            std::vector<int> job_of(64, -1);
            std::vector<Job> jobs;
            for (int i = 0; i < n; ++i)
                if (win.cpu[(size_t) i] && job_of[(size_t) win.ids[(size_t) i]] < 0) {
                    job_of[(size_t) win.ids[(size_t) i]] = (int) jobs.size();
                    jobs.push_back(Job{win.ids[(size_t) i]});
                }
            const int njobs = (int) jobs.size();
            if (njobs == 0) continue;
            std::vector<char> heavy_expert(64, 0);
            bool heavy_job[64] = {};
            for (int i = 0; i < n; ++i)
                if (win.cpu[(size_t) i] && defer_heavy_entry(win.w.data(), i, win.k, heavy, top)) {
                    heavy_job[job_of[(size_t) win.ids[(size_t) i]]] = true;
                    heavy_expert[(size_t) win.ids[(size_t) i]] = 1;
                }
            int nheavy = 0;
            for (int j = 0; j < njobs; ++j) nheavy += heavy_job[j] ? 1 : 0;
            std::vector<Job> ref = jobs;   // the split as before the helper: a stable partition, heavy first
            std::stable_partition(ref.begin(), ref.end(), [&](const Job& jb) { return heavy_expert[(size_t) jb.expert] != 0; });
            for (int fold : {0, 1, 2, 3, 64}) {
                std::vector<Job> order = jobs;
                const int nfirst = defer_order_jobs(order.data(), heavy_job, njobs, fold, scratch);
                bool same = true;
                for (int j = 0; j < njobs; ++j) same = same && order[(size_t) j].expert == ref[(size_t) j].expert;
                CHECK(same);
                const int nlight = njobs - nheavy;
                const bool fold_now = fold > 0 && nheavy > 0 && nlight > 0 && nlight <= fold;
                CHECK(nfirst == (fold_now ? njobs : nheavy));
                for (int j = 0; j < njobs; ++j)
                    if (heavy_expert[(size_t) order[(size_t) j].expert]) CHECK(j < nfirst);
                if (fold == 2) (fold_now ? folded : split) += 1;
            }
        }
        CHECK(folded > 0 && split > 0);
    }

    if (failed == 0) std::printf("defer_rank_test: ok\n");
    return failed == 0 ? 0 : 1;
}
