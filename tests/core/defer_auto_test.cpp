// MOD_PLAN 17.1 note: STRATA_VERIFY_DEFER_AUTO (core/defer_auto.hpp) - which verify windows defer their light CPU rows, the
// rank guard a window stages for that, and why a window staged at k keeps every CPU row on time (the rule of defer_rank.hpp
// that the host pool and the GPU's copy_rows_by_weight_kernel both apply).  No GPU, no model.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "strata/core/defer_auto.hpp"
#include "strata/core/defer_rank.hpp"

using namespace strata::core;

static int failed = 0;
static int checks = 0;
#define CHECK(x) do { ++checks; if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

namespace {

struct Job {
    int expert = -1;
};

const DeferWindowKind kAllKinds[] = {DeferWindowKind::Prompt, DeferWindowKind::Plain, DeferWindowKind::Model,
                                     DeferWindowKind::Lookup, DeferWindowKind::Chain};

void test_parse() {
    bool known = false;
    CHECK(!parse_defer_auto(nullptr, &known) && known);
    CHECK(!parse_defer_auto("", &known) && known);
    CHECK(!parse_defer_auto("0", &known) && known);
    CHECK(parse_defer_auto("1", &known) && known);
    for (const char* bad : {"2", "yes", "on", "1 ", " 1", "-1", "01"}) {
        known = true;
        CHECK(!parse_defer_auto(bad, &known) && !known);
    }
    CHECK(parse_defer_auto("1"));   // (known may be omitted)

    CHECK(parse_defer_auto_min_t(nullptr, &known) == 0 && known);
    CHECK(parse_defer_auto_min_t("", &known) == 0 && known);
    CHECK(parse_defer_auto_min_t("0", &known) == 0 && known);
    CHECK(parse_defer_auto_min_t("5", &known) == 5 && known);
    CHECK(parse_defer_auto_min_t("12", &known) == 12 && known);
    CHECK(parse_defer_auto_min_t("256", &known) == kDeferAutoMaxMinT && known);
    for (const char* bad : {"257", "-1", "5x", "x", "4.5", " "}) {
        known = true;
        CHECK(parse_defer_auto_min_t(bad, &known) == 0 && !known);
    }
}

void test_kind() {
    // a merged first window is the prompt's, whatever else is true
    CHECK(defer_window_kind(true, true, 3, 12) == DeferWindowKind::Prompt);
    CHECK(defer_window_kind(true, false, 0, 1) == DeferWindowKind::Prompt);
    // one row: no drafts
    CHECK(defer_window_kind(false, false, 0, 1) == DeferWindowKind::Plain);
    CHECK(defer_window_kind(false, true, 0, 1) == DeferWindowKind::Plain);
    // the lookup copy, the chained tail, the model's drafts
    CHECK(defer_window_kind(false, true, 0, 2) == DeferWindowKind::Lookup);
    CHECK(defer_window_kind(false, true, 0, 12) == DeferWindowKind::Lookup);
    CHECK(defer_window_kind(false, false, 3, 7) == DeferWindowKind::Chain);
    CHECK(defer_window_kind(false, false, 0, 4) == DeferWindowKind::Model);
}

void test_window_on() {
    // auto off: every window defers as configured (the default path asks nothing else)
    const DeferAutoConfig off{};
    for (DeferWindowKind kind : kAllKinds)
        for (int rows = 1; rows <= 16; ++rows) CHECK(defer_window_on(off, kind, rows));
    DeferAutoConfig off_with_t{false, 9};   // a MIN_T without the switch does nothing
    CHECK(defer_window_on(off_with_t, DeferWindowKind::Model, 2));

    // auto on, any size: lookup and chained windows of 2 rows or more
    const DeferAutoConfig on{true, 0};
    for (int rows = 1; rows <= 16; ++rows) {
        CHECK(defer_window_on(on, DeferWindowKind::Lookup, rows) == (rows >= 2));
        CHECK(defer_window_on(on, DeferWindowKind::Chain, rows) == (rows >= 2));
        CHECK(!defer_window_on(on, DeferWindowKind::Model, rows));
        CHECK(!defer_window_on(on, DeferWindowKind::Prompt, rows));
        CHECK(!defer_window_on(on, DeferWindowKind::Plain, rows));
    }
    // MIN_T 1 is MIN_T 2 (a window of one row has no drafts)
    const DeferAutoConfig on1{true, 1};
    CHECK(!defer_window_on(on1, DeferWindowKind::Lookup, 1));
    CHECK(defer_window_on(on1, DeferWindowKind::Lookup, 2));
    // MIN_T 5 (fast-hip: --spec 4, so no MTP window has 5 rows): a pure size split there
    const DeferAutoConfig on5{true, 5};
    CHECK(!defer_window_on(on5, DeferWindowKind::Lookup, 4));
    CHECK(defer_window_on(on5, DeferWindowKind::Lookup, 5));
    CHECK(defer_window_on(on5, DeferWindowKind::Lookup, 12));
    CHECK(!defer_window_on(on5, DeferWindowKind::Model, 12));
    // MIN_T 256: no window defers (the "closed" arm of the measurement plan: the deferral's graphs, every row on time)
    const DeferAutoConfig closed{true, kDeferAutoMaxMinT};
    for (DeferWindowKind kind : kAllKinds)
        for (int rows = 1; rows <= 255; ++rows) CHECK(!defer_window_on(closed, kind, rows));
}

void test_staged_top() {
    // auto off: whatever defer_top_now() gave, unchanged, whether or not a caller marked the window
    for (int top : {0, 1, 2, 3, 10, 63})
        for (bool marked : {false, true}) CHECK(defer_staged_top(false, marked, top, 10) == top);
    // auto on: a marked (deferring) window keeps it; an unmarked one stages k
    for (int top : {0, 1, 2, 10}) {
        CHECK(defer_staged_top(true, true, top, 10) == top);
        CHECK(defer_staged_top(true, false, top, 10) == 10);
        CHECK(defer_staged_top(true, false, top, 8) == 8);
    }
    // no k to say "every entry" with: as configured
    CHECK(defer_staged_top(true, false, 2, 0) == 2);
    CHECK(defer_staged_top(true, false, 0, -1) == 0);
}

// The claim the gate rests on: a window staged at top = k keeps every CPU entry on time.  defer_heavy_entry is the rule the
// host pool applies (expert_source.cpp) and copy_rows_by_weight_kernel repeats on the GPU (verify_kernels.cu): with top = k
// every entry's rank (0..k-1) is below top, so every entry is heavy, whatever the weights and the BELOW threshold.
void test_all_heavy_at_k() {
    std::mt19937 rng(1711);
    std::lognormal_distribution<float> logit(0.0f, 1.5f);
    int windows = 0;
    for (int k : {1, 2, 8, 10, 16}) {
        for (int rep = 0; rep < 200; ++rep) {
            const int n_tok = 1 + (int) (rng() % 12);
            std::vector<float> w((size_t) n_tok * k);
            for (int t = 0; t < n_tok; ++t) {
                float sum = 0.0f;
                for (int j = 0; j < k; ++j) sum += (w[(size_t) (t * k + j)] = logit(rng));
                for (int j = 0; j < k; ++j) w[(size_t) (t * k + j)] /= sum;
                if (rep % 5 == 0)   // ties: the rank's tie-break by index must still put every entry below k
                    for (int j = 0; j < k; ++j) w[(size_t) (t * k + j)] = 1.0f / (float) k;
                if (rep % 7 == 0 && k >= 2) w[(size_t) (t * k + 1)] = 0.0f;   // a zero weight is still among the top k
            }
            const int top = defer_staged_top(true, false, /*top_now=*/0, k);
            CHECK(top == k);
            for (float heavy : {0.1f, 0.5f, 0.99f, 2.0f}) {
                bool all = true;
                for (int64_t i = 0; i < (int64_t) w.size(); ++i) all = all && defer_heavy_entry(w.data(), i, k, heavy, top);
                CHECK(all);
                // and without the gate (top_now 0, BELOW 0.5) some entries are light, so the gate is what changed
                if (heavy == 0.5f && k >= 2 && rep % 5 != 0) {
                    bool some_light = false;
                    for (int64_t i = 0; i < (int64_t) w.size(); ++i)
                        some_light = some_light || !defer_heavy_entry(w.data(), i, k, heavy, 0);
                    CHECK(some_light);
                }
            }
            ++windows;
        }
    }
    CHECK(windows == 5 * 200);
}

// The host pool's split with every job heavy: one pass, the partial flag after all of them, the jobs in their own order
// (with or without STRATA_VERIFY_DEFER_CPU_FOLD), so the late addition finds no light row.
void test_pool_one_pass() {
    std::vector<Job> scratch;
    for (int njobs : {0, 1, 3, 17, 64}) {
        for (int fold : {0, 2, 100}) {
            std::vector<Job> jobs((size_t) njobs);
            for (int j = 0; j < njobs; ++j) jobs[(size_t) j].expert = 1000 + j;
            bool heavy_job[128] = {};
            for (int j = 0; j < njobs; ++j) heavy_job[j] = true;   // every entry heavy (test_all_heavy_at_k): every job too
            const int nfirst = defer_order_jobs(jobs.data(), heavy_job, njobs, fold, scratch);
            CHECK(nfirst == njobs);
            bool same_order = true;
            for (int j = 0; j < njobs; ++j) same_order = same_order && jobs[(size_t) j].expert == 1000 + j;
            CHECK(same_order);
        }
    }
}

}  // namespace

int main() {
    test_parse();
    test_kind();
    test_window_on();
    test_staged_top();
    test_all_heavy_at_k();
    test_pool_one_pass();
    if (failed != 0) {
        std::fprintf(stderr, "defer_auto_test: %d of %d checks FAILED\n", failed, checks);
        return 1;
    }
    std::printf("defer_auto_test: %d checks passed\n", checks);
    return 0;
}
