// epoch_bandit_test: MOD_PLAN A44, the STRATA_BANDIT epoch bandit (include/strata/core/epoch_bandit.hpp).  Host only,
// deterministic: the parse, the arms, the epochs (skip, cost, decay), both rules, the window typer, and a simulated mixed
// session (chat windows where the rank guard costs 7%, copy windows where it saves 17%) in which the bandit must beat
// every fixed arm.
#include "strata/core/epoch_bandit.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>

static int g_fail = 0;
#define CHECK(c)                                                                                  \
    do {                                                                                          \
        if (!(c)) { std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

using namespace strata::core;

using Env = std::map<std::string, std::string>;
static BanditConfig cfg_of(const Env& env, uint32_t available, std::string* msg = nullptr) {
    return bandit_config([&](const char* n) -> const char* {
        auto it = env.find(n);
        return it == env.end() ? nullptr : it->second.c_str();
    }, available, msg);
}

// a deterministic noise source for the simulations (independent of the bandit's own stream)
struct Lcg {
    uint64_t s;
    double next() {   // uniform in [0, 1)
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return (double) (s >> 11) * (1.0 / 9007199254740992.0);
    }
};

// one epoch of `n` windows of type t, each `ms_per_tok` (+-noise) per token, 3 tokens a window; true when it closed
static bool feed_epoch(EpochBandit& b, int t, int n, double ms_per_tok, Lcg& rng, double noise) {
    bool closed = false;
    for (int i = 0; i < n; ++i) closed = b.end(t, 3.0 * ms_per_tok * (1.0 + noise * (2.0 * rng.next() - 1.0)), 3) || closed;
    return closed;
}

static void test_parse() {
    std::string msg;
    BanditConfig c = cfg_of({}, kBanditAll);
    CHECK(!c.on());
    CHECK(c.switches == 0);

    c = cfg_of({{"STRATA_BANDIT", "top,kmiss"}}, kBanditAll);
    CHECK(c.on());
    CHECK(c.switches == (kBanditTop | kBanditKmiss));
    CHECK(c.epoch == 16 && c.skip == 2 && !c.ucb && c.eps == 0.1 && c.decay == 0.9 && c.seed == 1);
    CHECK(c.type_mode == kTypeReq && c.type_arg == 0.30);

    // a switch whose own env var is off is dropped and said
    msg.clear();
    c = cfg_of({{"STRATA_BANDIT", "top,kmiss,preread"}}, kBanditKmiss, &msg);
    CHECK(c.switches == kBanditKmiss);
    CHECK(msg.find("top dropped") != std::string::npos);
    CHECK(msg.find("preread dropped") != std::string::npos);

    // the exactness limit: 1 drops the lossy rank guard, 2 the rounding k-th miss too (said as the reason, before
    // availability: verify.cpp keeps no rank-guard word then, so top is also unavailable)
    msg.clear();
    c = cfg_of({{"STRATA_BANDIT", "all"}, {"STRATA_BANDIT_EXACT", "1"}}, kBanditKmiss | kBanditPreread, &msg);
    CHECK(c.exact == 1);
    CHECK(c.switches == (kBanditKmiss | kBanditPreread));
    CHECK(msg.find("top dropped: STRATA_BANDIT_EXACT=1 (lossy)") != std::string::npos);
    c = cfg_of({{"STRATA_BANDIT", "all"}, {"STRATA_BANDIT_EXACT", "1"}}, kBanditAll);
    CHECK(c.switches == (kBanditKmiss | kBanditPreread));
    c = cfg_of({{"STRATA_BANDIT", "all"}, {"STRATA_BANDIT_EXACT", "2"}}, kBanditAll);
    CHECK(c.switches == kBanditPreread);
    c = cfg_of({{"STRATA_BANDIT", "top"}, {"STRATA_BANDIT_EXACT", "1"}}, kBanditAll);
    CHECK(!c.on());

    // unknown names, bad numbers: said, defaults kept
    msg.clear();
    c = cfg_of({{"STRATA_BANDIT", "rank,bogus"}, {"STRATA_BANDIT_EPOCH", "1"}, {"STRATA_BANDIT_DECAY", "2"},
                {"STRATA_BANDIT_RULE", "greedy"}, {"STRATA_BANDIT_TYPE", "weird"}}, kBanditAll, &msg);
    CHECK(c.switches == kBanditTop);
    CHECK(c.epoch == 16 && c.decay == 0.9 && !c.ucb && c.type_mode == kTypeReq);
    CHECK(msg.find("bogus") != std::string::npos);
    CHECK(msg.find("STRATA_BANDIT_EPOCH") != std::string::npos);
    CHECK(msg.find("STRATA_BANDIT_RULE") != std::string::npos);
    CHECK(msg.find("STRATA_BANDIT_TYPE") != std::string::npos);

    c = cfg_of({{"STRATA_BANDIT", "pre"}, {"STRATA_BANDIT_EPOCH", "8"}, {"STRATA_BANDIT_SKIP", "3"},
                {"STRATA_BANDIT_RULE", "ucb:0.8"}, {"STRATA_BANDIT_DECAY", "1"}, {"STRATA_BANDIT_SEED", "42"},
                {"STRATA_BANDIT_TYPE", "size:6"}, {"STRATA_BANDIT_LOG", "1"}}, kBanditAll);
    CHECK(c.switches == kBanditPreread);
    CHECK(c.epoch == 8 && c.skip == 3 && c.ucb && c.ucb_c == 0.8 && c.decay == 1.0 && c.seed == 42 && c.log);
    CHECK(c.type_mode == kTypeSize && c.type_arg == 6.0);
    c = cfg_of({{"STRATA_BANDIT", "top"}, {"STRATA_BANDIT_RULE", "eps:0.25"}, {"STRATA_BANDIT_TYPE", "req:50"}}, kBanditAll);
    CHECK(!c.ucb && c.eps == 0.25 && c.type_mode == kTypeReq && c.type_arg == 0.5);
    c = cfg_of({{"STRATA_BANDIT", "top"}, {"STRATA_BANDIT_TYPE", "src"}}, kBanditAll);
    CHECK(c.type_mode == kTypeSrc);
    c = cfg_of({{"STRATA_BANDIT", "top"}, {"STRATA_BANDIT_TYPE", "one"}}, kBanditAll);
    CHECK(c.type_mode == kTypeOne);
    c = cfg_of({{"STRATA_BANDIT", "top"}, {"STRATA_BANDIT_TYPE", "ctx"}}, kBanditAll);
    CHECK(c.type_mode == kTypeCtx && c.type_arg == 4096.0);
    c = cfg_of({{"STRATA_BANDIT", "off"}}, kBanditAll);
    CHECK(!c.on());
}

static void test_names() {
    CHECK(bandit_arm_name(0) == "off");
    CHECK(bandit_arm_name(kBanditTop) == "top");
    CHECK(bandit_arm_name(kBanditTop | kBanditKmiss) == "top+kmiss");
    CHECK(bandit_arm_name(kBanditAll) == "top+kmiss+preread");
    CHECK(bandit_arm_class(0) == 0);
    CHECK(bandit_arm_class(kBanditPreread) == 0);
    CHECK(bandit_arm_class(kBanditKmiss | kBanditPreread) == 1);
    CHECK(bandit_arm_class(kBanditTop | kBanditPreread) == 2);
}

static void test_arms_and_epochs() {
    BanditConfig c = cfg_of({{"STRATA_BANDIT", "top,preread"}, {"STRATA_BANDIT_EPOCH", "4"}, {"STRATA_BANDIT_SKIP", "1"},
                             {"STRATA_BANDIT_DECAY", "1"}}, kBanditAll);
    EpochBandit b(c, 2);
    CHECK(b.on());
    CHECK(b.n_arms() == 4);
    CHECK(b.arm(0) == 0 && b.arm(1) == kBanditTop && b.arm(2) == kBanditPreread && b.arm(3) == (kBanditTop | kBanditPreread));
    CHECK(b.begin(0) == 0 && b.begin(1) == 0);   // "off" first

    // the skipped window's cost (huge) is not counted; the epoch closes after 1 + 4 windows
    CHECK(!b.end(0, 1000.0, 1));
    CHECK(!b.end(0, 4.0, 2));
    CHECK(!b.end(0, 4.0, 2));
    CHECK(!b.end(0, 0.0, 2));    // no time: not counted, not even seen
    CHECK(!b.end(0, 4.0, 0));    // no token: the same
    CHECK(!b.end(0, 4.0, 2));
    CHECK(b.end(0, 8.0, 2));
    EpochBandit::Decision d = b.last_decision();
    CHECK(d.type == 0 && d.epoch == 1 && d.arm == 0 && d.windows == 4);
    CHECK(std::fabs(d.cost - 20.0 / 8.0) < 1e-12);
    CHECK(std::strcmp(d.why, "untried") == 0 && d.next == kBanditTop);
    CHECK(b.begin(0) == kBanditTop);
    CHECK(b.begin(1) == 0);       // the other type has its own epochs
    CHECK(b.windows(0, 0) == 5);  // windows run with "off": the skipped one included
    CHECK(b.pulls(0, 0) == 1 && b.epochs(0) == 1 && b.epochs(1) == 0);

    // the remaining arms once each, in order, then the best (decay 1: plain means)
    Lcg rng{7};
    const double cost[4] = {2.5, 2.0, 3.0, 2.2};
    for (int i = 1; i < 4; ++i) {
        CHECK(b.begin(0) == b.arm(i));
        CHECK(feed_epoch(b, 0, 5, cost[i], rng, 0.0));
    }
    CHECK(b.pulls(0, 1) == 1 && b.pulls(0, 2) == 1 && b.pulls(0, 3) == 1);
    CHECK(std::fabs(b.mean(0, 1) - 2.0) < 1e-9);
    const char* why = b.last_decision().why;
    CHECK(std::strcmp(why, "greedy") == 0 || std::strcmp(why, "explore") == 0);
    if (std::strcmp(why, "greedy") == 0) CHECK(b.begin(0) == kBanditTop);
}

static void test_decay() {
    BanditConfig c = cfg_of({{"STRATA_BANDIT", "top"}, {"STRATA_BANDIT_EPOCH", "2"}, {"STRATA_BANDIT_SKIP", "0"},
                             {"STRATA_BANDIT_DECAY", "0.5"}, {"STRATA_BANDIT_RULE", "eps:0"}}, kBanditAll);
    EpochBandit b(c, 1);
    Lcg rng{1};
    CHECK(feed_epoch(b, 0, 2, 1.0, rng, 0.0));   // off: 1.0
    CHECK(feed_epoch(b, 0, 2, 2.0, rng, 0.0));   // top: 2.0 -> greedy picks off
    CHECK(b.begin(0) == 0);
    CHECK(feed_epoch(b, 0, 2, 4.0, rng, 0.0));   // off drifts to 4.0: (1.0*0.25 + 4.0) / (0.25 + 1) = 3.4 > top's 2.0
    CHECK(std::fabs(b.mean(0, 0) - 3.4) < 1e-9);
    CHECK(std::fabs(b.mean(0, 1) - 2.0) < 1e-9);  // a decay scales n and the sum alike: the mean stays
    CHECK(b.begin(0) == kBanditTop);
}

static void test_determinism() {
    auto run = [](uint64_t seed, bool ucb) {
        Env env = {{"STRATA_BANDIT", "top,kmiss"}, {"STRATA_BANDIT_EPOCH", "4"}, {"STRATA_BANDIT_SEED", std::to_string(seed)},
                   {"STRATA_BANDIT_RULE", ucb ? "ucb:0.5" : "eps:0.3"}};
        EpochBandit b(cfg_of(env, kBanditAll), 2);
        Lcg rng{99};
        std::string trace;
        for (int e = 0; e < 200; ++e) {
            const int t = e % 2;
            const uint32_t arm = b.begin(t);
            const double base = t == 0 ? 2.0 : 1.0;
            feed_epoch(b, t, 6, base * (1.0 + 0.05 * (double) arm), rng, 0.1);
            trace += (char) ('a' + b.begin(t));
        }
        return trace;
    };
    CHECK(run(1, false) == run(1, false));
    CHECK(run(1, false) != run(2, false));   // another explorer stream (eps 0.3 over 200 epochs)
    CHECK(run(5, true) == run(5, true));
}

// the A4 picture: the rank guard (top) costs 7% on chat windows and saves 17% on copy windows; k-th miss saves 4% on
// copy and costs 6% on chat.  A session alternates 20 chat epochs and 20 copy epochs.  The bandit must beat every fixed
// arm and come close to the per-type oracle.
static void test_mixed_session(bool ucb) {
    Env env = {{"STRATA_BANDIT", "top,kmiss"}, {"STRATA_BANDIT_EPOCH", "16"}, {"STRATA_BANDIT_SKIP", "2"},
               {"STRATA_BANDIT_SEED", "3"}};
    if (ucb) env["STRATA_BANDIT_RULE"] = "ucb";   // the default c = 0.1
    const BanditConfig c = cfg_of(env, kBanditAll);
    auto cost = [](int t, uint32_t arm) {   // ms per committed token
        double x = t == 0 ? 30.0 : 12.0;
        if (arm & kBanditTop) x *= t == 0 ? 1.07 : 0.83;
        if (arm & kBanditKmiss) x *= t == 0 ? 1.06 : 0.96;
        return x;
    };
    const int kWindows = 16 + 2, kRounds = 40;
    double fixed[4] = {0, 0, 0, 0}, oracle = 0.0, bandit = 0.0;
    EpochBandit b(c, 2);
    Lcg rng{11};
    for (int r = 0; r < kRounds; ++r)
        for (int t = 0; t < 2; ++t)
            for (int e = 0; e < 20; ++e)
                for (int w = 0; w < kWindows; ++w) {
                    const double noise = 1.0 + 0.15 * (2.0 * rng.next() - 1.0);   // window-to-window jitter
                    for (int a = 0; a < 4; ++a) fixed[a] += cost(t, b.arm(a)) * noise;
                    oracle += (t == 0 ? cost(0, 0) : cost(1, kBanditTop | kBanditKmiss)) * noise;
                    const uint32_t arm = b.begin(t);
                    const double ms = 3.0 * cost(t, arm) * noise;
                    bandit += ms / 3.0;
                    b.end(t, ms, 3);
                }
    double best_fixed = fixed[0];
    for (double f : fixed) best_fixed = std::min(best_fixed, f);
    std::printf("mixed session (%s): bandit %.4f of the best fixed arm, %.4f of the oracle; most pulled chat=%s copy=%s\n",
                ucb ? "ucb" : "eps", bandit / best_fixed, bandit / oracle,
                bandit_arm_name(b.arm([&] { int k = 0; for (int i = 1; i < 4; ++i) if (b.pulls(0, i) > b.pulls(0, k)) k = i; return k; }())).c_str(),
                bandit_arm_name(b.arm([&] { int k = 0; for (int i = 1; i < 4; ++i) if (b.pulls(1, i) > b.pulls(1, k)) k = i; return k; }())).c_str());
    CHECK(bandit < 0.97 * best_fixed);    // the best fixed arm (top+kmiss or off) loses on one of the two types
    CHECK(bandit < 1.03 * oracle);        // exploration costs a few percent at most
    // the most pulled arm of each type is its best
    int k0 = 0, k1 = 0;
    for (int i = 1; i < 4; ++i) {
        if (b.pulls(0, i) > b.pulls(0, k0)) k0 = i;
        if (b.pulls(1, i) > b.pulls(1, k1)) k1 = i;
    }
    CHECK(b.arm(k0) == 0);
    CHECK(b.arm(k1) == (kBanditTop | kBanditKmiss));
}

static void test_typer() {
    BanditTyper req(kTypeReq, 0.30);
    CHECK(req.types() == 2);
    req.begin_request();
    CHECK(req.type(true, 12, 100) == 0);   // a request starts as chat
    for (int i = 0; i < 3; ++i) req.note(true);
    CHECK(req.share() > 0.30);              // 1 - (7/8)^3 = 0.33
    CHECK(req.type(false, 3, 100) == 1);
    for (int i = 0; i < 2; ++i) req.note(false);
    CHECK(req.type(false, 3, 100) == 0);    // 0.33 * (7/8)^2 = 0.25
    req.begin_request();
    CHECK(req.share() == 0.0);
    CHECK(std::strcmp(req.name(0), "chat") == 0 && std::strcmp(req.name(1), "copy") == 0);

    BanditTyper src(kTypeSrc, 0);
    CHECK(src.type(true, 2, 0) == 1 && src.type(false, 12, 0) == 0);
    BanditTyper size(kTypeSize, 4);
    CHECK(size.type(true, 4, 0) == 0 && size.type(false, 5, 0) == 1);
    BanditTyper ctx(kTypeCtx, 4096);
    CHECK(ctx.type(false, 1, 4095) == 0 && ctx.type(false, 1, 4096) == 1);
    BanditTyper one(kTypeOne, 0);
    CHECK(one.types() == 1 && one.type(true, 16, 1 << 20) == 0);
}

static void test_summary() {
    BanditConfig c = cfg_of({{"STRATA_BANDIT", "top"}, {"STRATA_BANDIT_EPOCH", "2"}, {"STRATA_BANDIT_SKIP", "0"}}, kBanditAll);
    EpochBandit b(c, 2);
    BanditTyper ty(kTypeReq, 0.3);
    CHECK(b.request_summary(ty) == "none");
    b.end(0, 1.0, 1);
    b.end(0, 1.0, 1);   // closes: next arm top (untried)
    b.end(0, 1.0, 1);
    b.end(1, 1.0, 1);
    CHECK(b.request_summary(ty) == "chat:off*2,top*1/copy:off*1");
    b.begin_request();
    CHECK(b.request_summary(ty) == "none");
    CHECK(b.windows(0, 0) == 2 && b.windows(0, 1) == 1);   // the totals stay
}

int main() {
    test_parse();
    test_names();
    test_arms_and_epochs();
    test_decay();
    test_determinism();
    test_mixed_session(false);
    test_mixed_session(true);
    test_typer();
    test_summary();
    if (g_fail == 0) std::printf("epoch_bandit_test: all passed\n");
    return g_fail == 0 ? 0 : 1;
}
