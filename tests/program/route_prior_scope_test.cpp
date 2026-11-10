// MOD_PLAN F13: the route prior's scope gate (STRATA_ROUTE_PRIOR_SCOPE): which windows may take the residency bonus.  No GPU, no model.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "strata/program/route_prior_scope.hpp"

using namespace strata::program;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

namespace {
constexpr int32_t kStart = 248045, kEnd = 248046, kTool = 248058, kEndTool = 248059, kThink = 248068, kEndThink = 248069,
                  kFence = 71093, kFenceSp = 52451, kA = 11, kB = 12, kC = 13;

bool open(const PriorScopeState& s, PriorScope sc, std::vector<int32_t> w) { return s.window_open(sc, w.data(), (int) w.size()); }
}  // namespace

int main() {
    // the scope words
    bool known = true;
    CHECK(parse_prior_scope(nullptr, &known) == PriorScope::Everywhere && known);
    CHECK(parse_prior_scope("", &known) == PriorScope::Everywhere && known);
    CHECK(parse_prior_scope("answer") == PriorScope::Answer);
    CHECK(parse_prior_scope("think") == PriorScope::Think);
    CHECK(parse_prior_scope("all") == PriorScope::All);
    CHECK(parse_prior_scope("answr", &known) == PriorScope::Everywhere && !known);

    // a prompt ending in the assistant's opened thinking: the state is read from the LAST turn only
    {
        std::vector<int32_t> ids = {kStart, kA, kFence, kTool, kB, kStart, kA, kThink, kB};   // history has an unclosed fence and call
        PriorScopeState s;
        s.sync(ids.data(), ids.size());
        CHECK(s.think() && !s.tool() && !s.fence());
        // thinking: answer and think scopes are open, a plain window
        CHECK(open(s, PriorScope::Answer, {kC, kA, kB}));
        CHECK(open(s, PriorScope::Think, {kC, kA, kB}));
        // a boundary anywhere in the window (head or draft) closes it
        CHECK(!open(s, PriorScope::Answer, {kC, kEndThink, kB}));
        CHECK(!open(s, PriorScope::Think, {kC, kA, kTool}));
        CHECK(!open(s, PriorScope::Answer, {kFence, kA}));
        CHECK(!open(s, PriorScope::Answer, {kC, kEnd}));
        // `all` ignores the boundaries
        CHECK(open(s, PriorScope::All, {kC, kEndThink, kTool}));
    }
    // incremental sync reads only what was appended; the state follows think -> answer -> tool call -> fence
    {
        std::vector<int32_t> ids = {kStart, kA, kThink};
        PriorScopeState s;
        s.sync(ids.data(), ids.size());
        CHECK(s.think());
        ids.push_back(kB); ids.push_back(kEndThink);
        s.sync(ids.data(), ids.size());
        CHECK(!s.think());
        CHECK(open(s, PriorScope::Answer, {kA, kB}));      // the reply's prose
        CHECK(!open(s, PriorScope::Think, {kA, kB}));      // not thinking
        ids.push_back(kTool);
        s.sync(ids.data(), ids.size());
        CHECK(s.tool());
        CHECK(!open(s, PriorScope::Answer, {kA, kB}));     // inside the call: exact
        ids.push_back(kFence);                             // a ``` inside a call is data
        s.sync(ids.data(), ids.size());
        CHECK(s.tool() && !s.fence());
        ids.push_back(kEndTool);
        s.sync(ids.data(), ids.size());
        CHECK(!s.tool() && open(s, PriorScope::Answer, {kA}));
        ids.push_back(kFenceSp);                           // the " ```" token opens a code fence
        s.sync(ids.data(), ids.size());
        CHECK(s.fence() && !open(s, PriorScope::Answer, {kA, kB}));
        ids.push_back(kFence);
        s.sync(ids.data(), ids.size());
        CHECK(!s.fence() && open(s, PriorScope::Answer, {kA, kB}));
    }
    // the head token counts: a window whose head is <tool_call> is computed exactly (its state is "inside"), one whose head is
    // </tool_call> is a boundary too
    {
        std::vector<int32_t> ids = {kStart, kA, kEndThink, kB};
        PriorScopeState s;
        s.sync(ids.data(), ids.size());
        CHECK(!open(s, PriorScope::Answer, {kTool, kA}));
        CHECK(!open(s, PriorScope::Answer, {kEndTool, kA}));
        CHECK(open(s, PriorScope::Answer, {kA, kB, kC}));
        CHECK(!open(s, PriorScope::Answer, {}));
    }
    // a fence inside the thinking is closed by </think>; a rewind (a shorter history) starts again
    {
        std::vector<int32_t> ids = {kStart, kThink, kFence, kA};
        PriorScopeState s;
        s.sync(ids.data(), ids.size());
        CHECK(s.think() && s.fence());
        CHECK(!open(s, PriorScope::Answer, {kA}));
        ids.push_back(kEndThink);
        s.sync(ids.data(), ids.size());
        CHECK(!s.fence() && !s.think());
        std::vector<int32_t> shorter = {kStart, kThink};
        s.sync(shorter.data(), shorter.size());
        CHECK(s.think() && !s.fence());
    }
    // no turn marker in the history: read from the start
    {
        std::vector<int32_t> ids = {kA, kThink, kB};
        PriorScopeState s;
        s.sync(ids.data(), ids.size());
        CHECK(s.think());
    }
    // the marker spec
    {
        PriorMarks m;
        std::string err;
        CHECK(parse_prior_marks(nullptr, m, err) && m.tool_open == 248058);
        CHECK(parse_prior_marks("think=1,endthink=2,fence=10:11:12", m, err));
        CHECK(m.think_open == 1 && m.think_close == 2 && m.n_fence == 3 && m.fence[2] == 12 && m.tool_open == 248058);
        PriorScopeState s(m);
        std::vector<int32_t> ids = {kStart, 1};
        s.sync(ids.data(), ids.size());
        CHECK(s.think());
        PriorMarks keep = m;
        CHECK(!parse_prior_marks("think=x", m, err) && m.think_open == keep.think_open);
        CHECK(!parse_prior_marks("nope=3", m, err));
        CHECK(!parse_prior_marks("fence=1:2:3:4:5", m, err));
        CHECK(!parse_prior_marks("think", m, err));
    }
    if (failed == 0) std::printf("route_prior_scope_test: ok\n");
    return failed == 0 ? 0 : 1;
}
