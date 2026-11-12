#pragma once
// MOD_PLAN F13 (route-prior guards for turbo), the scope half: where in a reply may STRATA_ROUTE_PRIOR's residency bonus
// act?  zsariboga/Strata 8d7042ad97 (STRATA_ROUTE_PRIOR_SCOPE), adapted: the serve loop feeds every committed token to a
// small state machine (think / tool call / code fence) and, before each verify window, writes 1 (bonus on) or 0 (the router's
// own top-10 stands) into the mapped float route_prior.cu reads.  Header-only, no GPU, no model: tests/program/
// route_prior_scope_test.cpp.
//
//   STRATA_ROUTE_PRIOR_SCOPE = answer | think | all     (unset: the bonus acts everywhere, as turbo always did)
//     answer  the bonus is on in the reply's prose and in the thinking, never inside <tool_call> ... </tool_call> or a ``` fence
//     think   only inside <think> ... </think> (and not inside a tool call in it)
//     all     everywhere, but still off while the prompt is read (a "gated all": measurement of the prior with the gate in the graph)
//   In every scope except `all`, a window whose rows (the fed-back head or a draft) hold a boundary token (think / tool /
//   fence marker, <|im_end|>) is computed exactly: the state at the row after a marker is not known when the window is built.
//   STRATA_ROUTE_PRIOR_ON_PROMPT=1 keeps the bonus on while the prompt is read (a measurement switch: teacher-forced KL of the
//   bonus itself); the default is off there, so a prompt's K/V is exactly the router's.
//   STRATA_ROUTE_PRIOR_MARKS=name=id[,name=id...]   the marker token ids; the defaults are Qwen 3.8 / Flash-Next's (checked against
//   the tokenizer's GGUF vocabulary: <|im_start|> 248045, <|im_end|> 248046, <tool_call> 248058, </tool_call> 248059, <think> 248068,
//   </think> 248069, and the two tokens that contain ``` : "```" 71093 and " ```" 52451).  Names: im_start, im_end, think, endthink, tool,
//   endtool, fence (fence takes ids joined with ':', up to 4).  Another family: name its ids here.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

namespace strata::program {

enum class PriorScope : int { Everywhere = -1, Think = 0, Answer = 1, All = 2 };

/// The text of STRATA_ROUTE_PRIOR_SCOPE: "answer", "think", "all"; nullptr / "" is Everywhere (no gate).  An unknown word is also
/// Everywhere and `known` (when given) says false, so the caller can warn.
inline PriorScope parse_prior_scope(const char* v, bool* known = nullptr) {
    if (known != nullptr) *known = true;
    if (v == nullptr || v[0] == 0) return PriorScope::Everywhere;
    if (std::strcmp(v, "answer") == 0) return PriorScope::Answer;
    if (std::strcmp(v, "think") == 0) return PriorScope::Think;
    if (std::strcmp(v, "all") == 0) return PriorScope::All;
    if (known != nullptr) *known = false;
    return PriorScope::Everywhere;
}

struct PriorMarks {
    int32_t im_start = 248045, im_end = 248046;
    int32_t think_open = 248068, think_close = 248069;
    int32_t tool_open = 248058, tool_close = 248059;
    int32_t fence[4] = {71093, 52451, -1, -1};
    int n_fence = 2;
};

/// STRATA_ROUTE_PRIOR_MARKS: false with `err` set on a malformed spec (m is then left as it was).
inline bool parse_prior_marks(const char* spec, PriorMarks& m, std::string& err) {
    if (spec == nullptr || spec[0] == 0) return true;
    PriorMarks r = m;
    const std::string s = spec;
    size_t at = 0;
    while (at <= s.size()) {
        size_t end = s.find(',', at);
        if (end == std::string::npos) end = s.size();
        const std::string item = s.substr(at, end - at);
        at = end + 1;
        if (item.empty()) continue;
        const size_t eq = item.find('=');
        if (eq == std::string::npos || eq == 0 || eq + 1 >= item.size()) { err = "'" + item + "' is not name=id"; return false; }
        const std::string name = item.substr(0, eq), val = item.substr(eq + 1);
        auto one = [&](int32_t& out) {
            char* e = nullptr;
            const long v = std::strtol(val.c_str(), &e, 10);
            if (e == val.c_str() || *e != 0 || v < 0 || v > 0x7fffffffL) return false;
            out = (int32_t) v;
            return true;
        };
        bool ok = true;
        if (name == "im_start") ok = one(r.im_start);
        else if (name == "im_end") ok = one(r.im_end);
        else if (name == "think") ok = one(r.think_open);
        else if (name == "endthink") ok = one(r.think_close);
        else if (name == "tool") ok = one(r.tool_open);
        else if (name == "endtool") ok = one(r.tool_close);
        else if (name == "fence") {
            r.n_fence = 0;
            size_t a = 0;
            while (a <= val.size() && ok) {
                size_t b = val.find(':', a);
                if (b == std::string::npos) b = val.size();
                const std::string t = val.substr(a, b - a);
                a = b + 1;
                char* e = nullptr;
                const long v = std::strtol(t.c_str(), &e, 10);
                if (t.empty() || e == t.c_str() || *e != 0 || v < 0 || v > 0x7fffffffL || r.n_fence >= 4) ok = false;
                else r.fence[r.n_fence++] = (int32_t) v;
            }
        } else { err = "unknown name '" + name + "'"; return false; }
        if (!ok) { err = "bad value in '" + item + "'"; return false; }
    }
    m = r;
    return true;
}

/// Reads a reply token by token and says whether it is inside a thinking block, a tool call or a code fence.  The rules are
/// zsariboga's: <think> opens thinking and closes any fence, </think> closes both, <tool_call> / </tool_call> toggle the call, a
/// fence token toggles a fence unless a tool call is open (a ``` inside a call's JSON string is data), <|im_start|> resets all.
class PriorScopeState {
public:
    explicit PriorScopeState(const PriorMarks& m = PriorMarks()) : m_(m) {}

    void reset() { think_ = tool_ = fence_ = false; fed_ = 0; seeded_ = false; }
    bool think() const { return think_; }
    bool tool() const { return tool_; }
    bool fence() const { return fence_; }

    void feed(int32_t id) {
        if (id == m_.im_start) think_ = tool_ = fence_ = false;
        else if (id == m_.think_open) { think_ = true; fence_ = false; }
        else if (id == m_.think_close) { think_ = false; fence_ = false; }
        else if (id == m_.tool_open) tool_ = true;
        else if (id == m_.tool_close) tool_ = false;
        else if (!tool_ && is_fence(id)) fence_ = !fence_;
    }

    /// Brings the state up to `ids[0..n)`, the request's tokens so far (prompt, then every committed token).  The first call reads
    /// the prompt's last turn only (from its last <|im_start|>: earlier turns' fences and calls are history); later calls read just
    /// what was appended.  A shorter `ids` than last time (a rewind) starts again.
    void sync(const int32_t* ids, size_t n) {
        if (!seeded_ || n < fed_) {
            think_ = tool_ = fence_ = false;
            size_t s = 0;
            for (size_t i = n; i > 0; --i)
                if (ids[i - 1] == m_.im_start) { s = i - 1; break; }
            fed_ = s;
            seeded_ = true;
        }
        for (; fed_ < n; ++fed_) feed(ids[fed_]);
    }

    bool is_boundary(int32_t id) const {
        return id == m_.think_open || id == m_.think_close || id == m_.tool_open || id == m_.tool_close || id == m_.im_end ||
               id == m_.im_start || is_fence(id);
    }

    /// May the bonus act in a window of T rows `win` (win[0] is the fed-back head, not yet in the state; the rest are drafts)?
    /// The state includes win[0] (row 0's hidden state is the one that produces the token after it); a boundary token anywhere
    /// in the window closes it (scopes think and answer).
    bool window_open(PriorScope scope, const int32_t* win, int T) const {
        if (scope == PriorScope::All || scope == PriorScope::Everywhere) return true;
        if (T <= 0) return false;
        PriorScopeState a = *this;
        a.feed(win[0]);
        const bool in_scope = scope == PriorScope::Answer ? (!a.tool_ && !a.fence_) : (a.think_ && !a.tool_);
        if (!in_scope) return false;
        for (int t = 0; t < T; ++t)
            if (is_boundary(win[t])) return false;
        return true;
    }

private:
    bool is_fence(int32_t id) const {
        for (int i = 0; i < m_.n_fence; ++i)
            if (id == m_.fence[i]) return true;
        return false;
    }
    PriorMarks m_;
    bool think_ = false, tool_ = false, fence_ = false, seeded_ = false;
    size_t fed_ = 0;
};

}  // namespace strata::program
