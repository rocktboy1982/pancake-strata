#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::program {

inline int64_t message_checkpoint_boundary(const std::vector<int64_t>& ids, int64_t resume,
        int64_t turn_at, int64_t turn_token, int64_t min_prefix = 8192, int64_t max_tail = 1024) {
    if (resume < 0 || turn_token < 0 || turn_at <= resume || turn_at >= int64_t(ids.size()) ||
        ids[std::size_t(turn_at)] != turn_token || min_prefix < 0 || max_tail <= 0) return -1;
    for (int64_t i = turn_at - 1; i > resume; --i) {
        if (turn_at - i > max_tail) return -1;
        if (ids[std::size_t(i)] == turn_token) return i >= min_prefix ? i : -1;
    }
    return -1;
}

/// MOD_PLAN 16 (STRATA_CACHE_TOOL_BOUNDARY): which of the server's `marks=` positions (a Qwen prompt's tool-call and
/// tool-result boundaries, serve/server.py qwen_tool_marks) this request reads to and checkpoints.  A mark counts only
/// inside the part read now (read_from < m < n - 1) and only where it pays: it must save at least `min_gain` tokens over
/// the deepest point below it that already has or gets a checkpoint (`held`: the kept chain's lengths and this request's
/// other checkpointed cuts; a request that leaves the prompt between them would otherwise go back that far), and sit at
/// least `min_room` tokens before the next such point above it (or the prompt's end): one just before another saves
/// almost nothing.  The deepest marks are taken first (a branch near the end is the likely one), at most `max_marks`;
/// a mark below one already taken must leave that one `min_gain` tokens too, or the deeper one would save less.
/// Returned sorted, without duplicates; empty when none qualifies.
inline std::vector<int64_t> tool_checkpoint_marks(std::vector<int64_t> marks, int64_t read_from, int64_t n,
        const std::vector<int64_t>& held, int64_t min_gain, int64_t min_room, int64_t max_marks) {
    std::vector<int64_t> out;
    if (max_marks <= 0 || n < 2 || min_gain < 0 || min_room < 0) return out;
    std::sort(marks.begin(), marks.end());
    marks.erase(std::unique(marks.begin(), marks.end()), marks.end());
    std::vector<int64_t> points;   // the checkpointed points: the held ones and the end
    for (int64_t h : held)
        if (h > 0) points.push_back(h);
    points.push_back(n - 1);
    for (auto it = marks.rbegin(); it != marks.rend() && (int64_t) out.size() < max_marks; ++it) {
        const int64_t m = *it;
        if (m <= read_from || m >= n - 1) continue;
        int64_t floor = 0, ceil = n - 1;
        bool same = false;
        for (int64_t h : points) {
            if (h == m) { same = true; break; }
            if (h < m) floor = std::max(floor, h);
            else ceil = std::min(ceil, h);
        }
        const int64_t above = out.empty() ? INT64_MAX : out.back();   // the shallowest taken so far (all are above m)
        if (same || m - floor < min_gain || ceil - m < min_room || above - m < min_gain) continue;
        out.push_back(m);
    }
    std::sort(out.begin(), out.end());
    return out;
}

} // namespace strata::program
