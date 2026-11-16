#include "strata/program/message_boundary.hpp"

#include <cstdio>

int main() {
    using strata::program::message_checkpoint_boundary;
    constexpr int64_t token = 248045;
    std::vector<int64_t> ids(33100, 7);
    ids[0] = ids[12] = ids[32800] = ids[33000] = token;
    int checks = 0;
    auto check = [&](int64_t resume, int64_t turn, int64_t expected) {
        ++checks;
        return message_checkpoint_boundary(ids, resume, turn, token) == expected;
    };
    if (!check(0, 33000, 32800) || !check(22016, 33000, 32800) ||
        !check(32800, 33000, -1) || !check(33000, 33000, -1) ||
        !check(-1, 33000, -1) || !check(0, -1, -1) ||
        !check(0, 33100, -1) || !check(0, 32999, -1) ||
        !check(0, 12, -1) || !check(0, 32800, -1)) return 1;
    for (int tail : {1, 63, 64, 512, 1024, 1025, 2048}) {
        std::vector<int64_t> fixture(8192 + tail + 1, 9);
        fixture[8192] = fixture[8192 + tail] = token;
        if (message_checkpoint_boundary(fixture, 0, 8192 + tail, token) !=
            (tail <= 1024 ? 8192 : -1)) return 1;
        ++checks;
    }
    if (message_checkpoint_boundary({}, 0, 0, token) != -1 ||
        message_checkpoint_boundary(ids, 0, 33000, -1) != -1 ||
        message_checkpoint_boundary(ids, 0, 33000, token, 8192, 0) != -1 ||
        message_checkpoint_boundary(ids, 0, 33000, token, -1) != -1) return 1;
    checks += 4;
    {   // MOD_PLAN 16: tool_checkpoint_marks
        using strata::program::tool_checkpoint_marks;
        using V = std::vector<int64_t>;
        auto same = [&](const V& got, const V& want) { ++checks; return got == want; };
        // a 30K prompt resumed at 20,002 (the live state past the assistant's call); the call's end at 20,004, the
        // result's end (the turn) at 29,000; the kept chain 3,000 (root) and 20,000 (the previous turn)
        const V held = {3000, 20000, 29000};
        if (!same(tool_checkpoint_marks({20004, 29000}, 20002, 30000, held, 1024, 64, 2), {}) ||   // 4 past a checkpoint
            !same(tool_checkpoint_marks({20004, 29000}, 20002, 30000, held, 0, 64, 2), {20004}) ||
            !same(tool_checkpoint_marks({20004, 29000}, 20002, 30000, {3000, 29000}, 1024, 64, 2), {20004}) ||
            !same(tool_checkpoint_marks({20004}, 20004, 30000, {3000}, 1024, 64, 2), {}) ||          // not past resume
            !same(tool_checkpoint_marks({29990}, 0, 30000, {}, 1024, 64, 2), {}) ||                   // 9 before the end
            !same(tool_checkpoint_marks({29999, 30000, -5}, 0, 30000, {}, 0, 0, 2), {}) ||          // junk
            !same(tool_checkpoint_marks({12000, 9000, 6000, 12000}, 0, 30000, {}, 1024, 64, 2), {9000, 12000}) ||
            !same(tool_checkpoint_marks({12000, 12500, 6000}, 0, 30000, {}, 1024, 64, 3), {6000, 12500}) ||   // 12,000: 500 below a taken one
            !same(tool_checkpoint_marks({6000}, 0, 30000, {}, 1024, 64, 0), {}) ||
            !same(tool_checkpoint_marks({6000}, 0, 30000, {6000}, 0, 0, 2), {}))                      // already held
            return 1;
    }
    {   // STRATA_CACHE_FORK_POINT (LEADS.md FK-27, Lamina ab808b133e): fork_checkpoint_points
        using strata::program::fork_checkpoint_points;
        using A = std::array<int64_t, 2>;
        constexpr int64_t im_end = 248046;
        auto same = [&](const A& got, const A& want) { ++checks; return got == want; };
        // a 6,100-token shared context, then two different questions: [im_end \n im_start] user ... im_end \n im_start(turn)
        auto prompt = [&](int64_t tail_tok) {
            std::vector<int64_t> v(6100, 5);
            v.push_back(im_end); v.push_back(10); v.push_back(token);   // the context's message ends at 6100
            for (int i = 0; i < 40; ++i) v.push_back(tail_tok);        // the question
            v.push_back(im_end); v.push_back(10); v.push_back(token); v.push_back(77);   // the reply's turn header
            return v;
        };
        const auto a = prompt(31), b = prompt(32);
        const int64_t turn_b = (int64_t) b.size() - 2, end_b = turn_b - 2;   // the turn token, the question's im_end
        // the second question parts from the first at 6103 (the first question token); the message end 4 before its im_end
        if (!same(fork_checkpoint_points(a, b, 0, turn_b, im_end), A{6103, end_b - 4}) ||
            // a resume past the fork (by min_gap): no fork point
            !same(fork_checkpoint_points(a, b, 6000, turn_b, im_end), A{-1, -1}) ||
            // no previous prompt: the message end alone
            !same(fork_checkpoint_points({}, b, 0, turn_b, im_end), A{-1, end_b - 4}) ||
            // the same prompt again: it does not part before its last token
            !same(fork_checkpoint_points(b, b, 0, turn_b, im_end), A{-1, end_b - 4}) ||
            // no im_end token known: the fork alone
            !same(fork_checkpoint_points(a, b, 0, turn_b, -1), A{6103, -1}) ||
            // a smaller gap rule still needs the point past resume
            !same(fork_checkpoint_points(a, b, 6102, turn_b, im_end, 0), A{6103, end_b - 4}) ||
            !same(fork_checkpoint_points(a, b, 6103, turn_b, im_end, 0), A{-1, end_b - 4}) ||
            // no turn given: searched below the prompt's end
            !same(fork_checkpoint_points(a, b, 0, -1, im_end), A{6103, end_b - 4}) ||
            !same(fork_checkpoint_points(a, {}, 0, -1, im_end), A{-1, -1}))
            return 1;
    }
    std::printf("message_boundary_test OK (%d checks)\n", checks);
}
