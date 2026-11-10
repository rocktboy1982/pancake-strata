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
    std::printf("message_boundary_test OK (%d checks)\n", checks);
}
