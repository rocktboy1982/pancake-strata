// MOD_PLAN.md section 6 item 42: STRATA_CKPT_STRIDE's rule (include/strata/program/ckpt_stride.hpp).
#include "strata/program/ckpt_stride.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace strata::program;

static int checks = 0;
static void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

// the positions a request that resumed at `resume` checkpoints at, reading chunks of `chunk` up to `total`
static std::vector<int64_t> positions(const CkptStride& c, int64_t resume, int64_t total, int64_t chunk, int64_t every) {
    std::vector<int64_t> at;
    int64_t next = ckpt_next_check(c, resume, resume, every);
    for (int64_t done = resume + chunk;; done += chunk) {
        if (done > total) done = total;
        if (done >= next) {
            at.push_back(done);
            next = ckpt_next_check(c, resume, done, every);
        }
        if (done == total) break;
    }
    return at;
}

int main() {
    const int64_t INF = INT64_MAX;
    CkptStride off = ckpt_stride_from_env(nullptr, nullptr);
    check(off.stride == 0 && off.max_count == 3, "unset: off, max 3");
    check(ckpt_stride_from_env("", "").stride == 0, "empty: off");
    check(ckpt_stride_from_env("-5", nullptr).stride == 0, "negative: off");
    check(ckpt_stride_from_env("4096", "5").stride == 4096 && ckpt_stride_from_env("4096", "5").max_count == 5, "parsed");
    // unset: exactly done + every, and every 0 is INT64_MAX (byte-identical to the old rule)
    check(ckpt_next_check(off, 1000, 1000, 16384) == 17384, "off: resume + every");
    check(ckpt_next_check(off, 1000, 17400, 16384) == 33784, "off: done + every");
    check(ckpt_next_check(off, 0, 0, 0) == INF, "every 0: none");
    CkptStride s = ckpt_stride_from_env("4096", nullptr);
    check(ckpt_next_check(s, 0, 0, 0) == INF, "stride never turns periodic checkpoints on");
    check(ckpt_next_check(s, 1000, 1000, 16384) == 5096, "stride: first at resume + N");
    // 40K-token prompt resumed at 1000, 2048-token chunks: three checkpoints then none
    const auto p = positions(s, 1000, 40000, 2048, 16384);
    check(p.size() == 3, "stride 4096: three checkpoints (max 3)");
    check(p[0] >= 5096 && p[0] < 5096 + 2048, "first is the first chunk end at or past resume + N");
    check(p[1] - p[0] >= 4096 && p[2] - p[1] >= 4096, "each at least N past the one before");
    // off: the old 16384 spacing, 2 checkpoints in 40K
    check(positions(off, 1000, 40000, 2048, 16384).size() == 2, "old rule: every 16384");
    // max 1
    check(positions(ckpt_stride_from_env("4096", "1"), 0, 40000, 2048, 16384).size() == 1, "max 1");
    check(positions(ckpt_stride_from_env("4096", "0"), 0, 40000, 2048, 16384).empty(), "max 0: none");
    // short prompt: nothing past the stride
    check(positions(s, 0, 3000, 1024, 16384).empty(), "prompt shorter than the stride: none");
    // fuzz regression (local fuzz): a stride or every near INT64_MAX saturates to "none" instead of overflowing done + stride
    {
        const int64_t none = INT64_MAX;
        check(ckpt_next_check(ckpt_stride_from_env("9223372036854775807", nullptr), 0, 1000, 16384) == none, "huge stride: none, no overflow");
        check(ckpt_next_check(off, 0, 1000, INT64_MAX) == none, "huge every with no stride: none, no overflow");
        check(ckpt_next_check(off, 0, 1000, 16384) == 1000 + 16384, "old rule unchanged");
    }
    std::printf("ckpt_stride_test:%d checks passed\n", checks);
    return 0;
}
