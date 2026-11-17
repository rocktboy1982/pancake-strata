// LEADS.md Part P lead L1: STRATA_LOOKUP_COPYLANE (include/strata/spec/copy_lane.hpp).  The first five cases are the behaviours of
// hipfire's MtpCopyContext tests (crates/hipfire-runtime/src/ngram_mod.rs, Apache-2.0) re-expressed against this C++ port; the
// parity case compares every proposal with a brute-force reference (the most recent occurrence of the longest key).
#include "strata/spec/copy_lane.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>

using strata::spec::CopyLane;
using strata::spec::CopyLaneSource;
using V = std::vector<int32_t>;

static int checks = 0;
static void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

static CopyLane::Config cfg(size_t cap, int n_match, int n_min, int n_max) {
    CopyLane::Config c;
    c.capacity = cap; c.n_match = n_match; c.n_min = n_min; c.n_max = n_max;
    return c;
}

// the committed history `h` then a proposal asked after `tail` (committed suffix ++ n_pending speculative tokens)
static V ask(CopyLane& l, const V& tail, int n_pending, int max_k, int* key = nullptr) {
    V out((size_t) std::max(max_k, 1));
    int k_len = 0;
    const int k = l.propose_after(tail.data(), (int) tail.size(), n_pending, max_k, out.data(), &k_len);
    if (key != nullptr) *key = k_len;
    out.resize((size_t) k);
    return out;
}

int main() {
    // invalid config and the key ladder
    check(!CopyLane(cfg(3, 2, 1, 4)).ok(), "capacity not a power of two");
    check(!CopyLane(cfg(16, 0, 1, 4)).ok(), "n_match 0");
    check(!CopyLane(cfg(16, 2, 5, 4)).ok(), "n_min > n_max");
    check(CopyLane(cfg(1 << 16, 4, 1, 63)).keys() == std::vector<int>({32, 16, 8, 4}), "keys 32 16 8 4");
    check(CopyLane(cfg(1 << 16, 5, 1, 63)).keys() == std::vector<int>({32, 16, 8, 5}), "keys 32 16 8 5");
    check(CopyLane(cfg(1 << 16, 16, 1, 63)).keys() == std::vector<int>({32, 16}), "keys 32 16");
    check(CopyLane(cfg(1 << 16, 40, 1, 63)).keys() == std::vector<int>({40}), "keys 40");
    check(CopyLane(cfg(1 << 16, 4, 1, 63)).slots_per_key() == (size_t) (1 << 14), "slots per key");
    check(CopyLane(cfg(1 << 16, 12, 1, 63)).slots_per_key() == (size_t) (1 << 14), "three keys round down");

    {   // copies ONE earlier occurrence verbatim: the 8-token key follows the first occurrence of 2..9, not the later 7 8 9
        V prompt;
        for (int i = 1; i <= 16; ++i) prompt.push_back(i);
        for (int t : {50, 51, 52, 53, 54, 7, 8, 9, 90, 91, 92}) prompt.push_back(t);
        CopyLane l(cfg(1 << 16, 3, 1, 8));
        l.begin_request();
        l.append(prompt.data(), prompt.size());
        // history ends ... 90 91 92: ask with the committed suffix 1..9 as the context? the history has no such suffix,
        // so the context is passed as pending tokens (speculative prefix), which is how the engine asks after the MTP drafts
        int key = 0;
        V got = ask(l, V{1, 2, 3, 4, 5, 6, 7, 8, 9}, 9, 8, &key);
        check(got == V({10, 11, 12, 13, 14, 15, 16, 50}) && key == 8, "copy follows the longest key's occurrence");
        got = ask(l, V{99, 7, 8, 9}, 4, 8, &key);
        check(got == V({90, 91, 92}) && key == 3, "only the 3-token key matches: its most recent occurrence, capped by the history end");
    }
    {   // caps, n_min fall-through, the history end
        V prompt;
        for (int i = 100; i < 110; ++i) prompt.push_back(i);
        CopyLane l(cfg(1 << 16, 2, 2, 6));
        l.begin_request();
        l.append(prompt.data(), prompt.size());
        check(ask(l, V{100, 101}, 2, 6) == V({102, 103, 104, 105, 106, 107}), "n_max cap 6");
        check(ask(l, V{100, 101}, 2, 3) == V({102, 103, 104}), "caller cap 3");
        check(ask(l, V{100, 101}, 2, 1).empty(), "caller cap below n_min: none");
        check(ask(l, V{100, 101, 7}, 3, 6).empty(), "unknown suffix: none");
        CopyLane l2(cfg(1 << 16, 2, 1, 6));
        l2.begin_request();
        l2.append(prompt.data(), prompt.size());
        check(ask(l2, V{108, 109}, 2, 6).empty(), "108 109 is the history end: nothing follows");
        CopyLane l3(cfg(1 << 16, 2, 3, 6));
        l3.begin_request();
        l3.append(prompt.data(), prompt.size());
        check(ask(l3, V{105, 106}, 2, 6) == V({107, 108, 109}), "a copy of exactly n_min tokens is kept");
        check(ask(l3, V{107, 108}, 2, 6).empty(), "a copy shorter than n_min falls through and finds nothing");
    }
    {   // committed tokens only: speculative ones are never learned
        CopyLane l(cfg(1 << 16, 2, 1, 4));
        l.begin_request();
        V p = {1, 2, 3};
        l.append(p.data(), p.size());
        check(ask(l, V{3, 1, 2}, 2, 4) == V({3}), "pending 1 2 after committed 3: copies 3");  
        check(l.history() == V({1, 2, 3}), "the proposal did not touch the history");
        V e = {1, 2, 9};
        l.append(e.data(), e.size());
        check(l.history() == V({1, 2, 3, 1, 2, 9}), "emitted tokens joined the history");
        check(ask(l, V{9, 1, 2}, 2, 4) == V({9}), "1 2 was last continued with 9");
    }
    {   // a new request reuses the tables without clearing: an old slot never proposes into the new history
        CopyLane l(cfg(1 << 10, 3, 1, 8));
        l.begin_request();
        V a = {10, 11, 12, 13, 14, 15};
        l.append(a.data(), a.size());
        check(ask(l, V{10, 11, 12}, 3, 4) == V({13, 14, 15}), "first request");
        l.begin_request();
        V b = {20, 21, 22, 23};
        l.append(b.data(), b.size());
        check(ask(l, V{10, 11, 12}, 3, 4).empty(), "the old request's key proposes nothing in the new one");
        check(ask(l, V{20, 21, 22}, 3, 4) == V({23}), "the new request's own key");
    }
    {   // parity against a brute-force reference on pseudo-random text with repeats, incremental appends, pending tails
        uint64_t s = 12345;
        auto rnd = [&](int m) { s = s * 6364136223846793005ull + 1442695040888963407ull; return (int32_t) ((s >> 33) % (uint64_t) m); };
        CopyLane l(cfg(1 << 18, 5, 2, 20));
        l.begin_request();
        V hist;
        int proposals = 0, nonempty = 0;
        for (int round = 0; round < 400; ++round) {
            V chunk;
            if (hist.size() > 60 && rnd(3) != 0) {   // a stretch of earlier text again (the copy case), else fresh tokens from a small alphabet
                const int len = 4 + rnd(14);
                const size_t at = (size_t) rnd((int) hist.size() - len - 1);
                chunk.assign(hist.begin() + (std::ptrdiff_t) at, hist.begin() + (std::ptrdiff_t) at + len);
            } else {
                for (int i = 0, add = 1 + rnd(7); i < add; ++i) chunk.push_back(rnd(12));
            }
            l.append(chunk.data(), chunk.size());
            hist.insert(hist.end(), chunk.begin(), chunk.end());
            const int n_pending = rnd(4);
            V pend;
            for (int i = 0; i < n_pending; ++i) pend.push_back(rnd(12));
            V tail;
            const size_t from_hist = std::min<size_t>(hist.size(), 60);
            tail.assign(hist.end() - (std::ptrdiff_t) from_hist, hist.end());
            tail.insert(tail.end(), pend.begin(), pend.end());
            int key = 0;
            const V got = ask(l, tail, n_pending, 20, &key);
            ++proposals;
            // reference: keys 32 16 8 5, longest first; the context is hist ++ pend; most recent position pos < hist.size(),
            // pos >= n, with hist[pos-n..pos] == the key; take = min(20, size - pos) >= 2
            V full = hist;
            full.insert(full.end(), pend.begin(), pend.end());
            V want;
            int want_key = 0;
            for (int n : {32, 16, 8, 5}) {
                if (full.size() < (size_t) n) continue;
                const V k(full.end() - n, full.end());
                for (size_t pos = hist.size() - 1; pos >= (size_t) n && pos < hist.size(); --pos) {
                    if (V(hist.begin() + (std::ptrdiff_t) pos - n, hist.begin() + (std::ptrdiff_t) pos) != k) continue;
                    const size_t take = std::min<size_t>(20, hist.size() - pos);
                    if (take >= 2) { want.assign(hist.begin() + (std::ptrdiff_t) pos, hist.begin() + (std::ptrdiff_t) (pos + take)); want_key = n; }
                    break;   // the most recent occurrence only: a shorter copy falls to the next key
                }
                if (!want.empty()) break;
            }
            // the table keeps the most recent position per hash slot, so a slot overwritten by another key's later position
            // can lose a match the brute force still finds: the lane may return nothing, but anything it returns is exact
            if (!got.empty()) { ++nonempty; check(got == want && key == want_key, "lane proposal equals the reference"); }
        }
        check(nonempty > 40, "the random text produced real proposals");
        (void) proposals;
    }
    {   // the DraftSource wrapper: match = key length, reset starts a request
        CopyLaneSource src(cfg(1 << 16, 3, 1, 8));
        src.reset();
        V p = {1, 2, 3, 4, 5, 6};
        src.append(p.data(), p.size());
        V tail = {1, 2, 3};
        V out(8);
        int match = -1;
        int k = src.propose(tail.data(), 3, 3, 8, out.data(), &match);
        check(k == 3 && match == 3 && out[0] == 4 && out[2] == 6, "source: copy of 4 5 6 with match 3");
        k = src.propose(tail.data(), 3, 3, 0, out.data(), &match);
        check(k == 0 && match == 0, "source: max_k 0");
        check(std::string(src.name()) == "copy-lane", "source name");
    }
    std::printf("copy_lane_test: %d checks passed\n", checks);
    return 0;
}
