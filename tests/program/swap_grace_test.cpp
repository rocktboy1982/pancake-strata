// STRATA_SWAP_GRACE (LEADS.md Part Q4): a swapped-in expert is no victim for N tier rounds (include/strata/program/swap_grace.hpp).
// No GPU.  Off by default: held() is false for everything and note_in() records nothing.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "strata/program/swap_grace.hpp"

using strata::program::SwapGrace;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    // the environment parse: unset, empty, 0, negative and junk are off
    CHECK(SwapGrace::rounds_from(nullptr) == 0 && SwapGrace::rounds_from("") == 0 && SwapGrace::rounds_from("0") == 0);
    CHECK(SwapGrace::rounds_from("-3") == 0 && SwapGrace::rounds_from("abc") == 0);
    CHECK(SwapGrace::rounds_from("3") == 3 && SwapGrace::rounds_from("12") == 12);
    // default constructed and N = 0: off, nothing is held, nothing is stored
    {
        SwapGrace d;
        CHECK(!d.on());
        d.round();
        d.note_in(0, 0);
        CHECK(!d.held(0, 0));
        SwapGrace z(4, 8, 0);
        CHECK(!z.on());
        z.round();
        z.note_in(1, 2);
        CHECK(!z.held(1, 2));
    }
    // N = 3: swapped in during round r, held in rounds r + 1 .. r + 3, free in r + 4; other experts and layers never held
    {
        SwapGrace g(4, 8, 3);
        CHECK(g.on() && g.rounds() == 3);
        g.round();   // round 1
        CHECK(!g.held(2, 5));
        g.note_in(2, 5);
        CHECK(!g.held(2, 6) && !g.held(1, 5) && !g.held(3, 5));
        g.round();   // round 2
        CHECK(g.held(2, 5));
        g.round();   // round 3
        CHECK(g.held(2, 5));
        g.round();   // round 4 = r + 3
        CHECK(g.held(2, 5));
        g.round();   // round 5 = r + 4
        CHECK(!g.held(2, 5));
        // swapped in again later: held again from that round
        g.note_in(2, 5);
        g.round();
        CHECK(g.held(2, 5));
        // out-of-range ids are harmless
        g.note_in(99, 99);
        CHECK(!g.held(99, 99) && !g.held(-1, 0));
    }
    // N = 1: held for exactly the next round
    {
        SwapGrace g(1, 4, 1);
        g.round();
        g.note_in(0, 1);
        g.round();
        CHECK(g.held(0, 1));
        g.round();
        CHECK(!g.held(0, 1));
    }
    // a victim filter built on it never offers a held expert (the use in generate.cpp: held experts are left out of the list)
    {
        SwapGrace g(1, 6, 2);
        g.round();
        g.note_in(0, 2);
        g.note_in(0, 4);
        g.round();
        std::vector<int> vict;
        for (int e = 0; e < 6; ++e)
            if (!g.held(0, e)) vict.push_back(e);
        CHECK((vict == std::vector<int>{0, 1, 3, 5}));
    }
    if (failed == 0) std::printf("swap_grace_test: ok\n");
    return failed == 0 ? 0 : 1;
}
