// tools/pancake/disco_parity_driver.cpp - A22: feeds DiscoPolicy::end_round() the rounds on stdin and writes its trace CSV,
// so test_disco_parity.py can compare the engine's per-draft features with disco_fit.py's replay of the same rounds.
//
//   disco_parity_driver OUT.csv < rounds       one round a line:  T accepted pos match | p_1 .. p_T-1 | miss_0 .. miss_T-1
//
// Build (host only):  g++ -std=c++20 -O1 -Iinclude tools/pancake/disco_parity_driver.cpp src/spec/disco.cpp
//                         src/spec/draft_policy.cpp src/spec/controller.cpp src/spec/suffix_drafter.cpp src/spec/draft_source.cpp
#include "strata/spec/disco.hpp"

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    strata::spec::DiscoPolicy disco;
    if (!disco.open_trace(argv[1])) return 3;
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        int T = 0, accepted = 0, match = 0;
        long long pos = 0;
        std::string bar;
        if (!(in >> T >> accepted >> pos >> match >> bar) || bar != "|") return 4;
        std::vector<float> p((size_t) (T - 1));
        for (float& v : p) in >> v;
        in >> bar;
        std::vector<int32_t> miss((size_t) T);
        for (int32_t& v : miss) in >> v;
        strata::spec::DiscoPolicy::Round r;
        r.p = p.data();
        r.T = T;
        r.accepted = accepted;
        r.mtp_window = true;
        r.round_ms = 1.0;
        r.match = match;
        r.pos = pos;
        r.row_miss = miss.data();
        r.row_miss_n = T;
        disco.end_round(r);
    }
    return 0;
}
