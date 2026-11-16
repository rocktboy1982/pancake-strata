// src/core/subfit_trace_test.cpp - STRATA_SUBFIT_TRACE: the switch parse and the report line (core/subfit_trace.hpp), CPU only.
#include "strata/core/subfit_trace.hpp"

#include <cstdio>
#include <string>

using namespace strata::core::subfit_trace;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    CHECK(!on_env(nullptr));
    CHECK(!on_env(""));
    CHECK(!on_env("0"));
    CHECK(!on_env("11"));
    CHECK(!on_env("true"));
    CHECK(on_env("1"));

    Tally t;
    const std::string empty = line(t);   // no replays: zero means, no division by zero
    CHECK(empty.find("0 replays") != std::string::npos);
    CHECK(empty.find("copy_blob(e) 0 us/replay") != std::string::npos);

    t.add(kCopyE, 3000);
    t.add(kCopyE, 1000);
    t.add(kExpertS, 500);
    t.job();
    t.job();
    const std::string s = line(t);
    CHECK(s.find("2 replays") != std::string::npos);
    CHECK(s.find("worker 4 ms in all") != std::string::npos);     // 4500 us -> 4 ms
    CHECK(s.find("copy_blob(e) 2000 us/replay") != std::string::npos);
    CHECK(s.find("expert(s) 250 us/replay") != std::string::npos);
    CHECK(s.find("copy_blob(s) 0 us/replay") != std::string::npos);
    CHECK(s.back() == '\n');

    if (fails == 0) std::printf("subfit_trace_test: ok\n");
    return fails == 0 ? 0 : 1;
}
