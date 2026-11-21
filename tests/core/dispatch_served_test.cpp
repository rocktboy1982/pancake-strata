// tests/core/dispatch_served_test.cpp - LEADS.md Part U U3: the served-path census of include/strata/core/dispatch_stats.hpp (host only, no GPU).
// One process per mode (the Stats singleton reads its environment once): argv[1] = off | on | report | stats0.
//   off     nothing set: served() / served_if() count nothing, json() has no "served" key and is the old report byte for byte
//   on      STRATA_DISPATCH_SERVED=1: hits / declines per name, texts merged across literals, sorted, coexisting with kernel() / note()
//   report  STRATA_DISPATCH_REPORT=<path>: the census is on as well (the harness sets this one)
//   stats0  STRATA_DISPATCH_STATS=0 wins over both: off
#include "strata/core/dispatch_stats.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_fail;
}
void put(const char* k, const char* v) {
#ifdef _WIN32
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}
bool has(const std::string& j, const char* t) { return j.find(t) != std::string::npos; }
}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "off";
    if (mode == "on") put("STRATA_DISPATCH_SERVED", "1");
    if (mode == "report") put("STRATA_DISPATCH_REPORT", "dispatch_served_test_report.json");   // never written: the test exits through _Exit below
    if (mode == "stats0") { put("STRATA_DISPATCH_SERVED", "1"); put("STRATA_DISPATCH_STATS", "0"); }
    namespace d = strata::dispatch;

    if (mode == "off" || mode == "stats0") {
        check(!d::Stats::get().served_on(), mode + ": the census is off");
        d::served("a.b");
        check(d::served_if("a.c", true) == true, "served_if passes true through");
        check(d::served_if("a.c", false) == false, "served_if passes false through");
        if (mode == "off") {
            d::kernel(d::kDense, 1, 2, 3, 4, 5);
            d::note("x");
            const std::string j = d::Stats::get().json();
            check(!has(j, "served"), "no served key in a default report");
            check(!has(j, "a.b"), "no census name in a default report");
            check(has(j, "\"counts\""), "the old counts are still reported");
        } else {
            check(!has(d::Stats::get().json(), "served"), "stats0: no served key");
        }
    } else {
        check(d::Stats::get().served_on(), mode + ": the census is on");
        d::served("a.hit");
        d::served("a.hit");
        d::served("a.hit");
        check(d::served_if("b.maybe", true), "served_if true");
        check(d::served_if("b.maybe", true), "served_if true again");
        check(!d::served_if("b.maybe", false), "served_if false");
        char other[] = "a.hit";   // the same text behind another pointer: merged in the report
        d::served(other);
        d::kernel(d::kDense, 1, 2, 3, 4, 5);
        d::note("some reason");
        std::vector<std::thread> th;
        for (int t = 0; t < 4; ++t) th.emplace_back([] { for (int i = 0; i < 1000; ++i) d::served("t.many"); });
        for (auto& x : th) x.join();
        const auto snap = d::Stats::get().served_snapshot();
        check(snap.at("a.hit").first == 4 && snap.at("a.hit").second == 0, "a.hit: 4 hits (3 + the merged copy), 0 declined");
        check(snap.at("b.maybe").first == 2 && snap.at("b.maybe").second == 1, "b.maybe: 2 hits, 1 declined");
        check(snap.at("t.many").first == 4000, "4 threads x 1000 counts");
        const std::string j = d::Stats::get().json();
        check(has(j, "\"served\": {"), "the served key");
        check(has(j, "\"a.hit\": {\"n\": 4, \"declined\": 0}"), "a.hit line");
        check(has(j, "\"b.maybe\": {\"n\": 2, \"declined\": 1}"), "b.maybe line");
        check(j.find("\"a.hit\"") < j.find("\"b.maybe\"") && j.find("\"b.maybe\"") < j.find("\"t.many\""), "names sorted");
        check(has(j, "\"cat\": \"str\"") && has(j, "\"launches\""), "the old keys coexist");
        check(j.back() == '\n' && j[j.size() - 2] == '}', "the report ends with a closed object");
    }
    if (g_fail) { std::fprintf(stderr, "%d failure(s)\n", g_fail); std::fflush(stderr); std::_Exit(1); }
    std::printf("dispatch_served_test %s ok\n", mode.c_str());
    std::fflush(stdout);
    std::_Exit(0);   // skip the atexit report writer (STRATA_DISPATCH_REPORT mode)
}
