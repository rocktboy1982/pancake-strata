// src/core/calibrate_ratio_test.cpp - MOD_PLAN A45: the CPU-vs-promote ratio derived from a probe, and its cache file (no GPU needed).
#include "strata/core/calibrate_ratio.hpp"

#include <cmath>
#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)
static bool near(double a, double b) { return std::fabs(a - b) < 1e-9; }

int main() {
    using namespace strata;
    // env switches
    CHECK(!calib_enabled(nullptr));
    CHECK(!calib_enabled(""));
    CHECK(!calib_enabled("0"));
    CHECK(calib_enabled("1"));
    CHECK(near(calib_min_share(nullptr), 0.35));
    CHECK(near(calib_min_share("0.2"), 0.2));
    CHECK(near(calib_min_share("7"), 0.9));
    CHECK(near(calib_min_share("-1"), 0.0));

    // equal costs: half the misses on each lane
    {
        CalibMeasure m;
        m.cpu_us = 100; m.copy_us = 100; m.gpu_us = 10;
        const CalibResult r = calib_derive(m, 0.95, 0.35);
        CHECK(r.valid && near(r.ratio, 1.0) && near(r.share_raw, 0.5) && near(r.share, 0.5));
        CHECK(near(r.break_even_reuse, 100.0 / 90.0));
    }
    // a link that moves an expert 4x faster than the CPU computes it: the link takes 0.8, the ceiling caps it
    {
        CalibMeasure m;
        m.cpu_us = 400; m.copy_us = 100;
        CHECK(near(calib_derive(m, 0.95, 0.35).share, 0.80));
        CHECK(near(calib_derive(m, 0.60, 0.35).share, 0.60));
    }
    // this box's order of magnitude: CPU 45 us per expert, PCIe 3.0 x16 (12 GB/s) 1.4 MB = 117 us -> share_raw 0.28 < 0.35 -> no PCIe share
    {
        CalibMeasure m;
        m.cpu_us = 45; m.copy_us = 117; m.gpu_us = 8;
        const CalibResult r = calib_derive(m, 0.95, 0.35);
        CHECK(r.valid && near(r.share_raw, 45.0 / 162.0) && near(r.share, 0.0));
        CHECK(near(r.break_even_reuse, 117.0 / 37.0));
        // with the threshold at 0 the rounded raw share is used (0.05 steps, floor 0.10)
        CHECK(near(calib_derive(m, 0.95, 0.0).share, 0.30));
    }
    // the contended figures win when both were measured; one missing -> the quiet pair
    {
        CalibMeasure m;
        m.cpu_us = 50; m.copy_us = 100; m.cpu_c_us = 100; m.copy_c_us = 100;
        CHECK(near(calib_derive(m, 0.95, 0.35).share_raw, 0.5));
        m.copy_c_us = 0;
        CHECK(near(calib_derive(m, 0.95, 0.0).share_raw, 50.0 / 150.0));
    }
    // a promoted expert never repays when the CPU is no slower than the GPU read
    {
        CalibMeasure m;
        m.cpu_us = 20; m.copy_us = 20; m.gpu_us = 30;
        CHECK(calib_derive(m, 0.95, 0.0).break_even_reuse < 0.0);
    }
    // an unusable probe: not valid, share 0
    {
        CalibMeasure m;
        CHECK(!calib_derive(m, 0.95, 0.35).valid && near(calib_derive(m, 0.95, 0.35).share, 0.0));
        m.cpu_us = 10;
        CHECK(!calib_derive(m, 0.95, 0.35).valid);
        m.copy_us = std::nan("");
        CHECK(!calib_derive(m, 0.95, 0.35).valid);
    }
    // the share is always 0, or a 0.05 step in [0.10, ceiling]
    for (double cpu = 5.0; cpu <= 2000.0; cpu *= 1.3)
        for (double copy = 5.0; copy <= 2000.0; copy *= 1.3) {
            CalibMeasure m;
            m.cpu_us = cpu; m.copy_us = copy;
            const CalibResult r = calib_derive(m, 0.95, 0.35);
            CHECK(r.valid && r.share >= 0.0 && r.share <= 0.95 + 1e-9);
            CHECK(r.share == 0.0 || (r.share >= 0.10 - 1e-9 && near(std::round(r.share * 20.0) / 20.0, r.share)));
            CHECK((r.share_raw < 0.35) == (r.share == 0.0));
        }
    // agrees with the static rule when its factor is 1: link / (link + R) with link = bytes/copy, R = bytes/cpu
    {
        CalibMeasure m;
        m.cpu_us = 60; m.copy_us = 90;
        const double bytes = 1.4e6, link = bytes / (m.copy_us * 1e-6) / 1e9, rate = bytes / (m.cpu_us * 1e-6) / 1e9;
        CHECK(near(calib_derive(m, 0.95, 0.0).share_raw, link / (link + rate)));
    }

    // cache file: round trip, key mismatch, damaged file
    {
        CalibMeasure m;
        m.cpu_us = 44.25; m.cpu_c_us = 51.5; m.copy_us = 110.75; m.copy_c_us = 125.0; m.gpu_us = 7.5; m.blob_bytes = 1409024;
        const std::string key = "gpu=AMD Radeon RX 7900 XT workers=15 layers=48 blob=1409024";
        const std::string txt = calib_cache_format(m, key);
        CalibMeasure o;
        CHECK(calib_cache_parse(txt, key, &o));
        CHECK(near(o.cpu_us, 44.25) && near(o.cpu_c_us, 51.5) && near(o.copy_us, 110.75) && near(o.copy_c_us, 125.0));
        CHECK(near(o.gpu_us, 7.5) && near(o.blob_bytes, 1409024.0));
        CalibMeasure z;
        CHECK(!calib_cache_parse(txt, key + "x", &z) && z.cpu_us == 0.0);   // another machine or pack: ignored
        CHECK(!calib_cache_parse("", key, &z));
        CHECK(!calib_cache_parse("garbage\nkey=" + key + "\n", key, &z));
        CHECK(!calib_cache_parse(txt.substr(0, txt.find("cpu_us=")), key, &z));          // cut short
        CHECK(!calib_cache_parse("strata-calibrate 1\nkey=" + key + "\ncpu_us=0\ncpu_c_us=0\ncopy_us=0\ncopy_c_us=0\ngpu_us=0\nblob_bytes=1\n", key, &z));
    }
    if (fails == 0) std::printf("calibrate_ratio_test: all passed\n");
    return fails == 0 ? 0 : 1;
}
