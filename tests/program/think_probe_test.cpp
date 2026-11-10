// MOD_PLAN Phase 16 A19: the learned thinking stop's host side (STRATA_HIDDEN_DUMP / STRATA_THINK_PROBE): the settings, the
// check points, the stop rule, the probe file and the dump file's bytes.  No GPU, no model.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <chrono>
#include <filesystem>
#include <vector>

#include "strata/program/think_probe.hpp"

using namespace strata::program::think_probe;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

namespace {
std::vector<unsigned char> slurp(const std::string& p) {
    std::vector<unsigned char> b;
    if (std::FILE* f = std::fopen(p.c_str(), "rb")) {
        unsigned char c[4096];
        size_t n;
        while ((n = std::fread(c, 1, sizeof c, f)) > 0) b.insert(b.end(), c, c + n);
        std::fclose(f);
    }
    return b;
}
uint32_t u32(const std::vector<unsigned char>& b, size_t at) { return detail::get_u32(b.data() + at); }
float f32(const std::vector<unsigned char>& b, size_t at) { return detail::get_f32(b.data() + at); }
}  // namespace

int main() {
    std::string err;
    // ---- STRATA_HIDDEN_DUMP
    {
        DumpConfig c;
        CHECK(parse_dump_config(nullptr, c, err) && !c.on());
        CHECK(parse_dump_config("", c, err) && !c.on());
        CHECK(parse_dump_config("/tmp/x", c, err) && c.on() && c.dir == "/tmp/x" && c.every == kDefaultEvery);
        CHECK(parse_dump_config("/tmp/y,every=32", c, err) && c.dir == "/tmp/y" && c.every == 32);
        DumpConfig keep = c;
        CHECK(!parse_dump_config("/tmp/z,every=0", c, err) && c.dir == keep.dir && !err.empty());
        CHECK(!parse_dump_config("/tmp/z,every=x", c, err));
        CHECK(!parse_dump_config("/tmp/z,evry=4", c, err));
        CHECK(!parse_dump_config(",every=4", c, err));
    }
    // ---- STRATA_THINK_PROBE
    {
        ProbeConfig c;
        CHECK(parse_probe_config(nullptr, c, err) && !c.on());
        CHECK(parse_probe_config("p.sprb,0.8", c, err) && c.on() && c.path == "p.sprb" && std::fabs(c.threshold - 0.8f) < 1e-6f &&
              c.checks == kDefaultChecks && c.min_tokens == 0);
        CHECK(parse_probe_config("p.sprb,0.9,k=2,min=256", c, err) && c.checks == 2 && c.min_tokens == 256);
        CHECK(parse_probe_config("p.sprb,-1.5", c, err) && c.threshold == -1.5f);   // an identity-link probe's raw score
        CHECK(!parse_probe_config("p.sprb", c, err));
        CHECK(!parse_probe_config("p.sprb,abc", c, err));
        CHECK(!parse_probe_config("p.sprb,0.5,k=0", c, err));
        CHECK(!parse_probe_config("p.sprb,0.5,z=1", c, err));
        CHECK(!parse_probe_config("p.sprb,nan", c, err));
    }
    // ---- paragraphs: "\n\n" in one token or split over two; anything else in between resets
    {
        ParagraphTracker t;
        CHECK(!t.at_boundary());
        t.feed("Step one.");
        CHECK(!t.at_boundary());
        t.feed("\n\n");
        CHECK(t.at_boundary());
        t.feed("Next");
        CHECK(!t.at_boundary());
        t.feed(".\n");
        CHECK(!t.at_boundary());
        t.feed("\n");
        CHECK(t.at_boundary());
        t.feed("\n");
        CHECK(t.at_boundary());
        t.reset();
        CHECK(!t.at_boundary());
    }
    // ---- check points
    {
        CHECK(point_flags(0, true, false, 64) == 0);             // nothing before the first thinking token
        CHECK(point_flags(0, false, true, 64) == kEnd);          // ... except the model's own close
        CHECK(point_flags(63, false, false, 64) == 0);
        CHECK(point_flags(64, false, false, 64) == kPeriodic);
        CHECK(point_flags(128, true, false, 64) == (kPeriodic | kParagraph));
        CHECK(point_flags(70, true, true, 64) == (kParagraph | kEnd));
        CHECK(point_flags(64, false, false, 0) == 0);            // every 0: paragraphs only
    }
    // ---- the stop rule: K in a row, the minimum, once
    {
        StopRule r(0.8f, 3, 100);
        CHECK(!r.check(0.9f, 10) && !r.check(0.9f, 20));
        CHECK(!r.check(0.5f, 30));                               // the run breaks
        CHECK(!r.check(0.9f, 40) && !r.check(0.85f, 50));
        CHECK(!r.check(0.8f, 60));                               // 3 in a row but under min
        CHECK(r.check(0.95f, 128));                              // 4 in a row, past min: fires
        CHECK(r.fired() && !r.check(0.99f, 192));                // once
        r.reset();
        CHECK(!r.fired() && r.streak() == 0);
        CHECK(!r.check(NAN, 200) && r.streak() == 0);
        StopRule one(0.5f, 1, 0);
        CHECK(one.check(0.5f, 1));                               // at or above
    }
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path tmp = fs::temp_directory_path(ec) /
        ("think_probe_test_" + std::to_string((long long) std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(tmp, ec);
    CHECK(!ec);
    if (ec) return 1;
    const std::string d = tmp.generic_string();
    // ---- the probe file: save / load / score, dimension and header checks
    {
        Probe p;
        p.set({0.5f, -1.0f, 2.0f}, 0.25f, kLogistic);
        CHECK(p.save(d + "/p.sprb"));
        Probe q;
        CHECK(q.load(d + "/p.sprb", 3, err) && q.dim() == 3 && q.link() == kLogistic && q.bias() == 0.25f);
        const float h[3] = {1.0f, 2.0f, 0.5f};                  // 0.25 + 0.5 - 2 + 1 = -0.25
        CHECK(std::fabs(q.score(h) - (float) (1.0 / (1.0 + std::exp(0.25)))) < 1e-6f);
        Probe r;
        CHECK(!r.load(d + "/p.sprb", 4, err) && !r.loaded() && err.find("inputs") != std::string::npos);
        CHECK(r.load(d + "/p.sprb", 0, err));                    // 0: any dimension
        Probe lin;
        lin.set({1.0f, 1.0f, 1.0f}, -1.0f, kIdentity);
        CHECK(lin.save(d + "/l.sprb") && r.load(d + "/l.sprb", 3, err) && std::fabs(r.score(h) - 2.5f) < 1e-6f);
        if (std::FILE* f = std::fopen((d + "/bad.sprb").c_str(), "wb")) { std::fputs("XXXXjunkjunkjunkjunk", f); std::fclose(f); }
        CHECK(!r.load(d + "/bad.sprb", 0, err));
        CHECK(!r.load(d + "/missing.sprb", 0, err));
        // truncated weights
        std::vector<unsigned char> b = slurp(d + "/p.sprb");
        if (std::FILE* f = std::fopen((d + "/t.sprb").c_str(), "wb")) { std::fwrite(b.data(), 1, b.size() - 2, f); std::fclose(f); }
        CHECK(!r.load(d + "/t.sprb", 0, err));
    }
    // ---- the dump file: nothing until the first record, then the header, fixed-size records and the .ids companion
    {
        DumpWriter w;
        w.begin(d, 4, 64, 1234567890123ULL, 777, 5, 4242);
        w.note_id(11);
        w.note_id(12);
        CHECK(!w.is_open());
        const float h1[4] = {1.0f, -2.0f, 3.5f, 0.0f}, h2[4] = {0.25f, 0.5f, 0.75f, 1.0f};
        CHECK(w.record(64, 99, kPeriodic | kParagraph, NAN, h1));
        w.note_id(13);
        CHECK(w.record(70, 248069, kEnd | kFired, 0.875f, h2));
        CHECK(w.records() == 2);
        const std::string path = w.path();
        CHECK(path == d + "/hd-1234567890123-4242-5.bin");
        w.close();
        const auto b = slurp(path);
        CHECK(b.size() == kHeaderBytes + 2 * (kRecordHeadBytes + 16));
        if (b.size() == kHeaderBytes + 2 * (kRecordHeadBytes + 16)) {
            CHECK(std::memcmp(b.data(), "SHDM", 4) == 0);
            CHECK(u32(b, 4) == kVersion && u32(b, 8) == 4 && u32(b, 12) == 64);
            CHECK(u32(b, 16) == (uint32_t) (1234567890123ULL & 0xffffffffu) && u32(b, 20) == (uint32_t) (1234567890123ULL >> 32));
            CHECK(u32(b, 24) == 777 && u32(b, 28) == 5);
            size_t r = kHeaderBytes;
            CHECK(u32(b, r) == 64 && u32(b, r + 4) == 99 && u32(b, r + 8) == (kPeriodic | kParagraph) && std::isnan(f32(b, r + 12)));
            CHECK(f32(b, r + 16) == 1.0f && f32(b, r + 20) == -2.0f && f32(b, r + 24) == 3.5f && f32(b, r + 28) == 0.0f);
            r += kRecordHeadBytes + 16;
            CHECK(u32(b, r) == 70 && u32(b, r + 4) == 248069 && u32(b, r + 8) == (kEnd | kFired) && f32(b, r + 12) == 0.875f);
            CHECK(f32(b, r + 28) == 1.0f);
        }
        const auto ids = slurp(d + "/hd-1234567890123-4242-5.ids");
        CHECK(ids.size() == 12 && u32(ids, 0) == 11 && u32(ids, 4) == 12 && u32(ids, 8) == 13);
        // a request with no record leaves no file
        DumpWriter none;
        none.begin(d, 4, 64, 1ULL, 1, 6, 4242);
        none.note_id(1);
        none.close();
        CHECK(slurp(d + "/hd-1-4242-6.bin").empty() && slurp(d + "/hd-1-4242-6.ids").empty());
        // an unwritable directory: the writer goes off for the request, no crash
        DumpWriter bad;
        bad.begin(d + "/no/such/dir", 4, 64, 2ULL, 1, 7, 4242);
        CHECK(!bad.record(64, 1, kPeriodic, 0.0f, h1) && bad.failed() && !bad.record(128, 1, kPeriodic, 0.0f, h1));
        std::remove((d + "/hd-1234567890123-4242-5.bin").c_str());
        std::remove((d + "/hd-1234567890123-4242-5.ids").c_str());
    }
    CHECK(tprobe_line(320, 0.91234f) == "TPROBE 320 0.9123");
    fs::remove_all(tmp, ec);
    if (failed) { std::fprintf(stderr, "think_probe_test: %d failure(s)\n", failed); return 1; }
    std::printf("think_probe_test: ok\n");
    return 0;
}
