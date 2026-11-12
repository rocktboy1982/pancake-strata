// MOD_PLAN "Project features discovered" item 1: core/switch_report.hpp writes each switch's "acted" and "will not act"
// line once per process, in the fixed format tools/pancake/switch_report.py reads, also when many threads report at once.
// No GPU, no model.
#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#define dup _dup
#define dup2 _dup2
#define fileno _fileno
#define close _close
#else
#include <unistd.h>
#endif

#include "strata/core/switch_report.hpp"

namespace sr = strata::switch_report;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stdout, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

namespace {

// what `body` writes to stderr (fd 2 pointed at a temporary file while it runs)
template <typename F> std::string capture_stderr(F body) {
    std::FILE* tmp = std::tmpfile();
    if (tmp == nullptr) return "<no tmpfile>";
    std::fflush(stderr);
    const int saved = dup(fileno(stderr));
    dup2(fileno(tmp), fileno(stderr));
    body();
    std::fflush(stderr);
    dup2(saved, fileno(stderr));
    close(saved);
    std::string out;
    std::rewind(tmp);
    for (int c; (c = std::fgetc(tmp)) != EOF;) out.push_back((char) c);
    std::fclose(tmp);
    return out;
}

int count_lines(const std::string& s) {
    int n = 0;
    for (char c : s) n += c == '\n';
    return n;
}

}  // namespace

int main() {
    {   // the format, once per kind, the two kinds independent
        sr::Switch s{"STRATA_TEST_A"};
        bool first = false, second = false, refused = false, refused2 = false;
        const std::string out = capture_stderr([&] {
            first = sr::acted(s, "plans of %d words use the %s copy", 1320, "wide");
            second = sr::acted(s, "plans of %d words use the %s copy", 2000, "wide");
            refused = sr::will_not_act(s, "a plan of %d words", 330);
            refused2 = sr::will_not_act(s, "another reason");
        });
        CHECK(first && !second && refused && !refused2);
        CHECK(out == "strata switch: STRATA_TEST_A acted: plans of 1320 words use the wide copy\n"
                     "strata switch: STRATA_TEST_A will not act: a plan of 330 words\n");
    }
    {   // a line break in the text stays on one line
        sr::Switch s{"STRATA_TEST_B"};
        const std::string out = capture_stderr([&] { sr::will_not_act(s, "%s", "two\nlines\r"); });
        CHECK(out == "strata switch: STRATA_TEST_B will not act: two lines \n");
    }
    {   // format_line: what the harness parses (tools/pancake/switch_report.py)
        char buf[128];
        sr::format_line(buf, sizeof buf, "STRATA_X", sr::kActed, "d");
        CHECK(std::strcmp(buf, "strata switch: STRATA_X acted: d") == 0);
        sr::format_line(buf, sizeof buf, "STRATA_X", sr::kWillNotAct, "r");
        CHECK(std::strcmp(buf, "strata switch: STRATA_X will not act: r") == 0);
    }
    {   // a switch nobody reports writes nothing
        sr::Switch s{"STRATA_TEST_C"};
        const std::string out = capture_stderr([] {});
        CHECK(out.empty() && !s.acted_said.load() && !s.refused_said.load());
    }
    {   // 16 threads x 2000 calls of each kind, released together: one line of each kind, exactly one caller told it wrote
        sr::Switch s{"STRATA_TEST_THREADS"};
        std::atomic<int> wrote_acted{0}, wrote_refused{0}, ready{0};
        std::atomic<bool> go{false};
        const int n_threads = 16;
        const std::string out = capture_stderr([&] {
            std::vector<std::thread> th;
            for (int i = 0; i < n_threads; ++i)
                th.emplace_back([&, i] {
                    ready.fetch_add(1);
                    while (!go.load()) {}
                    for (int k = 0; k < 2000; ++k) {
                        if (sr::acted(s, "thread %d", i)) wrote_acted.fetch_add(1);
                        if (sr::will_not_act(s, "thread %d", i)) wrote_refused.fetch_add(1);
                    }
                });
            while (ready.load() < n_threads) {}
            go.store(true);
            for (auto& t : th) t.join();
        });
        CHECK(wrote_acted.load() == 1 && wrote_refused.load() == 1);
        CHECK(count_lines(out) == 2);
        CHECK(out.find("strata switch: STRATA_TEST_THREADS acted: thread ") != std::string::npos);
        CHECK(out.find("strata switch: STRATA_TEST_THREADS will not act: thread ") != std::string::npos);
    }
    {   // the reported switches carry the names the engine reads (the harness matches them)
        CHECK(std::strcmp(sr::wait_copy256.name, "STRATA_WAIT_COPY256") == 0);
        CHECK(std::strcmp(sr::route_fast.name, "STRATA_ROUTE_FAST") == 0);
        CHECK(std::strcmp(sr::router_fuse.name, "STRATA_ROUTER_FUSE") == 0);
        CHECK(std::strcmp(sr::ic_warm.name, "STRATA_IC_WARM_MB") == 0);
        CHECK(std::strcmp(sr::expdec_swiglu.name, "STRATA_EXPDEC_SWIGLU") == 0);
        CHECK(std::strcmp(sr::hip_global_weights.name, "STRATA_HIP_GLOBAL_WEIGHTS") == 0);
    }
    std::printf("switch_report_test: %s\n", failed ? "FAILED" : "ok");
    return failed ? 1 : 0;
}
