// Host-only test of src/prefill/hipblaslt_policy.hpp (LEADS.md Part T, T2 / T3 / T4): no GPU, no ROCm, no model.
// The library sits behind HeuristicBackend / WarmBackend, so a fake stands in for hipBLASLt.  What it does NOT cover: the real
// hipblasLtMatmulAlgoGetHeuristic adapter, the catalog unpack time and the engine start (src/prefill/gemm.cu; box only).
#include "hipblaslt_policy.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace strata::prefill::hipblaslt;

namespace {

int g_total = 0;
int g_fail = 0;

void check(bool ok, const char* what) {
    ++g_total;
    if (!ok) ++g_fail;
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
}

bool has(const std::string& hay, const std::string& needle) { return hay.find(needle) != std::string::npos; }

EnvGetter env_of(const std::vector<std::pair<std::string, std::string>>& kv) {
    return [kv](const char* name) -> const char* {
        for (const auto& p : kv)
            if (p.first == name) return p.second.c_str();
        return nullptr;
    };
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-hipblaslt-policy-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

void write_table(const fs::path& dir, const std::string& arch, int version) {
    std::ofstream out(dir / (arch + "-hipblaslt-" + std::to_string(version) + ".txt"));
    out << "STRATA_HIPBLASLT_TUNING_V1 " << arch << " " << version << "\n"
        << "bf16 2560 320 2560 4096 1035\n"
        << "bf16 2560 320 2560 8192 1036\n"
        << "f16 128 2560 128 4096 7\n";
}

// a fake library: candidates by shape, scripted per call
struct FakeHeuristic final : HeuristicBackend {
    std::vector<HeuristicCandidate> script;
    bool fail = false;
    int calls = 0;
    int last_requested = 0;
    bool query(const HeuristicShape&, int requested, std::vector<HeuristicCandidate>& out, std::string& err) override {
        ++calls;
        last_requested = requested;
        if (fail) {
            err = "fake library error";
            return false;
        }
        for (int i = 0; i < requested && i < (int) script.size(); ++i) out.push_back(script[(size_t) i]);
        return true;
    }
};

struct FakeWarm final : WarmBackend {
    std::set<WarmShape> ran;
    std::set<WarmShape> refuse;
    bool run(const WarmShape& s) override {
        if (refuse.count(s)) return false;
        ran.insert(s);
        return true;
    }
};

}  // namespace

int main() {
    // ---- switches: everything default off, values parsed once
    {
        const Switches off = Switches::from_env(env_of({}));
        check(!off.strict && !off.nearest && !off.heuristic && !off.warmup, "all four switches default off");
        check(off.heuristic_count == 1 && off.warmup_max_bytes == (768ull << 20), "defaults: top 1, warm-up cap 768 MiB");
        const Switches on = Switches::from_env(env_of({{"STRATA_HIPBLASLT_TABLE_STRICT", "1"},
                                                       {"STRATA_HIPBLASLT_TABLE_NEAREST", "on"},
                                                       {"STRATA_HIPBLASLT_HEURISTIC", "TRUE"},
                                                       {"STRATA_HIPBLASLT_HEURISTIC_COUNT", "5"},
                                                       {"STRATA_HIPBLASLT_WARMUP", "yes"},
                                                       {"STRATA_HIPBLASLT_WARMUP_MAX_MIB", "64"}}));
        check(on.strict && on.nearest && on.heuristic && on.warmup && on.heuristic_count == 5 &&
                  on.warmup_max_bytes == (64ull << 20) && on.note.empty(),
              "1 / on / TRUE / yes switch on; count and cap parse");
        const Switches zero = Switches::from_env(env_of({{"STRATA_HIPBLASLT_TABLE_STRICT", "0"},
                                                         {"STRATA_HIPBLASLT_HEURISTIC", "off"},
                                                         {"STRATA_HIPBLASLT_WARMUP", ""}}));
        check(!zero.strict && !zero.heuristic && !zero.warmup, "0 / off / empty stay off");
        const Switches bad = Switches::from_env(env_of({{"STRATA_HIPBLASLT_HEURISTIC_COUNT", "99"},
                                                        {"STRATA_HIPBLASLT_WARMUP_MAX_MIB", "x"}}));
        check(bad.heuristic_count == 1 && bad.warmup_max_bytes == (768ull << 20) && has(bad.note, "HEURISTIC_COUNT") &&
                  has(bad.note, "WARMUP_MAX_MIB"),
              "a bad count or cap keeps the default and says so");
    }

    // ---- T2: names
    {
        TableName n;
        check(parse_table_name("gfx1100-hipblaslt-100500.txt", n) && n.arch == "gfx1100" && n.version == 100500, "parse a table name");
        check(!parse_table_name("gfx1100-rocblas-5.6.0.txt", n) && !parse_table_name("gfx1100-hipblaslt-.txt", n) &&
                  !parse_table_name("gfx1100-hipblaslt-10x.txt", n) && !parse_table_name("-hipblaslt-1.txt", n) &&
                  !parse_table_name("gfx1100-hipblaslt-100500.txt.bak", n),
              "other names are not tables");
    }

    // ---- T2: lookup, loud miss, strict, nearest (synthetic folder, loaded by the real TuningTable)
    {
        TempDir dir;
        write_table(dir.path, "gfx1100", 100200);
        write_table(dir.path, "gfx1100", 100401);
        write_table(dir.path, "gfx1151", 100500);
        const auto names = list_table_names(dir.path.string());
        check(names.size() == 3, "the folder lists three tables");

        const Switches def;
        auto exact = plan_table(names, "gfx1100", 100401, def);
        check(exact.match == TableMatch::Exact && exact.file == "gfx1100-hipblaslt-100401.txt" && exact.message.empty(),
              "100401 on gfx1100: exact match, no message");

        auto miss = plan_table(names, "gfx1100", 100500, def);
        check(miss.match == TableMatch::Missing && miss.same_arch_other_version, "100500 on gfx1100: missing, same card has other versions");
        check(has(miss.message, "WARNING") && has(miss.message, "gfx1100-hipblaslt-100500") &&
                  has(miss.message, "gfx1100-hipblaslt-100401") && has(miss.message, "gfx1100-hipblaslt-100200") &&
                  has(miss.message, "gfx1151-hipblaslt-100500"),
              "the warning names the key looked for and every key found");
        check(has(miss.message, "plain hipBLASEx") && has(miss.message, "STRICT") && has(miss.message, "NEAREST"),
              "the default miss says what runs instead and how to change it");

        Switches strict;
        strict.strict = true;
        auto sm = plan_table(names, "gfx1100", 100500, strict);
        check(sm.match == TableMatch::Missing && has(sm.message, "refusing to start"), "strict: a miss is a refusal message");
        check(plan_table(names, "gfx1100", 100401, strict).match == TableMatch::Exact, "strict: an exact match still loads");

        Switches heur;
        heur.heuristic = true;
        check(has(plan_table(names, "gfx1100", 100500, heur).message, "heuristic"), "a miss with the heuristic on says the heuristic runs");

        auto other_card = plan_table(names, "gfx1030", 100401, def);
        check(other_card.match == TableMatch::Missing && !other_card.same_arch_other_version && has(other_card.message, "note:") &&
                  !has(other_card.message, "WARNING"),
              "a card with no table at all gets a note, not the upgrade-drift warning");

        Switches nearest;
        nearest.nearest = true;
        auto np = plan_table(names, "gfx1100", 100500, nearest);
        check(np.match == TableMatch::Nearest && np.file == "gfx1100-hipblaslt-100401.txt" && np.version == 100401 &&
                  has(np.message, "WARNING") && has(np.message, "different kernel"),
              "nearest: the newest OLDER table of the same card, with the risk said");
        TableName tn;
        check(parse_table_name(np.file, tn), "nearest names a loadable file");
        TuningTable table;
        std::string err;
        check(table.load((dir.path / np.file).string(), "gfx1100", np.version, err) && table.rows().size() == 3,
              "nearest table loads with its own version as the gate");
        TuningTable wrong;
        check(!wrong.load((dir.path / np.file).string(), "gfx1100", 100500, err),
              "without nearest the loader's own version gate still rejects a 100401 file on 100500");

        auto newer_only = plan_table({"gfx1100-hipblaslt-100500.txt"}, "gfx1100", 100401, nearest);
        check(newer_only.match == TableMatch::Missing && has(newer_only.message, "no older table"),
              "nearest never takes a NEWER table");
        check(plan_table({}, "gfx1100", 100500, nearest).match == TableMatch::Missing, "an empty folder is a miss");
        check(list_table_names((dir.path / "nope").string()).empty(), "a missing folder lists nothing");

        Switches both;
        both.nearest = true;
        both.strict = true;
        check(plan_table(names, "gfx1100", 100500, both).match == TableMatch::Nearest,
              "strict + nearest: nearest is a match (strict only refuses a true miss)");

        const std::string explicit_msg = no_table_message("gfx1100", 100500, {"gfx1100-hipblaslt-100401"}, def,
                                                           "hipBLASLt version mismatch: file=100401 runtime=100500", true);
        check(has(explicit_msg, "version mismatch") && has(explicit_msg, "gfx1100-hipblaslt-100500"),
              "a file the loader rejects is said with the loader's own reason");
    }

    // ---- the repo's own tables (the folder is passed by CMake; skipped when the test is built by hand)
#ifdef STRATA_TEST_TABLE_DIR
    {
        const auto names = list_table_names(STRATA_TEST_TABLE_DIR);
        const Switches def;
        check(plan_table(names, "gfx1100", 100401, def).match == TableMatch::Exact, "repo: gfx1100 100401 table found");
        check(plan_table(names, "gfx1100", 100500, def).match == TableMatch::Exact, "repo: gfx1100 100500 table found");
        Switches nearest;
        nearest.nearest = true;
        const auto future = plan_table(names, "gfx1100", 100600, nearest);
        check(future.match == TableMatch::Nearest && future.version == 100500, "repo: a future 100600 takes the 100500 table with nearest");
        TuningTable t;
        std::string err;
        const bool loaded = t.load(std::string(STRATA_TEST_TABLE_DIR) + "/gfx1100-hipblaslt-100401.txt", "gfx1100", 100401, err);
        check(loaded && !t.rows().empty(), "repo: the 100401 table loads");
        if (loaded) {
            const WarmPlan plan = warm_plan(t.rows(), 768ull << 20);
            check(plan.shapes.size() + plan.skipped.size() == t.rows().size(), "repo: warm-up plan covers every row of the table");
            FakeWarm fake;
            double clock = 0;
            const auto s = run_warmup(plan, fake, [&clock] { return clock += 0.5; });
            check(s.ran == (int) plan.shapes.size() && fake.ran.size() == plan.shapes.size(), "repo: the warm-up touches every planned shape");
        }
    }
#endif

    // ---- T3: the selection layer against a fake library
    {
        HeuristicShape shape;
        shape.t = 8192;
        shape.n = 2560;
        shape.k = 320;
        shape.ldy = 2560;
        FakeHeuristic lib;
        lib.script = {{true, 111, 0}, {true, 222, 0}, {true, 333, 0}};
        auto top1 = pick_heuristic(lib, shape, 1, 32u << 20);
        check(top1.index == 0 && top1.candidate.solution_id == 111 && lib.last_requested == 1, "top-1: the first candidate, one requested");

        lib.script = {{false, 111, 0}, {true, 222, 0}, {true, 333, 0}};
        auto bad1 = pick_heuristic(lib, shape, 1, 32u << 20);
        check(bad1.index == -1 && !bad1.why.empty(), "top-1 with a failed status: nothing (the caller runs hipBLASEx)");
        auto bad3 = pick_heuristic(lib, shape, 3, 32u << 20);
        check(bad3.index == 1 && bad3.candidate.solution_id == 222 && lib.last_requested == 3, "top-3: the first that succeeded");

        lib.script = {{true, 111, 64u << 20}, {true, 222, 8u << 20}};
        auto ws = pick_heuristic(lib, shape, 2, 32u << 20);
        check(ws.index == 1 && ws.candidate.solution_id == 222, "a candidate over the workspace limit is skipped");
        auto ws1 = pick_heuristic(lib, shape, 1, 32u << 20);
        check(ws1.index == -1 && has(ws1.why, "workspace"), "top-1 over the workspace limit: nothing");

        lib.script.clear();
        check(pick_heuristic(lib, shape, 1, 1).index == -1, "no candidate: nothing");
        lib.fail = true;
        auto failed = pick_heuristic(lib, shape, 1, 1);
        check(failed.index == -1 && has(failed.why, "fake library error"), "a failing library call is reported, not thrown");
        lib.fail = false;
        lib.script = {{true, 5, 0}};
        pick_heuristic(lib, shape, 1000, 1);
        check(lib.last_requested == Switches::kMaxHeuristicCount, "the requested count is clamped");
    }

    // ---- T4: warm-up plan and run
    {
        std::vector<TuningRow> rows;
        auto row = [&](InputType type, int n, int k, int ldy, int t) {
            TuningRow r;
            r.type = type;
            r.n = n;
            r.k = k;
            r.ldy = ldy;
            r.t_bucket = t;
            r.solution_id = 1;
            rows.push_back(r);
        };
        row(InputType::bf16, 2560, 320, 2560, 4096);
        row(InputType::bf16, 2560, 320, 2560, 8192);
        row(InputType::bf16, 2560, 320, 2560, 4096);   // a duplicate key (a table cannot have one, the plan is still unique)
        row(InputType::f16, 128, 2560, 128, 4096);
        row(InputType::bf16, 10240, 10240, 10240, 32768);   // far over any cap here
        const WarmPlan plan = warm_plan(rows, 256ull << 20);
        check(plan.shapes.size() == 3 && plan.skipped.size() == 1, "plan: 3 unique shapes, 1 over the cap");
        uint64_t biggest = 0;
        bool aligned = true;
        for (const auto& s : plan.shapes) {
            biggest = std::max(biggest, s.bytes());
            aligned = aligned && s.x_offset() % 256 == 0 && s.y_offset() % 256 == 0 && s.bytes() % 256 == 0 &&
                      s.x_offset() >= 2ull * s.n * s.k && s.y_offset() >= s.x_offset() + 2ull * s.t * s.k &&
                      s.bytes() >= s.y_offset() + 4ull * s.ldy * s.t;
        }
        check(plan.buffer_bytes == biggest && aligned, "plan: one buffer of the largest shape, parts 256-byte aligned and non-overlapping");

        FakeWarm fake;
        double clock = 0;
        const WarmSummary s = run_warmup(plan, fake, [&clock] { return clock += 1.0; });
        check(s.ran == 3 && s.failed == 0 && s.skipped == 1 && fake.ran.size() == 3, "run: every planned shape ran once");
        check(s.seconds > 0 && s.slowest_seconds > 0, "run: the time is measured");

        FakeWarm refusing;
        refusing.refuse.insert(plan.shapes.front());
        const WarmSummary r = run_warmup(plan, refusing, [&clock] { return clock += 1.0; });
        check(r.ran == 2 && r.failed == 1, "run: a shape the library refuses is counted as failed, the rest still run");

        check(warm_plan({}, 1ull << 30).shapes.empty(), "no table rows: nothing to warm");
    }

    std::printf("hipblaslt_policy_test: %d checks, %d failures\n", g_total, g_fail);
    return g_fail == 0 ? 0 : 1;
}
