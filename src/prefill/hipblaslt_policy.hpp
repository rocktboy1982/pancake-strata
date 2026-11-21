#pragma once
// Host-only policy layer of the hipBLASLt prompt-GEMM path (LEADS.md Part T, leads T2, T3, T4).  No HIP types in here: the engine
// (src/prefill/gemm.cu) supplies the library behind two thin interfaces and this file decides, so the decisions are unit-tested on a
// machine without ROCm (tests/prefill/hipblaslt_policy_test.cpp).
//
//   T2  table lookup by <arch>-hipblaslt-<hipBLASLt version>: a miss is said loudly (the key looked for, the keys found);
//       STRATA_HIPBLASLT_TABLE_STRICT=1 refuses to start instead; STRATA_HIPBLASLT_TABLE_NEAREST=1 takes the nearest OLDER table of
//       the same card.  Both default off: without them the engine falls back to plain hipBLASEx exactly as before, only louder.
//   T3  STRATA_HIPBLASLT_HEURISTIC=1 (default off): a call the table does not cover is asked of hipblasLtMatmulAlgoGetHeuristic
//       (STRATA_HIPBLASLT_HEURISTIC_COUNT candidates, default 1 = the top one); pick_heuristic() chooses among what came back.
//   T4  STRATA_HIPBLASLT_WARMUP=1 (default off): the table's shapes are run once at engine start so the library unpacks its
//       compressed solution catalogs before the first request; warm_plan() picks the shapes, run_warmup() runs and times them.
#include "hipblaslt_tuning.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace strata::prefill::hipblaslt {

using EnvGetter = std::function<const char*(const char*)>;

inline EnvGetter process_env() {
#if defined(_MSC_VER)
#pragma warning(suppress : 4996)
#endif
    return [](const char* name) { return std::getenv(name); };
}

// A default-off switch: on for 1 / on / true / yes (any case), off for unset, empty, 0, off, false, no and anything else.
inline bool switch_on(const char* v) {
    if (v == nullptr) return false;
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s == "1" || s == "on" || s == "true" || s == "yes";
}

struct Switches {
    bool strict = false;                       // STRATA_HIPBLASLT_TABLE_STRICT
    bool nearest = false;                      // STRATA_HIPBLASLT_TABLE_NEAREST
    bool heuristic = false;                    // STRATA_HIPBLASLT_HEURISTIC
    int heuristic_count = 1;                   // STRATA_HIPBLASLT_HEURISTIC_COUNT, 1..kMaxHeuristicCount
    bool warmup = false;                       // STRATA_HIPBLASLT_WARMUP
    uint64_t warmup_max_bytes = 768ull << 20;  // STRATA_HIPBLASLT_WARMUP_MAX_MIB: device bytes the warm-up may allocate
    std::string note;                          // a value that was not understood (the caller prints it)

    static constexpr int kMaxHeuristicCount = 32;

    static Switches from_env(const EnvGetter& get) {
        Switches s;
        s.strict = switch_on(get("STRATA_HIPBLASLT_TABLE_STRICT"));
        s.nearest = switch_on(get("STRATA_HIPBLASLT_TABLE_NEAREST"));
        s.heuristic = switch_on(get("STRATA_HIPBLASLT_HEURISTIC"));
        s.warmup = switch_on(get("STRATA_HIPBLASLT_WARMUP"));
        if (const char* c = get("STRATA_HIPBLASLT_HEURISTIC_COUNT"); c != nullptr && *c) {
            char* end = nullptr;
            const long v = std::strtol(c, &end, 10);
            if (end == c || *end || v < 1 || v > kMaxHeuristicCount)
                s.note += std::string("STRATA_HIPBLASLT_HEURISTIC_COUNT=") + c + " is not 1.." +
                          std::to_string(kMaxHeuristicCount) + "; using 1. ";
            else
                s.heuristic_count = (int) v;
        }
        if (const char* m = get("STRATA_HIPBLASLT_WARMUP_MAX_MIB"); m != nullptr && *m) {
            char* end = nullptr;
            const long long v = std::strtoll(m, &end, 10);
            if (end == m || *end || v < 1 || v > (1ll << 20))
                s.note += std::string("STRATA_HIPBLASLT_WARMUP_MAX_MIB=") + m + " is not a number of MiB; using 768. ";
            else
                s.warmup_max_bytes = (uint64_t) v << 20;
        }
        return s;
    }
};

// ---------------------------------------------------------------------------------------------------------------- T2: table lookup

inline std::string table_key(const std::string& arch, int version) {
    return arch + "-hipblaslt-" + std::to_string(version);
}

struct TableName {
    std::string arch;
    int version = 0;
};

// "<arch>-hipblaslt-<version>.txt" -> arch, version.  False for any other name.
inline bool parse_table_name(const std::string& file, TableName& out) {
    static const std::string mid = "-hipblaslt-";
    static const std::string ext = ".txt";
    const auto m = file.find(mid);
    if (m == std::string::npos || m == 0 || file.size() <= ext.size() ||
        file.compare(file.size() - ext.size(), ext.size(), ext) != 0)
        return false;
    const std::string digits = file.substr(m + mid.size(), file.size() - ext.size() - (m + mid.size()));
    if (digits.empty() || digits.size() > 9 || digits.find_first_not_of("0123456789") != std::string::npos) return false;
    out.arch = file.substr(0, m);
    out.version = std::atoi(digits.c_str());
    return true;
}

// The file names in a tuning folder (empty when the folder is missing).
inline std::vector<std::string> list_table_names(const std::string& dir) {
    std::vector<std::string> names;
    std::error_code ec;
    for (std::filesystem::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        std::error_code e2;
        if (it->is_regular_file(e2)) names.push_back(it->path().filename().string());
    }
    std::sort(names.begin(), names.end());
    return names;
}

enum class TableMatch { Exact, Nearest, Missing };

struct TablePlan {
    TableMatch match = TableMatch::Missing;
    std::string file;                      // the file name to load (Exact, Nearest)
    int version = 0;                       // the version of that file
    bool same_arch_other_version = false;  // Missing, but this card has tables for other versions: the upgrade-drift case
    std::vector<std::string> found_keys;   // every <arch>-hipblaslt-<version> in the folder
    std::string message;                   // what to print ("" for an Exact match)
};

inline std::string join_keys(const std::vector<std::string>& keys) {
    if (keys.empty()) return "none";
    std::string s;
    for (const auto& k : keys) s += (s.empty() ? "" : ", ") + k;
    return s;
}

// The loud line for a table that does not load: the key looked for, the keys found, what the engine does instead.
// `reason` is the loader's own error for an explicit STRATA_HIPBLASLT_TUNING file ("" for a folder miss).
inline std::string no_table_message(const std::string& arch, int version, const std::vector<std::string>& found,
                                    const Switches& sw, const std::string& reason, bool drift) {
    std::string m = std::string(drift || !reason.empty() ? "WARNING: " : "note: ") + "no hipBLASLt tuning table for key " +
                    table_key(arch, version) + " (looked for " + table_key(arch, version) + ".txt";
    m += reason.empty() ? ")" : "; " + reason + ")";
    m += "; keys found: " + join_keys(found) + ". ";
    if (sw.strict)
        m += "STRATA_HIPBLASLT_TABLE_STRICT=1: refusing to start.";
    else if (sw.heuristic)
        m += "STRATA_HIPBLASLT_HEURISTIC=1: the prompt GEMMs use hipBLASLt's own heuristic choice.";
    else
        m += "The prompt GEMMs run on plain hipBLASEx (the T-096 gap, about -28% prompt speed with a table on 100401). "
             "Tune a table for this key (tools/pancake/box_chains/tune-hipblaslt-table.sh), or set STRATA_HIPBLASLT_TABLE_NEAREST=1 "
             "(older table of this card, solution ids may name other kernels), or STRATA_HIPBLASLT_HEURISTIC=1, "
             "or STRATA_HIPBLASLT_TABLE_STRICT=1 to refuse to start without a table.";
    return m;
}

inline TablePlan plan_table(const std::vector<std::string>& names, const std::string& arch, int version,
                            const Switches& sw) {
    TablePlan p;
    int best_older = -1;
    for (const auto& n : names) {
        TableName tn;
        if (!parse_table_name(n, tn)) continue;
        p.found_keys.push_back(table_key(tn.arch, tn.version));
        if (tn.arch != arch) continue;
        if (tn.version == version) {
            p.match = TableMatch::Exact;
            p.file = n;
            p.version = version;
        } else {
            p.same_arch_other_version = true;
            if (tn.version < version && tn.version > best_older) best_older = tn.version;
        }
    }
    std::sort(p.found_keys.begin(), p.found_keys.end());
    if (p.match == TableMatch::Exact) return p;
    if (sw.nearest && best_older > 0) {
        p.match = TableMatch::Nearest;
        p.version = best_older;
        p.file = arch + "-hipblaslt-" + std::to_string(best_older) + ".txt";
        p.message = "WARNING: no hipBLASLt tuning table for key " + table_key(arch, version) + "; keys found: " +
                    join_keys(p.found_keys) + ". STRATA_HIPBLASLT_TABLE_NEAREST=1 uses " + table_key(arch, best_older) +
                    ": its solution ids belong to hipBLASLt " + std::to_string(best_older) + ", not " +
                    std::to_string(version) + ", so a row may name a different kernel than the one tuned (a slower one, or one "
                    "the per-solution gate rejects and plain hipBLASEx runs) and the last bits of a result can differ from the "
                    "tuned table's. Tune a table for this key to remove the risk.";
        return p;
    }
    p.match = TableMatch::Missing;
    p.message = no_table_message(arch, version, p.found_keys, sw,
                                 sw.nearest ? "STRATA_HIPBLASLT_TABLE_NEAREST=1 found no older table of " + arch : "",
                                 p.same_arch_other_version);
    return p;
}

// ---------------------------------------------------------------------------------------------------------------- T3: heuristic

struct HeuristicShape {
    InputType type = InputType::bf16;
    int t = 0, n = 0, k = 0, ldy = 0;
    float beta = 0.0f;
};

struct HeuristicCandidate {
    bool ok = false;               // the library's per-result status was success
    int solution_id = -1;
    uint64_t workspace_bytes = 0;
};

// The library behind the selection.  gemm.cu implements it with hipblasLtMatmulAlgoGetHeuristic (and keeps the returned algorithms
// by index); the host test fakes it.
class HeuristicBackend {
public:
    virtual ~HeuristicBackend() = default;
    // At most `requested` candidates, best first.  False (and `err`) when the library call itself failed.
    virtual bool query(const HeuristicShape& shape, int requested, std::vector<HeuristicCandidate>& out, std::string& err) = 0;
};

struct HeuristicPick {
    int index = -1;                // into the vector the backend returned; -1 = none, the caller falls back to hipBLASEx
    HeuristicCandidate candidate;
    int returned = 0;
    std::string why;               // set when index < 0
};

// The first returned candidate that succeeded and fits `workspace_limit` bytes.  With requested = 1 that is the top choice or nothing.
inline HeuristicPick pick_heuristic(HeuristicBackend& backend, const HeuristicShape& shape, int requested,
                                    uint64_t workspace_limit) {
    HeuristicPick pick;
    requested = std::max(1, std::min(requested, Switches::kMaxHeuristicCount));
    std::vector<HeuristicCandidate> got;
    std::string err;
    if (!backend.query(shape, requested, got, err)) {
        pick.why = "heuristic query failed" + (err.empty() ? std::string() : ": " + err);
        return pick;
    }
    pick.returned = (int) got.size();
    if (got.empty()) {
        pick.why = "the heuristic returned no algorithm";
        return pick;
    }
    const int look = std::min<int>((int) got.size(), requested);
    for (int i = 0; i < look; ++i) {
        if (!got[(size_t) i].ok) continue;
        if (got[(size_t) i].workspace_bytes > workspace_limit) continue;
        pick.index = i;
        pick.candidate = got[(size_t) i];
        return pick;
    }
    pick.why = "none of the " + std::to_string(look) + " returned algorithm(s) is supported within " +
               std::to_string(workspace_limit) + " workspace bytes";
    return pick;
}

// ---------------------------------------------------------------------------------------------------------------- T4: warm-up

struct WarmShape {
    InputType type = InputType::bf16;
    int n = 0, k = 0, ldy = 0, t = 0;

    bool operator<(const WarmShape& o) const {
        return std::tie(type, n, k, ldy, t) < std::tie(o.type, o.n, o.k, o.ldy, o.t);
    }
    // One device buffer holds a call: W[n,k] and X[t,k] in 16 bit, Y[ldy,t] in f32, each part on a 256-byte boundary.
    static uint64_t up256(uint64_t v) { return (v + 255ull) & ~255ull; }
    uint64_t x_offset() const { return up256(2ull * (uint64_t) n * (uint64_t) k); }
    uint64_t y_offset() const { return x_offset() + up256(2ull * (uint64_t) t * (uint64_t) k); }
    uint64_t bytes() const { return y_offset() + up256(4ull * (uint64_t) ldy * (uint64_t) t); }
};

struct WarmPlan {
    std::vector<WarmShape> shapes;     // unique, sorted
    std::vector<WarmShape> skipped;    // over the byte cap
    uint64_t buffer_bytes = 0;         // the largest of `shapes`: one allocation serves them all
};

inline WarmPlan warm_plan(const std::vector<TuningRow>& rows, uint64_t max_bytes) {
    std::set<WarmShape> unique;
    for (const auto& r : rows) {
        WarmShape s;
        s.type = r.type;
        s.n = r.n;
        s.k = r.k;
        s.ldy = r.ldy;
        s.t = r.t_bucket;
        unique.insert(s);
    }
    WarmPlan plan;
    for (const auto& s : unique) {
        if (s.bytes() > max_bytes) {
            plan.skipped.push_back(s);
            continue;
        }
        plan.shapes.push_back(s);
        plan.buffer_bytes = std::max(plan.buffer_bytes, s.bytes());
    }
    return plan;
}

class WarmBackend {
public:
    virtual ~WarmBackend() = default;
    // Run one GEMM of this shape through the engine's own call path and wait for it.  False when it did not run on hipBLASLt.
    virtual bool run(const WarmShape& shape) = 0;
};

struct WarmSummary {
    int ran = 0;
    int failed = 0;
    int skipped = 0;
    double seconds = 0.0;
    double slowest_seconds = 0.0;
    WarmShape slowest;
};

// `now` returns seconds (a steady clock in the engine, a counter in the test).
inline WarmSummary run_warmup(const WarmPlan& plan, WarmBackend& backend, const std::function<double()>& now) {
    WarmSummary s;
    s.skipped = (int) plan.skipped.size();
    const double t0 = now();
    for (const auto& shape : plan.shapes) {
        const double a = now();
        const bool ok = backend.run(shape);
        const double d = now() - a;
        (ok ? s.ran : s.failed) += 1;
        if (d > s.slowest_seconds) {
            s.slowest_seconds = d;
            s.slowest = shape;
        }
    }
    s.seconds = now() - t0;
    return s;
}

}  // namespace strata::prefill::hipblaslt
