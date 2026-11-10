// MOD_PLAN 86: see include/strata/platform/mem_pressure.hpp
#include "strata/platform/mem_pressure.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include <thread>

namespace strata::platform {

namespace {

int64_t now_ns() {
    return (int64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool read_text(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

bool parse_double(const char* s, double& out) {
    if (s == nullptr || s[0] == '\0') return false;
    char* end = nullptr;
    const double v = std::strtod(s, &end);
    if (end == s || *end != '\0' || !(v >= 0.0)) return false;
    out = v;
    return true;
}

}  // namespace

std::optional<PsiSample> parse_psi(const std::string& text) {
    PsiSample out;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string kind;
        if (!(ls >> kind) || (kind != "some" && kind != "full")) continue;
        PsiFigures f;
        bool a10 = false, tot = false;
        std::string tok;
        while (ls >> tok) {
            const size_t eq = tok.find('=');
            if (eq == std::string::npos) continue;
            const std::string key = tok.substr(0, eq), val = tok.substr(eq + 1);
            char* end = nullptr;
            if (key == "total") {
                const unsigned long long v = std::strtoull(val.c_str(), &end, 10);
                if (end != val.c_str() && *end == '\0') { f.total_us = (uint64_t) v; tot = true; }
            } else {
                const double v = std::strtod(val.c_str(), &end);
                if (end == val.c_str() || *end != '\0') continue;
                if (key == "avg10") { f.avg10 = v; a10 = true; }
                else if (key == "avg60") f.avg60 = v;
                else if (key == "avg300") f.avg300 = v;
            }
        }
        f.present = a10 && tot;
        if (!f.present) continue;
        (kind == "some" ? out.some : out.full) = f;
    }
    if (!out.some.present) return std::nullopt;
    return out;
}

std::string cgroup_memory_pressure_path(const std::string& self_cgroup_text, const std::string& cgroup_root) {
    std::istringstream in(self_cgroup_text);
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("0::", 0) != 0) continue;
        std::string path = line.substr(3);
        while (!path.empty() && (path.back() == '\n' || path.back() == '\r' || path.back() == ' ')) path.pop_back();
        if (path.empty() || path[0] != '/') return std::string();
        if (path == "/") return cgroup_root + "/memory.pressure";
        return cgroup_root + path + "/memory.pressure";
    }
    return std::string();
}

double psi_recent_pct(uint64_t prev_total_us, uint64_t total_us, double seconds) {
    if (!(seconds > 0.0) || total_us < prev_total_us) return -1.0;
    const double pct = (double) (total_us - prev_total_us) / (seconds * 1e6) * 100.0;
    return std::min(pct, 100.0);
}

MemoryPressureProbe::MemoryPressureProbe() : MemoryPressureProbe(Paths{}) {}

MemoryPressureProbe::MemoryPressureProbe(const Paths& paths) {
#if defined(__linux__)
    std::string text;
    if (read_text(paths.system, text) && parse_psi(text)) sources_.push_back({"system", paths.system});
    std::string cg;
    if (read_text(paths.self_cgroup, cg)) {
        const std::string p = cgroup_memory_pressure_path(cg, paths.cgroup_root);
        if (!p.empty() && read_text(p, text) && parse_psi(text)) sources_.push_back({"cgroup", p});
    }
#else
    (void) paths;
#endif
}

PressureReading MemoryPressureProbe::read() {
    PressureReading r;
    for (Source& s : sources_) {
        std::string text;
        if (!read_text(s.path, text)) continue;
        const auto psi = parse_psi(text);
        if (!psi) continue;
        const int64_t t = now_ns();
        double recent = -1.0;
        if (s.have_prev) recent = psi_recent_pct(s.prev_total, psi->some.total_us, (double) (t - s.prev_ns) / 1e9);
        s.have_prev = true;
        s.prev_total = psi->some.total_us;
        s.prev_ns = t;
        char buf[96];
        std::snprintf(buf, sizeof buf, "%s%s %.1f%%", r.known ? " / " : "", s.label.c_str(), psi->some.avg10);
        r.where += buf;
        r.avg10 = r.known ? std::max(r.avg10, psi->some.avg10) : psi->some.avg10;
        r.recent = std::max(r.recent, recent);
        r.known = true;
    }
    return r;
}

PressureConfig pressure_config(const char* guard, const char* pause_pct, const char* resume_pct, const char* max_pause_ms,
                               const char* park_pct) {
    PressureConfig c;
    if (guard != nullptr && guard[0] == '0') c.on = false;
    double v = 0.0;
    if (parse_double(pause_pct, v) && v > 0.0 && v <= 100.0) c.pause_pct = v;
    c.park_pct = c.pause_pct;
    if (parse_double(resume_pct, v) && v <= 100.0) c.resume_pct = v;
    if (c.resume_pct >= c.pause_pct) c.resume_pct = c.pause_pct / 2.0;
    if (parse_double(max_pause_ms, v)) c.max_pause_ms = (uint64_t) v;
    if (parse_double(park_pct, v) && v > 0.0 && v <= 100.0) c.park_pct = v;
    return c;
}

PressureConfig pressure_config_from_env() {
    return pressure_config(std::getenv("STRATA_PSI_GUARD"), std::getenv("STRATA_PSI_PAUSE_PCT"),
                           std::getenv("STRATA_PSI_RESUME_PCT"), std::getenv("STRATA_PSI_MAX_PAUSE_MS"),
                           std::getenv("STRATA_PSI_PARK_PCT"));
}

#if defined(__linux__)
namespace {

constexpr int64_t kSampleNs = 200 * 1000 * 1000;   // one PSI read per 200 ms across all readers
constexpr int kNapMs = 50;                          // a held reader looks again after this

struct LoaderGuard {
    std::mutex mu;                       // the sampler: begin/end and one reader at a time
    int depth = 0;                       // under mu
    std::atomic<bool> active{false};     // a load is running and PSI is readable
    std::atomic<bool> paused{false};
    std::atomic<int64_t> last_sample{0};
    // under mu
    PressureConfig cfg;
    std::optional<MemoryPressureProbe> probe;
    std::string what;
    int64_t t_begin = 0, pause_start = 0;
    uint64_t paused_ns = 0;
    int pauses = 0;
    bool budget_spent = false;
    double peak_avg10 = 0.0, peak_recent = -1.0;
    std::string peak_where;
};

LoaderGuard& guard() {
    static LoaderGuard g;
    return g;
}

// Under g.mu: one sample and the pause/resume decision.
void sample_locked(LoaderGuard& g, int64_t t, bool decide = true) {
    g.last_sample.store(t, std::memory_order_relaxed);
    const PressureReading r = g.probe->read();
    if (!r.known) return;
    if (r.avg10 >= g.peak_avg10) { g.peak_avg10 = r.avg10; g.peak_where = r.where; }
    g.peak_recent = std::max(g.peak_recent, r.recent);
    if (!g.cfg.on || !decide) return;
    const uint64_t budget_ns = g.cfg.max_pause_ms * 1000000ull;
    if (!g.paused.load(std::memory_order_relaxed)) {
        if (g.budget_spent || !pressure_at_or_above(r.avg10, r.recent, g.cfg.pause_pct)) return;
        g.pause_start = t;
        ++g.pauses;
        g.paused.store(true, std::memory_order_release);
        std::fprintf(stderr, "strata load: memory pressure %.1f%% (some, last %d ms; avg10 %s) >= %.0f%%: holding the expert "
                             "readers until it falls below %.0f%% (STRATA_PSI_GUARD=0 turns this off)\n",
                     r.recent >= 0.0 ? r.recent : r.avg10, (int) (kSampleNs / 1000000), r.where.c_str(), g.cfg.pause_pct,
                     g.cfg.resume_pct);
        return;
    }
    const uint64_t held = (uint64_t) (t - g.pause_start);
    const bool calm = !pressure_at_or_above(r.avg10, r.recent, g.cfg.resume_pct);
    const bool spent = g.paused_ns + held >= budget_ns;
    if (!calm && !spent) return;
    g.paused_ns += held;
    g.budget_spent = spent && !calm;
    g.paused.store(false, std::memory_order_release);
    if (calm)
        std::fprintf(stderr, "strata load: memory pressure %.1f%% (avg10 %s): expert readers resumed after %.0f ms\n",
                     r.recent >= 0.0 ? r.recent : r.avg10, r.where.c_str(), (double) held / 1e6);
    else
        std::fprintf(stderr, "strata load: memory pressure still %.1f%% (avg10 %s) after %.0f ms held: the pause budget of %llu ms "
                             "is spent, the readers go on for the rest of this load (STRATA_PSI_MAX_PAUSE_MS)\n",
                     r.recent >= 0.0 ? r.recent : r.avg10, r.where.c_str(), (double) held / 1e6,
                     (unsigned long long) g.cfg.max_pause_ms);
}

}  // namespace

namespace {
LoaderPressureStats g_last;   // under guard().mu

void begin_impl(const char* what, const MemoryPressureProbe::Paths* paths) {
    LoaderGuard& g = guard();
    std::lock_guard<std::mutex> lk(g.mu);
    if (g.depth++ > 0) return;
    g.cfg = pressure_config_from_env();
    g_last = LoaderPressureStats{};
    // a fresh probe per load: the first sample then has no recent figure from the gap since the last load (avg10 decides)
    if (paths != nullptr) g.probe.emplace(*paths);
    else g.probe.emplace();
    g.what = what != nullptr ? what : "experts";
    g.t_begin = now_ns();
    g.paused_ns = 0;
    g.pauses = 0;
    g.budget_spent = false;
    g.peak_avg10 = 0.0;
    g.peak_recent = -1.0;
    g.peak_where.clear();
    g.paused.store(false);
    if (!g.probe->usable()) { g.active.store(false); return; }
    sample_locked(g, g.t_begin);   // the baseline for the recent figure (and a load that starts under pressure)
    g.active.store(true, std::memory_order_release);
}
}  // namespace

void loader_pressure_begin(const char* what) { begin_impl(what, nullptr); }
void loader_pressure_begin_with(const char* what, const MemoryPressureProbe::Paths& paths) { begin_impl(what, &paths); }

LoaderPressureStats loader_pressure_last() {
    std::lock_guard<std::mutex> lk(guard().mu);
    return g_last;
}

void loader_pressure_check() {
    LoaderGuard& g = guard();
    if (!g.active.load(std::memory_order_relaxed)) return;
    for (;;) {
        const bool held = g.paused.load(std::memory_order_acquire);
        if (!held && now_ns() - g.last_sample.load(std::memory_order_relaxed) < kSampleNs) return;
        if (g.mu.try_lock()) {
            const int64_t t = now_ns();
            if (g.active.load(std::memory_order_relaxed) && t - g.last_sample.load(std::memory_order_relaxed) >= kSampleNs)
                sample_locked(g, t);
            g.mu.unlock();
        }
        if (!g.paused.load(std::memory_order_acquire)) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(kNapMs));
    }
}

void loader_pressure_end() {
    LoaderGuard& g = guard();
    std::lock_guard<std::mutex> lk(g.mu);
    if (g.depth == 0 || --g.depth > 0) return;
    if (!g.active.load()) return;
    const int64_t t = now_ns();
    if (g.paused.load()) { g.paused_ns += (uint64_t) (t - g.pause_start); g.paused.store(false); }
    sample_locked(g, t, /*decide=*/false);   // the last stretch, for the peak only
    g.active.store(false);
    g_last.pauses = g.pauses;
    g_last.paused_ms = (double) g.paused_ns / 1e6;
    g_last.budget_spent = g.budget_spent;
    std::fprintf(stderr, "strata load: memory pressure during the %s load (%.1f s): peak some avg10 %.1f%% (%s), peak over %d ms "
                         "%.1f%%; readers held %d time%s, %.0f ms in all%s\n",
                 g.what.c_str(), (double) (t - g.t_begin) / 1e9, g.peak_avg10,
                 g.peak_where.empty() ? "n/a" : g.peak_where.c_str(), (int) (kSampleNs / 1000000),
                 std::max(g.peak_recent, 0.0), g.pauses, g.pauses == 1 ? "" : "s", (double) g.paused_ns / 1e6,
                 g.cfg.on ? "" : " (guard off: STRATA_PSI_GUARD=0)");
}

bool memory_pressure_blocks_parking(std::string* why) {
    static std::mutex mu;
    static std::optional<MemoryPressureProbe> probe;
    static PressureConfig cfg;
    std::lock_guard<std::mutex> lk(mu);
    if (!probe) { probe.emplace(); cfg = pressure_config_from_env(); }
    if (!cfg.on || !probe->usable()) return false;
    const PressureReading r = probe->read();
    if (!r.known || r.avg10 < cfg.park_pct) return false;
    if (why != nullptr) {
        char buf[64];
        std::snprintf(buf, sizeof buf, " >= %.0f%%", cfg.park_pct);
        *why = "memory pressure some avg10 " + r.where + buf;
    }
    return true;
}

#else   // no PSI: nothing to watch

void loader_pressure_begin(const char*) {}
void loader_pressure_begin_with(const char*, const MemoryPressureProbe::Paths&) {}
void loader_pressure_check() {}
void loader_pressure_end() {}
LoaderPressureStats loader_pressure_last() { return {}; }
bool memory_pressure_blocks_parking(std::string*) { return false; }

#endif

}  // namespace strata::platform
