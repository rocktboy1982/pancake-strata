// strata/platform/mem_pressure.hpp - MOD_PLAN 86 (first slice of 69): memory pressure (Linux PSI) for the expert loader and the
// conversation cache.
//
// The 33 GiB expert arena is filled while systemd-oomd watches the memory pressure of the user's slice (it kills a scope once
// the slice sits above 50% for 30 s).  The kernel publishes that pressure (PSI) system wide in /proc/pressure/memory and per
// cgroup in <group>/memory.pressure:
//     some avg10=1.23 avg60=0.40 avg300=0.10 total=123456
//     full avg10=0.50 avg60=0.10 avg300=0.02 total=45678
// `some` is the share of wall time in which at least one task waited for memory; `total` is that time in microseconds.
//
// The loader guard: each expert reader calls loader_pressure_check() once per chunk.  At most every 200 ms one reader samples
// both files; when the pressure over the last interval (from `total`; the first sample uses avg10) is at or above the pause
// threshold, every reader sleeps until it falls below the resume threshold, for at most a total pause budget per load.  The
// readers write the same bytes either way: only the timing changes.  The conversation cache skips parking a snapshot while
// avg10 is at or above its threshold (the next turn reads its prompt again, the answer is the same).
//
// Switches (read once per load / per process):
//   STRATA_PSI_GUARD=0             off (no pause, no parking skip; the end-of-load pressure line still prints)
//   STRATA_PSI_PAUSE_PCT=20        pause the readers at this `some` pressure, percent
//   STRATA_PSI_RESUME_PCT=10       resume below this
//   STRATA_PSI_MAX_PAUSE_MS=30000  the longest the readers are held in one load, in all
//   STRATA_PSI_PARK_PCT=20         skip parking a conversation while `some avg10` is at or above this
// Other systems than Linux: no PSI files, every call is a no-op.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace strata::platform {

struct PsiFigures {
    bool present = false;
    double avg10 = 0.0, avg60 = 0.0, avg300 = 0.0;   ///< percent
    uint64_t total_us = 0;
};

struct PsiSample {
    PsiFigures some, full;
};

/// One PSI file's text.  nullopt when there is no well-formed `some` line (an empty or foreign file).
std::optional<PsiSample> parse_psi(const std::string& text);

/// The memory.pressure file of this process's cgroup (v2), from the text of /proc/self/cgroup ("0::/path").  Empty when the
/// process has no v2 group (cgroup v1 only, or the text is not one).
std::string cgroup_memory_pressure_path(const std::string& self_cgroup_text, const std::string& cgroup_root);

/// Percent of wall time stalled between two samples of `total` (microseconds) `seconds` apart; -1 when it cannot be told
/// (no time passed, or the counter went backwards).
double psi_recent_pct(uint64_t prev_total_us, uint64_t total_us, double seconds);

/// The trip rule: the recent figure when there is one, else avg10.
inline bool pressure_at_or_above(double avg10, double recent, double threshold_pct) {
    return (recent >= 0.0 ? recent : avg10) >= threshold_pct;
}

struct PressureReading {
    bool known = false;      ///< at least one PSI file was read
    double avg10 = 0.0;      ///< the largest `some avg10` of the sources
    double recent = -1.0;    ///< the largest `some` share since the previous read (-1: no previous read)
    std::string where;       ///< "system 1.2% / cgroup 3.4%" (avg10 per source), for the log
};

/// Reads the system file and the cgroup's file (when present).  Keeps the previous `total` per source for the recent figure.
/// Not thread safe: one owner, or a lock around it.
class MemoryPressureProbe {
public:
    struct Paths {
        std::string system = "/proc/pressure/memory";
        std::string self_cgroup = "/proc/self/cgroup";
        std::string cgroup_root = "/sys/fs/cgroup";
    };
    MemoryPressureProbe();
    explicit MemoryPressureProbe(const Paths& paths);
    bool usable() const { return !sources_.empty(); }
    PressureReading read();

private:
    struct Source {
        std::string label, path;
        bool have_prev = false;
        uint64_t prev_total = 0;
        int64_t prev_ns = 0;
    };
    std::vector<Source> sources_;
};

struct PressureConfig {
    bool on = true;
    double pause_pct = 20.0;
    double resume_pct = 10.0;
    uint64_t max_pause_ms = 30000;
    double park_pct = 20.0;
};

/// From the switches' values (null = unset); a value that is not a number keeps the default; resume is kept below pause.
PressureConfig pressure_config(const char* guard, const char* pause_pct, const char* resume_pct, const char* max_pause_ms,
                               const char* park_pct);
/// The same from the environment.
PressureConfig pressure_config_from_env();

/// The loader guard (see the top of this file).  begin/end nest: only the outermost pair resets and prints.  `what` names the
/// load in the end line.
void loader_pressure_begin(const char* what);
void loader_pressure_check();
void loader_pressure_end();

/// Tests: begin with the probe on other files (`paths`), and what the last finished load did.
void loader_pressure_begin_with(const char* what, const MemoryPressureProbe::Paths& paths);
struct LoaderPressureStats {
    int pauses = 0;
    double paused_ms = 0.0;
    bool budget_spent = false;
};
LoaderPressureStats loader_pressure_last();

struct LoaderPressureScope {
    explicit LoaderPressureScope(const char* what) { loader_pressure_begin(what); }
    ~LoaderPressureScope() { loader_pressure_end(); }
    LoaderPressureScope(const LoaderPressureScope&) = delete;
    LoaderPressureScope& operator=(const LoaderPressureScope&) = delete;
};

/// --serve parking: true when memory pressure (some avg10) is at or above STRATA_PSI_PARK_PCT; `why` says the figures.
/// False when the guard is off, PSI cannot be read, or on another system.  Thread safe.
bool memory_pressure_blocks_parking(std::string* why);

}  // namespace strata::platform
