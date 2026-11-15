#pragma once
// STRATA_ENV_DUMP=1 (opt-in, read once at startup): the engine prints every STRATA_* variable of its environment, sorted,
// one per line, before it touches a GPU, so a run's log carries the switch state it ran under (the A/B tools record the
// environment they launch with, but not what a preset or a wrapper script added).  Diagnostic only: nothing is changed.
// Unset: nothing here runs.  Lead: LEADS.md Part F (window overhead: "Engine does not print its switch state").

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if !defined(_WIN32)
extern char** environ;
#endif

namespace strata::core {

/// The STRATA_* entries ("NAME=value") of an environment block, sorted by name.  Pure, for the startup dump and its test.
inline std::vector<std::string> strata_env_entries(char** env) {
    std::vector<std::string> out;
    if (env == nullptr) return out;
    for (char** e = env; *e != nullptr; ++e) {
        if (std::strncmp(*e, "STRATA_", 7) == 0) out.emplace_back(*e);
    }
    std::sort(out.begin(), out.end());
    return out;
}

/// Prints the STRATA_* environment to `f` when STRATA_ENV_DUMP=1; returns how many entries were printed (0 when off).
inline size_t dump_strata_env(std::FILE* f) {
    const char* v = std::getenv("STRATA_ENV_DUMP");
    if (v == nullptr || v[0] != '1' || f == nullptr) return 0;
#if defined(_WIN32)
    char** env = _environ;
#else
    char** env = environ;
#endif
    const std::vector<std::string> all = strata_env_entries(env);
    std::fprintf(f, "strata: STRATA_ENV_DUMP: %zu STRATA_* variables set\n", all.size());
    for (const std::string& s : all) std::fprintf(f, "strata:   %s\n", s.c_str());
    return all.size();
}

}  // namespace strata::core
