#pragma once
// MOD_PLAN A45 (STRATA_CALIBRATE=1, opt-in): the CPU-vs-promote ratio of the decode miss split, measured at start instead of assumed.
//
// The static rule (pcie_share_rule.hpp, PcieAuto) sets the PCIe share of a layer's missed experts from the probed link and an ASSUMED
// pool rate (4 GB/s of expert bytes per worker, capped at 65) and an empirical factor k.  Here the machine is measured once, after the
// weights and the CPU pool are up and before serving:
//   cpu_us     one missed expert on the CPU pool (the decode's own call: a native pack's run_split_multi_native)
//   cpu_c_us   the same while the PCIe link is kept busy by host->device copies (the DDR reads compete: this is the pool's real cost)
//   copy_us    one expert's bytes host -> device, from pinned memory (link alone)
//   copy_c_us  the same while the CPU pool runs
//   gpu_us     one expert read on the GPU (device-to-device copy of the blob: a VRAM-read proxy for a hit's expert kernel)
// The share of the misses the link carries that balances the two lanes is  f = cpu / (cpu + copy)  (both lanes finish together); it is also
// the fraction of the CPU-only miss time the split saves.  Below `min_share` the saving does not pay for the path's fixed cost (the flag-B
// wait and the launches of the PCIe share), so the share is 0 and the verify windows leave the path out (--pcie-frac 0): that is the result
// MOD_PLAN A45 expects on this box (a static PcieAuto share lost 6-13% here).  Policy only: the same experts are computed, only on which lane,
// so the output is exact up to the CPU lane's own rounding (the share already moves between runs by design).
// Pure functions, no GPU: the unit test is src/core/calibrate_ratio_test.cpp.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace strata {

/// What the probe measures; every time is microseconds per expert, 0 = not measured.
struct CalibMeasure {
    double cpu_us = 0.0, cpu_c_us = 0.0;
    double copy_us = 0.0, copy_c_us = 0.0;
    double gpu_us = 0.0;
    double blob_bytes = 0.0;   ///< mean expert blob of the pack
};

struct CalibResult {
    bool valid = false;
    double cpu_us = 0.0, copy_us = 0.0;   ///< the two figures the ratio uses (contended when measured)
    double ratio = 0.0;                   ///< cpu_us / copy_us: > 1 means the link moves an expert faster than the CPU computes it
    double share_raw = 0.0;               ///< cpu / (cpu + copy)
    double share = 0.0;                   ///< what the engine uses: 0.05 steps, 0 below min_share, at most the ceiling
    double break_even_reuse = -1.0;       ///< hits a promoted expert needs to repay its copy: copy / (cpu - gpu); < 0 = never repays
};

/// STRATA_CALIBRATE: unset, empty or 0 = off.
inline bool calib_enabled(const char* env) { return env != nullptr && env[0] != 0 && std::atoi(env) != 0; }

/// STRATA_CALIBRATE_MIN_SHARE (nullptr unset): clamped 0.0..0.9; default 0.35.
inline double calib_min_share(const char* env) {
    if (env != nullptr && env[0] != 0) return std::min(0.9, std::max(0.0, std::atof(env)));
    return 0.35;
}

/// The ratio, the share and the promotion break-even from a probe.  The contended figures are used when both were measured (they are the
/// ones the decode sees: the pool reads RAM while the DMA writes VRAM from it); otherwise the quiet ones.  `ceiling` is the rule's ceiling
/// (pcie_share_ceiling).  A reading that is not positive (a probe that did not run) gives valid = false and share = 0.
inline CalibResult calib_derive(const CalibMeasure& m, double ceiling, double min_share) {
    CalibResult r;
    const bool both_c = m.cpu_c_us > 0.0 && m.copy_c_us > 0.0;
    r.cpu_us = both_c ? m.cpu_c_us : m.cpu_us;
    r.copy_us = both_c ? m.copy_c_us : m.copy_us;
    if (!(r.cpu_us > 0.0) || !(r.copy_us > 0.0) || !std::isfinite(r.cpu_us) || !std::isfinite(r.copy_us)) return r;
    r.valid = true;
    r.ratio = r.cpu_us / r.copy_us;
    r.share_raw = r.cpu_us / (r.cpu_us + r.copy_us);
    if (r.share_raw < min_share) {
        r.share = 0.0;
    } else {
        r.share = std::min(ceiling, std::max(0.10, std::round(r.share_raw * 20.0) / 20.0));
    }
    const double saving = r.cpu_us - (m.gpu_us > 0.0 ? m.gpu_us : 0.0);
    r.break_even_reuse = saving > 0.0 ? r.copy_us / saving : -1.0;
    return r;
}

/// The cache file: one `key=...` line naming the machine and the pack, then `name=value` lines.  The key must match to be used, so a file
/// from another GPU, another pool size or another pack is ignored and re-measured.
inline std::string calib_cache_format(const CalibMeasure& m, const std::string& key) {
    char b[512];
    std::snprintf(b, sizeof b,
                  "strata-calibrate 1\nkey=%s\ncpu_us=%.4f\ncpu_c_us=%.4f\ncopy_us=%.4f\ncopy_c_us=%.4f\ngpu_us=%.4f\nblob_bytes=%.1f\n",
                  key.c_str(), m.cpu_us, m.cpu_c_us, m.copy_us, m.copy_c_us, m.gpu_us, m.blob_bytes);
    return b;
}

/// True and `out` filled when `text` is a cache file of this format whose key equals `key` and whose figures are usable (cpu and copy > 0).
inline bool calib_cache_parse(const std::string& text, const std::string& key, CalibMeasure* out) {
    size_t at = text.find('\n');
    if (at == std::string::npos || text.compare(0, at, "strata-calibrate 1") != 0) return false;
    CalibMeasure m;
    bool key_ok = false;
    int seen = 0;
    while (at < text.size()) {
        const size_t s = at + 1;
        size_t e = text.find('\n', s);
        if (e == std::string::npos) e = text.size();
        if (e > s) {
            const std::string line = text.substr(s, e - s);
            const size_t eq = line.find('=');
            if (eq != std::string::npos) {
                const std::string k = line.substr(0, eq), v = line.substr(eq + 1);
                if (k == "key") key_ok = v == key;
                else if (k == "cpu_us") { m.cpu_us = std::atof(v.c_str()); ++seen; }
                else if (k == "cpu_c_us") { m.cpu_c_us = std::atof(v.c_str()); ++seen; }
                else if (k == "copy_us") { m.copy_us = std::atof(v.c_str()); ++seen; }
                else if (k == "copy_c_us") { m.copy_c_us = std::atof(v.c_str()); ++seen; }
                else if (k == "gpu_us") { m.gpu_us = std::atof(v.c_str()); ++seen; }
                else if (k == "blob_bytes") { m.blob_bytes = std::atof(v.c_str()); ++seen; }
            }
        }
        at = e;
    }
    if (!key_ok || seen < 6 || !(m.cpu_us > 0.0) || !(m.copy_us > 0.0)) return false;
    *out = m;
    return true;
}

}  // namespace strata
