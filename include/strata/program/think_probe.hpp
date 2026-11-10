#pragma once
// MOD_PLAN Phase 16 A19 (item 21): a learned thinking stop.  Thought Calibration (MIT 2025, arXiv 2505.18404) and Learning
// How Hard to Think (ICLR'25) read a linear probe off the model's hidden state at reasoning-step boundaries and stop the
// thinking once the probe says the answer would no longer change.  This header is the engine's host side of it: the
// dump format, the probe file, the step boundaries and the stop rule.  Header-only, no GPU, no model:
// tests/program/think_probe_test.cpp; the offline half (reader, labels, training, calibration, export) is
// tools/pancake/thinking_probe.py.
//
//   STRATA_HIDDEN_DUMP=<dir>[,every=N]   (opt-in, exact: the output does not change)
//     While the reply is inside <think> ... </think>, the vector the LM head reads (the final hyper-connection read with
//     output_hc_norm, `head_mixed_` of the verify window, n_embd floats) is copied to the host for the rows that are
//     check points - every N thinking tokens (default 64), after every "\n\n" in the thinking, and at the model's own
//     </think> - and appended to <dir>/hd-<unix ms>-<pid>-<seq>.bin, one file per request (opened at its first point).
//     The thinking's token ids go to the same name with .ids (int32, every thinking token, for the replay labels).
//     Nothing is copied on other rows; with the variable unset nothing at all runs.
//   STRATA_THINK_PROBE=<weights.sprb>,<threshold>[,k=K][,min=M]   (opt-in, LOSSY: it shortens the thinking)
//     At the same check points the probe's score sigmoid(w.h + b) (or w.h + b for a ridge probe) is computed on the host;
//     K checks in a row (default 3) at or above the threshold, once M thinking tokens (default 0) are written, make the
//     engine print `TPROBE <thinking tokens> <score>` after that window's T lines.  The server then closes the thinking
//     the way its soft budget does (at the next sentence or line end, the wrap-up and </think>); once per request.  The
//     per-effort budget (reasoning_budget_by_effort) stays the safety net.  The check points are STRATA_HIDDEN_DUMP's
//     (every 64 when only the probe is set).
//
// Dump file (little-endian), version 1:
//   header 32 bytes: char magic[4] = "SHDM", u32 version, u32 dim, u32 every, u64 start_unix_ms, u32 prompt_tokens,
//                    u32 request_seq
//   record 16 + 4 * dim bytes: u32 think_index (thinking tokens written before the token this state produces: stopping
//                    "here" keeps exactly that many), i32 token (what this state produced), u32 flags (kPeriodic,
//                    kParagraph, kEnd: the produced token is the model's own </think>, kFired: the probe fired here),
//                    f32 score (NaN without a probe), f32 h[dim]
// Probe file (little-endian), version 1: char magic[4] = "SPRB", u32 version, u32 dim, u32 link (0 logistic, 1 identity),
//   f32 bias, f32 w[dim] (the standardisation folded in: score = link(bias + sum w[i] * h[i]) on the raw vector).
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace strata::program::think_probe {

constexpr uint32_t kVersion = 1;
constexpr uint32_t kHeaderBytes = 32;
constexpr uint32_t kRecordHeadBytes = 16;
constexpr int kDefaultEvery = 64;
constexpr int kDefaultChecks = 3;

enum PointFlags : uint32_t { kPeriodic = 1u, kParagraph = 2u, kEnd = 4u, kFired = 8u };
enum Link : uint32_t { kLogistic = 0u, kIdentity = 1u };

namespace detail {
inline bool parse_int(std::string_view s, long long lo, long long hi, long long& out) {
    if (s.empty() || s.size() > 18) return false;
    long long v = 0;
    for (char c : s) {
        if (c < '0' || c > '9') return false;
        v = v * 10 + (c - '0');
    }
    if (v < lo || v > hi) return false;
    out = v;
    return true;
}
inline bool parse_float(std::string_view s, float& out) {
    if (s.empty() || s.size() > 32) return false;
    const std::string t(s);
    char* end = nullptr;
    errno = 0;
    const double v = std::strtod(t.c_str(), &end);
    if (end != t.c_str() + t.size() || errno != 0 || !std::isfinite(v)) return false;
    out = (float) v;
    return true;
}
/// Splits on ',' (no escaping: a path with a comma is not supported).
inline std::vector<std::string_view> split(std::string_view s) {
    std::vector<std::string_view> out;
    size_t a = 0;
    for (size_t i = 0; i <= s.size(); ++i)
        if (i == s.size() || s[i] == ',') { out.push_back(s.substr(a, i - a)); a = i + 1; }
    return out;
}
inline void put_u32(unsigned char* p, uint32_t v) { for (int i = 0; i < 4; ++i) p[i] = (unsigned char) (v >> (8 * i)); }
inline void put_u64(unsigned char* p, uint64_t v) { for (int i = 0; i < 8; ++i) p[i] = (unsigned char) (v >> (8 * i)); }
inline uint32_t get_u32(const unsigned char* p) { uint32_t v = 0; for (int i = 3; i >= 0; --i) v = (v << 8) | p[i]; return v; }
inline void put_f32(unsigned char* p, float f) { uint32_t v; std::memcpy(&v, &f, 4); put_u32(p, v); }
inline float get_f32(const unsigned char* p) { const uint32_t v = get_u32(p); float f; std::memcpy(&f, &v, 4); return f; }
inline bool little_endian() { const uint16_t x = 1; unsigned char b; std::memcpy(&b, &x, 1); return b == 1; }
}  // namespace detail

/// STRATA_HIDDEN_DUMP.
struct DumpConfig {
    std::string dir;
    int every = kDefaultEvery;
    bool on() const { return !dir.empty(); }
};

/// nullptr / "" -> off (true).  "<dir>[,every=N]" with 1 <= N <= 1e6.  false with `err` on a malformed value (c unchanged).
inline bool parse_dump_config(const char* v, DumpConfig& c, std::string& err) {
    if (v == nullptr || v[0] == 0) { c = DumpConfig(); return true; }
    const auto parts = detail::split(v);
    DumpConfig r;
    if (parts[0].empty()) { err = "STRATA_HIDDEN_DUMP: the directory is empty"; return false; }
    r.dir = std::string(parts[0]);
    for (size_t i = 1; i < parts.size(); ++i) {
        long long n = 0;
        if (parts[i].substr(0, 6) == "every=" && detail::parse_int(parts[i].substr(6), 1, 1000000, n)) r.every = (int) n;
        else { err = "STRATA_HIDDEN_DUMP: unknown or malformed option '" + std::string(parts[i]) + "' (every=N, N >= 1)"; return false; }
    }
    c = r;
    return true;
}

/// STRATA_THINK_PROBE.
struct ProbeConfig {
    std::string path;
    float threshold = 0.5f;
    int checks = kDefaultChecks;
    long long min_tokens = 0;
    bool on() const { return !path.empty(); }
};

/// nullptr / "" -> off (true).  "<weights>,<threshold>[,k=K][,min=M]": a finite threshold, 1 <= K <= 1000, M >= 0.
inline bool parse_probe_config(const char* v, ProbeConfig& c, std::string& err) {
    if (v == nullptr || v[0] == 0) { c = ProbeConfig(); return true; }
    const auto parts = detail::split(v);
    ProbeConfig r;
    if (parts.size() < 2 || parts[0].empty()) { err = "STRATA_THINK_PROBE: expected <weights>,<threshold>[,k=K][,min=M]"; return false; }
    r.path = std::string(parts[0]);
    if (!detail::parse_float(parts[1], r.threshold)) { err = "STRATA_THINK_PROBE: the threshold '" + std::string(parts[1]) + "' is not a number"; return false; }
    for (size_t i = 2; i < parts.size(); ++i) {
        long long n = 0;
        if (parts[i].substr(0, 2) == "k=" && detail::parse_int(parts[i].substr(2), 1, 1000, n)) r.checks = (int) n;
        else if (parts[i].substr(0, 4) == "min=" && detail::parse_int(parts[i].substr(4), 0, 1000000000LL, n)) r.min_tokens = n;
        else { err = "STRATA_THINK_PROBE: unknown or malformed option '" + std::string(parts[i]) + "' (k=K, min=M)"; return false; }
    }
    c = r;
    return true;
}

/// The thinking's text so far ends a paragraph: it ends with "\n\n" (the step boundary of Thought Calibration).  Fed with
/// each thinking token's text in order; a "\n\n" split over two tokens counts.
class ParagraphTracker {
public:
    void reset() { nl_ = 0; }
    void feed(std::string_view text) {
        for (char ch : text) nl_ = ch == '\n' ? nl_ + 1 : 0;
    }
    bool at_boundary() const { return nl_ >= 2; }

private:
    int nl_ = 0;   // newlines at the end of the text so far
};

/// The check point flags of the state whose row produces `token` after `think_index` thinking tokens: every `every` tokens,
/// after a paragraph, and at the model's own close (`is_close`).  0: not a check point.  No point before the first token.
inline uint32_t point_flags(long long think_index, bool paragraph, bool is_close, int every) {
    uint32_t f = 0;
    if (is_close) f |= kEnd;
    if (think_index <= 0) return f;
    if (every > 0 && think_index % every == 0) f |= kPeriodic;
    if (paragraph) f |= kParagraph;
    return f;
}

/// K checks in a row at or above the threshold (once `min_tokens` thinking tokens are written) fire the stop, once.
class StopRule {
public:
    StopRule() = default;
    StopRule(float threshold, int checks, long long min_tokens) : thr_(threshold), need_(checks < 1 ? 1 : checks), min_(min_tokens) {}
    void reset() { streak_ = 0; fired_ = false; }
    /// True exactly once: at the check that completes the run.  A NaN score breaks the run.
    bool check(float score, long long think_index) {
        if (fired_) return false;
        streak_ = (score >= thr_) ? streak_ + 1 : 0;   // (NaN compares false)
        if (streak_ >= need_ && think_index >= min_) { fired_ = true; return true; }
        return false;
    }
    bool fired() const { return fired_; }
    int streak() const { return streak_; }

private:
    float thr_ = 0.5f;
    int need_ = kDefaultChecks;
    long long min_ = 0;
    int streak_ = 0;
    bool fired_ = false;
};

/// The probe: score = link(bias + w . h).
class Probe {
public:
    bool loaded() const { return !w_.empty(); }
    size_t dim() const { return w_.size(); }
    uint32_t link() const { return link_; }
    float bias() const { return b_; }
    const std::vector<float>& weights() const { return w_; }

    void set(std::vector<float> w, float bias, uint32_t link) { w_ = std::move(w); b_ = bias; link_ = link; }

    /// Reads an SPRB file; `dim` > 0 must match its dimension.  false with `err` (the probe is left as it was).
    bool load(const std::string& path, size_t dim, std::string& err) {
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (f == nullptr) { err = "cannot open " + path; return false; }
        unsigned char h[20];
        bool ok = std::fread(h, 1, sizeof h, f) == sizeof h;
        const uint32_t ver = ok ? detail::get_u32(h + 4) : 0, n = ok ? detail::get_u32(h + 8) : 0, link = ok ? detail::get_u32(h + 12) : 0;
        if (!ok || std::memcmp(h, "SPRB", 4) != 0) err = path + " is not a probe file (no SPRB header)";
        else if (ver != kVersion) err = path + ": probe file version " + std::to_string(ver) + ", this engine reads " + std::to_string(kVersion);
        else if (n == 0 || n > (1u << 20)) err = path + ": bad dimension " + std::to_string(n);
        else if (dim > 0 && n != dim) err = path + ": the probe has " + std::to_string(n) + " inputs, the model's hidden state " + std::to_string(dim);
        else if (link != kLogistic && link != kIdentity) err = path + ": unknown link " + std::to_string(link);
        else {
            std::vector<unsigned char> raw((size_t) n * 4);
            if (std::fread(raw.data(), 1, raw.size(), f) != raw.size()) err = path + ": truncated weights";
            else {
                std::vector<float> w(n);
                for (uint32_t i = 0; i < n; ++i) w[i] = detail::get_f32(raw.data() + 4 * (size_t) i);
                bool finite = std::isfinite(detail::get_f32(h + 16));
                for (float x : w) finite = finite && std::isfinite(x);
                if (!finite) err = path + ": non-finite weights";
                else { set(std::move(w), detail::get_f32(h + 16), link); std::fclose(f); return true; }
            }
        }
        std::fclose(f);
        return false;
    }

    /// Writes an SPRB file (the tests; the Python tool writes the real ones).
    bool save(const std::string& path) const {
        std::FILE* f = std::fopen(path.c_str(), "wb");
        if (f == nullptr) return false;
        std::vector<unsigned char> b(20 + 4 * w_.size());
        std::memcpy(b.data(), "SPRB", 4);
        detail::put_u32(b.data() + 4, kVersion);
        detail::put_u32(b.data() + 8, (uint32_t) w_.size());
        detail::put_u32(b.data() + 12, link_);
        detail::put_f32(b.data() + 16, b_);
        for (size_t i = 0; i < w_.size(); ++i) detail::put_f32(b.data() + 20 + 4 * i, w_[i]);
        const bool ok = std::fwrite(b.data(), 1, b.size(), f) == b.size();
        return std::fclose(f) == 0 && ok;
    }

    float score(const float* h) const {
        double s = b_;
        for (size_t i = 0; i < w_.size(); ++i) s += (double) w_[i] * (double) h[i];
        if (link_ == kIdentity) return (float) s;
        return (float) (1.0 / (1.0 + std::exp(-s)));
    }

private:
    std::vector<float> w_;
    float b_ = 0.0f;
    uint32_t link_ = kLogistic;
};

/// One request's dump file (opened at the first record) and its .ids companion.
class DumpWriter {
public:
    DumpWriter() = default;
    DumpWriter(const DumpWriter&) = delete;
    DumpWriter& operator=(const DumpWriter&) = delete;
    ~DumpWriter() { close(); }

    /// The request's identity (nothing is written yet).
    void begin(const std::string& dir, uint32_t dim, uint32_t every, uint64_t start_unix_ms, uint32_t prompt_tokens,
               uint32_t seq, long pid) {
        close();
        dir_ = dir; dim_ = dim; every_ = every; start_ms_ = start_unix_ms; prompt_ = prompt_tokens; seq_ = seq; pid_ = pid;
        failed_ = false;
        records_ = 0;
    }
    bool is_open() const { return f_ != nullptr; }
    bool failed() const { return failed_; }
    long long records() const { return records_; }
    const std::string& path() const { return path_; }

    /// Appends one record; opens the file first.  false: the file could not be opened or written (the writer then stays off
    /// for this request).
    bool record(uint32_t think_index, int32_t token, uint32_t flags, float score, const float* h) {
        if (!open()) return false;
        std::vector<unsigned char> b(kRecordHeadBytes + 4 * (size_t) dim_);
        detail::put_u32(b.data(), think_index);
        detail::put_u32(b.data() + 4, (uint32_t) token);
        detail::put_u32(b.data() + 8, flags);
        detail::put_f32(b.data() + 12, score);
        if (detail::little_endian()) std::memcpy(b.data() + kRecordHeadBytes, h, 4 * (size_t) dim_);
        else for (uint32_t i = 0; i < dim_; ++i) detail::put_f32(b.data() + kRecordHeadBytes + 4 * (size_t) i, h[i]);
        if (std::fwrite(b.data(), 1, b.size(), f_) != b.size()) { fail(); return false; }
        ++records_;
        return true;
    }
    /// One thinking token id for the .ids companion (buffered; written at close).  Only kept while a file is open or
    /// may still open (the ids before the first record count too).
    void note_id(int32_t id) { if (!failed_) ids_.push_back(id); }

    void close() {
        if (f_ != nullptr) {
            std::fflush(f_);
            std::fclose(f_);
            f_ = nullptr;
            if (!ids_.empty()) {
                std::string ip = path_.substr(0, path_.size() - 4) + ".ids";
                if (std::FILE* g = std::fopen(ip.c_str(), "wb")) {
                    std::vector<unsigned char> b(4 * ids_.size());
                    for (size_t i = 0; i < ids_.size(); ++i) detail::put_u32(b.data() + 4 * i, (uint32_t) ids_[i]);
                    (void) std::fwrite(b.data(), 1, b.size(), g);
                    std::fclose(g);
                }
            }
        }
        ids_.clear();
    }

private:
    bool open() {
        if (f_ != nullptr) return true;
        if (failed_ || dir_.empty() || dim_ == 0) return false;
        path_ = dir_ + "/hd-" + std::to_string((unsigned long long) start_ms_) + "-" + std::to_string(pid_) + "-" +
                std::to_string(seq_) + ".bin";
        f_ = std::fopen(path_.c_str(), "wb");
        if (f_ == nullptr) { failed_ = true; return false; }
        unsigned char h[kHeaderBytes];
        std::memcpy(h, "SHDM", 4);
        detail::put_u32(h + 4, kVersion);
        detail::put_u32(h + 8, dim_);
        detail::put_u32(h + 12, every_);
        detail::put_u64(h + 16, start_ms_);
        detail::put_u32(h + 24, prompt_);
        detail::put_u32(h + 28, seq_);
        if (std::fwrite(h, 1, sizeof h, f_) != sizeof h) { fail(); return false; }
        return true;
    }
    void fail() {
        if (f_ != nullptr) std::fclose(f_);
        f_ = nullptr;
        failed_ = true;
        ids_.clear();
    }
    std::FILE* f_ = nullptr;
    std::string dir_, path_;
    uint32_t dim_ = 0, every_ = 0, prompt_ = 0, seq_ = 0;
    uint64_t start_ms_ = 0;
    long pid_ = 0;
    bool failed_ = false;
    long long records_ = 0;
    std::vector<int32_t> ids_;
};

/// The `TPROBE` line the engine prints when the stop fires (the server reads it; serve/server.py).
inline std::string tprobe_line(long long think_index, float score) {
    char b[64];
    std::snprintf(b, sizeof b, "TPROBE %lld %.4f", think_index, (double) score);
    return b;
}

}  // namespace strata::program::think_probe
