// src/spec/suffix_index.cpp - see include/strata/spec/suffix_index.hpp (MOD_PLAN A24).
#include "strata/spec/suffix_index.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::spec {

namespace {
uint64_t finish(uint64_t h) {
    h ^= h >> 32;
    h *= 0xD6E8FEB86659FD93ull;
    return h ^ (h >> 32);
}
}  // namespace

SuffixIndex::SuffixIndex(size_t bytes, Params p) : p_(p) {
    size_t cap = 4096;
    while (cap * 2 * kBytesPerToken <= bytes) cap *= 2;
    tok_.assign(cap, kMark);
    mask_ = cap - 1;
    for (Level& l : lv_) {
        l.back.assign(cap, 0);
        l.head.assign(cap, 0);
    }
}

void SuffixIndex::clear() {
    std::fill(tok_.begin(), tok_.end(), kMark);
    for (Level& l : lv_) {
        std::fill(l.back.begin(), l.back.end(), 0u);
        std::fill(l.head.begin(), l.head.end(), 0ull);
    }
    end_ = run_ = outputs_ = 0;
}

// the hash of the k tokens that end at ring place `at`, oldest first
uint64_t SuffixIndex::hash_ring(uint64_t at, int k) const {
    uint64_t h = 0x9E3779B97F4A7C15ull * (uint64_t) k;
    for (int i = k - 1; i >= 0; --i) h = (h ^ (uint32_t) tok_[(at - (uint64_t) i) & mask_]) * 0x100000001B3ull;
    return finish(h);
}

// the same hash of the k tokens that end at `last` (a context outside the ring)
uint64_t SuffixIndex::hash_seq(const int32_t* last, int k) {
    uint64_t h = 0x9E3779B97F4A7C15ull * (uint64_t) k;
    for (int i = k - 1; i >= 0; --i) h = (h ^ (uint32_t) last[-i]) * 0x100000001B3ull;
    return finish(h);
}

void SuffixIndex::put(int32_t t, bool indexed) {
    const uint64_t at = end_;
    tok_[(size_t) (at & mask_)] = t;
    ++end_;
    run_ = t == kMark ? 0 : run_ + 1;
    const size_t i = (size_t) (at & mask_);
    for (int li = 0; li < kLevels; ++li) {
        const int k = kLen[li];
        Level& l = lv_[li];
        if (!indexed || run_ < (uint64_t) k) {
            l.back[i] = 0;
            continue;
        }
        uint64_t& h = l.head[(size_t) (hash_ring(at, k) & mask_)];
        l.back[i] = h != 0 && at - (h - 1) <= UINT32_MAX ? (uint32_t) (at - (h - 1)) : 0;
        h = at + 1;
    }
}

void SuffixIndex::insert(const int32_t* lead, size_t n_lead, const int32_t* out, size_t n_out) {
    if (n_out == 0) return;
    if (end_ > 0 && tok_[(size_t) ((end_ - 1) & mask_)] != kMark) put(kMark, false);
    // the lead: matched into; only its last place (followed by the output's first token) is a draft's place
    const size_t keep = std::min<size_t>(n_lead, (size_t) kLead);
    for (size_t i = n_lead - keep; i < n_lead; ++i) put(lead[i], i + 1 == n_lead);
    for (size_t i = 0; i < n_out; ++i) put(out[i], true);
    ++outputs_;
}

int SuffixIndex::propose(const int32_t* pat, int n_pat, int max_k, int32_t* out, int* match, double* prob) const {
    *match = 0;
    max_k = std::min(max_k, kMaxDraft);
    if (max_k < 1 || n_pat < kLen[kLevels - 1] || end_ == 0) return 0;
    const int32_t* last = pat + n_pat - 1;
    const uint64_t low = first();
    uint64_t place[kWalk];
    int32_t next[kWalk];
    int n = 0, best = 0;
    for (int li = 0; li < kLevels && n == 0; ++li) {
        const int k = kLen[li];
        if (n_pat < k) continue;
        const Level& l = lv_[li];
        uint64_t p = l.head[(size_t) (hash_seq(last, k) & mask_)];
        for (int walked = 0; p != 0 && walked < kWalk; ++walked) {
            const uint64_t q = p - 1;
            if (q < low) break;   // older places are gone too
            const int most = (int) std::min<uint64_t>(std::min<uint64_t>((uint64_t) n_pat, (uint64_t) kDepth), q - low + 1);
            int m = 0;
            while (m < most && tok_[(size_t) ((q - (uint64_t) m) & mask_)] == last[-m]) ++m;
            if (m >= k) {   // a hash collision in the table matches less than k: not a place of this node
                if (m > best) best = m, n = 0;
                if (m == best) place[n++] = q;
            }
            const uint32_t d = l.back[(size_t) (q & mask_)];
            p = d == 0 || d > q ? 0 : q - d + 1;
        }
    }
    if (n == 0) return 0;
    const int lim = std::min(max_k, std::max(1, (int) (p_.factor * (double) best)));
    double sure = 1.0;
    int k = 0;
    for (; k < lim; ++k) {
        // what each place that goes on has next; places at an output's end say nothing
        int m = 0;
        for (int i = 0; i < n; ++i) {
            const uint64_t nx = place[i] + 1 + (uint64_t) k;
            if (nx >= end_) continue;
            const int32_t t = tok_[(size_t) (nx & mask_)];
            if (t == kMark) continue;
            place[m] = place[i];
            next[m++] = t;
        }
        if (m == 0) break;
        // the most frequent (the frequency score); of equals, the newest place's (places are newest first)
        int bi = 0, bn = 0;
        for (int i = 0; i < m; ++i) {
            int c = 0;
            for (int j = 0; j < m; ++j) c += next[j] == next[i];
            if (c > bn) bi = i, bn = c;
        }
        sure *= (double) bn / (double) m;
        if (sure < p_.min_prob) break;
        out[k] = next[bi];
        if (prob != nullptr) prob[k] = sure;
        n = 0;
        for (int i = 0; i < m; ++i)
            if (next[i] == out[k]) place[n++] = place[i];
    }
    if (k > 0) *match = best;
    return k;
}

// ---------------------------------------------------------------------------------------- the log (STRATA_SUFFIX_INDEX_FILE)
namespace {
constexpr char kLogMagic[4] = {'S', 'X', 'I', '1'};
constexpr uint32_t kLogMaxOut = 1u << 24;   // a record claiming more output tokens than this is damage, not an output

struct LogRecord {
    size_t at;       // offset of its first token in the file's bytes
    uint32_t lead;   // tokens
    uint32_t out;
};

bool write_record(std::FILE* f, const int32_t* lead, size_t n_lead, const int32_t* out, size_t n_out) {
    const size_t keep = std::min<size_t>(n_lead, (size_t) SuffixIndex::kLead);
    const uint32_t hdr[2] = {(uint32_t) keep, (uint32_t) n_out};
    return std::fwrite(hdr, sizeof(hdr), 1, f) == 1 &&
           (keep == 0 || std::fwrite(lead + (n_lead - keep), sizeof(int32_t), keep, f) == keep) &&
           (n_out == 0 || std::fwrite(out, sizeof(int32_t), n_out, f) == n_out);
}
}  // namespace

SharedSuffixIndex::~SharedSuffixIndex() {
    if (log_ != nullptr) std::fclose(log_);
}

void SharedSuffixIndex::append_log(const int32_t* lead, size_t n_lead, const int32_t* out, size_t n_out) {
    if (n_out == 0 || n_out > kLogMaxOut) return;
    if (!write_record(log_, lead, n_lead, out, n_out) || std::fflush(log_) != 0) {
        std::fprintf(stderr, "strata: STRATA_SUFFIX_INDEX_FILE: a write failed; the index keeps running in memory only\n");
        std::fclose(log_);
        log_ = nullptr;
    }
}

bool SharedSuffixIndex::attach_log(const char* path, std::string& err, uint64_t* loaded) {
    std::unique_lock<std::shared_mutex> lk(mu_);
    if (loaded != nullptr) *loaded = 0;
    if (log_ != nullptr) { err = "a log is attached already"; return false; }
    // the whole log (2x the ring at most once it was compacted: 8 MiB at the default 64 MiB index)
    std::vector<char> bytes;
    if (std::FILE* f = std::fopen(path, "rb")) {
        char buf[1 << 16];
        for (size_t n; (n = std::fread(buf, 1, sizeof(buf), f)) > 0;) bytes.insert(bytes.end(), buf, buf + n);
        std::fclose(f);
    }
    std::vector<LogRecord> recs;
    bool damaged = false;
    uint64_t tokens = 0;
    if (!bytes.empty()) {
        if (bytes.size() < sizeof(kLogMagic) || std::memcmp(bytes.data(), kLogMagic, sizeof(kLogMagic)) != 0) {
            err = std::string("not a suffix index log (no SXI1 header): ") + path;
            return false;   // never overwrite a file that is not ours
        }
        size_t at = sizeof(kLogMagic);
        while (at < bytes.size()) {
            uint32_t hdr[2];
            if (bytes.size() - at < sizeof(hdr)) { damaged = true; break; }
            std::memcpy(hdr, bytes.data() + at, sizeof(hdr));
            const uint64_t need = ((uint64_t) hdr[0] + hdr[1]) * sizeof(int32_t);
            if (hdr[0] > (uint32_t) SuffixIndex::kLead || hdr[1] == 0 || hdr[1] > kLogMaxOut ||
                bytes.size() - at - sizeof(hdr) < need) {
                damaged = true;
                break;
            }
            recs.push_back({at + sizeof(hdr), hdr[0], hdr[1]});
            tokens += (uint64_t) hdr[0] + hdr[1] + 1;
            at += sizeof(hdr) + (size_t) need;
        }
    }
    // the newest records that fit in the ring (older ones would be overwritten at once)
    size_t first = recs.size();
    uint64_t fit = 0;
    while (first > 0 && fit + recs[first - 1].lead + recs[first - 1].out + 1 <= idx_.capacity()) {
        --first;
        fit += recs[first].lead + recs[first].out + 1;
    }
    std::vector<int32_t> lead, out;
    for (size_t r = first; r < recs.size(); ++r) {
        lead.resize(recs[r].lead);
        out.resize(recs[r].out);
        if (recs[r].lead > 0) std::memcpy(lead.data(), bytes.data() + recs[r].at, recs[r].lead * sizeof(int32_t));
        std::memcpy(out.data(), bytes.data() + recs[r].at + recs[r].lead * sizeof(int32_t), recs[r].out * sizeof(int32_t));
        idx_.insert(lead.data(), lead.size(), out.data(), out.size());
    }
    if (loaded != nullptr) *loaded = recs.size() - first;
    // a cut-off tail or a log twice the ring: rewrite it with the loaded records (a new file, then renamed over it)
    if (damaged || tokens > 2 * (uint64_t) idx_.capacity() || bytes.empty()) {
        const std::string tmp = std::string(path) + ".tmp";
        std::FILE* f = std::fopen(tmp.c_str(), "wb");
        bool ok = f != nullptr && std::fwrite(kLogMagic, sizeof(kLogMagic), 1, f) == 1;
        for (size_t r = first; ok && r < recs.size(); ++r) {
            lead.resize(recs[r].lead);
            out.resize(recs[r].out);
            if (recs[r].lead > 0) std::memcpy(lead.data(), bytes.data() + recs[r].at, recs[r].lead * sizeof(int32_t));
            std::memcpy(out.data(), bytes.data() + recs[r].at + recs[r].lead * sizeof(int32_t), recs[r].out * sizeof(int32_t));
            ok = write_record(f, lead.data(), lead.size(), out.data(), out.size());
        }
        if (f != nullptr) ok = std::fclose(f) == 0 && ok;
        // (Windows' rename does not replace an existing file: the old log goes first; it was read in full above)
        if (ok && std::rename(tmp.c_str(), path) != 0) ok = std::remove(path) == 0 && std::rename(tmp.c_str(), path) == 0;
        if (!ok) {
            std::remove(tmp.c_str());
            err = std::string("cannot rewrite ") + path;
            return false;
        }
    }
    log_ = std::fopen(path, "ab");
    if (log_ == nullptr) {
        err = std::string("cannot open ") + path + " for appending";
        return false;
    }
    return true;
}

namespace {
double env_num(const char* name, double def, double lo, double hi) {
    const char* s = std::getenv(name);
    if (s == nullptr || s[0] == '\0') return def;
    char* e = nullptr;
    const double x = std::strtod(s, &e);
    if (e == s) return def;
    return std::clamp(x, lo, hi);
}
}  // namespace

bool suffix_index_fallback_only() {
    static const bool fb = [] {
        const char* v = std::getenv("STRATA_SUFFIX_INDEX_RULE");
        return v != nullptr && std::strcmp(v, "fallback") == 0;
    }();
    return fb;
}

SharedSuffixIndex* shared_suffix_index() {
    static SharedSuffixIndex* const idx = []() -> SharedSuffixIndex* {
        const char* v = std::getenv("STRATA_SUFFIX_INDEX");
        if (v == nullptr || std::strcmp(v, "1") != 0) return nullptr;
        const double mb = env_num("STRATA_SUFFIX_INDEX_MB", 64.0, 1.0, 16384.0);
        SuffixIndex::Params p;
        p.min_prob = env_num("STRATA_SUFFIX_INDEX_MIN_PROB", 0.1, 0.0, 1.0);
        p.factor = env_num("STRATA_SUFFIX_INDEX_FACTOR", 1.0, 0.1, 32.0);
        auto* s = new SharedSuffixIndex((size_t) (mb * 1048576.0), p);   // lives as long as the process
        std::string log_txt = "memory only";
        if (const char* f = std::getenv("STRATA_SUFFIX_INDEX_FILE"); f != nullptr && f[0] != '\0') {
            std::string err;
            uint64_t loaded = 0;
            log_txt = s->attach_log(f, err, &loaded)
                          ? "log " + std::string(f) + ", " + std::to_string(loaded) + " earlier outputs loaded"
                          : "memory only (STRATA_SUFFIX_INDEX_FILE: " + err + ")";
        }
        size_t held = 0, cap = 0;
        uint64_t outs = 0;
        s->stats(&held, &cap, &outs);
        std::fprintf(stderr,
                     "strata: STRATA_SUFFIX_INDEX=1: cross-session suffix index (A24), %zu tokens in %zu MiB, min_prob %.2f, "
                     "factor %.2f, rule %s, %s\n",
                     cap, s->bytes() >> 20, p.min_prob, p.factor, suffix_index_fallback_only() ? "fallback" : "longest",
                     log_txt.c_str());
        return s;
    }();
    return idx;
}

}  // namespace strata::spec
