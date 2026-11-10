// MOD_PLAN item 39: see conversation_disk_tier.hpp.
//
// Files in a subfolder: lock, key.txt, and per entry c-<serial>.strs (a session file, conversation_file.cpp) and
// c-<serial>.idx (its index, written after the .strs is whole: an entry without a valid index is a leftover and is
// deleted at open).  Temporary files are hidden (".<name>.<hex>.tmp" from session_file_write, ".<name>.tmp" here).
//
// An index file: magic "SCDX", version, the key's hash, serial, stamp, the .strs size, flags (cvec, tool), the number
// of kept checkpoints, then the live state's and each kept checkpoint's ids and pictures, and a checksum seeded with
// the key's hash and the serial, so an index is good only under its own name and key.
#include "strata/core/conversation_disk_tier.hpp"

#include <algorithm>
#include <iterator>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <set>
#include <sstream>
#include <system_error>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <io.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace strata::core {
namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr uint32_t kIdxMagic = 0x58444353;   // "SCDX"
constexpr uint32_t kIdxVersion = 1;
constexpr uint64_t kKeySeed = 0x53434b4559763031ull;
constexpr uint64_t kMaxIdxTokens = 1ull << 26, kMaxIdxImages = 1ull << 20, kMaxIdxMarks = 64;
constexpr uint64_t kMaxIdxBytes = 1ull << 30;

uint64_t now_ns() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string hex16(uint64_t v) {
    char b[17];
    std::snprintf(b, sizeof b, "%016llx", (unsigned long long) v);
    return b;
}

// The serial in "c-<16 hex><ext>".
bool serial_in(const std::string& name, const char* ext, uint64_t& serial) {
    const size_t e = std::strlen(ext);
    if (name.size() != 2 + 16 + e || name.compare(0, 2, "c-") != 0 || name.compare(18, e, ext) != 0) return false;
    serial = 0;
    for (size_t i = 2; i < 18; ++i) {
        const char c = name[i];
        const int d = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
        if (d < 0) return false;
        serial = serial << 4 | (uint64_t) d;
    }
    return serial != 0;
}

bool is_temp(const std::string& name) {
    return name.size() > 5 && name[0] == '.' && name.compare(name.size() - 4, 4, ".tmp") == 0;
}

bool read_small(const fs::path& p, std::vector<uint8_t>& out, uint64_t limit) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    const std::streamoff n = f.tellg();
    if (n < 0 || (uint64_t) n > limit) return false;
    f.seekg(0, std::ios::beg);
    out.resize((size_t) n);
    return n == 0 || (bool) f.read(reinterpret_cast<char*>(out.data()), n);
}

// A small file written whole under a hidden temporary name, flushed and renamed over `path`.
bool write_small(const fs::path& path, const void* data, size_t n, bool durable, std::string& error) {
    const fs::path tmp = path.parent_path() / ("." + path.filename().string() + ".tmp");
#ifdef _WIN32
    FILE* f = _wfopen(tmp.c_str(), L"wb");
    if (!f) { error = "cannot create " + tmp.string(); return false; }
    bool ok = n == 0 || std::fwrite(data, 1, n, f) == n;
    ok = std::fflush(f) == 0 && ok;
    if (durable) ok = _commit(_fileno(f)) == 0 && ok;
    ok = std::fclose(f) == 0 && ok;
    ok = ok && MoveFileExW(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0) { error = "cannot create " + tmp.string() + ": " + std::strerror(errno); return false; }
    bool ok = true;
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t at = 0; ok && at < n;) {
        const ssize_t w = ::write(fd, p + at, n - at);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) { ok = false; break; }
        at += (size_t) w;
    }
    if (ok && durable) ok = ::fsync(fd) == 0;
    ok = ::close(fd) == 0 && ok;
    ok = ok && std::rename(tmp.c_str(), path.c_str()) == 0;
    if (ok && durable) {
        const int d = ::open(path.parent_path().c_str(), O_RDONLY | O_CLOEXEC);
        if (d >= 0) { (void) ::fsync(d); ::close(d); }   // (a filesystem that cannot flush a folder: not an error)
    }
#endif
    if (!ok) {
        error = "cannot write " + path.string();
        std::error_code ec;
        fs::remove(tmp, ec);
    }
    return ok;
}

void remove_quiet(const fs::path& p) {
    std::error_code ec;
    fs::remove(p, ec);
}

struct Out {
    std::vector<uint8_t> b;
    template<class T> void put(T v) {
        const size_t at = b.size();
        b.resize(at + sizeof v);
        std::memcpy(b.data() + at, &v, sizeof v);
    }
    void raw(const void* p, size_t n) {
        const size_t at = b.size();
        b.resize(at + n);
        if (n) std::memcpy(b.data() + at, p, n);
    }
};
struct In {
    const uint8_t* p;
    size_t left;
    template<class T> bool get(T& v) {
        if (left < sizeof v) return false;
        std::memcpy(&v, p, sizeof v);
        p += sizeof v, left -= sizeof v;
        return true;
    }
};

void put_mark(Out& o, const ConversationCheckpoint& c) {
    o.put((uint64_t) c.ids.size());
    o.raw(c.ids.data(), c.ids.size() * sizeof(int32_t));
    o.put((uint64_t) c.imgs.size());
    for (const auto& k : c.imgs) o.put(k.start), o.put(k.hash);
}
bool get_mark(In& in, ConversationCheckpoint& c) {
    uint64_t n = 0;
    if (!in.get(n) || n > kMaxIdxTokens || n * sizeof(int32_t) > in.left) return false;
    c.ids.resize((size_t) n);
    if (n) std::memcpy(c.ids.data(), in.p, (size_t) n * sizeof(int32_t));
    in.p += n * sizeof(int32_t), in.left -= (size_t) n * sizeof(int32_t);
    uint64_t m = 0;
    if (!in.get(m) || m > kMaxIdxImages || m * 16 > in.left) return false;
    c.imgs.resize((size_t) m);
    for (auto& k : c.imgs) if (!in.get(k.start) || !in.get(k.hash)) return false;
    return true;
}

ConversationCheckpoint marks_of(const ConversationCheckpoint& c) {
    ConversationCheckpoint m;
    m.ids = c.ids;
    m.imgs = c.imgs;
    return m;
}

const ConversationCheckpoint* deepest_mark(const ConversationDiskEntry& e) {
    const ConversationCheckpoint* d = nullptr;
    for (const auto& c : e.marks)
        if (!d || c.ids.size() > d->ids.size()) d = &c;
    return d;
}

#ifndef _WIN32
// a lock file's descriptor + 1 (0: none)
void* take_lock(const fs::path& p) {
    const int fd = ::open(p.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0) return nullptr;
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) { ::close(fd); return nullptr; }
    return (void*) (intptr_t) (fd + 1);
}
void drop_lock(void* h) { if (h) ::close((int) (intptr_t) h - 1); }
#else
void* take_lock(const fs::path& p) {
    const HANDLE h = CreateFileW(p.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS,
                                 FILE_ATTRIBUTE_NORMAL, nullptr);
    return h == INVALID_HANDLE_VALUE ? nullptr : (void*) h;
}
void drop_lock(void* h) { if (h) CloseHandle((HANDLE) h); }
#endif

}  // namespace

// ---- the key ----------------------------------------------------------------------------------------------------

const std::vector<std::string>& conversation_disk_key_env_names() {
    // the routing and numerics switches that change the computed state but are read from the environment, not from
    // the arguments the config fingerprint covers (generate.cpp session_identity)
    static const std::vector<std::string> names = {
        "STRATA_ROUTE_PRIOR",        "STRATA_ROUTE_PRIOR_DEPTH", "STRATA_ROUTE_RESIDENT",
        "STRATA_ROUTE_RESIDENT_RANKS", "STRATA_ROUTE_RESIDENT_MTP", "STRATA_ROUTE_TAIL_SKIP",
        "STRATA_VERIFY_DEFER_CPU",   "STRATA_PF_FUSED_F16",      "STRATA_HC_Q8",
        "STRATA_HC_REQ8",            "STRATA_EXPDEC",            "STRATA_EXPDEC_SWIGLU",
        "STRATA_EXPDEC_DOWN",        "STRATA_ROUTER_FUSE",       "STRATA_QFUSE",
    };
    return names;
}

std::vector<std::pair<std::string, std::string>>
conversation_disk_key_env(const std::function<const char*(const char*)>& get_env) {
    std::vector<std::pair<std::string, std::string>> out;
    auto add = [&](const std::string& name) {
        const char* v = get_env ? get_env(name.c_str()) : nullptr;
        out.emplace_back(name, v ? std::string(v) : std::string("-"));
    };
    for (const auto& n : conversation_disk_key_env_names()) add(n);
    // MOD_PLAN F13: the route-prior guards change the computed state too, but they enter the key only when set, so a session file written
    // before they existed (or by a run that does not use them) keeps its key
    static const char* const kWhenSet[] = {"STRATA_ROUTE_PRIOR_PROTECT", "STRATA_ROUTE_PRIOR_KEEP", "STRATA_ROUTE_PRIOR_KEEP_P",
                                           "STRATA_ROUTE_PRIOR_SCOPE", "STRATA_ROUTE_PRIOR_ON_PROMPT", "STRATA_ROUTE_PRIOR_MARKS"};
    for (const char* n : kWhenSet)
        if (get_env && get_env(n) != nullptr) add(n);
    std::set<std::string> extra;
    if (const char* x = get_env ? get_env("STRATA_CONVERSATION_DISK_KEY_ENV") : nullptr) {
        std::stringstream ss(x);
        for (std::string n; std::getline(ss, n, ',');) {
            n.erase(0, n.find_first_not_of(" \t"));
            n.erase(n.find_last_not_of(" \t") + 1);
            if (!n.empty() && std::find(conversation_disk_key_env_names().begin(), conversation_disk_key_env_names().end(),
                                        n) == conversation_disk_key_env_names().end() &&
                std::find_if(std::begin(kWhenSet), std::end(kWhenSet), [&](const char* w) { return n == w; }) == std::end(kWhenSet))
                extra.insert(n);
        }
    }
    for (const auto& n : extra) add(n);
    return out;
}

std::string conversation_disk_key(const ConversationDiskKeyInputs& in) {
    std::string k = "strata conversation disk 1\n";
    k += "model " + in.model + "\n";
    k += "model_fingerprint " + hex16(in.id.model) + "\n";
    k += "config_fingerprint " + hex16(in.id.config) + "\n";
    for (const auto& [name, value] : in.env) {
        std::string v = value;
        for (char& c : v) if (c == '\n' || c == '\r') c = ' ';
        k += "env " + name + "=" + v + "\n";
    }
    return k;
}

fs::path conversation_disk_subfolder(const fs::path& dir, const std::string& model, const std::string& key) {
    std::string stem;
    for (const char ch : fs::path(model).stem().string()) {
        if (stem.size() >= 48) break;
        const bool plain = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') ||
                           ch == '.' || ch == '-' || ch == '_';
        stem += plain ? ch : '_';
    }
    if (stem.empty()) stem = "model";
    return dir / (stem + "-" + hex16(session_hash64(key.data(), key.size(), kKeySeed)));
}

// ---- choice, index, policy --------------------------------------------------------------------------------------

std::vector<size_t> conversation_disk_checkpoints(const std::vector<ConversationCheckpoint>& chain) {
    if (chain.empty()) return {};
    const ConversationCheckpoint* deepest = session_deepest_checkpoint(chain);
    size_t root = 0;
    for (size_t i = 1; i < chain.size(); ++i)
        if (chain[i].ids.size() < chain[root].ids.size()) root = i;
    const size_t d = (size_t) (deepest - chain.data());
    if (d == root || chain[d].ids == chain[root].ids) return {d};
    return root < d ? std::vector<size_t>{root, d} : std::vector<size_t>{d, root};
}

bool conversation_ended_in_tool_call(const std::vector<int32_t>& ids, int64_t turn_token, int64_t tool_token) {
    if (turn_token < 0 || tool_token < 0) return false;
    for (size_t i = ids.size(); i-- > 0;) {
        if ((int64_t) ids[i] == tool_token) return true;
        if ((int64_t) ids[i] == turn_token) return false;
    }
    return false;
}

std::vector<uint8_t> conversation_disk_index_encode(const ConversationDiskEntry& e, uint64_t key_hash) {
    Out o;
    o.b.reserve(64 + (e.live.ids.size() + 64) * 4 * (1 + e.marks.size()));
    o.put(kIdxMagic), o.put(kIdxVersion), o.put(key_hash), o.put(e.serial), o.put(e.stamp), o.put(e.file_bytes);
    o.put((uint32_t) ((e.cvec ? 1u : 0u) | (e.tool ? 2u : 0u))), o.put((uint32_t) e.marks.size());
    put_mark(o, e.live);
    for (const auto& c : e.marks) put_mark(o, c);
    o.put(session_hash64(o.b.data(), o.b.size(), key_hash ^ e.serial));
    return std::move(o.b);
}

bool conversation_disk_index_decode(const uint8_t* data, size_t n, uint64_t key_hash, ConversationDiskEntry& out,
                                    std::string& error) {
    if (n < 56 + 8) { error = "index too short"; return false; }
    In in{data, n - 8};
    uint32_t magic = 0, version = 0, flags = 0, marks = 0;
    uint64_t key = 0;
    ConversationDiskEntry e;
    if (!in.get(magic) || !in.get(version) || !in.get(key) || !in.get(e.serial) || !in.get(e.stamp) ||
        !in.get(e.file_bytes) || !in.get(flags) || !in.get(marks)) { error = "index header"; return false; }
    if (magic != kIdxMagic || version != kIdxVersion) { error = "not an index file"; return false; }
    if (key != key_hash) { error = "index made for another key"; return false; }
    if (marks > kMaxIdxMarks || (flags & ~3u) != 0) { error = "index fields out of range"; return false; }
    uint64_t sum = 0;
    std::memcpy(&sum, data + n - 8, 8);
    if (sum != session_hash64(data, n - 8, key_hash ^ e.serial)) { error = "index checksum mismatch"; return false; }
    e.cvec = (flags & 1u) != 0;
    e.tool = (flags & 2u) != 0;
    if (!get_mark(in, e.live)) { error = "index live tokens"; return false; }
    e.marks.resize(marks);
    for (auto& c : e.marks) if (!get_mark(in, c)) { error = "index checkpoint tokens"; return false; }
    if (in.left != 0) { error = "index has unparsed bytes"; return false; }
    out = std::move(e);
    return true;
}

ConversationDiskEntry conversation_disk_entry_of(const SavedConversation& image, const std::vector<size_t>& keep) {
    ConversationDiskEntry e;
    e.cvec = image.cvec;
    e.live = marks_of(image.live);
    for (size_t i : keep)
        if (i < image.checkpoints.size()) e.marks.push_back(marks_of(image.checkpoints[i]));
    return e;
}

std::vector<size_t> conversation_disk_superseded(const std::vector<ConversationDiskEntry>& entries,
                                                 const SavedConversation& image) {
    auto held = [&](const ConversationCheckpoint& c) {
        if (c.ids == image.live.ids && c.imgs == image.live.imgs) return true;
        for (const auto& k : image.checkpoints)
            if (k.ids == c.ids && k.imgs == c.imgs) return true;
        return false;
    };
    std::vector<size_t> out;
    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& e = entries[i];
        if (e.cvec != image.cvec) continue;
        const ConversationCheckpoint* d = deepest_mark(e);
        // an entry without a checkpoint is superseded only by the same live state again
        if (d ? (!d->ids.empty() && held(*d)) : (!e.live.ids.empty() && held(e.live))) out.push_back(i);
    }
    return out;
}

std::vector<size_t> conversation_disk_victims(const std::vector<uint64_t>& bytes, const std::vector<uint64_t>& stamps,
                                              uint64_t incoming, uint64_t cap) {
    uint64_t total = incoming;
    for (uint64_t b : bytes) total += b;
    if (total <= cap) return {};
    std::vector<size_t> order(bytes.size());
    std::iota(order.begin(), order.end(), size_t(0));
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return stamps[a] < stamps[b]; });
    std::vector<size_t> out;
    for (size_t i : order) {
        if (total <= cap) break;
        out.push_back(i);
        total -= bytes[i];
    }
    return out;
}

// ---- the tier ---------------------------------------------------------------------------------------------------

ConversationDiskTier::~ConversationDiskTier() {
    {
        std::lock_guard<std::mutex> g(m_);
        stop_ = true;
    }
    cv_.notify_all();
    if (th_.joinable()) th_.join();
    drop_lock(lock_);
}

void ConversationDiskTier::say(const std::string& line) const {
    if (s_.log) s_.log("strata serve: conversation disk: " + line);
}

fs::path ConversationDiskTier::strs_path(uint64_t serial) const { return sub_ / ("c-" + hex16(serial) + ".strs"); }
fs::path ConversationDiskTier::idx_path(uint64_t serial) const { return sub_ / ("c-" + hex16(serial) + ".idx"); }

bool ConversationDiskTier::open(const Settings& s) {
    s_ = s;
    if (s_.max_bytes == 0) { say("--conversation-disk-max-gib is 0: the tier is off"); return false; }
    std::error_code ec;
    fs::create_directories(s_.dir, ec);
    if (ec) { say("cannot make " + s_.dir.string() + " (" + ec.message() + "): the tier is off"); return false; }
    sub_ = conversation_disk_subfolder(s_.dir, s_.model, s_.key);
    fs::create_directories(sub_, ec);
    if (ec) { say("cannot make " + sub_.string() + " (" + ec.message() + "): the tier is off"); return false; }
    lock_ = take_lock(sub_ / "lock");
    if (!lock_) { say(sub_.string() + " is in use by another engine (or cannot be locked): the tier is off"); return false; }
    key_hash_ = session_hash64(s_.key.data(), s_.key.size(), kKeySeed);

    // key.txt: a subfolder whose key is not ours (a hash collision, an edited file) is emptied first
    std::vector<uint8_t> old;
    const bool had_key = read_small(sub_ / "key.txt", old, 1 << 20);
    if (!had_key || std::string(old.begin(), old.end()) != s_.key) {
        size_t gone = 0;
        for (const auto& e : fs::directory_iterator(sub_, ec)) {
            const std::string name = e.path().filename().string();
            if (name.rfind("c-", 0) == 0 || is_temp(name)) gone += fs::remove(e.path(), ec) ? 1 : 0;
        }
        std::string why;
        if (!write_small(sub_ / "key.txt", s_.key.data(), s_.key.size(), s_.durable, why)) {
            say(why + ": the tier is off");
            return false;
        }
        if (had_key) say("the files in " + sub_.string() + " were made for another key; " + std::to_string(gone) + " deleted");
    }

    // what is there: an index whose .strs is whole is an entry; the rest are leftovers
    std::vector<std::pair<uint64_t, fs::path>> idxs;
    std::vector<std::pair<uint64_t, uint64_t>> strs;   // serial, size
    std::vector<fs::path> gone;
    for (const auto& e : fs::directory_iterator(sub_, ec)) {
        const std::string name = e.path().filename().string();
        uint64_t serial = 0;
        if (is_temp(name)) gone.push_back(e.path());
        else if (serial_in(name, ".idx", serial)) idxs.emplace_back(serial, e.path());
        else if (serial_in(name, ".strs", serial)) {
            std::error_code se;
            const uint64_t size = e.file_size(se);
            strs.emplace_back(serial, se ? 0 : size);
        }
        if (serial) next_serial_ = std::max(next_serial_, serial + 1);
    }
    for (const auto& p : gone) remove_quiet(p);
    std::set<uint64_t> valid;
    size_t bad = 0;
    for (const auto& [serial, path] : idxs) {
        std::vector<uint8_t> b;
        ConversationDiskEntry entry;
        std::string why;
        const auto it = std::find_if(strs.begin(), strs.end(), [&](const auto& x) { return x.first == serial; });
        const bool ok = read_small(path, b, kMaxIdxBytes) &&
                        conversation_disk_index_decode(b.data(), b.size(), key_hash_, entry, why) &&
                        entry.serial == serial && it != strs.end() && it->second == entry.file_bytes;
        if (!ok) {
            ++bad;
            remove_quiet(path);
            continue;
        }
        valid.insert(serial);
        entries_.push_back(std::move(entry));
    }
    for (const auto& [serial, size] : strs)
        if (!valid.count(serial)) remove_quiet(strs_path(serial));   // (written, but its index never was)
    if (bad) say(std::to_string(bad) + " index file" + (bad == 1 ? " was" : "s were") + " cut short or not whole and deleted");

    // the other subfolders (other keys): counted against the cap and deleted oldest first; one that another engine
    // holds is left alone
    for (const auto& d : fs::directory_iterator(s_.dir, ec)) {
        std::error_code de;
        if (!d.is_directory(de) || d.path() == sub_) continue;
        std::vector<uint8_t> k;
        if (!read_small(d.path() / "key.txt", k, 1 << 20)) continue;
        void* l = take_lock(d.path() / "lock");
        if (!l) continue;
        const uint64_t kh = session_hash64(k.data(), k.size(), kKeySeed);
        for (const auto& f : fs::directory_iterator(d.path(), de)) {
            uint64_t serial = 0;
            if (!serial_in(f.path().filename().string(), ".idx", serial)) continue;
            std::vector<uint8_t> b;
            ConversationDiskEntry entry;
            std::string why;
            Foreign fo;
            fo.idx = f.path();
            fo.strs = d.path() / ("c-" + hex16(serial) + ".strs");
            std::error_code fe;
            const uint64_t size = fs::file_size(fo.strs, fe);
            if (fe || !read_small(f.path(), b, kMaxIdxBytes) ||
                !conversation_disk_index_decode(b.data(), b.size(), kh, entry, why))
                continue;
            fo.bytes = size + b.size();
            fo.stamp = entry.stamp;
            foreign_.push_back(std::move(fo));
        }
        drop_lock(l);
    }

    uint64_t own = 0, other = 0;
    for (const auto& e : entries_) own += e.file_bytes;
    for (const auto& f : foreign_) other += f.bytes;
    {
        std::lock_guard<std::mutex> g(m_);
        make_room_locked(0);
    }
    open_ = true;
    th_ = std::thread([this] { run(); });
    char line[512];
    std::snprintf(line, sizeof line, "%s: %zu conversation%s from before (%.2f GiB; other keys %.2f GiB), cap %.2f GiB",
                  sub_.string().c_str(), entries_.size(), entries_.size() == 1 ? "" : "s", own / 1073741824.0,
                  other / 1073741824.0, s_.max_bytes / 1073741824.0);
    say(line);
    return true;
}

void ConversationDiskTier::remove_entry_locked(size_t i) {
    remove_quiet(idx_path(entries_[i].serial));   // the index first: a crash between the two leaves a leftover .strs
    remove_quiet(strs_path(entries_[i].serial));
    entries_.erase(entries_.begin() + (std::ptrdiff_t) i);
}

bool ConversationDiskTier::make_room_locked(uint64_t incoming) {
    if (incoming > s_.max_bytes) return false;
    std::vector<uint64_t> bytes, stamps;
    for (const auto& e : entries_) bytes.push_back(e.file_bytes), stamps.push_back(e.stamp);
    for (const auto& f : foreign_) bytes.push_back(f.bytes), stamps.push_back(f.stamp);
    std::vector<size_t> victims = conversation_disk_victims(bytes, stamps, incoming, s_.max_bytes);
    if (victims.empty()) return true;
    totals_.capped += victims.size();
    // own entries by descending index (erase keeps the earlier ones' indices), foreign likewise
    std::sort(victims.rbegin(), victims.rend());
    const size_t n_own = entries_.size();
    for (size_t v : victims) {
        if (v >= n_own) {
            Foreign& f = foreign_[v - n_own];
            remove_quiet(f.idx);
            remove_quiet(f.strs);
            foreign_.erase(foreign_.begin() + (std::ptrdiff_t) (v - n_own));
        } else {
            remove_entry_locked(v);
        }
    }
    return true;
}

bool ConversationDiskTier::write_index(const ConversationDiskEntry& e, std::string& error) const {
    const std::vector<uint8_t> b = conversation_disk_index_encode(e, key_hash_);
    return write_small(idx_path(e.serial), b.data(), b.size(), s_.durable, error);
}

bool ConversationDiskTier::keep_hot(const SavedConversation& image) const {
    return conversation_ended_in_tool_call(image.live.ids, s_.turn_token, s_.tool_token);
}

void ConversationDiskTier::spill(SavedConversation&& image) {
    // the copies that may still fill the image land here, on the calling (device) thread: the writer thread runs
    // no device call
    image.pending.reset();
    image.live_pending.reset();
    for (auto& k : image.kv) k.pending.reset();
    if (!on()) return;
    if ((int64_t) image.live.ids.size() < s_.min_tokens || image.live.ids.empty() || !image.stage_images.empty()) {
        std::lock_guard<std::mutex> g(m_);
        ++totals_.skipped;
        return;
    }
    Job j;
    const std::vector<size_t> keep = conversation_disk_checkpoints(image.checkpoints);
    j.entry = conversation_disk_entry_of(image, keep);
    j.entry.tool = keep_hot(image);
    {
        std::lock_guard<std::mutex> g(m_);
        for (size_t i : conversation_disk_superseded(entries_, image)) j.superseded_serials.push_back(entries_[i].serial);
    }
    std::vector<ConversationCheckpoint> kept;
    kept.reserve(keep.size());
    for (size_t i : keep) kept.push_back(std::move(image.checkpoints[i]));
    image.checkpoints = std::move(kept);
    j.image = std::make_unique<SavedConversation>(std::move(image));
    std::unique_lock<std::mutex> g(m_);
    // backpressure: one image waits behind the one being written, no more (each holds its RAM until written)
    idle_.wait(g, [&] { return q_.empty() || stop_; });
    if (stop_) return;
    q_.push_back(std::move(j));
    cv_.notify_all();
}

void ConversationDiskTier::drain() {
    if (!open_) return;
    std::unique_lock<std::mutex> g(m_);
    idle_.wait(g, [&] { return q_.empty() && !busy_; });
}

void ConversationDiskTier::run() {
    std::unique_lock<std::mutex> g(m_);
    for (;;) {
        cv_.wait(g, [&] { return stop_ || !q_.empty(); });
        if (q_.empty()) break;   // stop_ with nothing left: what was queued is written before the thread ends
        Job j = std::move(q_.front());
        q_.pop_front();
        busy_ = true;
        idle_.notify_all();
        g.unlock();
        write_job(j);
        j.image.reset();   // its RAM goes back here, outside the lock
        g.lock();
        busy_ = false;
        idle_.notify_all();
    }
}

void ConversationDiskTier::write_job(Job& j) {
    if (broken_.load()) return;
    const auto t0 = Clock::now();
    uint64_t serial = 0;
    const uint64_t estimate = j.image->bytes();
    {
        std::lock_guard<std::mutex> g(m_);
        // the older copies of this conversation go first (they hold nothing the new file does not)
        for (uint64_t s : j.superseded_serials)
            for (size_t i = 0; i < entries_.size(); ++i)
                if (entries_[i].serial == s) { remove_entry_locked(i); ++totals_.superseded; break; }
        if (!make_room_locked(estimate)) {
            ++totals_.skipped;
            say("skip a " + std::to_string(estimate >> 20) + " MiB conversation: larger than the cap");
            return;
        }
        serial = next_serial_++;
    }
    SessionWriteOptions wo;
    wo.min_free_bytes = s_.min_free_bytes;
    wo.durable = s_.durable;
    SessionStatus st;
    size_t bytes = 0;
    std::string err;
    const fs::path path = strs_path(serial);
    if (!session_file_write(path.string(), *j.image, s_.id, bytes, err, wo, &st)) {
        std::lock_guard<std::mutex> g(m_);
        ++totals_.failed;
        if (st.published) remove_quiet(path);
        if (st.error == SessionError::storage) {
            broken_.store(true);
            say("no room on the disk (" + err + "): the tier is off for this run; parked conversations are freed again");
        } else {
            say("write failed (" + err + "); the conversation is dropped");
        }
        return;
    }
    j.entry.serial = serial;
    j.entry.file_bytes = bytes;
    j.entry.stamp = now_ns();
    if (!write_index(j.entry, err)) {
        std::lock_guard<std::mutex> g(m_);
        ++totals_.failed;
        remove_quiet(path);
        say("index write failed (" + err + "); the conversation is dropped");
        return;
    }
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    std::lock_guard<std::mutex> g(m_);
    const int64_t tokens = j.entry.tokens();
    const size_t marks = j.entry.marks.size();
    const bool tool = j.entry.tool;
    entries_.push_back(std::move(j.entry));
    make_room_locked(0);   // (the exact size now; the estimate made room before)
    ++totals_.written;
    totals_.write_bytes += bytes;
    totals_.write_ms += ms;
    uint64_t own = 0;
    for (const auto& e : entries_) own += e.file_bytes;
    char line[320];
    std::snprintf(line, sizeof line, "wrote %lld tokens (%zu checkpoint%s, last reply %s) %.1f MiB in %.1f ms (%.2f GB/s); "
                  "on disk %zu conversation%s, %.2f GiB",
                  (long long) tokens, marks, marks == 1 ? "" : "s", tool ? "a tool call" : "a final answer",
                  bytes / 1048576.0, ms, ms > 0 ? bytes / ms / 1e6 : 0.0, entries_.size(), entries_.size() == 1 ? "" : "s",
                  own / 1073741824.0);
    say(line);
}

bool ConversationDiskTier::load(const ConversationDiskMatch& m, const SessionReadLimits& limits, SavedConversation& image,
                                size_t& bytes, std::string& error, SessionStatus* status) {
    drain();
    fs::path path;
    {
        std::lock_guard<std::mutex> g(m_);
        const auto it = std::find_if(entries_.begin(), entries_.end(), [&](const auto& e) { return e.serial == m.serial; });
        if (it == entries_.end()) { error = "the entry is gone"; return false; }
        path = strs_path(m.serial);
    }
    SessionStatus st;
    const bool ok = session_file_read(path.string(), s_.id, image, bytes, error, limits, &st);
    if (status) *status = st;
    std::lock_guard<std::mutex> g(m_);
    const auto it = std::find_if(entries_.begin(), entries_.end(), [&](const auto& e) { return e.serial == m.serial; });
    if (it == entries_.end()) return ok;
    if (!ok) {
        ++totals_.failed;
        // a refusal for RAM says nothing about the file; anything else does: it goes
        if (st.error != SessionError::memory) remove_entry_locked((size_t) (it - entries_.begin()));
        return false;
    }
    ++totals_.loaded;
    it->stamp = now_ns();   // the most recently used now (the size cap drops it last)
    std::string why;
    if (!write_index(*it, why)) say("cannot refresh " + idx_path(it->serial).string() + " (" + why + ")");
    return true;
}

size_t ConversationDiskTier::entries() const {
    std::lock_guard<std::mutex> g(m_);
    return entries_.size();
}

uint64_t ConversationDiskTier::bytes() const {
    std::lock_guard<std::mutex> g(m_);
    uint64_t n = 0;
    for (const auto& e : entries_) n += e.file_bytes;
    return n;
}

ConversationDiskTier::Totals ConversationDiskTier::totals() const {
    std::lock_guard<std::mutex> g(m_);
    return totals_;
}

} // namespace strata::core
