// MOD_PLAN item 39: the automatic disk tier for --serve's parked conversations (opt-in: --conversation-disk-dir).
//
// When the conversation cache evicts a parked conversation (ConversationCache::evict_oldest, the RAM budget or the
// physical-RAM gate), the image is handed to this tier instead of being freed.  A thread of its own writes it with
// session_file_write (direct 16 MiB blocks, flush, atomic rename: conversation_file.cpp) keeping TWO checkpoints - the
// root (the shallowest one, normally the end of the shared system prompt and tools) and the deepest one (the next
// turn's resume point) - and then a small index file beside it: the token lists of the live state and of the two
// checkpoints, so a request is matched against the disk without opening the large file.  A request whose prompt
// starts with a disk entry's tokens, deeper than anything in RAM, reads the file back through the same checks a
// /slots RESTORE makes (identity, read limits, RAM preflight, payload hash) and puts it into the conversation cache,
// where the ordinary parked-restore path (validate, take or borrow) uses it.  The files outlive the process, so a
// conversation parked before a restart is found by the next engine.
//
// The folder given holds one subfolder per key (strata-poly's conversation_disk idea): the key names the model file,
// the session identity (model + config fingerprints: conversation_file.hpp) and the lossy routing switches
// (STRATA_ROUTE_PRIOR, ...) that are not in the config fingerprint, so a state made by a lossy preset is never
// resumed by the exact one.  key.txt in the subfolder says what it is valid for; a lock file keeps a second engine
// out.  The whole folder is capped (--conversation-disk-max-gib): the least recently used entries go first, in any
// subfolder.
//
// Eviction policy (Bidaw, FAST'26): with the tier on, the cache demotes a conversation whose last reply ended in a
// final answer before one whose last reply ended in a tool call - the client comes straight back to the latter with
// the tool's result.  Pure host code: no CUDA.
#pragma once

#include "strata/core/conversation_file.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace strata::core {

// ---- the key ----------------------------------------------------------------------------------------------------

/// How a STRATA_* environment switch relates to the key (src/core/conversation_disk_env.def, the audit of every name in src/
/// and include/; tests/core/conversation_disk_env_test.cpp keeps the table complete).
enum class EnvClass {
    Key,        ///< changes what a state holds; always in the key ("-" when unset)
    KeyIfSet,   ///< changes what a state holds; in the key only when set, so a run without it keeps its key
    Unsure,     ///< may change what a state holds, unverified: keyed like KeyIfSet
    InConfig,   ///< changes what a state holds but is in the session config fingerprint already
    Speed,      ///< placement, scheduling, drafting policy or documented bit-identical: never in the key
    Inert,      ///< diagnostics, test hooks, paths, limits, io, tools, other engines: never in the key
};
struct EnvSwitch {
    const char* name;
    EnvClass cls;
    const char* note;
};
const std::vector<EnvSwitch>& conversation_env_switches();

/// The environment switches that change what a state holds without being in the session config fingerprint: the
/// lossy routing of the turbo presets and the output-changing numerics (the Key class of the table above; the KeyIfSet and
/// Unsure ones join conversation_disk_key_env only when set).  STRATA_CONVERSATION_DISK_KEY_ENV (a comma list) adds names.
const std::vector<std::string>& conversation_disk_key_env_names();
/// (name, value) of every key switch, "-" for an unset one, in a fixed order (names, then the extra ones sorted).
std::vector<std::pair<std::string, std::string>>
conversation_disk_key_env(const std::function<const char*(const char*)>& get_env);

struct ConversationDiskKeyInputs {
    SessionFileIdentity id;
    std::string model;   ///< the model file the engine serves (its path; the first native shard or the pack)
    std::vector<std::pair<std::string, std::string>> env;
};
/// What key.txt holds: one "name value" line per input.
std::string conversation_disk_key(const ConversationDiskKeyInputs& in);
/// The subfolder of `dir` for a key: the model file's stem (plain characters, at most 48) and 16 hex digits of the
/// key's hash.
std::filesystem::path conversation_disk_subfolder(const std::filesystem::path& dir, const std::string& model,
                                                  const std::string& key);

// ---- what is written and how it is chosen -----------------------------------------------------------------------

/// The checkpoints a disk entry keeps, as indices into `chain` in chain order: the root (the shallowest checkpoint)
/// and the deepest one (session_deepest_checkpoint's rule: the deepest pinned one when the chain has a pin).  One
/// index when they are the same, none for an empty chain.
std::vector<size_t> conversation_disk_checkpoints(const std::vector<ConversationCheckpoint>& chain);

/// The conversation's last reply ended in a tool call: `tool_token` (Qwen3.x <tool_call>) occurs after the last
/// `turn_token` (<|im_start|>, the reply's own turn).  False when either token is negative or the reply has none.
bool conversation_ended_in_tool_call(const std::vector<int32_t>& ids, int64_t turn_token, int64_t tool_token);

/// An entry as its index file has it: the token lists (ids + pictures only; the running state stays in the large
/// file), whether its last reply ended in a tool call, its size and when it was last used.
struct ConversationDiskEntry {
    uint64_t serial = 0;       ///< c-<serial>.strs / c-<serial>.idx
    uint64_t stamp = 0;        ///< last use (ns since the epoch): the size cap drops the lowest first
    uint64_t file_bytes = 0;   ///< the .strs file's size
    bool cvec = true, tool = false;
    ConversationCheckpoint live;               ///< ids + imgs only
    std::vector<ConversationCheckpoint> marks; ///< the kept checkpoints' ids + imgs, in file order
    int64_t tokens() const { return (int64_t) live.ids.size(); }
};
std::vector<uint8_t> conversation_disk_index_encode(const ConversationDiskEntry& e, uint64_t key_hash);
/// Bounded parse: false (and a reason) for a short, foreign, corrupt or over-long file.
bool conversation_disk_index_decode(const uint8_t* data, size_t n, uint64_t key_hash, ConversationDiskEntry& e,
                                    std::string& error);
/// The entry's index from a parked image: its live tokens and the checkpoints `keep` names.
ConversationDiskEntry conversation_disk_entry_of(const SavedConversation& image, const std::vector<size_t>& keep);

struct ConversationDiskMatch {
    size_t index = SIZE_MAX;
    int64_t tokens = 0;
    uint64_t serial = 0;
};
/// The deepest entry state (live or kept checkpoint) this prompt continues: conversation_prefix's rule (same tokens,
/// same pictures below, shorter than the prompt), same steering.  Ties prefer the most recently used entry.
template<class Token>
ConversationDiskMatch conversation_disk_best(const std::vector<ConversationDiskEntry>& entries,
                                             const std::vector<Token>& prompt,
                                             const std::vector<ConversationImageKey>& images, bool cvec) {
    ConversationDiskMatch best;
    uint64_t best_stamp = 0;
    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& e = entries[i];
        if (e.cvec != cvec) continue;
        auto consider = [&](const ConversationCheckpoint& c) {
            const int64_t n = conversation_prefix(c, prompt, images);
            if (n > best.tokens || (n > 0 && n == best.tokens && e.stamp > best_stamp)) {
                best = {i, n, e.serial};
                best_stamp = e.stamp;
            }
        };
        consider(e.live);
        for (const auto& c : e.marks) consider(c);
    }
    return best;
}

/// #342 on disk: the entries a new image supersedes - its chain (live state or any checkpoint, before the two are
/// chosen) holds the entry's deepest kept checkpoint, so the entry is the same conversation a turn or more back.
std::vector<size_t> conversation_disk_superseded(const std::vector<ConversationDiskEntry>& entries,
                                                 const SavedConversation& image);

/// The size cap: indices of the entries to delete, lowest stamp first, so that sum(bytes) + incoming <= cap.  Empty
/// when it already fits; every entry when even that is not enough (the caller then skips the new one if it still
/// does not fit).
std::vector<size_t> conversation_disk_victims(const std::vector<uint64_t>& bytes, const std::vector<uint64_t>& stamps,
                                              uint64_t incoming, uint64_t cap);

// ---- the tier ---------------------------------------------------------------------------------------------------

class ConversationDiskTier {
public:
    struct Settings {
        std::filesystem::path dir;        ///< --conversation-disk-dir (the top folder; subfolders per key)
        std::string model;                ///< the model file (subfolder name)
        std::string key;                  ///< conversation_disk_key
        SessionFileIdentity id;           ///< what session_file_write binds a file to
        uint64_t max_bytes = 0;           ///< --conversation-disk-max-gib, for the whole folder
        uint64_t min_free_bytes = 0;      ///< the volume keeps this much free (--session-min-free-mib)
        int64_t min_tokens = 0;           ///< a shorter conversation is not written (re-reading it is cheap)
        int64_t turn_token = -1, tool_token = -1;   ///< conversation_ended_in_tool_call
        bool durable = true;              ///< flush each file before its rename (tests: off)
        std::function<void(const std::string&)> log;
    };

    ConversationDiskTier() = default;
    ConversationDiskTier(const ConversationDiskTier&) = delete;
    ConversationDiskTier& operator=(const ConversationDiskTier&) = delete;
    /// Writes what is queued, then stops the writer.
    ~ConversationDiskTier();

    /// Takes the subfolder (lock, key check, leftovers removed), reads the index files, and starts the writer.  False
    /// (one log line) when it cannot be used: the engine then runs without the tier.
    bool open(const Settings& s);
    bool on() const { return open_ && !broken_.load(); }

    /// The eviction sink: the image goes to disk in the background (the caller has joined any copy still filling it).
    /// Too short, or nothing to keep: dropped.  Blocks while one image is already waiting behind the one being written.
    void spill(SavedConversation&& image);
    /// Waits until everything handed to spill() is written (or failed).
    void drain();

    template<class Token>
    ConversationDiskMatch best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images,
                               bool cvec) const {
        std::lock_guard<std::mutex> g(m_);
        return conversation_disk_best(entries_, prompt, images, cvec);
    }
    /// Reads the match's file (drain() first: no write runs meanwhile).  The entry stays on disk (it stays valid until
    /// the conversation moves on; a later spill of it supersedes it) and becomes the most recently used one.  A file
    /// that fails to read is deleted with its index.
    bool load(const ConversationDiskMatch& m, const SessionReadLimits& limits, SavedConversation& image, size_t& bytes,
              std::string& error, SessionStatus* status = nullptr);

    /// The keep-hot rule for ConversationCache::set_disk_tier: true while the image's last reply ended in a tool call.
    bool keep_hot(const SavedConversation& image) const;

    size_t entries() const;
    uint64_t bytes() const;   ///< this subfolder's files
    struct Totals {
        uint64_t written = 0, write_bytes = 0, skipped = 0, superseded = 0, capped = 0, loaded = 0, failed = 0;
        double write_ms = 0;
    };
    Totals totals() const;

private:
    struct Job {
        std::unique_ptr<SavedConversation> image;   ///< pruned to the kept checkpoints
        ConversationDiskEntry entry;
        std::vector<size_t> superseded_serials;     ///< (serials, computed on the main thread from the full chain)
    };
    struct Foreign {   ///< another subfolder's entry (another key): counted against the cap, deleted oldest first
        std::filesystem::path strs, idx;
        uint64_t bytes = 0, stamp = 0;
    };
    void run();
    void write_job(Job& j);
    bool write_index(const ConversationDiskEntry& e, std::string& error) const;
    void remove_entry_locked(size_t i);
    bool make_room_locked(uint64_t incoming);
    std::filesystem::path strs_path(uint64_t serial) const;
    std::filesystem::path idx_path(uint64_t serial) const;
    void say(const std::string& line) const;

    Settings s_;
    std::filesystem::path sub_;
    uint64_t key_hash_ = 0, next_serial_ = 1;
    bool open_ = false;
    std::atomic<bool> broken_{false};
    void* lock_ = nullptr;
    std::vector<ConversationDiskEntry> entries_;   // under m_
    std::vector<Foreign> foreign_;                 // under m_
    mutable std::mutex m_;
    std::condition_variable cv_, idle_;
    std::deque<Job> q_;
    bool busy_ = false, stop_ = false;
    Totals totals_;
    std::thread th_;
};

} // namespace strata::core
