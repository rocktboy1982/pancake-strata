#pragma once
// STRATA_MTP_SLOTS=<n> (opt-in, default off; LEADS.md Part Q3, ledger T-1092 / T-1047): the host's half of the MTP drafter's
// expert slots.  Idea from Strata-HotRod (MIT; commits 8423fd685, 5e13e762c, cb59a8130), re-implemented here, nothing copied.
// Only `n_slots` of the drafter layer's `n_expert` (512) routed experts live in VRAM; the other experts are read EXACTLY, from a
// pinned host copy of experts.bin the kernels reach by address (zero copy), so a miss only costs time, never a different draft.
// The drafter's grouping kernel (moe_group_resident_ptr) looks every routed id up in a device table of `n_expert` blob addresses:
// `dev_base + slot * blob` for a resident expert, `host_base + id * blob` for the others.  This class is the map behind that
// table and the policy that updates it:
//   * slot_of / expert_of : the map (a bijection between resident experts and the slots they fill; free slots do not exist
//     after init: init fills every slot, so an install is always an eviction),
//   * choose_installs     : which misses to copy into slots after a round and which resident expert each one replaces (CLOCK:
//     a reference bit per slot, a hand; a fresh install starts with its bit set = one pass of grace; an expert installed in the
//     same call is never a victim; a victim chosen from the map as it will be when the queued copies have run),
//   * mark_used           : the reference bits, from the drafter's `used` bitmap (the kernel marks every resident expert it read),
//   * pointer_table       : the table image the host uploads on the drafter's stream, after the slot copies (the order is written
//     down in TEST_LEDGER.md T-1047),
//   * counters            : hits / misses (routed entries), installs, evictions.
// No CUDA, no I/O: tests/core/mtp_slots_test.cpp drives it with random traces on any machine.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>

namespace strata::core {

struct MtpSlotInstall {
    int32_t expert = -1;   ///< the expert to copy into `slot`
    int32_t slot = -1;
    int32_t victim = -1;   ///< the expert that slot held (it goes back to its host address in the same table update)
};

class MtpSlots {
public:
    /// STRATA_MTP_SLOTS bounds: below 32 the cache is a toy, at 512 there is nothing to cache.
    static constexpr int kMinSlots = 32;

    /// The environment value: the slot count, 0 (off) when unset / empty / 0 / not a number / out of [kMinSlots, n_expert - 1]
    /// (`bad` is set for a value that is set but unusable, so the caller can say so instead of silently running the full blob).
    static int slots_from(const char* v, int n_expert, bool* bad = nullptr) {
        if (bad != nullptr) *bad = false;
        if (v == nullptr || v[0] == '\0') return 0;
        char* end = nullptr;
        const long n = std::strtol(v, &end, 10);
        if (end == v || *end != '\0') { if (bad != nullptr) *bad = true; return 0; }
        if (n == 0) return 0;
        if (n < kMinSlots || n >= n_expert) { if (bad != nullptr) *bad = true; return 0; }
        return (int) n;
    }

    /// Every slot full: `initial` lists the experts of slots 0..n_slots-1 (distinct, in [0, n_expert)); null = experts 0..n_slots-1.
    /// False (map left off) for an unusable shape or list.
    bool init(int n_slots, int n_expert, const int32_t* initial = nullptr) {
        *this = MtpSlots();
        if (n_expert < 2 || n_slots < 1 || n_slots > n_expert) return false;   // n_slots == n_expert: every expert resident (the identity table)
        std::vector<int32_t> slot_of(n_expert, -1), expert_of(n_slots, -1);
        for (int s = 0; s < n_slots; ++s) {
            const int e = initial != nullptr ? initial[s] : s;
            if (e < 0 || e >= n_expert || slot_of[e] >= 0) return false;
            slot_of[e] = s;
            expert_of[s] = e;
        }
        slot_of_ = std::move(slot_of);
        expert_of_ = std::move(expert_of);
        ref_.assign(n_slots, 0);
        fresh_.assign(n_slots, 0);
        n_slots_ = n_slots;
        n_expert_ = n_expert;
        return true;
    }

    bool active() const { return n_slots_ > 0; }
    int n_slots() const { return n_slots_; }
    int n_expert() const { return n_expert_; }
    int slot_of(int e) const { return e >= 0 && e < n_expert_ ? slot_of_[(size_t) e] : -1; }
    int expert_of(int s) const { return s >= 0 && s < n_slots_ ? expert_of_[(size_t) s] : -1; }
    bool resident(int e) const { return slot_of(e) >= 0; }
    bool ref_bit(int s) const { return s >= 0 && s < n_slots_ && ref_[(size_t) s] != 0; }
    int hand() const { return hand_; }

    /// The host model of what a round reads: counts each routed id as a hit (resident) or a miss.  The device kernel keeps its own
    /// counters (ingest); this is the reference the proxy test and the unit test hold the kernel's numbers against.
    void observe(const int32_t* ids, int n) {
        for (int i = 0; i < n; ++i) {
            if (resident(ids[i])) { ++hits_; mark_used(ids[i]); }
            else ++misses_;
        }
    }

    /// The drafter's `used` bitmap and the kernel's entry counters for the rounds since the last service.
    void ingest(const uint8_t* used, uint64_t hit_entries, uint64_t miss_entries) {
        hits_ += hit_entries;
        misses_ += miss_entries;
        if (used != nullptr)
            for (int e = 0; e < n_expert_; ++e)
                if (used[e] != 0) mark_used(e);
    }

    /// A resident expert was read: its slot's reference bit is set (not resident: nothing).
    void mark_used(int e) {
        const int s = slot_of(e);
        if (s >= 0) ref_[(size_t) s] = 1;
    }

    /// The service step.  `want` holds the experts the kernel saw missing at rank 0-1 (possibly repeated, possibly already
    /// installed since: the ring is read after the round, the map has moved on).  Returns the installs to queue, at most
    /// `max_installs` (and at most `out_cap`), and APPLIES them to the map: the map is the image of the device after the queued
    /// copies have run, which is the state the next round sees.  Only distinct experts that are not resident are installed;
    /// ids outside [0, n_expert) are ignored.
    int choose_installs(const int32_t* want, int n_want, int max_installs, MtpSlotInstall* out, int out_cap) {
        if (!active() || want == nullptr || out == nullptr) return 0;
        const int cap = max_installs < out_cap ? max_installs : out_cap;
        int n = 0;
        std::fill(fresh_.begin(), fresh_.end(), (uint8_t) 0);
        for (int i = 0; i < n_want && n < cap; ++i) {
            const int e = want[i];
            if (e < 0 || e >= n_expert_ || slot_of_[(size_t) e] >= 0) continue;   // garbage, resident, or a duplicate installed above
            const int s = pick_victim();
            if (s < 0) break;   // every slot is fresh: nothing left to evict
            const int old = expert_of_[(size_t) s];
            slot_of_[(size_t) old] = -1;
            slot_of_[(size_t) e] = s;
            expert_of_[(size_t) s] = e;
            ref_[(size_t) s] = 1;     // one pass of grace
            fresh_[(size_t) s] = 1;   // and not a victim again inside this call
            out[n++] = {e, s, old};
            ++installs_;
            ++evictions_;
        }
        return n;
    }

    /// The table image: element e is the address of expert e's blob.  Addresses are plain integers (the device base, the base of
    /// the mapped host copy as the device sees it), so the test can use any numbers.
    void pointer_table(uint64_t dev_base, uint64_t host_base, uint64_t blob, uint64_t* out) const {
        for (int e = 0; e < n_expert_; ++e) {
            const int s = slot_of_[(size_t) e];
            out[e] = s >= 0 ? dev_base + (uint64_t) s * blob : host_base + (uint64_t) e * blob;
        }
    }

    /// Whole-map invariant (the test calls it after every step): slot_of and expert_of are inverse bijections.
    bool consistent() const {
        if (!active()) return slot_of_.empty() && expert_of_.empty();
        int resident_n = 0;
        for (int e = 0; e < n_expert_; ++e) {
            const int s = slot_of_[(size_t) e];
            if (s < 0) continue;
            ++resident_n;
            if (s >= n_slots_ || expert_of_[(size_t) s] != e) return false;
        }
        if (resident_n != n_slots_) return false;
        for (int s = 0; s < n_slots_; ++s) {
            const int e = expert_of_[(size_t) s];
            if (e < 0 || e >= n_expert_ || slot_of_[(size_t) e] != s) return false;
        }
        return true;
    }

    uint64_t hits() const { return hits_; }
    uint64_t misses() const { return misses_; }
    uint64_t installs() const { return installs_; }
    uint64_t evictions() const { return evictions_; }
    double hit_rate() const { return hits_ + misses_ > 0 ? (double) hits_ / (double) (hits_ + misses_) : 0.0; }

private:
    /// CLOCK: the hand sweeps; a set bit is cleared and skipped, a clear bit is the victim.  Slots installed in this call are
    /// skipped without clearing.  At most two sweeps (the first clears every bit), -1 when every slot is fresh.
    int pick_victim() {
        for (int step = 0; step < 2 * n_slots_ + 1; ++step) {
            const int s = hand_;
            hand_ = (hand_ + 1) % n_slots_;
            if (fresh_[(size_t) s] != 0) continue;
            if (ref_[(size_t) s] != 0) { ref_[(size_t) s] = 0; continue; }
            return s;
        }
        return -1;
    }

    int n_slots_ = 0, n_expert_ = 0, hand_ = 0;
    std::vector<int32_t> slot_of_, expert_of_;
    std::vector<uint8_t> ref_, fresh_;
    uint64_t hits_ = 0, misses_ = 0, installs_ = 0, evictions_ = 0;
};

}  // namespace strata::core
