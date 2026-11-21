// STRATA_MTP_SLOTS (LEADS.md Part Q3, TEST_LEDGER.md T-1047): the host map, its CLOCK policy and the acceptance proxy.  No GPU, no files.
//   1. the environment parse and the shape checks (off = inactive),
//   2. the pointer table: every expert resident = the identity table (base + id * blob); a miss points at the host copy,
//   3. CLOCK: order of victims, the pass of grace, the fresh-install guard, no duplicate installs, the cap, garbage ids,
//   4. hit / miss accounting (observe, ingest),
//   5. a randomized trace: after every service the map is a bijection, the table image matches it, a model of the slot bytes
//      (copies applied in queue order) holds exactly the experts the map names, installs == evictions, hits + misses == entries,
//   6. the drafter-acceptance proxy: a toy drafter reads expert weights THROUGH the table (slot memory or host copy), with a
//      service between rounds on the model of the stream.  With every expert resident, with a large cache and with a small cache the
//      drafts equal the full-blob drafts bit for bit (a miss is an exact read; only the miss COUNT differs, and it equals the number of
//      routed entries the map did not hold).  The negative control uploads the table before the slot bytes land and runs a round in
//      between: its drafts differ, so a wrong stream order is detectable by this test's method.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <vector>

#include "strata/core/mtp_slots.hpp"

using strata::core::MtpSlotInstall;
using strata::core::MtpSlots;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

namespace {
struct Lcg {
    uint64_t s;
    explicit Lcg(uint64_t seed) : s(seed * 6364136223846793005ull + 1442695040888963407ull) {}
    uint32_t next() { s = s * 6364136223846793005ull + 1442695040888963407ull; return (uint32_t) (s >> 33); }
    uint32_t below(uint32_t n) { return next() % n; }
};

// ---- 1 ---------------------------------------------------------------------------------------------------------------------
void test_parse() {
    bool bad = true;
    CHECK(MtpSlots::slots_from(nullptr, 512, &bad) == 0 && !bad);
    CHECK(MtpSlots::slots_from("", 512, &bad) == 0 && !bad);
    CHECK(MtpSlots::slots_from("0", 512, &bad) == 0 && !bad);
    CHECK(MtpSlots::slots_from("256", 512, &bad) == 256 && !bad);
    CHECK(MtpSlots::slots_from("32", 512, &bad) == 32 && !bad);
    CHECK(MtpSlots::slots_from("511", 512, &bad) == 511 && !bad);
    CHECK(MtpSlots::slots_from("31", 512, &bad) == 0 && bad);     // set but unusable: reported, never silently the full blob
    CHECK(MtpSlots::slots_from("512", 512, &bad) == 0 && bad);
    CHECK(MtpSlots::slots_from("abc", 512, &bad) == 0 && bad);
    CHECK(MtpSlots::slots_from("12x", 512, &bad) == 0 && bad);
    CHECK(MtpSlots::slots_from("-5", 512, &bad) == 0 && bad);
    CHECK(MtpSlots::slots_from("256", 512, nullptr) == 256);
    MtpSlots d;   // default constructed = off: nothing resident, nothing installed
    CHECK(!d.active() && d.consistent() && d.slot_of(0) == -1 && !d.resident(3));
    const int32_t want[2] = {1, 2};
    MtpSlotInstall o[4];
    CHECK(d.choose_installs(want, 2, 4, o, 4) == 0);
}

// ---- 2 ---------------------------------------------------------------------------------------------------------------------
void test_table() {
    MtpSlots m;
    CHECK(!m.init(0, 8) && !m.init(9, 8) && !m.init(4, 1));
    const int32_t dup[3] = {1, 2, 1};
    CHECK(!m.init(3, 8, dup) && !m.active());
    const int32_t oob[3] = {1, 2, 8};
    CHECK(!m.init(3, 8, oob) && !m.active());
    // every expert resident: the identity table, nothing to install, every routed entry a hit
    CHECK(m.init(8, 8) && m.active() && m.consistent());
    uint64_t t[8];
    m.pointer_table(1000, 900000, 100, t);
    for (int e = 0; e < 8; ++e) CHECK(t[e] == 1000 + (uint64_t) e * 100);
    const int32_t ids[5] = {7, 0, 3, 3, 5};
    m.observe(ids, 5);
    CHECK(m.hits() == 5 && m.misses() == 0);
    const int32_t w[2] = {1, 2};
    MtpSlotInstall o[2];
    CHECK(m.choose_installs(w, 2, 2, o, 2) == 0);
    // 4 of 8 resident, an explicit initial set: slot s holds initial[s]
    const int32_t init4[4] = {6, 2, 5, 0};
    CHECK(m.init(4, 8, init4) && m.consistent());
    m.pointer_table(1000, 900000, 100, t);
    CHECK(t[6] == 1000 && t[2] == 1100 && t[5] == 1200 && t[0] == 1300);
    CHECK(t[1] == 900100 && t[3] == 900300 && t[4] == 900400 && t[7] == 900700);
    CHECK(m.slot_of(6) == 0 && m.expert_of(3) == 0 && m.slot_of(7) == -1 && m.expert_of(4) == -1 && m.slot_of(99) == -1);
}

// ---- 3 ---------------------------------------------------------------------------------------------------------------------
void test_clock() {
    MtpSlots m;
    CHECK(m.init(4, 8) && m.hand() == 0);   // slots hold 0 1 2 3, every reference bit clear
    MtpSlotInstall o[8];
    const int32_t a[2] = {4, 5};
    CHECK(m.choose_installs(a, 2, 8, o, 8) == 2);
    CHECK(o[0].expert == 4 && o[0].slot == 0 && o[0].victim == 0);
    CHECK(o[1].expert == 5 && o[1].slot == 1 && o[1].victim == 1);
    CHECK(m.ref_bit(0) && m.ref_bit(1) && !m.ref_bit(2) && m.consistent());
    CHECK(m.resident(4) && m.resident(5) && !m.resident(0) && !m.resident(1));
    // next call: slots 2, 3 are clear: they go before the fresh installs' grace runs out
    const int32_t b[3] = {6, 7, 0};
    CHECK(m.choose_installs(b, 3, 8, o, 8) == 3);
    CHECK(o[0].slot == 2 && o[1].slot == 3);
    // the third: every bit is set (4, 5 had grace; 6, 7 are fresh bits, which stay set): the sweep clears all of them from the hand
    // (at slot 0 after the wrap) and the victim is the first slot that came up clear, slot 0 = expert 4
    CHECK(o[2].expert == 0 && o[2].slot == 0 && o[2].victim == 4);
    CHECK(m.slot_of(0) == 0 && !m.resident(4) && m.consistent());
    // a use mark protects: in a fresh map expert 0 (slot 0, hand there) is marked used, so the hand clears its bit and moves on:
    // the victim is expert 1 in slot 1, and the next victim (bit still clear) is expert 2, not expert 0 (its pass is spent only once)
    MtpSlots u;
    CHECK(u.init(4, 8));
    u.mark_used(0);
    const int32_t c[1] = {4};
    CHECK(u.choose_installs(c, 1, 8, o, 8) == 1 && o[0].slot == 1 && o[0].victim == 1 && u.resident(0) && !u.ref_bit(0) && u.ref_bit(1));
    const int32_t c2[1] = {5};
    CHECK(u.choose_installs(c2, 1, 8, o, 8) == 1 && o[0].slot == 2 && o[0].victim == 2 && u.consistent());
    const int32_t c3[1] = {6};   // slot 3 (clear) goes, then the hand wraps onto slot 0 whose bit was cleared by the first sweep
    CHECK(u.choose_installs(c3, 1, 8, o, 8) == 1 && o[0].slot == 3 && o[0].victim == 3);
    const int32_t c4[1] = {7};   // hand 0: slot 0 (clear) is the victim: the mark bought exactly one sweep
    CHECK(u.choose_installs(c4, 1, 8, o, 8) == 1 && o[0].slot == 0 && o[0].victim == 0 && u.consistent());
    // no duplicate installs, residents and garbage are skipped, the cap is honoured
    MtpSlots n;
    CHECK(n.init(6, 12));
    const int32_t w[9] = {9, 9, 3, -1, 40, 10, 9, 11, 10};   // distinct misses: 9, 10, 11 (3 is resident)
    MtpSlotInstall p[8];
    CHECK(n.choose_installs(w, 9, 2, p, 8) == 2 && p[0].expert == 9 && p[1].expert == 10);
    CHECK(n.choose_installs(w, 9, 8, p, 8) == 1 && p[0].expert == 11);   // 9, 10 are resident now; 11 is the only miss left
    CHECK(n.choose_installs(w, 9, 8, p, 8) == 0);
    CHECK(n.installs() == 3 && n.evictions() == 3 && n.consistent());
    CHECK(n.choose_installs(w, 9, 8, p, 1) == 0);   // out_cap = 1: nothing left anyway
    // a call cannot evict what it installed: 2 slots, 3 misses wanted -> 2 installs
    MtpSlots q;
    CHECK(q.init(2, 6));
    const int32_t w3[3] = {3, 4, 5};
    MtpSlotInstall r[3];
    CHECK(q.choose_installs(w3, 3, 3, r, 3) == 2 && r[0].slot != r[1].slot && q.resident(3) && q.resident(4) && !q.resident(5) && q.consistent());
}

// ---- 4 ---------------------------------------------------------------------------------------------------------------------
void test_accounting() {
    MtpSlots m;
    CHECK(m.init(4, 8));
    const int32_t ids[6] = {0, 1, 5, 6, 0, 7};   // 0, 1, 0 resident (3 hits); 5, 6, 7 not (3 misses)
    m.observe(ids, 6);
    CHECK(m.hits() == 3 && m.misses() == 3 && m.hit_rate() == 0.5);
    CHECK(m.ref_bit(0) && m.ref_bit(1) && !m.ref_bit(2));
    uint8_t used[8] = {0, 0, 0, 1, 0, 0, 0, 0};
    m.ingest(used, 10, 4);   // the device's counters for the next rounds
    CHECK(m.hits() == 13 && m.misses() == 7 && m.ref_bit(3));
}

// ---- 5 ---------------------------------------------------------------------------------------------------------------------
// A model of the device: slot bytes (the expert id each slot's data belongs to) and a queue of copies applied in order.
struct DeviceModel {
    std::vector<int32_t> slot_data;   // which expert's bytes sit in each slot
    std::vector<uint64_t> table;      // the table the "kernel" reads
};

void test_random_trace() {
    const int shapes[][2] = {{32, 512}, {64, 512}, {256, 512}, {5, 40}, {1, 10}, {40, 41}};
    for (const auto& sh : shapes) {
        const int ns = sh[0], ne = sh[1];
        MtpSlots m;
        CHECK(m.init(ns, ne));
        DeviceModel dev;
        dev.slot_data.resize((size_t) ns);
        for (int s = 0; s < ns; ++s) dev.slot_data[(size_t) s] = s;
        dev.table.resize((size_t) ne);
        const uint64_t DEV = 1ull << 40, HOST = 1ull << 44, BLOB = 1382400;
        m.pointer_table(DEV, HOST, BLOB, dev.table.data());
        Lcg rng(1234 + (uint64_t) ns * 7 + (uint64_t) ne);
        uint64_t entries = 0;
        for (int round = 0; round < 20000; ++round) {
            // a skewed trace: most picks from a hot set that drifts, some uniform
            const int hot = (round / 500) * 37 % ne;
            int32_t ids[10];
            for (int i = 0; i < 10; ++i)
                ids[i] = (int32_t) (rng.below(4) != 0 ? (hot + rng.below((uint32_t) (ne / 4 > 0 ? ne / 4 : 1))) % ne : rng.below((uint32_t) ne));
            // the kernel's view: it reads the TABLE
            for (int i = 0; i < 10; ++i) {
                const uint64_t a = dev.table[(size_t) ids[i]];
                if (a >= HOST) CHECK(a == HOST + (uint64_t) ids[i] * BLOB);
                else {
                    const int s = (int) ((a - DEV) / BLOB);
                    CHECK(a == DEV + (uint64_t) s * BLOB && s < ns && dev.slot_data[(size_t) s] == ids[i]);   // the slot holds this expert's bytes
                }
            }
            m.observe(ids, 10);
            entries += 10;
            // service: want = rank 0-1 misses (the first two picks), distinct per call
            int32_t want[2];
            int nw = 0;
            for (int i = 0; i < 2; ++i)
                if (!m.resident(ids[i])) want[nw++] = ids[i];
            MtpSlotInstall inst[4];
            const int n = m.choose_installs(want, nw, 4, inst, 4);
            for (int i = 0; i < n; ++i) dev.slot_data[(size_t) inst[i].slot] = inst[i].expert;   // the queued copies, in order
            if (n > 0) m.pointer_table(DEV, HOST, BLOB, dev.table.data());                       // then the table
            if (!m.consistent()) { CHECK(m.consistent()); break; }
            // the table image equals the map, the slot model equals the map
            if (round % 997 == 0) {
                std::set<uint64_t> seen;
                for (int e = 0; e < ne; ++e) {
                    const int s = m.slot_of(e);
                    CHECK(dev.table[(size_t) e] == (s >= 0 ? DEV + (uint64_t) s * BLOB : HOST + (uint64_t) e * BLOB));
                    if (s >= 0) { CHECK(seen.insert(dev.table[(size_t) e]).second && dev.slot_data[(size_t) s] == e); }
                }
            }
        }
        CHECK(m.hits() + m.misses() == entries);
        CHECK(m.installs() == m.evictions());
        if (ns == ne) CHECK(m.misses() == 0 && m.installs() == 0);
        else CHECK(m.misses() > 0);
    }
}

// ---- 6 ---------------------------------------------------------------------------------------------------------------------
// The toy drafter: each expert has 4 weights; a round routes 10 experts from the context state, sums the weights it READS (through
// the table: slot memory or the host copy), and the draft is a hash of the sum.  Memory model: dev[slot][4], host[expert][4].
struct Toy {
    int ne;
    std::vector<float> host;   // host[e * 4 + k]: the pinned copy, never changes
    std::vector<float> dev;    // dev[s * 4 + k]: the slots' bytes
    Toy(int ne_, int ns) : ne(ne_), host((size_t) ne_ * 4), dev((size_t) ns * 4) {
        Lcg r(99);
        for (auto& x : host) x = (float) (r.below(2001) - 1000) / 64.0f;
    }
    void copy_in(int slot, int expert) { std::memcpy(&dev[(size_t) slot * 4], &host[(size_t) expert * 4], 4 * sizeof(float)); }
    // address model: [0, ns) = a slot index, [HOSTBASE + e) = host expert e
    static constexpr uint64_t HOSTBASE = 1u << 20;
    uint32_t draft(const int32_t* ids, const uint64_t* table) const {
        float acc[4] = {0, 0, 0, 0};
        for (int i = 0; i < 10; ++i) {
            const uint64_t a = table[ids[i]];
            const float* w = a >= HOSTBASE ? &host[(size_t) (a - HOSTBASE) * 4] : &dev[(size_t) a * 4];
            for (int k = 0; k < 4; ++k) acc[k] += w[k] * (float) (i + 1);
        }
        uint32_t h = 2166136261u;
        for (int k = 0; k < 4; ++k) { uint32_t b; std::memcpy(&b, &acc[k], 4); h = (h ^ b) * 16777619u; }
        return h;
    }
};

struct ProxyResult { std::vector<uint32_t> drafts; uint64_t misses = 0, hits = 0; };

// ordering: 0 = the real one (slot bytes, then the table, then the next round); 1 = the hazard (the table first, a round between,
// then the slot bytes land)
ProxyResult run_proxy(int ns, int ne, int rounds, int ordering) {
    Toy toy(ne, ns);
    MtpSlots m;
    CHECK(m.init(ns, ne));
    for (int s = 0; s < ns; ++s) toy.copy_in(s, s);
    auto image = [&](std::vector<uint64_t>& t) {
        t.resize((size_t) ne);
        for (int e = 0; e < ne; ++e) { const int s = m.slot_of(e); t[(size_t) e] = s >= 0 ? (uint64_t) s : Toy::HOSTBASE + (uint64_t) e; }
    };
    std::vector<uint64_t> table;
    image(table);
    Lcg rng(4242);
    ProxyResult res;
    uint64_t independent_misses = 0;
    std::vector<MtpSlotInstall> late;   // the hazard: slot copies that land after the next round
    for (int r = 0; r < rounds; ++r) {
        const int hot = (r / 300) * 53 % ne;
        int32_t ids[10];
        for (int i = 0; i < 10; ++i)
            ids[i] = (int32_t) (rng.below(5) != 0 ? (hot + rng.below((uint32_t) (ne / 5 > 0 ? ne / 5 : 1))) % ne : rng.below((uint32_t) ne));
        for (int i = 0; i < 10; ++i) independent_misses += table[(size_t) ids[i]] >= Toy::HOSTBASE ? 1 : 0;
        res.drafts.push_back(toy.draft(ids, table.data()));
        for (const MtpSlotInstall& in : late) toy.copy_in(in.slot, in.expert);   // (hazard) the bytes land after that round
        late.clear();
        m.observe(ids, 10);
        int32_t want[2];
        int nw = 0;
        for (int i = 0; i < 2; ++i)
            if (!m.resident(ids[i])) want[nw++] = ids[i];
        MtpSlotInstall inst[4];
        const int n = m.choose_installs(want, nw, 4, inst, 4);
        if (ordering == 0) {
            for (int i = 0; i < n; ++i) toy.copy_in(inst[i].slot, inst[i].expert);
            if (n > 0) image(table);
        } else {
            if (n > 0) image(table);                              // table first ...
            for (int i = 0; i < n; ++i) late.push_back(inst[i]);   // ... the bytes land after the next round
        }
    }
    res.misses = m.misses();
    res.hits = m.hits();
    if (ordering == 0) CHECK(independent_misses == m.misses());   // the declared misses are exactly the entries the map did not hold
    return res;
}

void test_proxy() {
    const int ne = 512, rounds = 6000;
    const ProxyResult full = run_proxy(ne, ne, rounds, 0);   // every expert resident: the reference (what the default drafter reads)
    CHECK(full.misses == 0 && full.hits == (uint64_t) rounds * 10);
    uint64_t prev_misses = 0;
    for (const int ns : {480, 256, 128, 32}) {
        const ProxyResult r = run_proxy(ns, ne, rounds, 0);
        CHECK(r.drafts == full.drafts);                       // exact zero copy: identical drafts, in every round
        CHECK(r.hits + r.misses == (uint64_t) rounds * 10);
        if (ns == 480) CHECK(r.misses < r.hits);              // a large cache mostly hits
        if (ns == 32) CHECK(r.misses > 0 && r.misses >= prev_misses);
        prev_misses = r.misses;
    }
    // the hazard control: the table points into a slot before its bytes arrived and a round reads it: some draft differs
    const ProxyResult bad = run_proxy(64, ne, rounds, 1);
    int diff = 0;
    for (size_t i = 0; i < bad.drafts.size(); ++i) diff += bad.drafts[i] != full.drafts[i];
    CHECK(diff > 0);
    std::printf("proxy: hazard control differs in %d of %d rounds; cache sizes 480/256/128/32 equal the full blob in all %d rounds\n", diff, rounds, rounds);
}
}  // namespace

int main() {
    test_parse();
    test_table();
    test_clock();
    test_accounting();
    test_random_trace();
    test_proxy();
    if (failed != 0) { std::fprintf(stderr, "mtp_slots_test: %d check(s) failed\n", failed); return 1; }
    std::printf("mtp_slots_test: ok\n");
    return 0;
}
