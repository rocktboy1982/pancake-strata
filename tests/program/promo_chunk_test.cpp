// A7 (STRATA_PROMO_CHUNK): chunked, cancellable promotions of the asynchronous adaptive tier.  No GPU: a slot is a host
// byte array, a chunk copy is a memcpy, and the rounds are walked as generate.cpp's adapt_tick walks them (plan the next
// chunk of every moving swap, land it at the next tick, apply the cancel rule, repeat; a drain copies the rest at once).
// The invariant checked: a swap that ends Landed holds the incoming blob byte for byte, one that ends Restored holds the
// evicted blob byte for byte, and a cancel copies back exactly the bytes written over the evicted expert.
#include <cstdio>
#include <cstring>
#include <vector>

#include "strata/program/promo_chunk.hpp"

using namespace strata::program;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

struct Swap {
    std::vector<unsigned char> in, out, slot;
    PromoProgress p;
};

static Swap make(size_t total, unsigned char a, unsigned char b) {
    Swap s;
    s.in.resize(total);
    s.out.resize(total);
    for (size_t i = 0; i < total; ++i) {
        s.in[i] = (unsigned char) (a + i * 7);
        s.out[i] = (unsigned char) (b + i * 13);
    }
    s.slot = s.out;   // the slot holds the evicted expert when the round starts (the arena's copy back is a no-op)
    s.p.total = total;
    return s;
}

// one tick's plan + copy (the job thread), returns the bytes copied
static size_t post(std::vector<Swap>& r, size_t chunk, bool all) {
    size_t bytes = 0;
    for (Swap& s : r) {
        const size_t len = promo_next_len(s.p, chunk, all);
        if (len == 0) continue;
        const std::vector<unsigned char>& from = s.p.st == PromoState::Copying ? s.in : s.out;
        std::memcpy(s.slot.data() + s.p.done, from.data() + s.p.done, len);
        s.p.posted = len;
        bytes += len;
    }
    return bytes;
}

static size_t land(std::vector<Swap>& r) {
    size_t fin = 0;
    for (Swap& s : r) fin += promo_chunk_landed(s.p) ? 1 : 0;
    return fin;
}

static bool moving(const std::vector<Swap>& r) {
    for (const Swap& s : r)
        if (promo_moving(s.p)) return true;
    return false;
}

static void parse_tests() {
    CHECK(promo_chunk_parse(nullptr) == 0);
    CHECK(promo_chunk_parse("") == 0);
    CHECK(promo_chunk_parse("0") == 0);
    CHECK(promo_chunk_parse("abc") == 0);
    CHECK(promo_chunk_parse("12x") == 0);
    CHECK(promo_chunk_parse("1") == 65536);           // at least 64 KiB
    CHECK(promo_chunk_parse("262144") == 262144);
    CHECK(promo_chunk_parse("262145") == 262144);     // 4 KiB multiple
    CHECK(promo_chunk_parse("256K") == 262144);
    CHECK(promo_chunk_parse("256k") == 262144);
    CHECK(promo_chunk_parse("1M") == 1048576);
    CHECK(promo_chunk_parse("1G") == 1073741824ull);
    CHECK(promo_chunk_parse("99999999999999999999") == 0 || promo_chunk_parse("99999999999999999999") >= 65536);
}

static void next_len_tests() {
    PromoProgress p;
    p.total = 1000;
    CHECK(promo_next_len(p, 300, false) == 300);
    CHECK(promo_next_len(p, 0, false) == 1000);       // chunk 0 = whole blob (the tier's old copy)
    CHECK(promo_next_len(p, 300, true) == 1000);      // a drain: the rest at once
    p.done = 900;
    CHECK(promo_next_len(p, 300, false) == 100);
    p.done = 1000;
    CHECK(promo_next_len(p, 300, false) == 0);
    p.st = PromoState::Landed;
    p.done = 0;
    CHECK(promo_next_len(p, 300, false) == 0);        // finished swaps copy nothing
    CHECK(!promo_chunk_landed(p));
}

static void cancel_rule_tests() {
    CHECK(promo_should_cancel(0.0f, 1.0f));
    CHECK(!promo_should_cancel(1.0f, 1.0f));          // the incoming one was routed: keep it
    CHECK(!promo_should_cancel(0.0f, 0.0f));          // neither: keep (the pick still stands)
    CHECK(!promo_should_cancel(2.0f, 0.0f));
    CHECK(!promo_should_cancel(-3.0f, -1.0f));        // counts reset under it: never cancel on a negative
}

// whole round without cancels: every swap lands after ceil(total / chunk) ticks, holding the incoming blob
static void plain_round(size_t total, size_t chunk) {
    std::vector<Swap> r;
    for (int i = 0; i < 5; ++i) r.push_back(make(total, (unsigned char) (10 * i), (unsigned char) (100 + i)));
    size_t bytes = post(r, chunk, false), ticks = 0;
    while (moving(r)) {
        land(r);
        ++ticks;
        bytes += post(r, chunk, false);
        CHECK(ticks < 1000);
        if (ticks >= 1000) return;
    }
    const size_t want = chunk == 0 ? 1 : (total + chunk - 1) / chunk;
    CHECK(ticks == want);
    CHECK(bytes == 5 * total);
    for (const Swap& s : r) {
        CHECK(s.p.st == PromoState::Landed);
        CHECK(s.slot == s.in);
    }
}

// cancel after k chunks landed: the evicted blob is back whole and the restore copied exactly the bytes written
static void cancel_round(size_t total, size_t chunk, size_t k) {
    std::vector<Swap> r;
    r.push_back(make(total, 3, 200));
    r.push_back(make(total, 5, 201));   // never cancelled
    for (size_t t = 0; t < k; ++t) {   // a tick: the chunks posted, landed at the next tick
        post(r, chunk, false);
        land(r);
    }
    if (r[0].p.st != PromoState::Copying) {   // it finished before the cancel point: a Landed swap stays (the rule is for Copying)
        CHECK(r[0].p.st == PromoState::Landed);
        CHECK(r[0].slot == r[0].in);
        return;
    }
    // between windows: swap 0's incoming expert was not routed since the pick, the evicted one was
    const size_t written = r[0].p.done;
    CHECK(written == k * chunk);
    CHECK(promo_should_cancel(0.0f, 2.0f));
    const bool at_once = promo_cancel(r[0].p);
    CHECK(at_once == (written == 0));
    CHECK(r[0].p.target == written);
    size_t restored = 0;
    for (int guard = 0; moving(r) && guard < 1000; ++guard) {
        if (r[0].p.st == PromoState::Restoring) restored += promo_next_len(r[0].p, chunk, false);
        post(r, chunk, false);
        land(r);
    }
    CHECK(r[0].p.st == PromoState::Restored);
    CHECK(promo_cancelled(r[0].p));
    CHECK(r[0].slot == r[0].out);
    CHECK(restored == written);
    CHECK(r[1].p.st == PromoState::Landed);
    CHECK(r[1].slot == r[1].in);
}

// a cancel while a chunk is still in flight: that chunk counts as written (the copies back queue behind it)
static void cancel_in_flight() {
    std::vector<Swap> r;
    r.push_back(make(8192, 7, 70));
    post(r, 4096, false);                // [0, 4096) in flight (already copied in this host model)
    CHECK(!promo_cancel(r[0].p));
    CHECK(r[0].p.target == 4096);
    CHECK(r[0].p.posted == 0);
    post(r, 4096, false);
    land(r);
    CHECK(r[0].p.st == PromoState::Restored);
    CHECK(r[0].slot == r[0].out);
}

// a cancel before any chunk was posted (the round's first tick): free, the evicted expert is resident at once
static void cancel_before_first() {
    Swap s = make(4096, 1, 2);
    CHECK(promo_cancel(s.p));
    CHECK(s.p.st == PromoState::Restored);
    CHECK(promo_cancelled(s.p));
    CHECK(!promo_moving(s.p));
    CHECK(s.slot == s.out);
}

// the incoming expert became resident elsewhere after landing: the whole blob goes back (target = total)
static void cancel_landed() {
    std::vector<Swap> r;
    r.push_back(make(3000, 9, 90));
    post(r, 1024, false);
    while (moving(r)) {
        land(r);
        post(r, 1024, false);
    }
    CHECK(r[0].p.st == PromoState::Landed);
    CHECK(!promo_cancel(r[0].p));
    CHECK(r[0].p.st == PromoState::Restoring);
    CHECK(r[0].p.target == 3000);
    post(r, 1024, false);
    while (moving(r)) {
        land(r);
        post(r, 1024, false);
    }
    CHECK(r[0].p.st == PromoState::Restored);
    CHECK(r[0].slot == r[0].out);
}

// a drain in the middle: the rest goes at once, the next landing finishes every swap (restores included)
static void drain_round() {
    std::vector<Swap> r;
    r.push_back(make(10000, 4, 40));
    r.push_back(make(10000, 6, 60));
    post(r, 2048, false);
    land(r);                     // 2048 of each landed
    post(r, 2048, false);
    land(r);                     // 4096
    CHECK(!promo_cancel(r[1].p));   // swap 1 restores 4096 bytes
    post(r, 2048, true);         // drain: rest at once
    CHECK(r[0].p.posted == 10000 - 4096);
    CHECK(r[1].p.posted == 4096);
    CHECK(land(r) == 2);
    CHECK(!moving(r));
    CHECK(r[0].slot == r[0].in);
    CHECK(r[1].slot == r[1].out);
}

int main() {
    parse_tests();
    next_len_tests();
    cancel_rule_tests();
    plain_round(1443840, 262144);   // the daily model's mean expert (1.41 MB) in 256 KiB chunks: 6 ticks
    plain_round(1443840, 0);        // chunk 0 = one copy per swap (the tier's old behaviour)
    plain_round(1000, 4096);        // chunk larger than the blob
    plain_round(65536, 65536);      // exact multiple
    for (size_t k = 0; k <= 7; ++k) cancel_round(1443840, 262144, k);
    cancel_round(5000, 1000, 2);
    cancel_before_first();
    cancel_in_flight();
    cancel_landed();
    drain_round();
    if (failed == 0) std::puts("promo_chunk_test: ok");
    return failed == 0 ? 0 : 1;
}
