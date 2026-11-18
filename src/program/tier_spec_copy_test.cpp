// LEADS.md B147: STRATA_ADAPT_SPEC_COPY (include/strata/program/tier_spec_copy.hpp), CPU only, no model, no GPU.
// Synthetic routing, deterministic.  The last test drives a mock "device" (byte arrays for the victim slots, the staging area and
// the host) through the baseline swap scheme and the staged scheme and compares every slot byte for byte.
#include "strata/program/tier_spec_copy.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <vector>

using strata::program::TierSpecCopy;
using Pred = TierSpecCopy::Pred;

namespace {
void check(bool ok, const char* label) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
void set_env(const char* k, const char* v) {
#if defined(_WIN32)
    _putenv_s(k, v ? v : "");
#else
    if (v) setenv(k, v, 1); else unsetenv(k);
#endif
}
TierSpecCopy make(int slots, int max_copies, float conf, int age, int64_t L, int64_t E) {
    TierSpecCopy::Params p;
    p.on = true;
    p.slots = slots;
    p.max_copies = max_copies;
    p.conf = conf;
    p.age = age;
    TierSpecCopy t;
    t.configure(p, L, E);
    return t;
}
constexpr float kDecay = 0.7f;
}  // namespace

int main() {
    // ---- off: nothing happens
    {
        TierSpecCopy t;
        t.configure(TierSpecCopy::Params{}, 4, 16);
        check(!t.on() && t.staging_slots() == 0, "off by default");
        std::vector<float> raw(64, 1.0f);
        t.begin_round(raw, kDecay);
        std::vector<Pred> preds{{5.0f, 0, 3}};
        check(t.plan(preds).empty(), "off: plan stages nothing");
        check(t.find(0, 3) < 0 && !t.consume(0, 3) && t.rounds() == 0, "off: no lookup, no round");
        std::vector<Pred> out;
        int32_t res[16];
        for (int e = 0; e < 16; ++e) res[e] = e < 8 ? e : -1;
        t.predict_layer(0, res, 2.0f, 1.5f, out);
        check(out.empty(), "off: no prediction");
    }
    // ---- env: parse and bounds (the unset default stays off)
    {
        set_env("STRATA_ADAPT_SPEC_COPY", nullptr);
        check(!TierSpecCopy::params_from_env().on, "env unset = off");
        set_env("STRATA_ADAPT_SPEC_COPY", "0");
        check(!TierSpecCopy::params_from_env().on, "env 0 = off");
        set_env("STRATA_ADAPT_SPEC_COPY", "1");
        TierSpecCopy::Params p = TierSpecCopy::params_from_env();
        check(p.on && p.slots == 4 && p.max_copies == 8 && p.age == 4 && p.conf == 1.0f, "env 1: defaults 4 / 8 / 4 / 1.0");
        set_env("STRATA_ADAPT_SPEC_SLOTS", "2");
        set_env("STRATA_ADAPT_SPEC_MAX", "16");
        set_env("STRATA_ADAPT_SPEC_AGE", "7");
        set_env("STRATA_ADAPT_SPEC_CONF", "2.5");
        p = TierSpecCopy::params_from_env();
        check(p.slots == 2 && p.max_copies == 16 && p.age == 7 && p.conf == 2.5f, "env values read");
        set_env("STRATA_ADAPT_SPEC_SLOTS", "99");
        set_env("STRATA_ADAPT_SPEC_MAX", "0");
        set_env("STRATA_ADAPT_SPEC_AGE", "x");
        set_env("STRATA_ADAPT_SPEC_CONF", "-1");
        p = TierSpecCopy::params_from_env();
        check(p.slots == 4 && p.max_copies == 8 && p.age == 4 && p.conf == 1.0f, "env out of range: defaults");
        for (const char* k : {"STRATA_ADAPT_SPEC_COPY", "STRATA_ADAPT_SPEC_SLOTS", "STRATA_ADAPT_SPEC_MAX", "STRATA_ADAPT_SPEC_AGE",
                              "STRATA_ADAPT_SPEC_CONF"})
            set_env(k, nullptr);
    }
    // ---- predictor: the tier's rule on decay x count + this window's increment
    {
        const int64_t L = 1, E = 16;
        TierSpecCopy t = make(4, 8, 1.0f, 4, L, E);
        std::vector<float> raw((size_t) E, 0.0f);
        raw[10] = 1.0f;                 // round 1: expert 10 routed once
        t.begin_round(raw, kDecay);
        raw[10] = raw[10] * kDecay + 3.0f;   // round 2: experts 10 (3 more), 11 (4) routed
        raw[11] = 4.0f;
        raw[0] = 0.0f;
        t.begin_round(raw, kDecay);
        int32_t res[16];
        for (int e = 0; e < 8; ++e) res[e] = e;          // 0..7 resident with count 0, 8..15 not
        for (int e = 8; e < 16; ++e) res[e] = -1;
        std::vector<Pred> out;
        t.predict_layer(0, res, 2.0f, 1.5f, out);
        // predicted: expert 10: 0.7 x 3.7 + 3 = 5.59; expert 11: 0.7 x 4 + 4 = 6.8; victims all 0
        check(out.size() == 2, "two predicted incoming experts");
        std::sort(out.begin(), out.end(), [](const Pred& a, const Pred& b) { return a.gain > b.gain; });
        check(out[0].expert == 11 && out[1].expert == 10, "best predicted gain first");
        check(std::abs(out[0].gain - 6.8f) < 1e-4f && std::abs(out[1].gain - 5.59f) < 1e-4f, "gain = predicted count - victim count");
        // conf above the second gain removes it
        TierSpecCopy t2 = make(4, 8, 6.0f, 4, L, E);
        std::vector<float> r2((size_t) E, 0.0f);
        t2.begin_round(r2, kDecay);
        r2[10] = 3.0f; r2[11] = 4.0f;
        t2.begin_round(r2, kDecay);
        std::vector<Pred> o2;
        t2.predict_layer(0, res, 2.0f, 1.5f, o2);
        check(o2.size() == 1 && o2[0].expert == 11, "STRATA_ADAPT_SPEC_CONF drops the weaker prediction");
        // a candidate below the tier's floor is not predicted
        TierSpecCopy t3 = make(4, 8, 0.0f, 4, L, E);
        std::vector<float> r3((size_t) E, 0.0f);
        t3.begin_round(r3, kDecay);
        r3[12] = 1.0f;
        t3.begin_round(r3, kDecay);   // predicted 0.7 + 1 = 1.7 < 2
        std::vector<Pred> o3;
        t3.predict_layer(0, res, 2.0f, 1.5f, o3);
        check(o3.empty(), "below the floor: no prediction");
    }
    // ---- plan: cap, keep-alive, full layer, slot uniqueness
    {
        const int64_t L = 3, E = 32;
        TierSpecCopy t = make(2, 3, 0.0f, 4, L, E);
        std::vector<float> raw((size_t) (L * E), 0.0f);
        t.begin_round(raw, kDecay);
        std::vector<Pred> preds;
        for (int l = 0; l < 3; ++l)
            for (int e = 0; e < 4; ++e) preds.push_back({(float) (10 - e - l), l, 20 + e});
        auto p1 = t.plan(preds);
        check(p1.size() == 3, "at most max_copies new copies");
        check(t.staged_total() == 3, "staged counter");
        std::set<int64_t> used;
        for (auto& s : p1) {
            check(s.slot >= 0 && s.slot < 2, "slot inside the layer's slots");
            check(used.insert(s.layer * 2 + s.slot).second, "no slot written twice in one plan");
            check(t.find(s.layer, s.expert) == s.layer * 2 + s.slot, "find returns the staging index");
        }
        // the same predictions again: staged ones are kept alive, no new copy of them
        t.begin_round(raw, kDecay);
        const uint64_t before = t.staged_total();
        std::vector<Pred> again;
        for (auto& s : p1) again.push_back({50.0f, s.layer, s.expert});
        auto p2 = t.plan(again);
        check(p2.empty() && t.staged_total() == before, "re-predicted staged expert: no second copy");
        // a layer with both slots taken drops its oldest for a new best prediction
        TierSpecCopy u = make(2, 8, 0.0f, 10, 1, E);
        std::vector<float> r1((size_t) E, 0.0f);
        u.begin_round(r1, kDecay);
        std::vector<Pred> a{{5, 0, 1}, {4, 0, 2}};
        u.plan(a);                       // round 1: experts 1, 2
        u.begin_round(r1, kDecay);
        std::vector<Pred> b{{9, 0, 3}};
        auto pb = u.plan(b);             // round 2: expert 3 needs a slot: the oldest goes
        check(pb.size() == 1 && u.wasted() == 1, "a full layer evicts one entry (counted unused)");
        check(u.find(0, 3) >= 0 && (u.find(0, 1) < 0 || u.find(0, 2) < 0), "the new one is in, one old one is out");
    }
    // ---- consume and ageing
    {
        TierSpecCopy t = make(2, 8, 0.0f, 2, 1, 32);
        std::vector<float> raw(32, 0.0f);
        t.begin_round(raw, kDecay);
        std::vector<Pred> pr{{5, 0, 7}, {4, 0, 9}};
        t.plan(pr);
        int64_t idx = -1;
        check(t.consume(0, 7, &idx) && idx >= 0, "incoming staged expert: hit");
        check(!t.consume(0, 7), "consumed once");
        check(!t.consume(0, 8), "not staged: miss");
        check(t.hits() == 1 && t.misses() == 2, "hit / miss counters");
        t.begin_round(raw, kDecay);
        t.begin_round(raw, kDecay);
        check(t.find(0, 9) >= 0, "kept for age rounds");
        t.begin_round(raw, kDecay);
        check(t.find(0, 9) < 0 && t.wasted() == 1, "aged out after age rounds, counted unused");
    }
    // ---- unstage: a planned copy that could not be issued is forgotten
    {
        TierSpecCopy t = make(2, 8, 0.0f, 4, 1, 32);
        std::vector<float> raw(32, 0.0f);
        t.begin_round(raw, kDecay);
        std::vector<Pred> pr{{5, 0, 7}};
        auto pl = t.plan(pr);
        check(pl.size() == 1 && t.find(0, 7) >= 0 && t.staged_total() == 1, "staged");
        t.unstage(0, 7);
        check(t.find(0, 7) < 0 && t.staged_total() == 0 && t.wasted() == 0, "unstage: no entry, no count");
    }
    // ---- the mock device: baseline scheme vs staged scheme, every slot byte for byte (lossless)
    {
        const int L = 4, E = 40, SL = 24, BYTES = 64, ROUNDS = 400;   // SL resident slots per layer (of E experts)
        std::vector<uint8_t> host((size_t) L * E * BYTES);
        for (size_t i = 0; i < host.size(); ++i) host[i] = (uint8_t) ((i * 2654435761u) >> 13);
        auto build = [&](bool spec, uint64_t* d2d_out, uint64_t* h2d_out) {
            std::vector<uint8_t> slots((size_t) L * SL * BYTES, 0);
            std::vector<std::vector<int>> res((size_t) L, std::vector<int>((size_t) E, -1));
            for (int l = 0; l < L; ++l)
                for (int s = 0; s < SL; ++s) {
                    res[(size_t) l][(size_t) s] = s;
                    std::memcpy(&slots[((size_t) l * SL + s) * BYTES], &host[((size_t) l * E + s) * BYTES], BYTES);
                }
            TierSpecCopy t = spec ? make(3, 6, 1.0f, 3, L, E) : TierSpecCopy{};
            std::vector<uint8_t> stage(spec ? (size_t) t.staging_slots() * BYTES : 0, 0);
            std::vector<float> usage((size_t) L * E, 0.0f);
            uint64_t s = 12345, d2d = 0, h2d = 0;
            auto rnd = [&] { s = s * 6364136223846793005ull + 1442695040888963407ull; return (uint32_t) (s >> 33); };
            for (int r = 0; r < ROUNDS; ++r) {
                // routing: a drifting hot set per layer, 6 experts per window
                for (int l = 0; l < L; ++l)
                    for (int k = 0; k < 6; ++k) usage[(size_t) l * E + (size_t) ((rnd() % 12 + (size_t) (r / 40) * 5 + (size_t) l) % E)] += 1.0f;
                if (spec) t.begin_round(usage, kDecay);
                // the tier's swap loop (the baseline rule): candidates by count >= 2, victims lowest count, margin 1.5
                for (int l = 0; l < L; ++l) {
                    std::vector<std::pair<float, int>> cand, vict;
                    for (int e = 0; e < E; ++e) {
                        const float u = usage[(size_t) l * E + e];
                        if (res[(size_t) l][(size_t) e] < 0) { if (u >= 2.0f) cand.emplace_back(u, e); }
                        else vict.emplace_back(u, e);
                    }
                    std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
                    std::sort(vict.begin(), vict.end(), [](auto& a, auto& b) { return a.first < b.first || (a.first == b.first && a.second < b.second); });
                    const size_t nc = std::min(cand.size(), vict.size());
                    for (size_t i = 0; i < nc && i < 5; ++i) {
                        if (cand[i].first < vict[i].first + 1.5f) break;
                        const int in = cand[i].second, out = vict[i].second, slot = res[(size_t) l][(size_t) out];
                        int64_t si = -1;
                        if (spec && t.consume(l, in, &si)) {   // device copy from the staging slot
                            std::memcpy(&slots[((size_t) l * SL + slot) * BYTES], &stage[(size_t) si * BYTES], BYTES);
                            ++d2d;
                        } else {                               // the normal host copy
                            std::memcpy(&slots[((size_t) l * SL + slot) * BYTES], &host[((size_t) l * E + in) * BYTES], BYTES);
                            ++h2d;
                        }
                        res[(size_t) l][(size_t) out] = -1;
                        res[(size_t) l][(size_t) in] = slot;
                    }
                }
                if (spec) {                                    // end of round: predict, stage
                    std::vector<Pred> preds;
                    std::vector<int32_t> row((size_t) E);
                    for (int l = 0; l < L; ++l) {
                        for (int e = 0; e < E; ++e) row[(size_t) e] = res[(size_t) l][(size_t) e];
                        t.predict_layer(l, row.data(), 2.0f, 1.5f, preds);
                    }
                    for (auto& st : t.plan(preds))
                        std::memcpy(&stage[((size_t) st.layer * 3 + st.slot) * BYTES], &host[((size_t) st.layer * E + st.expert) * BYTES], BYTES);
                }
                for (float& v : usage) v *= kDecay;
            }
            if (d2d_out) *d2d_out = d2d;
            if (h2d_out) *h2d_out = h2d;
            // the slot contents must be the host bytes of the expert the table says sits there
            for (int l = 0; l < L; ++l)
                for (int e = 0; e < E; ++e) {
                    const int sl = res[(size_t) l][(size_t) e];
                    if (sl >= 0) check(std::memcmp(&slots[((size_t) l * SL + sl) * BYTES], &host[((size_t) l * E + e) * BYTES], BYTES) == 0,
                                       "slot holds the resident expert's bytes");
                }
            return std::make_pair(slots, res);
        };
        uint64_t d2d0 = 0, h2d0 = 0, d2d1 = 0, h2d1 = 0;
        auto base = build(false, &d2d0, &h2d0);
        auto spec = build(true, &d2d1, &h2d1);
        check(base.first == spec.first, "staged scheme: every slot byte equals the baseline's");
        check(base.second == spec.second, "staged scheme: the residency tables equal the baseline's");
        check(d2d0 == 0 && h2d0 > 100, "baseline: only host copies");
        check(d2d1 + h2d1 == h2d0, "staged scheme: the same number of swaps");
        check(d2d1 > 0, "staged scheme: some swaps finalised from the staging area");
        std::printf("tier_spec_copy_test: mock device, %d rounds: %llu swaps, %llu from the staging area (%.1f%%), residency and slot bytes equal to the baseline\n",
                    ROUNDS, (unsigned long long) h2d0, (unsigned long long) d2d1, 100.0 * (double) d2d1 / (double) h2d0);
    }
    // ---- cost of the CPU side at the real model size: 48 layers x 512 experts, the full predict + plan per round
    {
        const int64_t L = 48, E = 512;
        TierSpecCopy t = make(4, 8, 1.0f, 4, L, E);
        std::vector<float> raw((size_t) (L * E), 0.0f);
        std::vector<int32_t> row((size_t) E);
        for (int e = 0; e < E; ++e) row[(size_t) e] = e < 226 ? e : -1;
        uint32_t s = 1;
        const int R = 24;
        double us = 0;
        for (int r = 0; r < R; ++r) {
            for (int i = 0; i < 480; ++i) { s = s * 1664525u + 1013904223u; raw[(s >> 8) % raw.size()] += 1.0f; }
            const auto t0 = std::chrono::steady_clock::now();
            t.begin_round(raw, kDecay);
            std::vector<Pred> preds;
            for (int64_t l = 0; l < L; ++l) t.predict_layer(l, row.data(), 2.0f, 1.5f, preds);
            t.plan(preds);
            us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            for (float& v : raw) v *= kDecay;
        }
        std::printf("tier_spec_copy_test: %.1f us per round at 48 layers x 512 experts (observe + predict + plan)\n", us / R);
    }
    std::printf("tier_spec_copy_test: ok\n");
    return 0;
}
