// include/strata/core/fragile_recheck.hpp - LEADS.md "HotRod fragile-token re-verify + brake" (STRATA_FRAGILE_RECHECK=1, opt-in,
// a quality guard for the lossy stand-ins).  Pure host rules, no GPU; verify.cpp holds one FragileState and its CPU test
// (src/core/fragile_recheck_test.cpp) drives the same functions.
//
// Ported from sangharshadhyeta/Strata-HotRod (024f644789 "fragile tokens re-decided exactly", 24 Oct; opt-in since a2ef1638cf,
// 9 Nov; route_control.hpp's Wilson bound; the brake of verify.cpp route_fragile_settle / route_brake_grow):
//  - A row r of a decode window is a close call when a stand-in moved it (shift[r] > 0: the sum over its layers and picks of the
//    replaced pick's router weight times its logit gap to the stand-in) and its output gap (top-1 minus top-2 raw logit) is below
//    k * shift[r] (a NaN gap counts as one).  The window is cut before its first close call r >= 1: rows 0..r-1 are kept and the
//    next window re-decides row r as its row 0 with the router's own picks (the hold).  Row 0 is never cut.
//  - k: until it is set, the first kLearnWindows windows only collect the moved rows' gap / shift (no cut) and k is their 10th
//    percentile (clamped to [0.005, 50]).  Then every 50 settled re-checks, k = 1.25 x the largest flipped ratio in the ring of
//    the last kRing re-checks (no flip: k stays), a step bounded to [0.5 k, 1.5 k].  STRATA_FRAGILE_K0=<k> pins k.
//  - Exploration: every 16th eligible window without a normal cut, the first moved row r >= 1 with ratio in [k, 8k) is cut.
//  - The brake (STRATA_FRAGILE_BRAKE, on with the re-check, =0 off): each request starts at budget scale 1; with >= kBrakeMin
//    re-checks and a Wilson 90% lower bound of flips / re-checks above 0.2, or >= 3 flips over >= 100 kept tokens with the
//    flips-per-token lower bound above 0.01, the scale halves (below 0.01 it is 0) and the counts reset.  Every kBrakeWindows
//    decode windows the scale grows x1.25 (from 0: 0.25; at most 1) when the upper bounds are <= 0.2 and <= 0.01.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace strata::core {

/// Wilson score interval at 90% (z = 1.645) for f events in n trials (0 and 1 when n is 0).
inline void fragile_wilson90(long long f, long long n, double& lo, double& hi) {
    if (n <= 0) {
        lo = 0.0;
        hi = 1.0;
        return;
    }
    const double z = 1.645, z2 = z * z, nn = (double) n, p = (double) f / nn;
    const double centre = p + z2 / (2.0 * nn);
    const double half = z * std::sqrt(p * (1.0 - p) / nn + z2 / (4.0 * nn * nn));
    const double den = 1.0 + z2 / nn;
    lo = (centre - half) / den;
    hi = (centre + half) / den;
}

struct FragileState {
    enum { kLearnWindows = 64, kRing = 200, kSamples = 20000, kBrakeMin = 15, kBrakeWindows = 64, kBrakeTokMin = 100, kBrakeFlips = 3 };
    float k = 4.0f;
    bool k_set = false, k_pinned = false;
    int learn_windows = 0;
    std::vector<float> learn_ratio;
    float ring_ratio[kRing] = {};
    bool ring_flip[kRing] = {};
    int ring_pos = 0, ring_n = 0;
    long long session = 0;      // re-checks settled in this process
    long long eligible = 0;     // windows seen once k is set (every 16th may explore)
    bool pend = false, pend_explore = false;
    float pend_ratio = 0.0f;
    int32_t pend_tok = -1;
    // totals and this request's counts (fragile_take reads and resets the request's)
    long long rechecks = 0, flips = 0;
    long long req_cuts = 0, req_explores = 0, req_moved = 0, req_flips = 0;
    // the brake
    bool brake_on = true;
    float req_scale = 1.0f;
    long long req_rc = 0, req_fl = 0, req_tok = 0, req_wins = 0, req_brakes = 0;
};

/// The first close call among rows 0..n-1 (only r >= 1 is cut), or -1.  explore (may be null) says it is an explore cut.
inline int fragile_first(FragileState& s, const float* gap, const float* shift, int n, bool* explore) {
    if (explore != nullptr) *explore = false;
    if (!s.k_set) {
        for (int r = 1; r < n; ++r)
            if (shift[r] > 0.0f && gap[r] == gap[r]) s.learn_ratio.push_back(gap[r] / shift[r]);
        if (++s.learn_windows >= FragileState::kLearnWindows) {
            std::vector<float> v;
            v.swap(s.learn_ratio);
            s.k_set = true;
            if (!v.empty()) {
                std::sort(v.begin(), v.end());
                s.k = std::min(50.0f, std::max(0.005f, v[v.size() / 10]));
            }
        }
        return -1;
    }
    int first = -1;
    for (int r = 0; r < n; ++r) {
        if (!(shift[r] > 0.0f)) continue;
        ++s.req_moved;
        if (r >= 1 && first < 0 && !(gap[r] >= s.k * shift[r])) first = r;
    }
    if (s.eligible++ % 16 == 0 && first < 0) {
        for (int r = 1; r < n && first < 0; ++r) {
            if (!(shift[r] > 0.0f)) continue;
            const float ratio = gap[r] / shift[r];
            if (ratio >= s.k && ratio < 8.0f * s.k) first = r;
        }
        if (first >= 0 && explore != nullptr) *explore = true;
    }
    return first;
}

/// A window cut at row r >= 1: its ratio gap / shift and the stand-in window's top-1 id wait for the exact re-decision.
inline void fragile_cut(FragileState& s, float ratio, int32_t stand_top1, bool explore) {
    ++s.req_cuts;
    if (explore) ++s.req_explores;
    s.pend = true;
    s.pend_explore = explore;
    s.pend_ratio = ratio;
    s.pend_tok = stand_top1;
}

/// The pending cut's exact decision (the next window's row-0 top-1 id).  Returns 1 flipped, 0 kept, -1 nothing was pending.
/// The brake and the k update run here (see the header comment).
inline int fragile_settle(FragileState& s, int32_t exact_top1) {
    if (!s.pend) return -1;
    s.pend = false;
    const bool flipped = exact_top1 != s.pend_tok;
    ++s.rechecks;
    ++s.req_rc;
    if (flipped) {
        ++s.flips;
        ++s.req_flips;
        ++s.req_fl;
    }
    if (s.brake_on && s.req_scale > 0.0f) {
        double lo = 0.0, hi = 0.0, tlo = 0.0, thi = 0.0;
        fragile_wilson90(s.req_fl, s.req_rc, lo, hi);
        fragile_wilson90(s.req_fl, s.req_tok, tlo, thi);
        const bool by_recheck = s.req_rc >= FragileState::kBrakeMin && lo > 0.2;
        const bool by_token = s.req_tok >= FragileState::kBrakeTokMin && s.req_fl >= FragileState::kBrakeFlips && tlo > 0.01;
        if (by_recheck || by_token) {
            const float cut = s.req_scale * 0.5f;
            s.req_scale = cut < 0.01f ? 0.0f : cut;
            ++s.req_brakes;
            s.req_rc = s.req_fl = s.req_tok = 0;
        }
    }
    s.ring_ratio[s.ring_pos] = s.pend_ratio;
    s.ring_flip[s.ring_pos] = flipped;
    s.ring_pos = (s.ring_pos + 1) % FragileState::kRing;
    if (s.ring_n < FragileState::kRing) ++s.ring_n;
    ++s.session;
    if (!s.k_pinned && s.session >= 50 && s.session % 50 == 0) {
        float mx = -1.0f;
        for (int i = 0; i < s.ring_n; ++i)
            if (s.ring_flip[i] && s.ring_ratio[i] == s.ring_ratio[i]) mx = std::max(mx, s.ring_ratio[i]);
        if (mx >= 0.0f) {
            const float old = s.k;
            float target = std::min(50.0f, std::max(0.005f, 1.25f * mx));
            if (target > old) target = std::min(target, 1.5f * old);
            if (target < old) target = std::max(target, 0.5f * old);
            s.k = target;
        }
    }
    return flipped ? 1 : 0;
}

/// A decode window's kept rows (the brake's tokens) and its grow check every kBrakeWindows windows of the request.
inline void fragile_window_done(FragileState& s, int kept) {
    s.req_tok += kept;
    if (!s.brake_on || ++s.req_wins % FragileState::kBrakeWindows != 0 || s.req_rc < FragileState::kBrakeMin) return;
    double lo = 0.0, hi = 0.0, tlo = 0.0, thi = 0.0;
    fragile_wilson90(s.req_fl, s.req_rc, lo, hi);
    fragile_wilson90(s.req_fl, s.req_tok, tlo, thi);
    if (hi > 0.2 || thi > 0.01) return;
    s.req_scale = s.req_scale <= 0.0f ? 0.25f : std::min(1.0f, s.req_scale * 1.25f);
}

/// STRATA_ROUTE_BUDGET_AUTO=1 (with the re-check; HotRod 024f644789 "self-tuning budget", simplified): one decode budget b for the
/// ranks below lo and the floor ranks alike, searched by measured speed.  Decode windows alternate in runs of kRun between the base
/// arm (b) and a candidate (b x (1 + step), or / (1 + step) going down); after kPerArm windows of each arm the candidate wins when
/// its kept tokens per second beat the base by kBar and its flips per kept token are not clearly worse (Wilson 90%: the candidate's
/// lower bound above twice the base's upper bound loses).  A win moves b there (step x 1.25, at most 1); a loss turns the direction
/// and shrinks the step (x 0.7, at least 0.05).  b stays in [0.02, 4].  STRATA_ROUTE_BUDGET_AUTO_B0=<b> starts it (default 0.1).
struct AutoBudget {
    enum { kRun = 8, kPerArm = 96 };
    static constexpr double kBar = 0.02;
    bool on = false;
    float b = 0.1f, step = 0.5f;
    int dir = 1, arm = 0, run_left = kRun, pend_arm = 0;
    double tok[2] = {0.0, 0.0}, sec[2] = {0.0, 0.0};
    long long win[2] = {0, 0}, fl[2] = {0, 0}, rc[2] = {0, 0};
    long long tests = 0, wins = 0;
};
inline float auto_arm_b(const AutoBudget& a, int arm) {
    if (arm == 0) return a.b;
    const float c = a.dir > 0 ? a.b * (1.0f + a.step) : a.b / (1.0f + a.step);
    return std::min(4.0f, std::max(0.02f, c));
}
/// One decode window of the running arm: kept tokens and its seconds.  Returns true when a test ended (b may have moved).
inline bool auto_window(AutoBudget& a, int kept, double secs) {
    if (!a.on || !(secs > 0.0)) return false;
    a.tok[a.arm] += kept;
    a.sec[a.arm] += secs;
    ++a.win[a.arm];
    if (--a.run_left <= 0) {
        a.arm ^= 1;
        a.run_left = AutoBudget::kRun;
    }
    if (a.win[0] < AutoBudget::kPerArm || a.win[1] < AutoBudget::kPerArm) return false;
    const double r0 = a.sec[0] > 0.0 ? a.tok[0] / a.sec[0] : 0.0, r1 = a.sec[1] > 0.0 ? a.tok[1] / a.sec[1] : 0.0;
    double lo0 = 0.0, hi0 = 0.0, lo1 = 0.0, hi1 = 0.0;
    fragile_wilson90(a.fl[0], (long long) a.tok[0], lo0, hi0);
    fragile_wilson90(a.fl[1], (long long) a.tok[1], lo1, hi1);
    const bool worse = lo1 > 2.0 * hi0;
    if (r1 > r0 * (1.0 + AutoBudget::kBar) && !worse) {
        a.b = auto_arm_b(a, 1);
        a.step = std::min(1.0f, a.step * 1.25f);
        ++a.wins;
    } else {
        a.dir = -a.dir;
        a.step = std::max(0.05f, a.step * 0.7f);
    }
    ++a.tests;
    a.tok[0] = a.tok[1] = a.sec[0] = a.sec[1] = 0.0;
    a.win[0] = a.win[1] = a.fl[0] = a.fl[1] = a.rc[0] = a.rc[1] = 0;
    a.arm = 0;
    a.run_left = AutoBudget::kRun;
    return true;
}

/// A request's start: the brake's scale and the request's counts start over (a pending cut of the last request is dropped).
inline void fragile_request_begin(FragileState& s) {
    s.pend = false;
    s.req_scale = 1.0f;
    s.req_cuts = s.req_explores = s.req_moved = s.req_flips = 0;
    s.req_rc = s.req_fl = s.req_tok = s.req_wins = s.req_brakes = 0;
}

}  // namespace strata::core
