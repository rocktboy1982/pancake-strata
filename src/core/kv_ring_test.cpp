// src/core/kv_ring_test.cpp - the depth-adaptive resident K/V ring (STRATA_KV_RING_ADAPT; header-only, CPU only).
#include "strata/core/kv_ring.hpp"

#include <cstdint>
#include <cstdio>

using namespace strata::core;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    const int64_t floor = kKvRingFloorCells;
    CHECK(floor == 32768);

    // the switch: unset, empty and 0 are off; a number other than 0 is on; text with no number is off
    CHECK(!kv_ring_adapt_enabled(nullptr));
    CHECK(!kv_ring_adapt_enabled(""));
    CHECK(!kv_ring_adapt_enabled("0"));
    CHECK(!kv_ring_adapt_enabled("yes"));
    CHECK(kv_ring_adapt_enabled("1"));
    CHECK(kv_ring_adapt_enabled("2"));

    // switch off: the cap is returned whatever the depth (the default path)
    CHECK(kv_ring_adapt(false, 49152, floor, 1) == 49152);
    CHECK(kv_ring_adapt(false, 49152, floor, 131072) == 49152);
    CHECK(kv_ring_adapt(false, 32768, floor, 131072) == 32768);
    CHECK(kv_ring_adapt(false, 0, floor, 131072) == 0);

    // KV streaming off (cap 0, or a negative cap that never gets here): unchanged
    CHECK(kv_ring_adapt(true, 0, floor, 131072) == 0);
    CHECK(kv_ring_adapt(true, -1, floor, 131072) == -1);

    // cap at or below the floor: the cap, at any depth (nothing to grow)
    CHECK(kv_ring_adapt(true, 32768, floor, 1) == 32768);
    CHECK(kv_ring_adapt(true, 32768, floor, 262144) == 32768);
    CHECK(kv_ring_adapt(true, 20480, floor, 1) == 20480);       // below the floor: not raised to it
    CHECK(kv_ring_adapt(true, 20480, floor, 262144) == 20480);
    CHECK(kv_ring_adapt(true, 1, floor, 1 << 20) == 1);

    // cap 49152 (the PR's measured cap): the 2 x cap = 98304 edge
    const int64_t cap = 49152;
    CHECK(kv_ring_adapt(true, cap, floor, 1) == floor);          // 1K: the floor
    CHECK(kv_ring_adapt(true, cap, floor, 1024) == floor);
    CHECK(kv_ring_adapt(true, cap, floor, 32768) == floor);
    CHECK(kv_ring_adapt(true, cap, floor, 49152) == floor);      // context == cap: the rejected rule would grow here
    CHECK(kv_ring_adapt(true, cap, floor, 65536) == floor);      // 64K: the measured no-op
    CHECK(kv_ring_adapt(true, cap, floor, 98303) == floor);      // one below 2 x cap
    CHECK(kv_ring_adapt(true, cap, floor, 98304) == cap);        // exactly 2 x cap
    CHECK(kv_ring_adapt(true, cap, floor, 98305) == cap);
    CHECK(kv_ring_adapt(true, cap, floor, 131072) == cap);       // 128K: the measured +23.9% case
    CHECK(kv_ring_adapt(true, cap, floor, 262144) == cap);
    CHECK(kv_ring_adapt(true, cap, floor, INT64_MAX) == cap);

    // empty or negative depth reads as the shallowest run
    CHECK(kv_ring_adapt(true, cap, floor, 0) == floor);
    CHECK(kv_ring_adapt(true, cap, floor, -5) == floor);

    // a cap just above the floor: edge at 2 x 32769
    CHECK(kv_ring_adapt(true, 32769, floor, 65537) == floor);
    CHECK(kv_ring_adapt(true, 32769, floor, 65538) == 32769);

    // a huge cap must not overflow the 2 x cap test
    CHECK(kv_ring_adapt(true, INT64_MAX, floor, INT64_MAX) == floor);
    CHECK(kv_ring_adapt(true, INT64_MAX / 4, floor, INT64_MAX) == INT64_MAX / 4);
    CHECK(kv_ring_adapt(true, INT64_MAX / 2 + 1, floor, INT64_MAX) == floor);   // 2 x cap is out of int64 range: never reached

    // a larger floor (STRATA_KV_RESIDENT_MIN above 32768) is honoured
    CHECK(kv_ring_adapt(true, 49152, 40000, 1024) == 40000);
    CHECK(kv_ring_adapt(true, 40000, 40000, 1024) == 40000);

    if (fails == 0) std::printf("kv_ring_test: ok\n");
    return fails == 0 ? 0 : 1;
}
