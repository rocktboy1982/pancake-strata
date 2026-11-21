// LEADS.md Part S row S13: STRATA_MTP_HEAD_HOST's placement rule and budget arithmetic (include/strata/core/mtp_head_host.hpp),
// host only.  Validated: the parse, the placement (default = VRAM), the VRAM and lend consequences, the slab gain on a 20 GB
// card.  NOT validated: the zero-copy reads of the draft kernels (needs ROCm and the model), the draft latency, the VRAM
// the real cache sizing then hands out.
#include "strata/core/mtp_head_host.hpp"

#include <cstdio>
#include <cstdlib>
#include <initializer_list>

using namespace strata::core;

static int checks = 0;
static void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

int main() {
    check(!mtp_head_host_parse(nullptr) && !mtp_head_host_parse("") && !mtp_head_host_parse("0") && !mtp_head_host_parse("2") &&
              !mtp_head_host_parse("11") && !mtp_head_host_parse("host") && mtp_head_host_parse("1"), "only the text 1 is on");

    const uint64_t kExperts = 675ull << 20;   // 512 experts of the drafter (--mtp-q4 variants)
    // default: the switch unset gives the device placement whatever the host could offer, and nothing changes
    for (bool ok : {false, true}) {
        const HeadPlacement p = head_placement(false, ok);
        check(p == HeadPlacement::Device, "switch off: the experts are in VRAM");
        check(head_vram_bytes(p, kExperts) == kExperts && head_lendable(p) && head_cache_gain_bytes(p, kExperts) == 0,
              "switch off: all in VRAM, lendable, no gain for the cache");
    }
    // on, and the mapped block was obtained
    {
        const HeadPlacement p = head_placement(true, true);
        check(p == HeadPlacement::HostMapped, "switch on and host memory obtained: mapped host");
        check(head_vram_bytes(p, kExperts) == 0, "no VRAM taken");
        check(!head_lendable(p), "not lendable to the prompt room");
        check(head_cache_gain_bytes(p, kExperts) == kExperts, "the whole block is the cache's gain");
    }
    // on, host memory refused: back to VRAM, as before
    check(head_placement(true, false) == HeadPlacement::Device, "no pinned room: VRAM");
    // the budget arithmetic (an illustrative 2 MiB blob, not the model's real slot size)
    {
        const uint64_t blob = 2ull << 20;
        const uint64_t slots = head_cache_gain_slots(HeadPlacement::HostMapped, kExperts, blob);
        check(slots == 337, "675 MiB at 2 MiB a slot: 337 more cache slots");
        check(head_cache_gain_slots(HeadPlacement::Device, kExperts, blob) == 0 && head_cache_gain_slots(HeadPlacement::HostMapped, kExperts, 0) == 0,
              "no gain on the device, none for a zero blob");
    }
    std::printf("mtp_head_host_test: %d checks passed\n", checks);
    return 0;
}
