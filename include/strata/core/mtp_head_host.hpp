#pragma once
// STRATA_MTP_HEAD_HOST=1 (LEADS.md Part S row S13, default off): the MTP head's 512 routed experts (experts.bin, 675 MiB with
// the --mtp-q4 variants) stay in page-locked, device-mapped host memory instead of VRAM; the draft pass reads them through the
// link (zero copy).  radiance's `--draft-expert-placement host` does the same; its journal W42m: 650 MiB more expert slab on a
// W7900 gave -3.3% pool and -1.0% prefill, decode not reported.  Here the freed VRAM goes to the expert cache because the
// cache is sized from what is free after the drafter loads (generate.cpp, "the weights, the session and the drafter").
// This header is the placement rule and the budget arithmetic, pure, so a ctest covers it; src/core/mtp.cpp uses it.
#include <cstdint>

namespace strata::core {

enum class HeadPlacement { Device, HostMapped };

/// STRATA_MTP_HEAD_HOST: only the text "1" switches it on (default off).
inline bool mtp_head_host_parse(const char* v) { return v != nullptr && v[0] == '1' && v[1] == '\0'; }

/// The switch asks for host memory; `host_ok` is whether the mapped page-locked block and its device alias were obtained.
/// Without the switch, or when the block cannot be had, the experts are in VRAM exactly as before.
constexpr HeadPlacement head_placement(bool switch_on, bool host_ok) {
    return switch_on && host_ok ? HeadPlacement::HostMapped : HeadPlacement::Device;
}
/// VRAM the experts take (what `vram_bytes()` adds): all of it on the device, none when mapped on the host.
constexpr uint64_t head_vram_bytes(HeadPlacement p, uint64_t bytes) { return p == HeadPlacement::Device ? bytes : 0; }
/// The prompt room (STRATA_MTP_ROOM_MIN) carves slots out of the experts' VRAM block: only a device block can be lent.
constexpr bool head_lendable(HeadPlacement p) { return p == HeadPlacement::Device; }
/// The VRAM the placement frees for the expert cache, and the cache slots of `blob_bytes` that are (the cache is sized
/// after the drafter loads, from what is free).
constexpr uint64_t head_cache_gain_bytes(HeadPlacement p, uint64_t bytes) { return p == HeadPlacement::HostMapped ? bytes : 0; }
constexpr uint64_t head_cache_gain_slots(HeadPlacement p, uint64_t bytes, uint64_t blob_bytes) {
    return blob_bytes == 0 ? 0 : head_cache_gain_bytes(p, bytes) / blob_bytes;
}

}  // namespace strata::core
