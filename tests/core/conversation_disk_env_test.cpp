// MOD_PLAN 16b "disk-key": the table of every STRATA_* switch (src/core/conversation_disk_env.def) against the sources and against
// the conversation disk tier's key.  CPU only, no model.
//
//   1. every "STRATA_NAME" string in src/ and include/ has a line in the table, and every line names a string that is still there:
//      a new switch without a classification fails here, with the file it was found in
//   2. the class and the key agree: a Key switch is always in the key, a KeyIfSet / Unsure one when it is set, and a Speed / Inert /
//      InConfig one never changes the key
//   3. the audit's decisions that must not slip back (F14's expert layouts, the master switch, the other-summation-order defaults)
#include "strata/core/conversation_disk_tier.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {
int checks = 0;
int failures = 0;
void check(bool ok, const std::string& label) {
    ++checks;
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", label.c_str());
    }
}

fs::path source_root() {
#ifdef STRATA_SRC_DIR
    return fs::path(STRATA_SRC_DIR);
#else
    return fs::path(__FILE__).parent_path().parent_path().parent_path();   // tests/core/<this file>
#endif
}

bool name_char(char c) { return (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_'; }

// name -> the first file that holds it as a "STRATA_NAME" string literal (the table itself is not a source of names)
std::map<std::string, std::string> names_in_sources(const fs::path& root, std::string& err) {
    std::map<std::string, std::string> found;
    const std::set<std::string> exts = {".cpp", ".hpp", ".h", ".cu", ".cuh", ".inc", ".inl", ".c", ".cc", ".mm"};
    for (const char* dir : {"src", "include"}) {
        std::error_code ec;
        fs::recursive_directory_iterator it(root / dir, ec), end;
        if (ec) {
            err = "cannot read " + (root / dir).string() + ": " + ec.message();
            return found;
        }
        for (; it != end; it.increment(ec)) {
            if (ec) break;
            if (!it->is_regular_file() || exts.count(it->path().extension().string()) == 0) continue;
            std::ifstream in(it->path(), std::ios::binary);
            const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            static const std::string open = "\"STRATA_";
            for (size_t p = text.find(open); p != std::string::npos; p = text.find(open, p + 1)) {
                size_t q = p + 1;
                while (q < text.size() && name_char(text[q])) ++q;
                if (q == p + open.size() || q >= text.size() || text[q] != '"') continue;   // not a plain "STRATA_NAME"
                found.emplace(text.substr(p + 1, q - p - 1), fs::relative(it->path(), root).generic_string());
            }
        }
    }
    return found;
}

const EnvSwitch* find(const std::string& name) {
    for (const EnvSwitch& s : conversation_env_switches())
        if (name == s.name) return &s;
    return nullptr;
}

std::string key_with(const std::map<std::string, std::string>& env) {
    return conversation_disk_key({{1, 2}, "/m/a.gguf", conversation_disk_key_env([&](const char* n) -> const char* {
                                      const auto it = env.find(n);
                                      return it == env.end() ? nullptr : it->second.c_str();
                                  })});
}

void test_sources_against_table() {
    std::string err;
    const auto sources = names_in_sources(source_root(), err);
    check(err.empty(), "the source tree is readable: " + err);
    check(sources.size() > 500, "the scan found the engine's switches (" + std::to_string(sources.size()) + " names)");

    std::set<std::string> table;
    for (const EnvSwitch& s : conversation_env_switches()) {
        check(table.insert(s.name).second, std::string("table: ") + s.name + " is listed once");
        check(std::string(s.name).rfind("STRATA_", 0) == 0, std::string("table: ") + s.name + " starts with STRATA_");
        check(s.note != nullptr && s.note[0] != '\0', std::string("table: ") + s.name + " has a note");
    }
    std::string missing;
    for (const auto& [name, file] : sources)
        if (table.count(name) == 0) missing += "\n    " + name + "  (" + file + ")";
    check(missing.empty(), "every STRATA_* string in src/ and include/ is in src/core/conversation_disk_env.def - ADD A LINE for:" + missing);
    std::string stale;
    for (const std::string& name : table)
        if (sources.count(name) == 0) stale += "\n    " + name;
    check(stale.empty(), "no line of conversation_disk_env.def names a string that is gone from src/ and include/ - REMOVE:" + stale);
}

void test_classes_against_key() {
    const std::string base = key_with({});
    size_t n_key = 0, n_if_set = 0, n_other = 0;
    for (const EnvSwitch& s : conversation_env_switches()) {
        const std::string name = s.name;
        // the key mechanism's own extra-names switch is Inert and handled by the key test of conversation_disk_tier_test
        if (name == "STRATA_CONVERSATION_DISK_KEY_ENV") continue;
        const std::string set = key_with({{name, "1"}}), other = key_with({{name, "2"}}), empty = key_with({{name, ""}});
        switch (s.cls) {
            case EnvClass::Key:
                ++n_key;
                check(set != base && set != other, name + " (Key) changes the key, value by value");
                break;
            case EnvClass::KeyIfSet:
            case EnvClass::Unsure:
                ++n_if_set;
                check(set != base && set != other, name + " (KeyIfSet/Unsure) changes the key when set, value by value");
                // presence counts: CPU_YMM, IQ3S_MT1, NO_IQ512, SELECT_OLD, EXP_MODE ... are read as getenv != nullptr, so "" is a setting
                check(empty != base && empty != set, name + " (KeyIfSet/Unsure) set but empty is keyed as set, apart from unset and from other values");
                break;
            case EnvClass::InConfig:
            case EnvClass::Speed:
            case EnvClass::Inert:
                ++n_other;
                check(set == base && other == base, name + " (Speed/Inert/InConfig) does not change the key");
                break;
        }
    }
    check(n_key == conversation_disk_key_env_names().size(), "the Key class is exactly the key's always-present list");
    check(conversation_disk_key_env({}).size() == n_key, "an environment with nothing set keys exactly the Key class");
    check(n_if_set > 0 && n_other > 0, "all the classes are used");
}

void test_decisions_that_must_hold() {
    auto cls = [](const char* name) {
        const EnvSwitch* s = find(name);
        return s ? static_cast<int>(s->cls) : -1;
    };
    auto keyed = [&](const char* name) {
        const int c = cls(name);
        return c == static_cast<int>(EnvClass::Key) || c == static_cast<int>(EnvClass::KeyIfSet) || c == static_cast<int>(EnvClass::Unsure);
    };
    // F14's per-format expert layouts (bitwise by bench on one card only, not gated at run time)
    for (const char* n : {"STRATA_EXP_MODE", "STRATA_EXP9_DOWN", "STRATA_EXP9_GU"}) check(keyed(n), std::string(n) + " is in the key when set");
    // the layer-major walk adds the routed experts in id order (FP32 rounding; the small-card rule turns it on by default), and its
    // siblings and the 128-bit IQ kernel (rel <= 1e-5 against ggml, not bitwise) change the geometry / the sum order too
    for (const char* n : {"STRATA_PREFILL_LAYER_MAJOR", "STRATA_PREFILL_LM_FORCE", "STRATA_PREFILL_LM_TOKENS", "STRATA_PREFILL_LM_EVEN",
                          "STRATA_PREFILL_LM_R", "STRATA_PREFILL_LM_MARGIN_MIB", "STRATA_NO_IQ128", "STRATA_NO_IQ256"})
        check(keyed(n), std::string(n) + " is in the key when set");
    // switches the sources read by presence alone (getenv != nullptr): an empty value turns them on, so it must change the key
    for (const char* n : {"STRATA_CPU_YMM", "STRATA_IQ3S_MT1", "STRATA_NO_IQ512", "STRATA_NO_IQ4NL", "STRATA_PROMPT_ATTN_V1", "STRATA_SELECT_OLD",
                          "STRATA_INDEXER_PER_TOKEN", "STRATA_ROUTE_PROJ_PER_TOKEN", "STRATA_HIP_ROUTER_OLD", "STRATA_HIP_ROUTER_FAST", "STRATA_EXP_MODE",
                          "STRATA_NO_IQ128", "STRATA_NO_IQ256"})
        check(keyed(n) && key_with({{n, ""}}) != key_with({}), std::string(n) + " set but empty is in the key");
    // the master switch and the defaults that are another summation order: a run with and without them shares a key otherwise
    for (const char* n : {"STRATA_DEFAULTS", "STRATA_GR_V3", "STRATA_PROMPT_ATTN_IMMA", "STRATA_SM120_PERF", "STRATA_GFX12_DEFAULTS"})
        check(keyed(n), std::string(n) + " is in the key when set");
    // the bit-changing prompt switches of the arch tables
    for (const char* n : {"STRATA_PF_FUSED", "STRATA_PF_GEMM", "STRATA_HIP_WMMA", "STRATA_SELECT_WMMA", "STRATA_PA_FAST", "STRATA_HC_UPMIX",
                          "STRATA_MMQ_KQ", "STRATA_PREFILL_NVFP4", "STRATA_PREFILL_CPU_SHARE", "STRATA_KVA", "STRATA_CPU_YMM", "STRATA_PF_SMALL"})
        check(keyed(n), std::string(n) + " is in the key when set");
    // the deferral switches (MOD_PLAN 16 item 8) and the routing ones were already there
    for (const char* n : {"STRATA_VERIFY_DEFER_CPU", "STRATA_VERIFY_DEFER_CPU_BELOW", "STRATA_VERIFY_DEFER_CPU_TOP", "STRATA_ROUTE_PRIOR",
                          "STRATA_ROUTE_TAIL_SKIP", "STRATA_ROUTE_RESIDENT"})
        check(cls(n) == static_cast<int>(EnvClass::Key), std::string(n) + " stays in the always-present key list");
    // what must stay out: the fold only moves a row between pool passes; Q2_FAST is gated by an in-process bitwise probe
    check(cls("STRATA_VERIFY_DEFER_CPU_FOLD") == static_cast<int>(EnvClass::Speed), "STRATA_VERIFY_DEFER_CPU_FOLD stays out of the key");
    check(cls("STRATA_Q2_FAST") == static_cast<int>(EnvClass::Speed) && cls("STRATA_Q2_LEGACY") == static_cast<int>(EnvClass::Speed),
          "STRATA_Q2_FAST / _LEGACY stay out of the key (the probe compares the kernels bit for bit and falls back)");
    // already in the session config fingerprint: not a second time
    check(cls("STRATA_KV_ROT") == static_cast<int>(EnvClass::InConfig), "STRATA_KV_ROT is covered by the config fingerprint");
}
}  // namespace

int main() {
    test_sources_against_table();
    test_classes_against_key();
    test_decisions_that_must_hold();
    if (failures != 0) {
        std::fprintf(stderr, "conversation_disk_env_test: %d of %d checks FAILED\n", failures, checks);
        return 1;
    }
    std::printf("conversation_disk_env_test: %d checks passed (%zu switches classified)\n", checks, conversation_env_switches().size());
    return 0;
}
