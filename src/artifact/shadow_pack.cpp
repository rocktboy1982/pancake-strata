// src/artifact/shadow_pack.cpp - the `strata_shadow_pack` CLI (MOD_PLAN 16.5 A27 / F4, offline prototype).  The
// converter is header-only in include/strata/artifact/shadow_pack.hpp so shadow_pack_test exercises the same code.
//
//   strata_shadow_pack --pack PACK --gguf SHARD1 --out DIR [--type iq1_m|iq1_s|iq2_xxs] [--layers 0,1,8] [--stride 8]
//                      [--eval 4] [--threads 2] [--flat]
//
// --flat encodes with an importance matrix of ones instead of ggml's no-imatrix weighting (each weight by its square).
// --stride s converts experts 0, s, 2s, ... only (a measurement slice: it reads about 1/s of each layer and writes no
// native_experts.txt).  Without --layers every layer whose gate/up rows the shadow makes smaller is converted; listed
// layers are converted even when they are not (a round trip of an IQ1_M layer shows the encoder's own error floor).
#include "strata/artifact/shadow_pack.hpp"

#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    strata::shadow::Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const char* v = i + 1 < argc ? argv[i + 1] : nullptr;
        auto take = [&]() { ++i; return std::string(v); };
        if (a == "--flat") { o.flat = true; continue; }
        if (!v) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); return 2; }
        if (a == "--pack") o.pack = take();
        else if (a == "--gguf") o.gguf = take();
        else if (a == "--out") o.out = take();
        else if (a == "--type") {
            const std::string t = take();
            o.type = t == "iq1_m" ? GGML_TYPE_IQ1_M : t == "iq2_xxs" ? GGML_TYPE_IQ2_XXS : t == "iq1_s" ? GGML_TYPE_IQ1_S : -1;
            if (o.type < 0) { std::fprintf(stderr, "--type: iq1_m, iq1_s or iq2_xxs\n"); return 2; }
        } else if (a == "--layers") {
            std::stringstream ss(take());
            for (std::string x; std::getline(ss, x, ',');) o.layers.push_back(std::atoll(x.c_str()));
        } else if (a == "--stride") o.stride = std::atoll(take().c_str());
        else if (a == "--eval") o.eval = std::atoi(take().c_str());
        else if (a == "--threads") o.threads = std::atoi(take().c_str());
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (o.pack.empty() || o.gguf.empty() || o.out.empty()) {
        std::fprintf(stderr, "usage: strata_shadow_pack --pack PACK --gguf SHARD1 --out DIR [--type iq1_m|iq1_s|iq2_xxs] "
                             "[--layers L,..] [--stride S] [--eval N] [--threads T] [--flat]\n");
        return 2;
    }
    std::vector<strata::shadow::LayerReport> rep;
    std::string err;
    if (!strata::shadow::convert(o, rep, err)) { std::fprintf(stderr, "strata_shadow_pack: %s\n", err.c_str()); return 1; }
    std::printf("layer src      dst      experts  gate+up KB/expert  saved   w rel err  h rel err  y rel err  s\n");
    uint64_t src = 0, dst = 0;
    strata::shadow::Err w, h, y;
    for (const auto& r : rep) {
        std::printf("%5lld %-8s %-8s %7lld  %8.1f -> %6.1f  %5.1f%%  %9.4f  %9.4f  %9.4f  %.1f\n", (long long) r.layer,
                    ::ggml_type_name((ggml_type) r.src_type), ::ggml_type_name((ggml_type) r.dst_type),
                    (long long) r.experts, r.src_bytes / 1000.0, r.dst_bytes / 1000.0,
                    100.0 * (1.0 - (double) r.dst_bytes / (double) r.src_bytes), r.w.rel(), r.h.rel(), r.y.rel(), r.seconds);
        src += r.src_bytes;
        dst += r.dst_bytes;
        w.add(r.w);
        h.add(r.h);
        y.add(r.y);
    }
    std::printf("all   gate+up bytes saved %.1f%%; relative error: weights %.4f, SwiGLU hidden %.4f, expert output %.4f "
                "(eval %d random N(0,1) activations per expert; 0 = not evaluated)\n",
                100.0 * (1.0 - (double) dst / (double) src), w.rel(), h.rel(), y.rel(), o.eval);
    std::printf("wrote %s/shadow-%s.gguf%s\n", o.out.c_str(), ::ggml_type_name((ggml_type) o.type),
                o.stride == 1 ? " and native_experts.txt" : " (a slice: no native_experts.txt)");
    return 0;
}
