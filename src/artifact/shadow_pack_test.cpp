// src/artifact/shadow_pack_test.cpp - the F4 shadow pack read back (MOD_PLAN 16.5 A27; CPU only, no model files).
//
// A synthetic two-layer source shard (the model's widths, 3 experts; layer 0 IQ2_S gate/up, layer 1 already IQ1_M,
// both Q2_0 down) and its native_experts.txt v3 are converted by strata::shadow::convert.  Then:
//  1. expert_layout_load reads the shadow's native_experts.txt: layer 0 is IQ1_M with gate/up in the shadow GGUF and
//     down in the source shard, layer 1 (not smaller as IQ1_M) is unchanged;
//  2. every (layer, role) file and offset names the tensor it claims, the checks of the engine's check_experts_gguf;
//  3. the shadow rows are, bit for bit, ggml's IQ1_M encoding of ggml's decoding of the source rows;
//  4. the engine's CPU expert kernels on a blob cut from the shadow layout match the float expert of the decoded
//     shadow weights (activation rounding only), and the shadow differs from the source by a bounded amount.
#include "strata/artifact/shadow_pack.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>

using namespace strata::kernels::cpu;
namespace fs = std::filesystem;

namespace {

constexpr int64_t kH = 2560, kFF = 640, kNE = 3;
int g_fail = 0;

void check(bool ok, const char* what) {
    std::printf("  %s %s\n", ok ? "ok    " : "FAILED", what);
    if (!ok) ++g_fail;
}

std::vector<uint8_t> random_rows(int type, int64_t rows, int64_t cols, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 0.02f);
    std::vector<float> f((size_t) (rows * cols));
    for (auto& v : f) v = nd(rng);
    std::vector<uint8_t> q(ggml_row_size((ggml_type) type, cols) * (size_t) rows);
    ggml_quantize_chunk((ggml_type) type, f.data(), q.data(), 0, rows, cols, nullptr);
    return q;
}

std::vector<float> to_f32(int type, const uint8_t* p, int64_t rows, int64_t cols) {
    std::vector<float> f((size_t) (rows * cols));
    const size_t rb = ggml_row_size((ggml_type) type, cols);
    for (int64_t r = 0; r < rows; ++r) ggml_get_type_traits((ggml_type) type)->to_float(p + r * rb, f.data() + r * cols, cols);
    return f;
}

std::vector<uint8_t> read_at(const std::string& path, uint64_t at, size_t n) {
    std::vector<uint8_t> b(n);
    std::ifstream f(path, std::ios::binary);
    f.seekg((std::streamoff) at);
    if (!f.read((char*) b.data(), (std::streamsize) n)) b.clear();
    return b;
}

double rel(const std::vector<float>& got, const std::vector<float>& ref) {
    strata::shadow::Err e;
    e.add(ref.data(), got.data(), ref.size());
    return e.rel();
}

}  // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / ("shadow_pack_test_" + std::to_string(std::random_device{}()));
    fs::create_directories(dir / "model");
    fs::create_directories(dir / "pack");
    const std::string src = (dir / "model" / "src.gguf").string();

    // ---- the synthetic source shard and its pack index ----
    const int gu_type[2] = {GGML_TYPE_IQ2_S, GGML_TYPE_IQ1_M};
    std::vector<strata::shadow::detail::OutTensor> ts;
    std::vector<std::vector<uint8_t>> data;
    std::mt19937 rng(7);
    for (int l = 0; l < 2; ++l)
        for (int r = 0; r < 3; ++r) {
            static const char* roles[3] = {"gate", "up", "down"};
            const int type = r < 2 ? gu_type[l] : GGML_TYPE_Q2_0;
            const int64_t rows = r < 2 ? kFF : kH, cols = r < 2 ? kH : kFF;
            data.push_back(random_rows(type, rows * kNE, cols, rng));
            ts.push_back({"blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight",
                          {(uint64_t) cols, (uint64_t) rows, (uint64_t) kNE}, type, 0, data.back().size()});
        }
    const std::string hdr = strata::shadow::detail::gguf_header(ts, {{"general.type", "test"}});
    {
        std::string file = hdr;
        for (size_t i = 0; i < ts.size(); ++i) {
            file.resize(hdr.size() + ts[i].offset, '\0');
            file.append((const char*) data[i].data(), data[i].size());
        }
        std::ofstream(src, std::ios::binary).write(file.data(), (std::streamsize) file.size());
        std::FILE* p = std::fopen((dir / "pack" / "native_experts.txt").string().c_str(), "w");
        std::fprintf(p, "# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard] "
                        "(n_expert %lld, total 0; test)\n", (long long) kNE);
        uint64_t at = 0;
        for (int l = 0; l < 2; ++l) {
            const uint64_t blob = (ts[3 * l].bytes + ts[3 * l + 1].bytes + ts[3 * l + 2].bytes) / kNE;
            std::fprintf(p, "%d %d %d %llu %llu %llu %llu %llu\n", l, gu_type[l], (int) GGML_TYPE_Q2_0,
                         (unsigned long long) at, (unsigned long long) blob, (unsigned long long) (hdr.size() + ts[3 * l].offset),
                         (unsigned long long) (hdr.size() + ts[3 * l + 1].offset),
                         (unsigned long long) (hdr.size() + ts[3 * l + 2].offset));
            at += blob * kNE;
        }
        std::fclose(p);
    }

    // ---- convert ----
    strata::shadow::Options o;
    o.pack = (dir / "pack").string();
    o.gguf = src;
    o.out = (dir / "shadow").string();
    o.eval = 2;
    std::vector<strata::shadow::LayerReport> rep;
    std::string err;
    const bool converted = strata::shadow::convert(o, rep, err);
    std::printf("convert: %s\n", converted ? "done" : err.c_str());
    check(converted && rep.size() == 1 && rep[0].layer == 0, "only layer 0 converted (layer 1 is already IQ1_M)");
    if (!converted || rep.empty()) return 1;
    std::printf("  layer 0: gate+up %.1f -> %.1f KB per expert, rel err weights %.3f, hidden %.3f, output %.3f\n",
                rep[0].src_bytes / 1000.0, rep[0].dst_bytes / 1000.0, rep[0].w.rel(), rep[0].h.rel(), rep[0].y.rel());
    check(rep[0].dst_bytes * 82 == rep[0].src_bytes * 56, "IQ1_M rows are 56/82 of IQ2_S rows");
    check(rep[0].w.rel() > 0.05 && rep[0].w.rel() < 0.8 && rep[0].y.rel() > 0.05 && rep[0].y.rel() < 1.5,
          "lossy, bounded error (requant of 2-bit rows)");

    // ---- 1. the engine's pack loader ----
    const bool loaded = expert_layout_load(o.out, 2, kNE, err);
    std::printf("expert_layout_load: %s\n", loaded ? "done" : err.c_str());
    check(loaded, "expert_layout_load reads the shadow's native_experts.txt");
    if (!loaded) return 1;
    const ExpertLayout& lay = expert_layout();
    const std::string rel_name = "../shadow/shadow-iq1_m.gguf";
    std::printf("  layer 0 gate file: %s\n", lay.gguf_file.empty() ? "(none)" : lay.gguf_file[0].c_str());
    check(lay.native && lay.version == 4 && lay.n_expert == kNE, "native v4 layout, 3 experts");
    check(lay.fmt[0].gu_type == GGML_TYPE_IQ1_M && lay.fmt[0].d_type == GGML_TYPE_Q2_0 && lay.fmt[1].gu_type == GGML_TYPE_IQ1_M,
          "layer 0 IQ1_M/Q2_0, layer 1 unchanged");
    check(lay.gguf_file.size() == 6 && lay.gguf_file[0] == rel_name && lay.gguf_file[1] == rel_name && lay.gguf_file[2].empty() &&
              lay.gguf_file[3].empty() && lay.gguf_file[4].empty() && lay.gguf_file[5].empty(),
          "layer 0 gate/up name the shadow GGUF, down and layer 1 the source shard");
    check(lay.gguf_off[5] == hdr.size() + ts[5].offset && lay.gguf_off[2] == hdr.size() + ts[2].offset,
          "down offsets are the source's");

    // ---- 2. every (layer, role) names its tensor (the engine's check_experts_gguf conditions) ----
    bool spans = true;
    for (int l = 0; l < 2; ++l)
        for (int r = 0; r < 3; ++r) {
            static const char* roles[3] = {"gate", "up", "down"};
            const std::string& name = lay.gguf_file[(size_t) (3 * l + r)];
            const std::string path = name.empty() ? src : (dir / "model" / name).string();
            strata::GgufFile g(path);
            const strata::TensorInfo* t = g.find("blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight");
            const uint64_t per[3] = {lay.fmt[(size_t) l].up_off, lay.fmt[(size_t) l].up_off,
                                     lay.bytes[(size_t) l] - lay.fmt[(size_t) l].down_off};
            spans = spans && t && (int) t->type == (r < 2 ? lay.fmt[(size_t) l].gu_type : lay.fmt[(size_t) l].d_type) &&
                    t->shape.size() == 3 && t->shape[0] == (uint64_t) (r < 2 ? kH : kFF) && t->shape[2] == (uint64_t) kNE &&
                    strata::tensor_payload_bytes(*t) == per[r] * kNE && g.data_start() + t->offset == lay.gguf_off[(size_t) (3 * l + r)] &&
                    t->offset + per[r] * kNE <= g.file_size() - g.data_start();
        }
    check(spans, "every file/offset is the tensor of the type, shape and size the layout says");

    // ---- 3. bit for bit ggml's encoding of the decoded source rows ----
    const std::string shadow = (dir / "model" / rel_name).string();
    const size_t per_src = ts[0].bytes / kNE, per_sh = lay.fmt[0].up_off;
    bool exact = true;
    for (int r = 0; r < 2; ++r)
        for (int64_t e = 0; e < kNE; ++e) {
            const auto f = to_f32(GGML_TYPE_IQ2_S, data[(size_t) r].data() + e * per_src, kFF, kH);
            std::vector<uint8_t> want(per_sh);
            ggml_quantize_chunk(GGML_TYPE_IQ1_M, f.data(), want.data(), 0, kFF, kH, nullptr);
            exact = exact && read_at(shadow, lay.gguf_off[(size_t) r] + e * per_sh, per_sh) == want;
        }
    check(exact, "shadow rows == ggml IQ1_M(decode(IQ2_S rows)), every expert");

    // ---- 4. the engine's CPU expert kernels read a shadow blob ----
    const int64_t e = 1;
    const NativeFmt& f = lay.fmt[0];
    std::vector<uint8_t> blob(f.bytes);
    const auto g = read_at(shadow, lay.gguf_off[0] + e * f.up_off, f.up_off);
    const auto u = read_at(shadow, lay.gguf_off[1] + e * f.up_off, f.up_off);
    const auto d = read_at(src, lay.gguf_off[2] + e * (f.bytes - f.down_off), f.bytes - f.down_off);
    std::memcpy(blob.data(), g.data(), g.size());
    std::memcpy(blob.data() + f.up_off, u.data(), u.size());
    std::memcpy(blob.data() + f.down_off, d.data(), d.size());
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(kH), ff(kFF), y(kH), h_ref(kFF), y_ref(kH), h_src(kFF), y_src(kH);
    for (auto& v : x) v = nd(rng);
    std::vector<uint8_t> act(kNativeActBytes), hq(kNativeHBytes);
    native_quant_act(f, x.data(), act.data());
    const void* a[1] = {act.data()};
    float* ffp[1] = {ff.data()};
    native_gu_rows(f, blob.data(), a, 1, ffp, 0, (int) kFF);
    native_quant_h(f, ff.data(), hq.data());
    const void* hp[1] = {hq.data()};
    float* yp[1] = {y.data()};
    native_down_rows(f, blob.data(), hp, 1, yp, 0, (int) kH);
    const auto gs = to_f32(GGML_TYPE_IQ1_M, g.data(), kFF, kH), us = to_f32(GGML_TYPE_IQ1_M, u.data(), kFF, kH);
    const auto ds = to_f32(GGML_TYPE_Q2_0, d.data(), kH, kFF);
    strata::shadow::expert_f32(gs.data(), us.data(), ds.data(), x.data(), kH, kFF, h_ref.data(), y_ref.data());
    const auto g0 = to_f32(GGML_TYPE_IQ2_S, data[0].data() + e * per_src, kFF, kH);
    const auto u0 = to_f32(GGML_TYPE_IQ2_S, data[1].data() + e * per_src, kFF, kH);
    strata::shadow::expert_f32(g0.data(), u0.data(), ds.data(), x.data(), kH, kFF, h_src.data(), y_src.data());
    const double e_kernel = rel(y, y_ref), e_shadow = rel(y_ref, y_src);
    std::printf("  expert %lld: engine kernels vs float shadow expert %.4f; float shadow vs float source %.4f\n",
                (long long) e, e_kernel, e_shadow);
    check(e_kernel < 0.03, "native_gu_rows/native_down_rows on the shadow blob == the float shadow expert (q8 rounding)");
    check(e_shadow > 0.05 && e_shadow < 1.5, "the shadow expert differs from the source expert, boundedly");

    std::error_code ec;
    fs::remove_all(dir, ec);
    std::printf("%s\n", g_fail ? "FAILED" : "PASS");
    return g_fail ? 1 : 0;
}
