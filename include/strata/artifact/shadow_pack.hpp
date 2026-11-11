// include/strata/artifact/shadow_pack.hpp - MOD_PLAN 16.5 A27 / Phase 15 F4: a sub-2-bit "shadow" of a native
// pack's routed experts (offline prototype; the engine has no switch that reads it yet).
//
// The gate and up rows of a layer are decoded with ggml's own dequantizer and encoded again with ggml's own quantizer
// into a smaller type (IQ1_M by default); down is not touched.  The result is a GGUF that holds only the shadow
// tensors (blk.L.ffn_gate_exps.weight / ffn_up_exps.weight, [n_embd, n_ff, n_expert]) and, for a full conversion, a
// native_experts.txt v4 whose converted layers name that GGUF for gate/up (a path relative to the --native shard's
// folder) and the model's own shard for down.  So the existing pack loader (expert_layout_load) and the CPU expert
// kernels (native_gu_rows) read a shadow layer like any other native layer.
//
// Lossy twice over: the source rows are already 2-bit, and no importance matrix ships with the model, so the encoder
// weighs each weight by its own square (ggml's default when no imatrix is given; a matrix of ones where ggml insists).
// Every converted expert is decoded again and compared with its source: weight error, and the expert's output error
// on random activations (the error that reaches the residual stream when a CPU miss reads the shadow).
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace strata::shadow {

/// One native_experts.txt line (v2-v4): per role (0 gate, 1 up, 2 down) its absolute offset and its file, a name
/// relative to the --native shard's folder ("" = the --native shard itself).
struct PackLine {
    int64_t layer = -1;
    int gu_type = -1, d_type = -1;
    uint64_t offset = 0, blob = 0;
    uint64_t off[3] = {0, 0, 0};
    std::string file[3];
};

struct Pack {
    int64_t n_expert = 512;
    std::vector<PackLine> lines;   // sorted by layer
};

inline bool read_pack(const std::string& dir, Pack& p, std::string& err) {
    std::ifstream in(dir + "/native_experts.txt");
    if (!in) { err = dir + "/native_experts.txt: cannot open"; return false; }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            const size_t at = line.find("(n_expert ");
            if (at != std::string::npos) p.n_expert = std::atoll(line.c_str() + at + 10);
            continue;
        }
        std::istringstream ss(line);
        PackLine l;
        if (!(ss >> l.layer >> l.gu_type >> l.d_type >> l.offset >> l.blob >> l.off[0] >> l.off[1] >> l.off[2])) {
            err = "native_experts.txt: a line without GGUF offsets (a pack older than v2): " + line;
            return false;
        }
        std::string files;
        if (ss >> files) {
            std::vector<std::string> parts;
            for (size_t from = 0;;) {
                const size_t comma = files.find(',', from);
                parts.push_back(files.substr(from, comma == std::string::npos ? comma : comma - from));
                if (comma == std::string::npos) break;
                from = comma + 1;
            }
            if (parts.size() == 1) parts = {parts[0], parts[0], parts[0]};
            if (parts.size() != 3) { err = "native_experts.txt: the shard column is one name or gate,up,down: " + line; return false; }
            for (int r = 0; r < 3; ++r) l.file[r] = parts[(size_t) r];
        }
        p.lines.push_back(l);
    }
    std::sort(p.lines.begin(), p.lines.end(), [](const PackLine& a, const PackLine& b) { return a.layer < b.layer; });
    return true;
}

inline size_t row_bytes(int type, int64_t cols) { return ggml_row_size((ggml_type) type, cols); }

/// rows x cols weights of `type` -> floats, with ggml's decoder.
inline void decode(int type, const uint8_t* src, int64_t rows, int64_t cols, float* dst) {
    const ggml_type_traits* tr = ggml_get_type_traits((ggml_type) type);
    const size_t rb = row_bytes(type, cols);
    for (int64_t r = 0; r < rows; ++r) tr->to_float(src + (size_t) r * rb, dst + r * cols, cols);
}

/// floats -> rows of `type`, with ggml's encoder (call ggml_quantize_init(type) once before using it from threads).
/// No importance matrix ships with the model: by default ggml weighs each weight by its own square (its no-imatrix
/// path); `flat` passes a matrix of ones instead (each weight by sqrt(sigma^2 + x^2), nearer a plain squared error).
/// A type that requires a matrix always gets the ones.
inline void encode(int type, const float* src, int64_t rows, int64_t cols, uint8_t* dst, bool flat = false) {
    std::vector<float> ones;
    if (flat || ggml_quantize_requires_imatrix((ggml_type) type)) ones.assign((size_t) cols, 1.0f);
    ggml_quantize_chunk((ggml_type) type, src, dst, 0, rows, cols, ones.empty() ? nullptr : ones.data());
}

/// Accumulated relative L2 error, sqrt(sum (got - ref)^2 / sum ref^2).
struct Err {
    double d2 = 0, s2 = 0;
    void add(const float* ref, const float* got, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            const double d = (double) got[i] - ref[i];
            d2 += d * d;
            s2 += (double) ref[i] * ref[i];
        }
    }
    void add(const Err& o) { d2 += o.d2; s2 += o.s2; }
    double rel() const { return s2 > 0 ? std::sqrt(d2 / s2) : 0.0; }
};

/// One expert in float: h = silu(gate . x) * (up . x) (gate/up [ff rows of embd]), y = down . h (down [embd rows of ff]).
inline void expert_f32(const float* g, const float* u, const float* d, const float* x, int64_t embd, int64_t ff,
                       float* h, float* y) {
    for (int64_t r = 0; r < ff; ++r) {
        double a = 0, b = 0;
        for (int64_t c = 0; c < embd; ++c) {
            a += (double) g[r * embd + c] * x[c];
            b += (double) u[r * embd + c] * x[c];
        }
        h[r] = (float) (a / (1.0 + std::exp(-a)) * b);
    }
    for (int64_t r = 0; r < embd; ++r) {
        double a = 0;
        for (int64_t c = 0; c < ff; ++c) a += (double) d[r * ff + c] * h[c];
        y[r] = (float) a;
    }
}

struct Options {
    std::string pack;               ///< the source pack's folder (native_experts.txt with GGUF offsets)
    std::string gguf;               ///< its --native shard (other shards are named relative to its folder)
    std::string out;                ///< output folder
    int type = GGML_TYPE_IQ1_M;     ///< the shadow's gate/up type
    std::vector<int64_t> layers;    ///< empty: every layer whose gate/up rows the shadow makes smaller
    int64_t stride = 1;             ///< convert experts 0, s, 2s, ...; > 1 is a measurement slice (no native_experts.txt)
    int eval = 4;                   ///< random activations per converted expert for the output error (0: none)
    int threads = 2;
    bool flat = false;              ///< encode with an importance matrix of ones (see encode)
};

struct LayerReport {
    int64_t layer = -1;
    int src_type = -1, dst_type = -1;
    int64_t experts = 0;            ///< converted
    uint64_t src_bytes = 0, dst_bytes = 0;   ///< gate + up per expert
    Err w, h, y;                    ///< weights (gate and up), the SwiGLU hidden, the expert output
    double seconds = 0;
};

namespace detail {

struct OutTensor {
    std::string name;
    uint64_t dims[3];
    int type;
    uint64_t offset;                ///< from the data start
    uint64_t bytes;
};

inline void put_u32(std::string& s, uint32_t v) { s.append((const char*) &v, 4); }
inline void put_u64(std::string& s, uint64_t v) { s.append((const char*) &v, 8); }
inline void put_str(std::string& s, const std::string& v) { put_u64(s, v.size()); s += v; }
inline uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

/// A GGUF v3 header (string metadata only, default alignment 32) for `ts`, whose offsets it assigns; padded to the
/// data start.
inline std::string gguf_header(std::vector<OutTensor>& ts, const std::vector<std::pair<std::string, std::string>>& kv) {
    uint64_t at = 0;
    for (auto& t : ts) { t.offset = at; at = align_up(at + t.bytes, 32); }
    std::string s = "GGUF";
    put_u32(s, 3);
    put_u64(s, ts.size());
    put_u64(s, kv.size());
    for (const auto& [k, v] : kv) { put_str(s, k); put_u32(s, 8); put_str(s, v); }
    for (const auto& t : ts) {
        put_str(s, t.name);
        put_u32(s, 3);
        for (uint64_t d : t.dims) put_u64(s, d);
        put_u32(s, (uint32_t) t.type);
        put_u64(s, t.offset);
    }
    s.resize(align_up(s.size(), 32), '\0');
    return s;
}

/// The source tensor of (layer, role) as the pack names it, checked the way the engine's check_experts_gguf does.
inline const uint8_t* source(std::map<std::string, std::unique_ptr<GgufFile>>& files, const std::string& gguf,
                             const PackLine& l, int r, int64_t n_expert, uint64_t& cols, uint64_t& rows, std::string& err) {
    static const char* roles[3] = {"gate", "up", "down"};
    const size_t cut = gguf.find_last_of("/\\");
    const std::string path = l.file[r].empty() ? gguf : gguf.substr(0, cut == std::string::npos ? 0 : cut + 1) + l.file[r];
    auto& f = files[path];
    if (!f) f = std::make_unique<GgufFile>(path);
    const std::string name = "blk." + std::to_string(l.layer) + ".ffn_" + roles[r] + "_exps.weight";
    const TensorInfo* t = f->find(name);
    const int want = r < 2 ? l.gu_type : l.d_type;
    if (!t || (int) t->type != want || t->shape.size() != 3 || t->shape[2] != (uint64_t) n_expert ||
        f->data_start() + t->offset != l.off[r] || tensor_payload_bytes(*t) == 0 ||
        t->offset + tensor_payload_bytes(*t) > f->file_size() - f->data_start()) {
        err = name + " in " + path + " is not the tensor native_experts.txt describes";
        return nullptr;
    }
    cols = t->shape[0];
    rows = t->shape[1];
    return f->tensor_data(*t);
}

}  // namespace detail

/// Convert the layers `o` selects; one report per converted layer.  Writes <out>/shadow-<type>.gguf and, when every
/// expert was converted (stride 1), <out>/native_experts.txt.
inline bool convert(const Options& o, std::vector<LayerReport>& rep, std::string& err) {
    using detail::OutTensor;
    Pack pack;
    if (!read_pack(o.pack, pack, err)) return false;
    if (o.stride < 1 || o.threads < 1 || pack.n_expert < 1) { err = "bad stride, threads or n_expert"; return false; }
    if (o.type < 0 || o.type >= GGML_TYPE_COUNT || !ggml_is_quantized((ggml_type) o.type)) {
        err = "the shadow type is not a ggml quantized type";
        return false;
    }

    std::map<std::string, std::unique_ptr<GgufFile>> files;
    struct Job { const PackLine* line; const uint8_t* src[3]; uint64_t cols, rows; };
    std::vector<Job> jobs;
    try {
        for (const PackLine& l : pack.lines) {
            const bool listed = std::find(o.layers.begin(), o.layers.end(), l.layer) != o.layers.end();
            if (!o.layers.empty() && !listed) continue;
            Job j{&l, {nullptr, nullptr, nullptr}, 0, 0};
            uint64_t dc = 0, dr = 0;
            for (int r = 0; r < 3; ++r) {
                uint64_t c = 0, w = 0;
                j.src[r] = detail::source(files, o.gguf, l, r, pack.n_expert, c, w, err);
                if (!j.src[r]) return false;
                if (r == 0) { j.cols = c; j.rows = w; }
                if (r == 1 && (c != j.cols || w != j.rows)) { err = "gate and up shapes differ"; return false; }
                if (r == 2) { dc = c; dr = w; }
            }
            if (dc != j.rows || dr != j.cols) { err = "down is not the transpose shape of gate/up"; return false; }
            if (j.cols % (uint64_t) ggml_blck_size((ggml_type) o.type)) { err = "n_embd is not a multiple of the shadow's block"; return false; }
            if (o.layers.empty() && row_bytes(o.type, (int64_t) j.cols) >= row_bytes(l.gu_type, (int64_t) j.cols)) continue;
            jobs.push_back(j);
        }
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    for (int64_t want : o.layers)
        if (std::none_of(jobs.begin(), jobs.end(), [&](const Job& j) { return j.line->layer == want; })) {
            err = "layer " + std::to_string(want) + " is not in the pack";
            return false;
        }
    if (jobs.empty()) { err = "no layer to convert"; return false; }

    const int64_t n_out = (pack.n_expert + o.stride - 1) / o.stride;
    std::vector<OutTensor> ts;
    for (const Job& j : jobs)
        for (const char* role : {"gate", "up"}) {
            const uint64_t per = row_bytes(o.type, (int64_t) j.cols) * j.rows;
            ts.push_back({"blk." + std::to_string(j.line->layer) + ".ffn_" + role + "_exps.weight",
                          {j.cols, j.rows, (uint64_t) n_out}, o.type, 0, per * (uint64_t) n_out});
        }
    const std::string src_name = std::filesystem::path(o.gguf).filename().string();
    const std::string header = detail::gguf_header(ts, {{"general.type", "strata-shadow-experts"},
                                                        {"strata.shadow.source", src_name},
                                                        {"strata.shadow.stride", std::to_string(o.stride)}});
    std::error_code ec;
    std::filesystem::create_directories(o.out, ec);
    const std::string type_name = ::ggml_type_name((ggml_type) o.type);
    const std::string out_gguf = o.out + "/shadow-" + type_name + ".gguf";
    std::FILE* fo = std::fopen(out_gguf.c_str(), "wb");
    if (!fo) { err = "cannot write " + out_gguf; return false; }
    bool io_ok = std::fwrite(header.data(), 1, header.size(), fo) == header.size();
    std::mutex io;
    // every expert of every tensor is written, so the file ends where the last tensor ends (padding between
    // tensors is a hole, read as zeros)
    auto put = [&](uint64_t at, const std::vector<uint8_t>& b) {
        std::lock_guard<std::mutex> lk(io);
#if defined(_WIN32)
        io_ok = io_ok && _fseeki64(fo, (int64_t) at, SEEK_SET) == 0;
#else
        io_ok = io_ok && fseeko(fo, (off_t) at, SEEK_SET) == 0;
#endif
        io_ok = io_ok && std::fwrite(b.data(), 1, b.size(), fo) == b.size();
    };

    ggml_quantize_init((ggml_type) o.type);
    for (size_t ji = 0; ji < jobs.size(); ++ji) {
        const Job& j = jobs[ji];
        const PackLine& l = *j.line;
        const int64_t cols = (int64_t) j.cols, rows = (int64_t) j.rows, n = rows * cols;
        const size_t src_per = row_bytes(l.gu_type, cols) * (size_t) rows;
        const size_t dst_per = row_bytes(o.type, cols) * (size_t) rows;
        const size_t down_per = row_bytes(l.d_type, rows) * (size_t) cols;
        const uint64_t at_gate = header.size() + ts[2 * ji].offset, at_up = header.size() + ts[2 * ji + 1].offset;
        LayerReport lr;
        lr.layer = l.layer;
        lr.src_type = l.gu_type;
        lr.dst_type = o.type;
        lr.experts = n_out;
        lr.src_bytes = 2 * src_per;
        lr.dst_bytes = 2 * dst_per;
        std::vector<LayerReport> part((size_t) o.threads);
        const auto t0 = std::chrono::steady_clock::now();
        auto work = [&](int tid) {
            std::vector<float> ref[2], sh[2], down_f((size_t) n);   // gate/up as the source has them and as the shadow does
            for (int r = 0; r < 2; ++r) { ref[r].resize((size_t) n); sh[r].resize((size_t) n); }
            std::vector<float> x((size_t) cols), h0((size_t) rows), h1((size_t) rows), y0((size_t) cols), y1((size_t) cols);
            std::vector<uint8_t> q(dst_per);
            for (int64_t eo = tid; eo < n_out; eo += o.threads) {
                const int64_t e = eo * o.stride;
                for (int r = 0; r < 2; ++r) {
                    decode(l.gu_type, j.src[r] + (size_t) e * src_per, rows, cols, ref[r].data());
                    encode(o.type, ref[r].data(), rows, cols, q.data(), o.flat);
                    decode(o.type, q.data(), rows, cols, sh[r].data());
                    part[(size_t) tid].w.add(ref[r].data(), sh[r].data(), (size_t) n);
                    put((r == 0 ? at_gate : at_up) + (uint64_t) eo * dst_per, q);
                }
                if (o.eval <= 0) continue;
                decode(l.d_type, j.src[2] + (size_t) e * down_per, cols, rows, down_f.data());
                std::mt19937 rng((uint32_t) (l.layer * 1000003 + e));
                std::normal_distribution<float> nd(0.f, 1.f);
                for (int i = 0; i < o.eval; ++i) {
                    for (auto& v : x) v = nd(rng);
                    expert_f32(ref[0].data(), ref[1].data(), down_f.data(), x.data(), cols, rows, h0.data(), y0.data());
                    expert_f32(sh[0].data(), sh[1].data(), down_f.data(), x.data(), cols, rows, h1.data(), y1.data());
                    part[(size_t) tid].h.add(h0.data(), h1.data(), h0.size());
                    part[(size_t) tid].y.add(y0.data(), y1.data(), y0.size());
                }
            }
        };
        std::vector<std::thread> pool;
        for (int t = 1; t < o.threads; ++t) pool.emplace_back(work, t);
        work(0);
        for (auto& th : pool) th.join();
        for (const auto& p : part) { lr.w.add(p.w); lr.h.add(p.h); lr.y.add(p.y); }
        lr.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        rep.push_back(lr);
    }
    io_ok = std::fclose(fo) == 0 && io_ok;
    if (!io_ok) { err = "write error on " + out_gguf; return false; }

    if (o.stride != 1) return true;   // a slice: the tensors hold n_out experts, not a loadable pack
    std::error_code rec;
    const std::filesystem::path gdir = std::filesystem::absolute(o.gguf).parent_path();
    const std::string rel = std::filesystem::relative(std::filesystem::absolute(out_gguf), gdir, rec).generic_string();
    if (rec || rel.empty() || rel.find(',') != std::string::npos) { err = "cannot name " + out_gguf + " relative to " + gdir.string(); return false; }
    std::ofstream idx(o.out + "/native_experts.txt");
    std::ostringstream body;
    uint64_t at = 0;
    for (const PackLine& l : pack.lines) {
        PackLine w = l;
        for (size_t ji = 0; ji < jobs.size(); ++ji) {
            if (jobs[ji].line != &l) continue;
            const uint64_t gu_old = 2 * row_bytes(l.gu_type, (int64_t) jobs[ji].cols) * jobs[ji].rows;
            const uint64_t gu_new = 2 * row_bytes(o.type, (int64_t) jobs[ji].cols) * jobs[ji].rows;
            w.gu_type = o.type;
            w.blob = l.blob - gu_old + gu_new;
            w.off[0] = header.size() + ts[2 * ji].offset;
            w.off[1] = header.size() + ts[2 * ji + 1].offset;
            w.file[0] = w.file[1] = rel;
        }
        w.offset = at;
        at += w.blob * (uint64_t) pack.n_expert;
        body << w.layer << ' ' << w.gu_type << ' ' << w.d_type << ' ' << w.offset << ' ' << w.blob << ' ' << w.off[0] << ' '
             << w.off[1] << ' ' << w.off[2];
        if (!(w.file[0].empty() && w.file[1].empty() && w.file[2].empty()))
            body << ' ' << w.file[0] << ',' << w.file[1] << ',' << w.file[2];
        body << '\n';
    }
    idx << "# strata native experts v4: layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard] (n_expert "
        << pack.n_expert << ", total " << at << "; " << type_name << " shadow of gate/up from "
        << o.pack << "; absolute offsets in " << src_name << ", or in the named file beside it)\n"
        << body.str();
    if (!idx) { err = "cannot write " + o.out + "/native_experts.txt"; return false; }
    return true;
}

}  // namespace strata::shadow
