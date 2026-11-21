// CPU-only test of the compact GDN snapshot store (STRATA_SNAP_STATE_F16, LEADS.md S6, follow-up to gdn_state_f16_test):
// the blob in whole and in blocks of any size (not a multiple of 4, starting inside a value), the size accounting, the disk
// header flag in both directions (an image written with the switch off loads with it on, and the other way round; a blob is
// never read as FP32 nor FP32 as a blob), the streamed live-state write and the staged streamed restore, and that a file written
// without the switch has the flag zero (default path unchanged: conversation_file_test's golden hash is the byte-for-byte check).
// Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no CUDA, no model.  The engine glue that needs a device (checkpoint_save /
// restore, ConversationStateTarget) is thin: it calls the helpers tested here.
#include "strata/core/conversation_file.hpp"
#include "strata/core/gdn_state_f16.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace strata::core;
namespace g16 = strata::core::gdn_f16;
namespace fs = std::filesystem;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

std::mt19937_64 rng(0x6d16a5eedull);

// FP32 state-like bytes: mostly small values with a few large, some zeros
std::vector<uint8_t> fp32_bytes(size_t n_values) {
    std::vector<uint8_t> v(n_values * 4);
    std::normal_distribution<float> nd(0.0f, 0.3f);
    for (size_t i = 0; i < n_values; ++i) {
        float f = (i % 17 == 0) ? 0.0f : nd(rng) * ((i % 101 == 0) ? 400.0f : 1.0f);
        std::memcpy(v.data() + 4 * i, &f, 4);
    }
    return v;
}
std::vector<uint8_t> rounded(const std::vector<uint8_t>& f32) {
    std::vector<uint8_t> r = f32;
    g16::round_bytes_inplace(r.data(), r.size());
    return r;
}
std::vector<uint8_t> blob_of(const std::vector<uint8_t>& f32) {
    std::vector<uint8_t> b;
    check(g16::encode(f32.data(), f32.size(), b), "encode");
    return b;
}
std::vector<uint8_t> small_bytes(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = uint8_t(i * 7u + seed);
    return v;
}

// blob_slice over an in-memory FP32 array
bool slice_of(const std::vector<uint8_t>& f32, size_t off, size_t n, uint8_t* dst) {
    return g16::blob_slice([&](size_t at, void* to, size_t bytes) {
        if (at % 4 || bytes % 4 || at + bytes > f32.size()) return false;
        std::memcpy(to, f32.data() + at, bytes);
        return true;
    }, f32.size() / 4, off, n, dst);
}

void test_slices_and_stage() {
    for (size_t nv : {size_t(0), size_t(1), size_t(2), size_t(3), size_t(7), size_t(1000), size_t(65537), size_t(300001)}) {
        const std::vector<uint8_t> f32 = fp32_bytes(nv);
        const std::vector<uint8_t> blob = blob_of(f32);
        check(blob.size() == g16::blob_bytes(nv) && blob.size() == g16::stored_bytes(true, f32.size()), "blob size accounting");
        check(g16::blob_raw_bytes(blob.size()) == f32.size(), "blob_raw_bytes inverts stored_bytes");
        // the whole blob in one slice
        std::vector<uint8_t> whole(blob.size());
        check(slice_of(f32, 0, whole.size(), whole.data()) && whole == blob, "whole slice equals encode");
        // random blocks of any size and offset (1 byte .. a few thousand, odd offsets included), and a streamed reassembly
        std::vector<uint8_t> got(blob.size());
        size_t at = 0, blocks = 0, odd = 0;
        g16::BlobStage stage(blob.size());
        while (at < blob.size()) {
            const size_t n = std::min<size_t>(blob.size() - at, 1 + rng() % std::max<size_t>(1, std::min<size_t>(4099, blob.size() / 3)));
            std::vector<uint8_t> blk(n);
            check(slice_of(f32, at, n, blk.data()), "block slice");
            check(std::memcmp(blk.data(), blob.data() + at, n) == 0, "block slice equals the blob bytes");
            check(stage.put(at, blk.data(), n), "stage accepts the next block");
            if (n % 4 || at % 4) ++odd;
            std::memcpy(got.data() + at, blk.data(), n);
            at += n; ++blocks;
        }
        check(got == blob, "blocks reassemble the blob");
        std::vector<uint8_t> back;
        check(stage.finish(back, f32.size()) && back == rounded(f32), "staged blocks decode to the FP16-rounded state");
        if (blob.size() > 64) check(odd > 0, "the test did use blocks that are not multiples of 4");
        (void) blocks;
    }
    // a slice outside the blob, a bad stage use
    const std::vector<uint8_t> f32 = fp32_bytes(100);
    std::vector<uint8_t> d(10);
    check(!slice_of(f32, g16::blob_bytes(100) - 5, 10, d.data()), "slice past the end refused");
    check(!g16::blob_slice([](size_t, void*, size_t) { return false; }, 100, 20, 10, d.data()), "a failing read fails the slice");
    g16::BlobStage st(32);
    uint8_t z[16] = {};
    check(!st.put(4, z, 4), "stage refuses a block that skips bytes");
    check(st.put(0, z, 16) && !st.put(0, z, 16), "stage refuses a repeated block");
    std::vector<uint8_t> o;
    check(!st.finish(o, 0), "stage refuses to finish incomplete");
    check(st.put(16, z, 16) && !st.finish(o, 0), "an all-zero blob (no magic) is not decoded");
    check(!st.put(32, z, 1), "stage refuses a block past the end");
    // blob / FP32 are never confused: the FP32 bytes of the same state do not decode, a blob is not the FP32 size
    const std::vector<uint8_t> raw = fp32_bytes(5000), blob = blob_of(raw);
    std::vector<uint8_t> junk;
    check(!g16::decode(raw.data(), raw.size(), junk), "FP32 bytes are not a blob");
    check(blob.size() != raw.size() && g16::stored_bytes(false, raw.size()) == raw.size(), "default stored size is the FP32 size");
    // a blob with one byte cut or added, or a flipped magic / count, is refused
    std::vector<uint8_t> cut(blob.begin(), blob.end() - 1), add = blob, magic = blob, cnt = blob;
    add.push_back(0); magic[0] ^= 1; cnt[8] ^= 1;
    check(!g16::decode(cut.data(), cut.size(), junk) && !g16::decode(add.data(), add.size(), junk) &&
          !g16::decode(magic.data(), magic.size(), junk) && !g16::decode(cnt.data(), cnt.size(), junk), "damaged blobs are refused");
}

ConversationCheckpoint cp(size_t tokens, uint8_t seed, const std::vector<uint8_t>& f32, bool blob) {
    ConversationCheckpoint c;
    for (size_t i = 0; i < tokens; ++i) c.ids.push_back(int32_t(1000 + i));
    c.imgs = {{3, 0x1234567890abcdefull ^ seed}};
    c.gdn = blob ? blob_of(f32) : f32;
    c.gdn_f16 = blob;
    c.ple = small_bytes(77, seed + 1);
    c.tails = small_bytes(301, seed + 2);
    c.dead = small_bytes(64, seed + 3);
    c.block_pos = small_bytes(8, seed + 4);
    c.used = 42 + seed;
    return c;
}

struct Parts {
    std::vector<uint8_t> live, a, b;   // the FP32 states
};

SavedConversation image(const Parts& p, bool live_blob, bool a_blob, bool b_blob) {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(100 + i);
    s.layer_lo = 0; s.layer_hi = 48;
    s.cvec = false;
    s.live = cp(1000, 1, p.live, live_blob);
    s.checkpoints.push_back(cp(10, 2, p.a, a_blob));
    s.checkpoints.push_back(cp(500, 3, p.b, b_blob));
    for (int layer = 0; layer < 2; ++layer) {
        ConversationKv kv;
        kv.format = 2 + layer; kv.cells = 64; kv.heads = 2; kv.head_dim = 16;
        kv.page_size = 64; kv.pooled_rows = 0; kv.idx_dim = 128;
        kv.k.resize(4096); kv.v.resize(4096);
        kv.k.visit(0, 4096, [&](uint8_t* q, size_t c, size_t at) { for (size_t i = 0; i < c; ++i) q[i] = uint8_t(at + i + layer); return true; });
        kv.v.visit(0, 4096, [&](uint8_t* q, size_t c, size_t at) { for (size_t i = 0; i < c; ++i) q[i] = uint8_t(at * 3 + i); return true; });
        s.kv.push_back(std::move(kv));
    }
    return s;
}

std::vector<char> slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
uint64_t flags_of(const std::vector<char>& g) {
    uint64_t v = 0;
    std::memcpy(&v, g.data() + 40, 8);
    return v;
}

bool write(const fs::path& p, const SavedConversation& s, const SessionFileIdentity& id) {
    size_t bytes = 0;
    std::string error;
    return session_file_write(p.string(), s, id, bytes, error);
}
bool read(const fs::path& p, const SessionFileIdentity& id, SavedConversation& out) {
    size_t bytes = 0;
    std::string error;
    return session_file_read(p.string(), id, out, bytes, error);
}

void test_disk(const fs::path& dir) {
    const SessionFileIdentity id{0x1111222233334444ull, 0x5555666677778888ull};
    Parts p{fp32_bytes(40009), fp32_bytes(40009), fp32_bytes(40009)};
    const fs::path f_off = dir / "off.bin", f_on = dir / "on.bin", f_a = dir / "a.bin", f_b = dir / "b.bin";

    // switch off: all FP32, header flags zero, reads back as FP32, bytes equal
    check(write(f_off, image(p, false, false, false), id), "write FP32 image");
    const auto g_off = slurp(f_off);
    check(flags_of(g_off) == 0, "default image: header flags are zero");
    SavedConversation r;
    check(read(f_off, id, r), "read FP32 image");
    check(!r.live.gdn_f16 && !r.checkpoints[0].gdn_f16 && !r.checkpoints[1].gdn_f16, "FP32 image reads as FP32");
    check(r.live.gdn == p.live && r.checkpoints[0].gdn == p.a && r.checkpoints[1].gdn == p.b, "FP32 image: bytes exact");
    // the same, written twice, is identical (deterministic): default path unchanged
    check(write(dir / "off2.bin", image(p, false, false, false), id) && slurp(dir / "off2.bin") == g_off, "FP32 image is byte-stable");

    // switch on: all blobs, flag 1, half the GDN bytes, reads back as blobs, decode == rounded
    check(write(f_on, image(p, true, true, true), id), "write blob image");
    const auto g_on = slurp(f_on);
    check(flags_of(g_on) == 1, "blob image: header flag set");
    check(g_on.size() < g_off.size() - 3 * 40009 * 4 / 2 + 3 * 64, "blob image is about 3 x half a state smaller");
    SavedConversation rb;
    check(read(f_on, id, rb), "read blob image");
    check(rb.live.gdn_f16 && rb.checkpoints[0].gdn_f16 && rb.checkpoints[1].gdn_f16, "blob image reads as blobs");
    check(rb.live.gdn == blob_of(p.live) && rb.checkpoints[1].gdn == blob_of(p.b), "blob image: blob bytes exact");
    std::vector<uint8_t> back;
    check(g16::decode(rb.live.gdn.data(), rb.live.gdn.size(), back) && back == rounded(p.live), "blob image decodes to the rounded state");
    check(rb.live.ple == r.live.ple && rb.kv.size() == 2 && rb.checkpoints[0].ids == r.checkpoints[0].ids, "the rest of the image is the same");

    // mixed in memory (a switch-off image loaded by a switch-on run and the other way round), written in the live part's format
    check(write(f_a, image(p, false, true, false), id), "write: live FP32, checkpoints partly blobs");
    check(flags_of(slurp(f_a)) == 0, "live FP32: flag zero");
    SavedConversation ra;
    check(read(f_a, id, ra), "read converted to FP32");
    check(!ra.checkpoints[0].gdn_f16 && ra.checkpoints[0].gdn == rounded(p.a) && ra.checkpoints[1].gdn == p.b &&
          ra.live.gdn == p.live, "a blob in an FP32 file is stored as its rounded FP32 values; FP32 parts are exact");
    check(write(f_b, image(p, true, false, true), id), "write: live blob, a checkpoint FP32");
    check(flags_of(slurp(f_b)) == 1, "live blob: flag set");
    SavedConversation rc;
    check(read(f_b, id, rc), "read converted to blobs");
    check(rc.checkpoints[0].gdn_f16 && rc.checkpoints[0].gdn == blob_of(p.a) && rc.checkpoints[1].gdn == blob_of(p.b),
          "an FP32 part in a blob file is stored as the blob");

    // a flag the reader does not know, with a valid header hash, is refused; a flipped flag with a stale hash is refused
    {
        auto g = g_on;
        uint64_t unknown = 1 | 2;
        std::memcpy(g.data() + 40, &unknown, 8);
        const uint64_t hh = session_hash64(g.data(), 56, 0);
        std::memcpy(g.data() + 56, &hh, 8);
        const fs::path q = dir / "unknown.bin";
        { std::ofstream f(q, std::ios::binary | std::ios::trunc); f.write(g.data(), (std::streamsize) g.size()); }
        SavedConversation x; size_t n = 0; std::string e;
        check(!session_file_read(q.string(), id, x, n, e) && e.find("invalid header fields") != std::string::npos, "an unknown header flag is refused");
        auto g2 = g_off;
        g2[40] = 1;   // FP32 payload claiming to be blobs, header hash not updated
        const fs::path q2 = dir / "flip.bin";
        { std::ofstream f(q2, std::ios::binary | std::ios::trunc); f.write(g2.data(), (std::streamsize) g2.size()); }
        e.clear();
        check(!session_file_read(q2.string(), id, x, n, e) && e.find("header checksum") != std::string::npos, "a flipped flag is refused by the header hash");
        // and with the hash fixed, an FP32 payload under the blob flag is read as blobs (tagged), never silently as FP32: the
        // engine's size check (state_sizes_valid: stored_bytes(blob, raw)) then refuses it; the file layer only tags
        const uint64_t one = 1;
        std::memcpy(g2.data() + 40, &one, 8);
        const uint64_t h2 = session_hash64(g2.data(), 56, 0);
        std::memcpy(g2.data() + 56, &h2, 8);
        { std::ofstream f(q2, std::ios::binary | std::ios::trunc); f.write(g2.data(), (std::streamsize) g2.size()); }
        SavedConversation y;
        check(session_file_read(q2.string(), id, y, n, e) && y.live.gdn_f16 && y.live.gdn.size() == p.live.size(), "flag 1 tags the parts as blobs");
        check(!g16::decode(y.live.gdn.data(), y.live.gdn.size(), back), "...and such FP32 bytes fail the blob decode");
    }

    // streamed live state: a source that reads the blob in odd blocks from FP32 (as state_source does from the device) writes the
    // same file bytes as the image with the blob
    {
        SessionSaveSources save;
        const SavedConversation full = image(p, true, true, true);
        save.meta.geometry = full.geometry; save.meta.layer_lo = full.layer_lo; save.meta.layer_hi = full.layer_hi;
        save.meta.cvec = full.cvec;
        save.meta.live.ids = full.live.ids; save.meta.live.imgs = full.live.imgs; save.meta.live.used = full.live.used;
        save.state = session_state_source(full.live);
        save.state.gdn_f16 = true;
        const std::vector<uint8_t>& f32 = p.live;
        save.state.sizes[0] = g16::blob_bytes(f32.size() / 4);
        size_t calls = 0;
        const auto base = save.state.read;
        save.state.read = [&f32, base, &calls](size_t part, size_t off, void* dst, size_t n) {
            if (part != 0) return base(part, off, dst, n);
            ++calls;
            // split into pieces of odd size: the reader may ask for any block
            for (size_t at = 0; at < n;) {
                const size_t k = std::min<size_t>(n - at, 1 + (at * 7 + off) % 1237);
                if (!slice_of(f32, off + at, k, static_cast<uint8_t*>(dst) + at)) return false;
                at += k;
            }
            return true;
        };
        for (const auto& c : full.checkpoints) save.checkpoints.push_back(&c);
        for (const auto& k : full.kv) {
            SessionKvSource src;
            src.format = k.format; src.cells = k.cells; src.heads = k.heads; src.head_dim = k.head_dim;
            src.page_size = k.page_size; src.pooled_rows = k.pooled_rows; src.idx_dim = k.idx_dim;
            const std::array<const ConversationBuffer*, 5> parts = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
            for (size_t i = 0; i < 5; ++i) src.sizes[i] = parts[i]->size();
            src.read = [parts](size_t part, size_t offset, void* dst, size_t n) { return parts[part]->read(dst, offset, n); };
            save.kv.push_back(std::move(src));
        }
        size_t bytes = 0; std::string e;
        const fs::path q = dir / "stream.bin";
        check(session_file_write(q.string(), save, id, bytes, e), "streamed write of a blob live state");
        check(calls > 0 && slurp(q) == g_on, "streamed blob live state: the file is byte-identical to the image write");
        // the same stream, with FP32 checkpoints converted to the file's blob format
        SessionSaveSources save2 = save;
        const SavedConversation mixed = image(p, true, false, false);
        save2.checkpoints.clear();
        for (const auto& c : mixed.checkpoints) save2.checkpoints.push_back(&c);
        const fs::path q2 = dir / "stream2.bin";
        check(session_file_write(q2.string(), save2, id, bytes, e) && slurp(q2) == g_on, "streamed write converts FP32 checkpoints to the file's blob format");
        // read it back as a streamed restore: the live blob arrives in blocks, is staged, and decodes to the rounded state
        SavedConversation meta; SessionStreamLayout layout; size_t n = 0;
        check(session_file_read_streamed(q.string(), id, meta, n, e, {}, layout), "streamed read of the blob file");
        check(meta.live.gdn_f16 && meta.live.gdn.empty() && layout.live_state[0] == g16::blob_bytes(f32.size() / 4), "streamed meta: blob flag, blob part size");
        g16::BlobStage stage(layout.live_state[0]);
        size_t blocks = 0;
        check(session_file_apply_state_streamed(q.string(), id, [&](size_t part, uint64_t off, const void* d, size_t k) {
            if (part != 0) return true;
            ++blocks;
            return stage.put((size_t) off, d, k);
        }, layout, e), "streamed state apply");
        check(blocks > 0 && stage.finish(back, f32.size()) && back == rounded(f32), "streamed restore decodes to the rounded state");
        // the same with a switch-off file: not a blob, sizes are the FP32 size
        SavedConversation meta0; SessionStreamLayout layout0;
        SessionSaveSources s0 = save;
        const SavedConversation raw = image(p, false, false, false);
        s0.state = session_state_source(raw.live);
        s0.checkpoints.clear();
        for (const auto& c : raw.checkpoints) s0.checkpoints.push_back(&c);
        const fs::path q0 = dir / "stream0.bin";
        check(session_file_write(q0.string(), s0, id, bytes, e) && slurp(q0) == g_off, "streamed FP32 write is byte-identical to the image write");
        check(session_file_read_streamed(q0.string(), id, meta0, n, e, {}, layout0) && !meta0.live.gdn_f16 && layout0.live_state[0] == f32.size(), "streamed FP32 meta: no blob flag");
    }
}
}  // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / ("strata-gdn-blob-test-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    test_slices_and_stage();
    test_disk(dir);
    // sizes the engine checks: a 111 MiB state halves (plus the 16 byte header)
    const size_t raw = 36u * (128u * 32u * 128u + 8192u * 3u) * 4u;
    check(g16::stored_bytes(false, raw) == raw && g16::stored_bytes(true, raw) == raw / 2 + 16, "size accounting at a real state size");
    std::error_code ec;
    fs::remove_all(dir, ec);
    std::printf("gdn_snap_blob_test: %d checks passed\n", checks);
    return 0;
}
