// LEADS.md B147: STRATA_ADAPT_SPEC_COPY on a card (CUDA; HIP builds need the usual hipify macros and are not covered).
// Standalone: no model, no engine.  SKIPs (exit 77) when no CUDA device is visible.
//
//  (a) the time of one expert-sized host-to-device copy: IQ2_XS preset sizes per layer format (DERIVED bytes, see
//      tools/pancake/replay_spec_copy.py) and a poly-sized blob, pinned and pageable source, one copy at a time, timed with events:
//      mean, SD, coefficient of variation, min, max, GB/s.  Is "every expert copies in about the same time" true?
//  (b) the staged scheme against the baseline serial scheme on a schedule written by tools/pancake/replay_spec_copy.py --schedule from a
//      real routing trace (each round: the tier's swaps with their victim slots, which incoming experts were staged, the copies
//      issued at the end of the round).  Baseline: every swap is one H2D copy on the adapt stream.  Staged: a swap whose expert was
//      staged is a D2D copy from the staging slot, the others are the same H2D copy, and the next round's staged copies go out on a
//      second stream behind the finalising copies (the staging slot is read before it is rewritten) while a spin kernel stands for the
//      decode window.  Per round: the host-measured latency from issuing the round's copies until they have all landed (what the next
//      window's wait sees), the same by events, and the whole round's wall time.  Repeats alternate the schemes; the 95 % interval is a
//      Student-t interval over the repeats.  The final content of every slot must be byte-identical between the schemes.
//
//  tier_spec_copy_gpu_test [--sched FILE] [--reps N] [--windows 24.8,2,0] [--copies N] [--quick]
//  Without --sched a synthetic schedule is generated (part (b) then proves the logic and the bytes only).
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "tier_spec_copy_gpu_test: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}
double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct Stats { double mean, sd, mn, mx, med; size_t n; };
Stats stats(std::vector<double> v) {
    Stats s{};
    s.n = v.size();
    if (v.empty()) return s;
    double sum = 0;
    for (double x : v) sum += x;
    s.mean = sum / (double) v.size();
    double q = 0;
    for (double x : v) q += (x - s.mean) * (x - s.mean);
    s.sd = v.size() > 1 ? std::sqrt(q / (double) (v.size() - 1)) : 0.0;
    std::sort(v.begin(), v.end());
    s.mn = v.front();
    s.mx = v.back();
    s.med = v[v.size() / 2];
    return s;
}
double tcrit(size_t n) {   // two-sided 95 %, n - 1 degrees of freedom
    static const double t[] = {0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131};
    return n < 2 ? 0.0 : (n - 1 < 16 ? t[n - 1] : 1.96);
}
double ci_half(const std::vector<double>& v) { const Stats s = stats(v); return s.n < 2 ? 0.0 : tcrit(s.n) * s.sd / std::sqrt((double) s.n); }

__global__ void spin_kernel(long long cycles) {
    const long long t0 = clock64();
    while (clock64() - t0 < cycles) {}
}
__global__ void fill_kernel(uint8_t* p, size_t n, uint32_t seed) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        uint32_t h = (uint32_t) i * 2654435761u ^ seed;
        h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
        p[i] = (uint8_t) h;
    }
}

// ---- IQ2_XS preset expert bytes (derived from the dims: hidden 2560, expert width 640; gate/up IQ2_S 820 B a row, IQ2_XXS 660, IQ1_M 560, down Q2_0 180)
constexpr int kLayers = 48;
size_t layer_bytes(int l) {
    const size_t down = 2560u * 180u;
    if (l < 34) return 2u * 640u * 820u + down;       // 1,510,400
    if (l < 45) return 2u * 640u * 660u + down;       // 1,305,600
    return 2u * 640u * 560u + down;                    // 1,177,600
}
constexpr size_t kStride = 1510400;                    // the largest blob (a multiple of 512)
constexpr size_t kPolyBytes = 3u * 2048u * 512u * 9u / 16u;   // Qwen3.6-35B-A3B Q4_K expert, 1,769,472 B (derived: 4.5 bit)

// ---------------------------------------------------------------- (a) copy time
void part_a(int copies) {
    std::printf("\n(a) one H2D copy of one expert, one at a time, %d copies per row, GTX 1660 Ti class card (NOT gfx1100); sources rotate over 256 MiB\n", copies);
    std::printf("%-28s %-9s %9s %8s %6s %9s %9s %9s %8s\n", "blob", "source", "mean us", "SD us", "CV", "min us", "max us", "median", "GB/s");
    const size_t span = 256u << 20;
    uint8_t* pinned = nullptr;
    ck(cudaHostAlloc((void**) &pinned, span, cudaHostAllocDefault), "pinned 256 MiB");
    uint8_t* pageable = (uint8_t*) std::malloc(span);
    std::memset(pageable, 0x5a, span);
    std::memset(pinned, 0xa5, span);
    uint8_t* dev = nullptr;
    ck(cudaMalloc((void**) &dev, 8u << 20), "device");
    cudaStream_t st;
    ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    cudaEvent_t a, b;
    ck(cudaEventCreate(&a), "ev");
    ck(cudaEventCreate(&b), "ev");
    struct Row { const char* name; size_t bytes; };
    const Row rows[] = {{"IQ1_M layer 1,177,600 B", 1177600}, {"IQ2_XXS layer 1,305,600 B", 1305600}, {"IQ2_S layer 1,510,400 B", 1510400},
                        {"poly Q4_K 1,769,472 B", kPolyBytes}, {"Q2_0 daily 1,382,400 B", 1382400}};
    for (const Row& r : rows)
        for (int pin = 1; pin >= 0; --pin) {
            const uint8_t* base = pin ? pinned : pageable;
            std::vector<double> us;
            for (int i = -5; i < copies; ++i) {   // 5 warm-up copies
                const size_t off = ((size_t) (i + 5) * 7919u * 4096u) % (span - r.bytes);
                ck(cudaEventRecord(a, st), "rec");
                ck(cudaMemcpyAsync(dev, base + (off & ~(size_t) 255), r.bytes, cudaMemcpyHostToDevice, st), "h2d");
                ck(cudaEventRecord(b, st), "rec");
                ck(cudaEventSynchronize(b), "sync");
                float ms = 0;
                ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
                if (i >= 0) us.push_back((double) ms * 1000.0);
            }
            const Stats s = stats(us);
            std::printf("%-28s %-9s %9.1f %8.1f %6.3f %9.1f %9.1f %9.1f %8.2f\n", r.name, pin ? "pinned" : "pageable", s.mean, s.sd, s.sd / s.mean, s.mn, s.mx, s.med,
                        (double) r.bytes / (s.mean * 1e-6) / 1e9);
        }
    {   // device to device (the finalising copy)
        uint8_t* dev2 = nullptr;
        ck(cudaMalloc((void**) &dev2, 8u << 20), "device2");
        std::vector<double> us;
        for (int i = -5; i < copies; ++i) {
            ck(cudaEventRecord(a, st), "rec");
            ck(cudaMemcpyAsync(dev2, dev, 1510400, cudaMemcpyDeviceToDevice, st), "d2d");
            ck(cudaEventRecord(b, st), "rec");
            ck(cudaEventSynchronize(b), "sync");
            float ms = 0;
            ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
            if (i >= 0) us.push_back((double) ms * 1000.0);
        }
        const Stats s = stats(us);
        std::printf("%-28s %-9s %9.1f %8.1f %6.3f %9.1f %9.1f %9.1f %8.2f  (read + write)\n", "IQ2_S layer 1,510,400 B", "D2D", s.mean, s.sd, s.sd / s.mean, s.mn, s.mx, s.med,
                    (double) 1510400 / (s.mean * 1e-6) / 1e9);
        cudaFree(dev2);
    }
    cudaStreamDestroy(st);
    cudaFree(dev);
    cudaFreeHost(pinned);
    std::free(pageable);
}

// ---------------------------------------------------------------- (b) schedules
struct Inc { int l, e, slot, stage; };
struct Stg { int l, e, sslot; };
struct Round { std::vector<Inc> inc; std::vector<Stg> stg; };
struct Sched { int nl = 0, ne = 0, S = 0; std::vector<Round> rounds; };

bool load_sched(const std::string& path, Sched& s) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream is(line);
        std::string k;
        is >> k;
        if (k == "SCHED") is >> s.nl >> s.ne >> s.S;
        else if (k == "R") s.rounds.emplace_back();
        else if (k == "I" && !s.rounds.empty()) { Inc x; is >> x.l >> x.e >> x.slot >> x.stage; s.rounds.back().inc.push_back(x); }
        else if (k == "S" && !s.rounds.empty()) { Stg x; is >> x.l >> x.e >> x.sslot; s.rounds.back().stg.push_back(x); }
    }
    return s.nl > 0 && !s.rounds.empty();
}
// a synthetic schedule: 26 swaps a round over random layers, 11 % of the incoming experts staged one round before
Sched synthetic(int rounds) {
    Sched s;
    s.nl = kLayers; s.ne = 512; s.S = 4;
    uint64_t r = 88172645463325252ull;
    auto rnd = [&] { r ^= r << 13; r ^= r >> 7; r ^= r << 17; return (uint32_t) (r >> 16); };
    std::vector<Stg> prev;
    std::vector<std::vector<int>> occ((size_t) kLayers, std::vector<int>((size_t) s.S, -1));   // staged expert per (layer, stage slot)
    for (int i = 0; i < rounds; ++i) {
        Round rd;
        for (int k = 0; k < 26; ++k) {
            Inc x{(int) (rnd() % kLayers), (int) (rnd() % 512), (int) (rnd() % 227), -1};
            for (int q = 0; q < s.S; ++q) if (occ[(size_t) x.l][(size_t) q] >= 0 && rnd() % 9 == 0) { x.e = occ[(size_t) x.l][(size_t) q]; x.stage = q; occ[(size_t) x.l][(size_t) q] = -1; break; }
            rd.inc.push_back(x);
        }
        for (int k = 0; k < 8; ++k) {
            Stg x{(int) (rnd() % kLayers), (int) (rnd() % 512), (int) (rnd() % s.S)};
            if (occ[(size_t) x.l][(size_t) x.sslot] >= 0) continue;
            occ[(size_t) x.l][(size_t) x.sslot] = x.e;
            rd.stg.push_back(x);
        }
        s.rounds.push_back(rd);
    }
    return s;
}

uint64_t fnv(const uint8_t* p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

struct Dev {
    int SLc = 8;                       // local slots per layer (the schedule's slot % SLc)
    int S = 4;
    size_t P = 192;                    // host pool blobs (pinned)
    uint8_t *host = nullptr, *slots = nullptr, *stage = nullptr;
    cudaStream_t adapt, spec, comp;
    cudaEvent_t evS, evF, evAd, evSp;
    long long cyc_per_ms = 0;
};
size_t pool_of(int l, int e, size_t P) { return ((size_t) l * 37u + (size_t) e) % P; }

struct RunOut { std::vector<double> lat_host, lat_ev, wall; uint64_t hash = 0; int d2d = 0, h2d = 0; };

RunOut run_scheme(Dev& d, const Sched& sc, bool spec, double window_ms, int max_rounds) {
    RunOut o;
    ck(cudaMemset(d.slots, 0, (size_t) kLayers * d.SLc * kStride), "clear slots");
    ck(cudaMemset(d.stage, 0, (size_t) kLayers * d.S * kStride), "clear stage");
    ck(cudaDeviceSynchronize(), "sync");
    const int nr = std::min<int>(max_rounds, (int) sc.rounds.size());
    // the first round's staged copies were issued at the end of round 0 of the schedule; the schedule starts at round 1 with nothing staged
    for (int r = 0; r < nr; ++r) {
        const Round& rd = sc.rounds[(size_t) r];
        const double t0 = now_us();
        if (spec) ck(cudaStreamWaitEvent(d.adapt, d.evSp, 0), "wait spec");
        ck(cudaEventRecord(d.evS, d.adapt), "rec");
        if (window_ms > 0) spin_kernel<<<1, 1, 0, d.comp>>>((long long) (window_ms * (double) d.cyc_per_ms));
        for (const Inc& x : rd.inc) {
            uint8_t* dst = d.slots + ((size_t) x.l * d.SLc + (size_t) (x.slot % d.SLc)) * kStride;
            if (spec && x.stage >= 0) {
                ck(cudaMemcpyAsync(dst, d.stage + ((size_t) x.l * d.S + (size_t) x.stage) * kStride, layer_bytes(x.l), cudaMemcpyDeviceToDevice, d.adapt), "d2d");
                ++o.d2d;
            } else {
                ck(cudaMemcpyAsync(dst, d.host + pool_of(x.l, x.e, d.P) * kStride, layer_bytes(x.l), cudaMemcpyHostToDevice, d.adapt), "h2d");
                ++o.h2d;
            }
        }
        ck(cudaEventRecord(d.evF, d.adapt), "rec");
        if (spec) {   // the staged copies of the NEXT round: after the finalising copies have read the staging slots
            ck(cudaEventRecord(d.evAd, d.adapt), "rec");
            ck(cudaStreamWaitEvent(d.spec, d.evAd, 0), "wait adapt");
            for (const Stg& x : rd.stg)
                ck(cudaMemcpyAsync(d.stage + ((size_t) x.l * d.S + (size_t) x.sslot) * kStride, d.host + pool_of(x.l, x.e, d.P) * kStride, layer_bytes(x.l),
                                   cudaMemcpyHostToDevice, d.spec), "stage h2d");
            ck(cudaEventRecord(d.evSp, d.spec), "rec");
        }
        ck(cudaEventSynchronize(d.evF), "sync F");
        const double t1 = now_us();
        float ms = 0;
        ck(cudaEventElapsedTime(&ms, d.evS, d.evF), "elapsed");
        o.lat_host.push_back(t1 - t0);
        o.lat_ev.push_back((double) ms * 1000.0);
        ck(cudaDeviceSynchronize(), "sync round");   // the window and the staged copies are done: the next round starts clean
        o.wall.push_back(now_us() - t0);
    }
    std::vector<uint8_t> h((size_t) kLayers * d.SLc * kStride);
    ck(cudaMemcpy(h.data(), d.slots, h.size(), cudaMemcpyDeviceToHost), "read slots");
    o.hash = fnv(h.data(), h.size());
    return o;
}

int part_b(const Sched& sc, int reps, const std::vector<double>& windows, bool synthetic_sched) {
    Dev d;
    d.S = sc.S;
    cudaDeviceProp prop;
    ck(cudaGetDeviceProperties(&prop, 0), "props");
    int clk_khz = 0;
    ck(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0), "clock");
    d.cyc_per_ms = (long long) clk_khz;   // kHz = cycles per ms
    ck(cudaHostAlloc((void**) &d.host, d.P * kStride, cudaHostAllocDefault), "pinned pool");
    ck(cudaMalloc((void**) &d.slots, (size_t) kLayers * d.SLc * kStride), "slots");
    ck(cudaMalloc((void**) &d.stage, (size_t) kLayers * d.S * kStride), "stage");
    {   // pool content: deterministic, generated on the card then read back
        uint8_t* tmp = nullptr;
        ck(cudaMalloc((void**) &tmp, d.P * kStride), "tmp");
        fill_kernel<<<256, 256>>>(tmp, d.P * kStride, 0x9e3779b9u);
        ck(cudaMemcpy(d.host, tmp, d.P * kStride, cudaMemcpyDeviceToHost), "pool");
        cudaFree(tmp);
    }
    ck(cudaStreamCreateWithFlags(&d.adapt, cudaStreamNonBlocking), "s");
    ck(cudaStreamCreateWithFlags(&d.spec, cudaStreamNonBlocking), "s");
    ck(cudaStreamCreateWithFlags(&d.comp, cudaStreamNonBlocking), "s");
    for (cudaEvent_t* e : {&d.evS, &d.evF, &d.evAd, &d.evSp}) ck(cudaEventCreate(e), "ev");
    ck(cudaEventRecord(d.evSp, d.spec), "rec");   // the first wait sees a finished event
    long inc = 0, hits = 0, stg = 0;
    for (const Round& r : sc.rounds) { inc += (long) r.inc.size(); stg += (long) r.stg.size(); for (const Inc& x : r.inc) hits += x.stage >= 0; }
    std::printf("\n(b) %s schedule: %zu rounds, %.2f swaps/round, %.1f%% of the incoming experts staged (%ld of %ld), %.2f staged copies/round; %d slots/layer here, "
                "staging %d/layer, host pool %zu pinned blobs; %s (%d MiB VRAM total)\n",
                synthetic_sched ? "SYNTHETIC" : "trace-driven", sc.rounds.size(), (double) inc / (double) sc.rounds.size(), 100.0 * (double) hits / (double) inc, hits, inc,
                (double) stg / (double) sc.rounds.size(), d.SLc, d.S, d.P, prop.name,
                (int) (((size_t) kLayers * (d.SLc + d.S) * kStride + d.P * 0) >> 20));
    int bad = 0;
    uint64_t ref = 0;
    bool have_ref = false;
    for (double w : windows) {
        std::vector<double> bh, bw, be, sh, sw, se, diff;
        int d2d = 0, h2d = 0;
        for (int rep = -1; rep < reps; ++rep) {   // rep -1 is a warm-up of both schemes
            RunOut b, s;
            if (rep % 2 == 0) { b = run_scheme(d, sc, false, w, 1 << 30); s = run_scheme(d, sc, true, w, 1 << 30); }
            else { s = run_scheme(d, sc, true, w, 1 << 30); b = run_scheme(d, sc, false, w, 1 << 30); }
            if (!have_ref) { ref = b.hash; have_ref = true; }
            if (b.hash != ref || s.hash != ref) { ++bad; std::printf("  MISMATCH: baseline %016llx staged %016llx reference %016llx (window %.1f ms, rep %d)\n",
                                                                  (unsigned long long) b.hash, (unsigned long long) s.hash, (unsigned long long) ref, w, rep); }
            if (rep < 0) continue;
            d2d = s.d2d; h2d = s.h2d;
            const double mb = stats(b.lat_host).mean, ms = stats(s.lat_host).mean;
            bh.push_back(mb); sh.push_back(ms); diff.push_back(ms - mb);
            be.push_back(stats(b.lat_ev).mean); se.push_back(stats(s.lat_ev).mean);
            bw.push_back(stats(b.wall).mean); sw.push_back(stats(s.wall).mean);
        }
        auto row = [&](const char* nm, const std::vector<double>& x) { const Stats q = stats(x); std::printf("    %-34s %9.1f us  +-%.1f (95%% CI over %zu reps), SD %.1f\n", nm, q.mean, ci_half(x), q.n, q.sd); };
        std::printf("  window stand-in %.1f ms; staged scheme: %d device copies + %d host copies per run (baseline %d host copies)\n", w, d2d, h2d, d2d + h2d);
        row("baseline: latency until landed", bh);
        row("staged:   latency until landed", sh);
        row("difference staged - baseline", diff);
        row("baseline: latency by events", be);
        row("staged:   latency by events", se);
        row("baseline: round wall", bw);
        row("staged:   round wall", sw);
        const Stats dm = stats(diff);
        const double half = ci_half(diff);
        std::printf("    paired verdict: staged is %+.1f%% of the baseline latency [%+.1f%%, %+.1f%%]\n", 100.0 * dm.mean / stats(bh).mean,
                    100.0 * (dm.mean - half) / stats(bh).mean, 100.0 * (dm.mean + half) / stats(bh).mean);
    }
    std::printf("  bytes: %s\n", bad == 0 ? "every slot byte-identical between the baseline and the staged scheme in every run" : "MISMATCH");
    cudaFreeHost(d.host);
    cudaFree(d.slots);
    cudaFree(d.stage);
    return bad == 0 ? 0 : 1;
}
}  // namespace

int main(int argc, char** argv) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        (void) cudaGetLastError();
        std::printf("tier_spec_copy_gpu_test: no CUDA device: SKIP\n");
        return 77;
    }
    std::string sched_path;
    int reps = 5, copies = 300;
    std::vector<double> windows{24.8, 2.0, 0.0};
    bool quick = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--sched" && i + 1 < argc) sched_path = argv[++i];
        else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (a == "--copies" && i + 1 < argc) copies = std::atoi(argv[++i]);
        else if (a == "--quick") quick = true;
        else if (a == "--windows" && i + 1 < argc) {
            windows.clear();
            std::stringstream ss(argv[++i]);
            std::string t;
            while (std::getline(ss, t, ',')) windows.push_back(std::atof(t.c_str()));
        }
    }
    if (quick) { reps = 2; copies = 30; windows = {2.0, 0.0}; }
    cudaDeviceProp prop;
    ck(cudaGetDeviceProperties(&prop, 0), "props");
    std::printf("tier_spec_copy_gpu_test: %s, sm_%d%d, %.1f GiB (numbers are for THIS card, not for gfx1100)\n", prop.name, prop.major, prop.minor,
                (double) prop.totalGlobalMem / (1 << 30));
    part_a(copies);
    Sched sc;
    bool syn = sched_path.empty();
    if (syn) sc = synthetic(quick ? 20 : 120);
    else if (!load_sched(sched_path, sc)) { std::fprintf(stderr, "cannot read schedule %s\n", sched_path.c_str()); return 2; }
    if (sc.nl != kLayers) { std::fprintf(stderr, "schedule has %d layers, the test %d\n", sc.nl, kLayers); return 2; }
    const int rc = part_b(sc, reps, windows, syn);
    std::printf("tier_spec_copy_gpu_test: %s\n", rc == 0 ? "ok" : "FAILED");
    return rc;
}
