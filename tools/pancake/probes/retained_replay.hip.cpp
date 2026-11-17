// retained_replay.hip.cpp - LEADS.md Part P lead L13: the CEILING of a retained-PM4 / AQL replay of a decode window, before any
// engine work.  hipfire's "Redline" keeps the kernel graph as pre-built command packets and replays it through a public ROCr
// queue instead of a HIP launch per kernel (their plain-AR decode on a dense 27B, 7900 XTX: HIP graph 53.73 tok/s against
// retained PM4 62.81 tok/s, in-loop dispatch gap about 0.7 us).  Our window is 2,248-2,648 kernels (MOD_PLAN.md section 10) in
// 24.8 ms and is CPU-wait bound, so the lead's own rule applies: ceiling first.
//
// QUESTION   What does one dependent empty-kernel launch cost per node in (a) plain stream launches, (b) a captured hipGraph
//            chain and (c) a raw AQL queue replay of N pre-built packets, for N = 2,248 and 2,648, on the 7900 XT; and so
//            how much of the 24.8 ms window could a replay take out: N x (graph per node - replay per node)?
// BUILD      hipcc -O2 --offload-arch=gfx1100 tools/pancake/probes/retained_replay.hip.cpp -o /tmp/retained_replay
//            For arm (c) also: hipcc --offload-arch=gfx1100 --genco tools/pancake/probes/retained_replay.hip.cpp -o /tmp/empty.co
//            and build with -DSTRATA_PROBE_HSA (needs the ROCr headers, /opt/rocm/include/hsa, Linux; the Windows HIP SDK has no
//            HSA: the file compiles there WITHOUT arm (c), which is then reported "not built").
// RUN        /tmp/retained_replay [--n 2248,2648] [--blocks 96] [--threads 256] [--reps 20] [--window-ms 24.8]
//                                 [--hsaco /tmp/empty.co] [--device 0]
//            Needs the GPU idle (stop the engine first); seconds.
// READ       per N: stream / graph / replay microseconds per node (median of reps, wall clock around submit .. completion, so the
//            CPU submit cost is inside every arm), then the ceiling line: saved = N x (graph - replay) in ms and as a share of
//            the window.  RULE (LEADS.md L13): below 3% of the window stop; at or above, the engine step (record the window's
//            launches, validate against the HIP run) is worth a spec.  Arm (c) is an UPPER bound on the replay gain: it submits
//            empty kernels, so it ignores that our nodes are separated by real work that hides part of the dispatch gap.
//            ONLY the empty-kernel launch overhead is measured here; no hipfire code is used (idea only, no code read).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(STRATA_PROBE_HSA) && __has_include(<hsa/hsa.h>)
#define HAVE_HSA 1
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#else
#define HAVE_HSA 0
#endif

#define CK(x)                                                                                      \
    do {                                                                                           \
        hipError_t e_ = (x);                                                                       \
        if (e_ != hipSuccess) {                                                                    \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

extern "C" __global__ void __launch_bounds__(256) empty_k() {}   // extern "C": the code object symbol is `empty_k`

using clk = std::chrono::steady_clock;
static double us_since(clk::time_point t0) { return std::chrono::duration<double, std::micro>(clk::now() - t0).count(); }
static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

static double stream_per_node(hipStream_t s, int n, int blocks, int threads, int reps) {
    std::vector<double> t;
    for (int r = 0; r < reps + 2; ++r) {
        const auto t0 = clk::now();
        for (int i = 0; i < n; ++i) empty_k<<<blocks, threads, 0, s>>>();
        CK(hipStreamSynchronize(s));
        if (r >= 2) t.push_back(us_since(t0) / n);
    }
    return median(t);
}

static double graph_per_node(hipStream_t s, int n, int blocks, int threads, int reps) {
    hipGraph_t g;
    hipGraphExec_t ge;
    CK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));
    for (int i = 0; i < n; ++i) empty_k<<<blocks, threads, 0, s>>>();
    CK(hipStreamEndCapture(s, &g));
    CK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    std::vector<double> t;
    for (int r = 0; r < reps + 2; ++r) {
        const auto t0 = clk::now();
        CK(hipGraphLaunch(ge, s));
        CK(hipStreamSynchronize(s));
        if (r >= 2) t.push_back(us_since(t0) / n);
    }
    CK(hipGraphExecDestroy(ge));
    CK(hipGraphDestroy(g));
    return median(t);
}

#if HAVE_HSA
// (c) raw AQL: N dependent (barrier-bit) kernel dispatch packets written into a user queue from a retained template, one doorbell.
// NOT compiled where this file was written (no HSA headers on the Windows HIP SDK): the flow follows the ROCr API reference
// (hsa_queue_create, hsa_code_object_reader_create_from_file, hsa_executable_*, AQL kernel dispatch packet, doorbell signal).
struct Aql {
    hsa_agent_t agent{};
    hsa_queue_t* q = nullptr;
    hsa_signal_t done{};
    uint64_t kernel_object = 0;
    uint32_t kernarg_size = 0, group = 0, priv = 0;
    void* kernarg = nullptr;
    bool ok = false;
};

static hsa_status_t find_gpu(hsa_agent_t a, void* d) {
    hsa_device_type_t t;
    hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t);
    if (t == HSA_DEVICE_TYPE_GPU) { *(hsa_agent_t*) d = a; return HSA_STATUS_INFO_BREAK; }
    return HSA_STATUS_SUCCESS;
}
static hsa_status_t find_kernarg(hsa_amd_memory_pool_t p, void* d) {
    hsa_amd_segment_t seg;
    hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg);
    uint32_t flags = 0;
    hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
    if (seg == HSA_AMD_SEGMENT_GLOBAL && (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT)) {
        *(hsa_amd_memory_pool_t*) d = p;
        return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
}

static bool aql_open(Aql& a, const char* hsaco) {
    if (hsa_init() != HSA_STATUS_SUCCESS) return false;
    hsa_iterate_agents(find_gpu, &a.agent);
    if (hsa_queue_create(a.agent, 4096, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, UINT32_MAX, UINT32_MAX, &a.q) != HSA_STATUS_SUCCESS)
        return false;
    FILE* f = std::fopen(hsaco, "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<char> buf((size_t) sz);
    if (std::fread(buf.data(), 1, buf.size(), f) != buf.size()) { std::fclose(f); return false; }
    std::fclose(f);
    hsa_code_object_reader_t rd;
    hsa_executable_t ex;
    if (hsa_code_object_reader_create_from_memory(buf.data(), buf.size(), &rd) != HSA_STATUS_SUCCESS) return false;
    if (hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr, &ex) != HSA_STATUS_SUCCESS) return false;
    if (hsa_executable_load_agent_code_object(ex, a.agent, rd, nullptr, nullptr) != HSA_STATUS_SUCCESS) return false;
    if (hsa_executable_freeze(ex, nullptr) != HSA_STATUS_SUCCESS) return false;
    hsa_executable_symbol_t sym;
    if (hsa_executable_get_symbol_by_name(ex, "empty_k.kd", &a.agent, &sym) != HSA_STATUS_SUCCESS) return false;
    hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &a.kernel_object);
    hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &a.kernarg_size);
    hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &a.group);
    hsa_executable_symbol_get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &a.priv);
    hsa_amd_memory_pool_t pool{};
    hsa_amd_agent_iterate_memory_pools(a.agent, find_kernarg, &pool);
    const size_t ka = std::max<size_t>(a.kernarg_size, 64);
    if (hsa_amd_memory_pool_allocate(pool, ka, 0, &a.kernarg) != HSA_STATUS_SUCCESS) return false;
    std::memset(a.kernarg, 0, ka);
    hsa_amd_agents_allow_access(1, &a.agent, nullptr, a.kernarg);
    if (hsa_signal_create(1, 0, nullptr, &a.done) != HSA_STATUS_SUCCESS) return false;
    a.ok = true;
    return true;
}

// the retained template: the packet body of one dispatch; replay = copy it N times (header last, barrier bit set), ring once
static double aql_per_node(Aql& a, int n, int blocks, int threads, int reps) {
    hsa_kernel_dispatch_packet_t tpl{};
    tpl.setup = 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    tpl.workgroup_size_x = (uint16_t) threads; tpl.workgroup_size_y = 1; tpl.workgroup_size_z = 1;
    tpl.grid_size_x = (uint32_t) blocks * threads; tpl.grid_size_y = 1; tpl.grid_size_z = 1;
    tpl.private_segment_size = a.priv; tpl.group_segment_size = a.group;
    tpl.kernel_object = a.kernel_object; tpl.kernarg_address = a.kernarg;
    std::vector<double> t;
    for (int r = 0; r < reps + 2; ++r) {
        const auto t0 = clk::now();
        hsa_signal_store_relaxed(a.done, 1);
        const uint64_t base = hsa_queue_add_write_index_relaxed(a.q, (uint64_t) n);
        auto* ring = (hsa_kernel_dispatch_packet_t*) a.q->base_address;
        for (int i = 0; i < n; ++i) {
            hsa_kernel_dispatch_packet_t* p = &ring[(base + i) & (a.q->size - 1)];
            hsa_signal_t sig = (i == n - 1) ? a.done : hsa_signal_t{0};
            uint16_t hdr = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) | (1 << HSA_PACKET_HEADER_BARRIER) |
                           (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
                           (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
            std::memcpy(((char*) p) + 4, ((char*) &tpl) + 4, sizeof(tpl) - 4);   // everything but the header word
            p->completion_signal = sig;
            __atomic_store_n((uint16_t*) p, hdr, __ATOMIC_RELEASE);
        }
        hsa_signal_store_screlease(a.q->doorbell_signal, (hsa_signal_value_t) (base + n - 1));
        while (hsa_signal_wait_scacquire(a.done, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE) != 0) {}
        if (r >= 2) t.push_back(us_since(t0) / n);
    }
    return median(t);
}
#endif

int main(int argc, char** argv) {
    std::vector<int> ns = {2248, 2648};
    int blocks = 96, threads = 256, reps = 20, device = 0;
    double window_ms = 24.8;
    std::string hsaco;
    for (int i = 1; i < argc; ++i) {
        auto arg = [&](const char* k) { return !std::strcmp(argv[i], k) && i + 1 < argc; };
        if (arg("--n")) {
            ns.clear();
            const std::string s = argv[++i];
            for (size_t pos = 0; pos < s.size();) {
                const size_t q = s.find(',', pos);
                ns.push_back(std::atoi(s.substr(pos, q == std::string::npos ? q : q - pos).c_str()));
                if (q == std::string::npos) break;
                pos = q + 1;
            }
        } else if (arg("--blocks")) blocks = std::atoi(argv[++i]);
        else if (arg("--threads")) threads = std::atoi(argv[++i]);
        else if (arg("--reps")) reps = std::atoi(argv[++i]);
        else if (arg("--window-ms")) window_ms = std::atof(argv[++i]);
        else if (arg("--hsaco")) hsaco = argv[++i];
        else if (arg("--device")) device = std::atoi(argv[++i]);
        else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    CK(hipSetDevice(device));
    hipDeviceProp_t p;
    CK(hipGetDeviceProperties(&p, device));
    std::printf("device %d: %s (%s), %d blocks x %d threads, %d reps, window %.1f ms\n", device, p.name, p.gcnArchName, blocks,
                threads, reps, window_ms);
    hipStream_t s;
    CK(hipStreamCreate(&s));
    for (int i = 0; i < 4000; ++i) empty_k<<<blocks, threads, 0, s>>>();   // clock ramp
    CK(hipStreamSynchronize(s));
#if HAVE_HSA
    Aql aql;
    const bool have_aql = !hsaco.empty() && aql_open(aql, hsaco.c_str());
    if (!have_aql) std::printf("arm (c) AQL replay: could not open the queue / code object (--hsaco %s)\n", hsaco.c_str());
#else
    const bool have_aql = false;
    std::printf("arm (c) AQL replay: not built (no HSA headers or no -DSTRATA_PROBE_HSA)\n");
#endif
    for (int n : ns) {
        const double st = stream_per_node(s, n, blocks, threads, reps);
        const double gr = graph_per_node(s, n, blocks, threads, reps);
        std::printf("N %d: stream %.3f us/node, graph %.3f us/node", n, st, gr);
        double rp = -1.0;
#if HAVE_HSA
        if (have_aql) rp = aql_per_node(aql, n, blocks, threads, reps);
#endif
        if (rp >= 0) std::printf(", AQL replay %.3f us/node", rp);
        std::printf("\n");
        const double graph_ms = gr * n / 1000.0;
        std::printf("   graph chain total %.2f ms = %.1f%% of the %.1f ms window\n", graph_ms, 100.0 * graph_ms / window_ms, window_ms);
        if (rp >= 0) {
            const double saved = (gr - rp) * n / 1000.0;
            std::printf("   CEILING of a replay: %.2f ms = %.1f%% of the window (rule: below 3%% stop)\n", saved, 100.0 * saved / window_ms);
        }
    }
    return 0;
}
