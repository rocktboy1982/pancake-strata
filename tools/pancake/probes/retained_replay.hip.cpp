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
//
// ARM (d), LEADS.md Part S lead S3 (radiance's "chained launches", idea only, none of their code read or copied): N dependent kernels with NO
//            AQL barrier bit and NO fence between the packets; node i spins in-kernel on a device counter until node i - 1 has finished
//            (the last of node i - 1's blocks to take a ticket stores the counter), then runs.  Two measurements: (d1) the same chain_k nodes on a
//            plain HIP stream (the HIP runtime sets the barrier bit: this is the COST of the in-kernel wait, no saving), and (d2) raw AQL, barrier
//            bit 0 on every node but the first (needs -DSTRATA_PROBE_HSA like arm (c)): the CEILING of S3 = N x (graph - chained) against the
//            window, rule 3% (and N x (barrier replay - chained) as the share the missing barrier bit alone buys).  Safety: the AQL processor
//            dispatches the workgroups of packet i before those of packet i + 1 (in-order dispatch), so a spinning node never starves its
//            predecessor while the grid of every node fits the card (the default 96 x 256 does); every spin has an iteration limit and the run
//            reports a timeout instead of hanging the card.  After every rep the counter must equal N and every node's ticket count its block
//            count (`chain check`).  The memory-ordering argument for an engine step is in TEST_LEDGER.md (the S3 row): an in-kernel acquire
//            (agent scope) on every thread after the wait, a release (agent scope) on every thread before the ticket; the scalar-cache hazard
//            (a uniform load of data an earlier node wrote may hit the scalar cache: radiance saw one stale value in 1,800 launches without the
//            invalidate) is why an engine step must read cross-node data with vector loads.
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

// Arm (d): one node of a chain.  No blockDim / gridDim (a raw AQL dispatch does not fill the implicit kernel arguments): the block count is an
// argument.  counter: nodes finished so far; tickets[idx]: blocks of node idx that finished.  spin_limit: iterations before the wait gives up
// (bad[0] = 1; the node then runs without the guarantee: only reached when the in-order dispatch assumption is false).
extern "C" __global__ void __launch_bounds__(256) chain_k(unsigned* counter, unsigned* tickets, unsigned* bad, unsigned idx, unsigned nblocks,
                                                          unsigned spin_limit) {
    if (threadIdx.x == 0) {
        unsigned n = 0;
        while (__hip_atomic_load(counter, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < idx) {
            __builtin_amdgcn_s_sleep(1);
            if (++n >= spin_limit) { __hip_atomic_store(bad, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); break; }
        }
    }
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");   // every thread: its own caches see what the previous node wrote
    // (the node's real work would be here)
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");   // every thread: its writes are visible before the ticket
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned old = __hip_atomic_fetch_add(&tickets[idx], 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        if (old == nblocks - 1) __hip_atomic_store(counter, idx + 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
}
struct ChainMem {
    unsigned *counter = nullptr, *tickets = nullptr, *bad = nullptr;
    int n = 0;
    void init(int nodes) {
        n = nodes;
        CK(hipMalloc(&counter, sizeof(unsigned)));
        CK(hipMalloc(&tickets, sizeof(unsigned) * (size_t) nodes));
        CK(hipMalloc(&bad, sizeof(unsigned)));
    }
    void release() {
        CK(hipFree(counter));
        CK(hipFree(tickets));
        CK(hipFree(bad));
    }
    void reset(hipStream_t s) {
        CK(hipMemsetAsync(counter, 0, sizeof(unsigned), s));
        CK(hipMemsetAsync(tickets, 0, sizeof(unsigned) * (size_t) n, s));
        CK(hipMemsetAsync(bad, 0, sizeof(unsigned), s));
        CK(hipStreamSynchronize(s));
    }
    // true when every node ran to its end and no wait timed out
    bool check(int blocks, const char* what) const {
        unsigned c = 0, b = 0;
        std::vector<unsigned> t((size_t) n);
        CK(hipMemcpy(&c, counter, sizeof c, hipMemcpyDeviceToHost));
        CK(hipMemcpy(&b, bad, sizeof b, hipMemcpyDeviceToHost));
        CK(hipMemcpy(t.data(), tickets, sizeof(unsigned) * (size_t) n, hipMemcpyDeviceToHost));
        bool ok = c == (unsigned) n && b == 0;
        for (int i = 0; i < n; ++i) ok = ok && t[(size_t) i] == (unsigned) blocks;
        if (!ok) std::printf("   chain check %s: FAILED (counter %u of %d, timeout flag %u)\n", what, c, n, b);
        return ok;
    }
};
static const unsigned kSpinLimit = 1u << 22;

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

// (d1) the chain on a plain HIP stream: the in-kernel wait costs, the runtime's barrier bit already orders the nodes
static double chain_stream_per_node(hipStream_t s, ChainMem& m, int n, int blocks, int threads, int reps, bool* ok) {
    std::vector<double> t;
    *ok = true;
    for (int r = 0; r < reps + 2; ++r) {
        m.reset(s);
        const auto t0 = clk::now();
        for (int i = 0; i < n; ++i) chain_k<<<blocks, threads, 0, s>>>(m.counter, m.tickets, m.bad, (unsigned) i, (unsigned) blocks, kSpinLimit);
        CK(hipStreamSynchronize(s));
        if (r >= 2) t.push_back(us_since(t0) / n);
        *ok = *ok && m.check(blocks, "stream");
    }
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
    // arm (d): the chain kernel's symbol, and the kernarg pool for one block a node
    uint64_t chain_object = 0;
    uint32_t chain_kernarg_size = 0, chain_group = 0, chain_priv = 0;
    hsa_amd_memory_pool_t kpool{};
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
    hsa_executable_symbol_t csym;
    if (hsa_executable_get_symbol_by_name(ex, "chain_k.kd", &a.agent, &csym) == HSA_STATUS_SUCCESS) {
        hsa_executable_symbol_get_info(csym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &a.chain_object);
        hsa_executable_symbol_get_info(csym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &a.chain_kernarg_size);
        hsa_executable_symbol_get_info(csym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &a.chain_group);
        hsa_executable_symbol_get_info(csym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &a.chain_priv);
    }
    hsa_amd_memory_pool_t pool{};
    hsa_amd_agent_iterate_memory_pools(a.agent, find_kernarg, &pool);
    a.kpool = pool;
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

// (d2) arm (d) on raw AQL: N chain_k packets, barrier bit 0 (node 0 keeps it, to wait for the previous rep), no acquire / release fence on the
// middle packets, the last one releases to the system and signals.  One kernarg block a node (the node index differs).  NOT compiled where this
// file was written (no HSA headers on the Windows HIP SDK).
static double aql_chain_per_node(Aql& a, ChainMem& m, int n, int blocks, int threads, int reps, bool* ok) {
    *ok = false;
    if (!a.chain_object) return -1.0;
    const size_t ka = std::max<size_t>(((a.chain_kernarg_size + 63) / 64) * 64, 64);
    void* kargs = nullptr;
    if (hsa_amd_memory_pool_allocate(a.kpool, ka * (size_t) n, 0, &kargs) != HSA_STATUS_SUCCESS) return -1.0;
    hsa_amd_agents_allow_access(1, &a.agent, nullptr, kargs);
    std::memset(kargs, 0, ka * (size_t) n);
    struct Args { unsigned long long counter, tickets, bad; unsigned idx, nblocks, spin; };
    static_assert(sizeof(Args) <= 64, "kernel arguments");
    for (int i = 0; i < n; ++i) {
        Args* p = (Args*) ((char*) kargs + ka * (size_t) i);
        p->counter = (unsigned long long) (uintptr_t) m.counter; p->tickets = (unsigned long long) (uintptr_t) m.tickets;
        p->bad = (unsigned long long) (uintptr_t) m.bad;
        p->idx = (unsigned) i; p->nblocks = (unsigned) blocks; p->spin = kSpinLimit;
    }
    hsa_kernel_dispatch_packet_t tpl{};
    tpl.setup = 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    tpl.workgroup_size_x = (uint16_t) threads; tpl.workgroup_size_y = 1; tpl.workgroup_size_z = 1;
    tpl.grid_size_x = (uint32_t) blocks * threads; tpl.grid_size_y = 1; tpl.grid_size_z = 1;
    tpl.private_segment_size = a.chain_priv; tpl.group_segment_size = a.chain_group;
    tpl.kernel_object = a.chain_object;
    std::vector<double> t;
    *ok = true;
    for (int r = 0; r < reps + 2; ++r) {
        m.reset(0);
        const auto t0 = clk::now();
        hsa_signal_store_relaxed(a.done, 1);
        const uint64_t base = hsa_queue_add_write_index_relaxed(a.q, (uint64_t) n);
        auto* ring = (hsa_kernel_dispatch_packet_t*) a.q->base_address;
        for (int i = 0; i < n; ++i) {
            hsa_kernel_dispatch_packet_t* p = &ring[(base + i) & (a.q->size - 1)];
            hsa_signal_t sig = (i == n - 1) ? a.done : hsa_signal_t{0};
            uint16_t hdr = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);
            if (i == 0) hdr |= (1 << HSA_PACKET_HEADER_BARRIER) | (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE);
            if (i == n - 1) hdr |= (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
            std::memcpy(((char*) p) + 4, ((char*) &tpl) + 4, sizeof(tpl) - 4);
            p->kernarg_address = (char*) kargs + ka * (size_t) i;
            p->completion_signal = sig;
            __atomic_store_n((uint16_t*) p, hdr, __ATOMIC_RELEASE);
        }
        hsa_signal_store_screlease(a.q->doorbell_signal, (hsa_signal_value_t) (base + n - 1));
        while (hsa_signal_wait_scacquire(a.done, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE) != 0) {}
        if (r >= 2) t.push_back(us_since(t0) / n);
        *ok = *ok && m.check(blocks, "AQL");
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
    ChainMem chain;
    chain.init(*std::max_element(ns.begin(), ns.end()));
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
        chain.n = n;
        bool ok_d1 = false;
        const double cs = chain_stream_per_node(s, chain, n, blocks, threads, reps, &ok_d1);
        std::printf("   (d1) chained nodes on a HIP stream %.3f us/node (the in-kernel wait costs %+.3f us/node against empty nodes)%s\n", cs, cs - st,
                    ok_d1 ? "" : "  CHAIN CHECK FAILED");
        double ch = -1.0;
#if HAVE_HSA
        bool ok_d2 = false;
        if (have_aql) {
            ch = aql_chain_per_node(aql, chain, n, blocks, threads, reps, &ok_d2);
            if (ch >= 0) std::printf("   (d2) AQL chained, no barrier bit %.3f us/node%s\n", ch, ok_d2 ? "" : "  CHAIN CHECK FAILED");
            else std::printf("   (d2) AQL chained: chain_k not in the code object (--hsaco must come from this file)\n");
        }
#endif
        const double graph_ms = gr * n / 1000.0;
        std::printf("   graph chain total %.2f ms = %.1f%% of the %.1f ms window\n", graph_ms, 100.0 * graph_ms / window_ms, window_ms);
        if (rp >= 0) {
            const double saved = (gr - rp) * n / 1000.0;
            std::printf("   CEILING of a replay: %.2f ms = %.1f%% of the window (rule: below 3%% stop)\n", saved, 100.0 * saved / window_ms);
        }
        if (ch >= 0) {
            const double saved = (gr - ch) * n / 1000.0;
            std::printf("   CEILING of S3 chained launches: %.2f ms = %.1f%% of the window against the graph chain (rule: below 3%% stop)", saved,
                        100.0 * saved / window_ms);
            if (rp >= 0) std::printf("; against the barrier replay %.2f ms = %.1f%%", (rp - ch) * n / 1000.0, 100.0 * (rp - ch) * n / 1000.0 / window_ms);
            std::printf("\n");
        }
    }
    chain.release();
    return 0;
}
