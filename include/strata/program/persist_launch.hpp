// include/strata/program/persist_launch.hpp - the host side of a persistent launch built on include/strata/program/persist_tape.hpp (LEADS.md Part O O7,
// STRATA_PERSIST_LAYER).  persist_tape.hpp is the protocol (tape, claim -> segment, wait targets, bases) and its CPU reference; this header turns a tape
// into the flat arrays a device kernel reads (include/strata/kernels/persist_dev.hpp) and keeps the bases between launches.  Pure host code, no GPU:
// tested by src/program/persist_launch_test.cpp, which runs the device's claim / wait rule on these arrays with std::thread workers.
//
// WHAT IS WIRED (honest stop point, MOD_PLAN.md section 6): the switch STRATA_PERSIST_LAYER (persist_layer_enabled), this image / bases layer, the device
// kernel template (persist_dev.hpp, checked on a card by src/kernels/persist_layer_parity.cu) and the dependency tape shape of one decode layer
// (layer_tape below).  NOT wired: the decode loop does not launch it.  Every stage of the real decode is a normal kernel whose blocks read blockIdx; a
// persistent launch runs a stage's task on whichever workgroup claimed it, so each stage kernel needs a body that takes the task number as an argument
// (attn_chunk_body in qsa_decode_attn.cu is the one stage that already has this form).  Converting the ~10 kernels of a layer is the remaining work.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <vector>

#include "strata/program/persist_tape.hpp"

namespace strata::program {

// STRATA_PERSIST_LAYER=1: the decode runs a layer's GPU-resident span as one persistent launch (not wired; see above)
inline bool persist_layer_enabled(const char* env) { return env != nullptr && env[0] == '1' && env[1] == '\0'; }
inline bool persist_layer_from_env() { return persist_layer_enabled(std::getenv("STRATA_PERSIST_LAYER")); }

// the arrays of one launch: prefix (nseg + 1 task numbers), dep per segment (-1 none), and for each segment the done-counter value its dependency must
// reach (bases + count), plus the claim base; all 64-bit counters are the host's, carried between launches
struct PersistImage {
    std::vector<int64_t> prefix;
    std::vector<int32_t> dep;
    std::vector<uint64_t> wait_target;   // per segment; 0 when dep < 0 (never read)
    uint64_t claim_base = 0;
    int nseg = 0;
};

class PersistLauncher {
public:
    explicit PersistLauncher(const PersistTape& t) : tape_(t), bases_(t.seg.size()) {}
    const PersistTape& tape() const { return tape_; }
    // the image for the next launch, from the current bases
    PersistImage image() const {
        PersistImage im;
        im.nseg = int(tape_.seg.size());
        im.prefix = tape_.prefix;
        im.claim_base = bases_.claim;
        im.dep.resize(size_t(im.nseg));
        im.wait_target.assign(size_t(im.nseg), 0);
        for (int s = 0; s < im.nseg; ++s) {
            im.dep[size_t(s)] = tape_.seg[size_t(s)].dep;
            if (tape_.seg[size_t(s)].dep >= 0) im.wait_target[size_t(s)] = bases_.wait_target(tape_, tape_.seg[size_t(s)].dep);
        }
        return im;
    }
    // after a launch that ran to the end by `workers` workgroups (every worker makes one last claim past the end of the tape)
    void advance(int workers) { bases_.advance(tape_, workers); }
    const PersistBases& bases() const { return bases_; }

private:
    PersistTape tape_;
    PersistBases bases_;
};

// The dependency tape of one decode layer's GPU-resident span, by task counts (what the converted kernels would launch as blocks), in claim order:
// attention, router, shared expert, routed gate / up, routed down, combine.  A segment has ONE dependency (persist_tape.hpp), so the fan-out of the real
// data flow (the shared expert and the routed experts both read the router, the combine reads both) is written as a chain: a tape that let the combine
// wait on only the routed down could start it while a shared-expert task is still running.  The cost is that the shared expert does not overlap the
// routed experts; the probe prices that.  The stage names are placeholders for the SHAPE, not measurements.
inline PersistTape layer_tape(int attn_tasks, int router_tasks, int shared_tasks, int gate_up_tasks, int down_tasks, int combine_tasks) {
    std::vector<TapeSegment> s;
    s.push_back({attn_tasks, -1});       // 0 attention (reads the previous layer's output)
    s.push_back({router_tasks, 0});      // 1 router
    s.push_back({shared_tasks, 1});      // 2 shared expert
    s.push_back({gate_up_tasks, 2});     // 3 routed experts gate / up
    s.push_back({down_tasks, 3});        // 4 routed experts down
    s.push_back({combine_tasks, 4});     // 5 combine
    return PersistTape::build(s);
}

}  // namespace strata::program
