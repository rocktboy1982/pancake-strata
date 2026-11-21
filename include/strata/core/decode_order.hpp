#pragma once
// STRATA_DECODE_ORDER_SMALL=1 (LEADS.md Part S row S11, default off): the launch order of the small ops of a QSA layer's
// window graph, as a table the host can test.  radiance PR 498 (Flash-Next, 156 paired requests, texts identical):
// launch order alone moved the step time -2.5% (ms/step) / +3.3% tok/s; the stated cause is that a verify pass is about 900
// launches a rank in about 10 ms against 6.4 ms of weight reads, so which small kernel goes first decides whether the
// next op's weights are already in flight.  Our weight read-ahead switches (IC_WARM_MB, IC_WARM_DEFER: T-018, T-643, T-895,
// T-963) are a side-stream read; the launch order itself was not tried, and the launch-fusion switches screened as noise
// (T-604, T-612, T-614, T-616).
//
// What is coded (the part that is exact by construction): in the QSA layer of Verifier::record_window the indexer's key
// projection (IdxKey) is launched right before the indexer's append (IdxAppend), behind the K / V projection, their norm and
// RoPE, the rotations and the cell appends (the attention side), instead of first.  Every op below writes only its own
// buffers and the table says which pairs are data-dependent; any order that keeps those pairs is the same arithmetic on
// the same inputs, so the outputs are bit-identical (the captured graph is one stream: the order is the launch order).
// What is NOT coded: PR 498's other moves (router weight read beside the mixer reduction, the shared down projection before the
// routed down, chain polling by grid size): LEADS.md row S11 says so.
#include <array>
#include <cstdlib>

namespace strata::core::decode_order {

/// The small ops of one QSA layer, from the projections to the indexer's query (the steps between `mixed_` and the block
/// scores in Verifier::record_window; the one-stream case: no STRATA_SIDE_MIXER).
enum class SmallOp : int {
    IdxKey,      ///< indexer key projection  (reads mixed_, writes idx_raw)
    KvProj,      ///< the K and V projections (mm_set / mm)
    KNormRope,   ///< K norm and RoPE
    KvRot,       ///< the K / V rotations (kv_rot / kv_hybrid) - may be a no-op
    IdxSnap,     ///< the indexer tail's snapshot
    KvAppend,    ///< the window's K / V cells appended
    IdxAppend,   ///< the indexer's keys appended (reads idx_raw)
    QChain,      ///< the attention's query projection, split, norm, RoPE, rotation
    IdxQuery,    ///< the indexer's query projection, norm, RoPE
    kCount
};
constexpr int kSmallOps = (int) SmallOp::kCount;
using Order = std::array<SmallOp, kSmallOps>;

/// The order the engine has always launched (the default).
constexpr Order default_order() {
    return {SmallOp::IdxKey,   SmallOp::KvProj,   SmallOp::KNormRope, SmallOp::KvRot,   SmallOp::IdxSnap,
            SmallOp::KvAppend, SmallOp::IdxAppend, SmallOp::QChain,   SmallOp::IdxQuery};
}
/// STRATA_DECODE_ORDER_SMALL=1: the indexer keys behind the attention side, right before the append that reads them.
constexpr Order small_order() {
    return {SmallOp::KvProj,   SmallOp::KNormRope, SmallOp::KvRot,    SmallOp::IdxSnap,  SmallOp::KvAppend,
            SmallOp::IdxKey,   SmallOp::IdxAppend, SmallOp::QChain,   SmallOp::IdxQuery};
}

struct Dep { SmallOp first, then; };   ///< `first` must be launched before `then`
/// The data dependencies of the table (producer before consumer; the snapshot before the append that overwrites the tail).
constexpr std::array<Dep, 5> deps() {
    return {{{SmallOp::KvProj, SmallOp::KNormRope},
             {SmallOp::KNormRope, SmallOp::KvRot},
             {SmallOp::KvRot, SmallOp::KvAppend},
             {SmallOp::IdxKey, SmallOp::IdxAppend},
             {SmallOp::IdxSnap, SmallOp::IdxAppend}}};
}

constexpr int position(const Order& o, SmallOp op) {
    for (int i = 0; i < kSmallOps; ++i)
        if (o[(size_t) i] == op) return i;
    return -1;
}

/// A permutation of all the ops that keeps every dependency.
constexpr bool order_valid(const Order& o) {
    for (int k = 0; k < kSmallOps; ++k)
        if (position(o, (SmallOp) k) < 0) return false;
    for (int i = 0; i < kSmallOps; ++i)
        for (int j = i + 1; j < kSmallOps; ++j)
            if (o[(size_t) i] == o[(size_t) j]) return false;
    for (const Dep& d : deps())
        if (position(o, d.first) >= position(o, d.then)) return false;
    return true;
}

/// STRATA_DECODE_ORDER_SMALL: only the text "1" switches it on (default off).
inline bool decode_order_small_parse(const char* v) { return v != nullptr && v[0] == '1' && v[1] == '\0'; }

/// The order in force (read once).  Unset: default_order().
inline const Order& active_order() {
    static const Order o = decode_order_small_parse(std::getenv("STRATA_DECODE_ORDER_SMALL")) ? small_order() : default_order();
    return o;
}

/// Whether the order launches the indexer's key projection after the K / V projection (the one place the engine reads the
/// table).  False for default_order(), true for small_order().
constexpr bool idx_key_late(const Order& o) { return position(o, SmallOp::IdxKey) > position(o, SmallOp::KvProj); }

}  // namespace strata::core::decode_order
