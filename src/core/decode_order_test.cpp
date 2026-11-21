// LEADS.md Part S row S11: STRATA_DECODE_ORDER_SMALL's launch-order table (include/strata/core/decode_order.hpp), host only.
// Validated: the default table is the order the engine has always launched, the small table is a dependency-respecting
// permutation that differs only in the position of the indexer key projection, the parse, and that the default engine query
// (idx_key_late of the active order) is false when the switch is unset.  NOT validated: that the captured graph really
// launches in this order (needs the verifier on a GPU) and any speed effect.
#include "strata/core/decode_order.hpp"

#include <cstdio>
#include <cstdlib>

using namespace strata::core::decode_order;

static int checks = 0;
static void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

int main() {
    // compile-time facts too
    static_assert(order_valid(default_order()), "default order must be valid");
    static_assert(order_valid(small_order()), "small order must be valid");
    static_assert(!idx_key_late(default_order()) && idx_key_late(small_order()), "only the small order launches the keys late");

    const Order d = default_order(), s = small_order();
    // the default order is the engine's current sequence, written out
    const SmallOp want_default[kSmallOps] = {SmallOp::IdxKey,   SmallOp::KvProj,    SmallOp::KNormRope, SmallOp::KvRot,  SmallOp::IdxSnap,
                                             SmallOp::KvAppend, SmallOp::IdxAppend, SmallOp::QChain,    SmallOp::IdxQuery};
    bool same = true;
    for (int i = 0; i < kSmallOps; ++i) same = same && d[(size_t) i] == want_default[i];
    check(same, "the default table is the sequence record_window launches today");
    check(kSmallOps == 9, "nine small ops in the table");

    check(order_valid(d) && order_valid(s), "both tables are valid permutations");
    // the small order: only IdxKey moves, to just before IdxAppend
    check(position(s, SmallOp::IdxKey) + 1 == position(s, SmallOp::IdxAppend), "the keys go right before the append that reads them");
    int moved = 0;
    for (int k = 0; k < kSmallOps; ++k) {
        // the relative order of every op but IdxKey is unchanged
        if (static_cast<SmallOp>(k) == SmallOp::IdxKey) continue;
        for (int m = k + 1; m < kSmallOps; ++m) {
            if (static_cast<SmallOp>(m) == SmallOp::IdxKey) continue;
            const bool a = position(d, (SmallOp) k) < position(d, (SmallOp) m);
            const bool b = position(s, (SmallOp) k) < position(s, (SmallOp) m);
            if (a != b) ++moved;
        }
    }
    check(moved == 0, "every other pair keeps its relative order");
    check(position(d, SmallOp::IdxKey) == 0 && position(s, SmallOp::KvProj) == 0, "default: keys first; small: the K/V projection first");

    // the validator rejects broken orders
    {
        Order bad = s;
        bad[(size_t) position(s, SmallOp::IdxKey)] = SmallOp::KvProj;   // a duplicate and a missing op
        check(!order_valid(bad), "a duplicate / missing op is invalid");
        Order wrong = s;   // the keys after the append that reads them
        const int a = position(wrong, SmallOp::IdxKey), b = position(wrong, SmallOp::IdxAppend);
        const SmallOp t = wrong[(size_t) a]; wrong[(size_t) a] = wrong[(size_t) b]; wrong[(size_t) b] = t;
        check(!order_valid(wrong), "the keys after the append that reads them is invalid");
        Order snap = d;   // the snapshot after the append that overwrites the tail
        const int p = position(snap, SmallOp::IdxSnap), q = position(snap, SmallOp::IdxAppend);
        const SmallOp u = snap[(size_t) p]; snap[(size_t) p] = snap[(size_t) q]; snap[(size_t) q] = u;
        check(!order_valid(snap), "the snapshot after the append is invalid");
    }

    // parse and the default engine query
    check(!decode_order_small_parse(nullptr) && !decode_order_small_parse("") && !decode_order_small_parse("0") &&
              !decode_order_small_parse("2") && !decode_order_small_parse("11") && decode_order_small_parse("1"), "only 1 is on");
#if defined(_WIN32)
    _putenv_s("STRATA_DECODE_ORDER_SMALL", "");   // empty = unset on Windows
#else
    unsetenv("STRATA_DECODE_ORDER_SMALL");
#endif
    check(!idx_key_late(active_order()), "switch unset: the engine's query says the default order (keys first)");
    {
        const Order& a = active_order();
        bool eq = true;
        for (int i = 0; i < kSmallOps; ++i) eq = eq && a[(size_t) i] == d[(size_t) i];
        check(eq, "switch unset: the active order IS the default table");
    }
    std::printf("decode_order_test: %d checks passed\n", checks);
    return 0;
}
