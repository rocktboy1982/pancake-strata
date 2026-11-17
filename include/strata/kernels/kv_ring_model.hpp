// include/strata/kernels/kv_ring_model.hpp - LEADS.md Part N N2: a host model of the streamed K/V ring with a LIVE slot count
// (STRATA_KV_GROW_RING design).  Host only.  src/kernels/cuda/kv_stream.cu keeps the resident blocks of a streamed state (`--kv-resident`) in a
// pool of n_slots pages chosen by a clock sweep (resolve_kernel); the K/V of every block always lives in the pinned host copy (the writers
// `kv_append_*` write the host copy ALWAYS and the slot only if the block is resident, include/strata/kernels/kv_stream.hpp).  That makes the pool a
// cache, and a cache can give slots back by dropping entries: no data moves.  The elastic form is
//     shrink(L):  every slot >= L loses its entry (page_table[block] = -1, slot_block = -1), live = L, then the pages of the slots >= L are
//                 unmapped (VmmRange);
//     grow(L):    the pages of the slots [live, L) are mapped, live = L (the new slots start empty);
//     resolve:    the clock sweep walks the first `live` slots only.
// The model is the sequential equivalent of the GPU sweep (a window of RT slots at a time, candidates taken in index order, reference bits of the
// slots passed cleared, the hand stopping just past the last slot taken: one slot at a time gives the same takes) and carries the host copy, so the
// property that matters can be tested: after any mix of shrinks, grows, appends and resolves, a block read through the pool equals the host copy.
// What the model cannot say: the speed, the graph-capture details (the kernels read `live` from a device word, so a captured graph follows it), and
// the mode-2 drafter ring (a STATIC table block % n_slots, no sweep: it cannot shrink without a re-fill and is out of this design).
#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

namespace strata::kernels::kvring {

struct Ring {
    int n_blocks = 0, n_slots = 0, live = 0;
    std::vector<int> page_table, slot_block, slot_stamp, slot_ref;
    std::vector<int64_t> host, slot_val;   // the host copy (all blocks) and the pool's pages (value per slot)
    int epoch = 0, hand = 0;
    bool overflow = false;                 // ctl[3]: a call needed more slots than are live
    long placed_beyond_live = 0;           // must stay 0

    void init(int blocks, int slots) {
        n_blocks = blocks; n_slots = slots; live = slots;
        page_table.assign(blocks, -1);
        slot_block.assign(slots, -1);
        slot_stamp.assign(slots, -1);
        slot_ref.assign(slots, 0);
        host.assign(blocks, 0);
        slot_val.assign(slots, 0);
    }
    /// the writer: host copy always, the slot only if the block is resident
    void append(int b, int64_t v) {
        host[b] = v;
        if (page_table[b] >= 0) slot_val[page_table[b]] = v;
    }
    /// make every block of `blocks` (distinct) resident: kv_stream resolve_kernel's three steps, over the live slots only
    void resolve(const std::vector<int>& blocks) {
        ++epoch;
        std::vector<int> miss;
        for (int b : blocks) {
            const int sl = page_table[b];
            if (sl >= 0) { slot_stamp[sl] = epoch; slot_ref[sl] = 1; } else miss.push_back(b);
        }
        std::vector<int> taken;
        int got = 0;
        const int need = (int) miss.size();
        for (int scanned = 0; got < need && scanned < 3 * live; ++scanned) {
            const int j = hand;
            hand = (hand + 1) % live;
            const bool mine = slot_stamp[j] == epoch;
            const bool cand = !mine && (slot_block[j] < 0 || slot_ref[j] == 0);
            if (cand) { taken.push_back(j); slot_stamp[j] = epoch; ++got; } else if (!mine) slot_ref[j] = 0;
        }
        if (got < need) overflow = true;
        for (int k = 0; k < got; ++k) {
            const int b = miss[k], sl = taken[k];
            if (sl >= live) ++placed_beyond_live;
            if (slot_block[sl] >= 0) page_table[slot_block[sl]] = -1;
            slot_block[sl] = b;
            slot_stamp[sl] = epoch;
            slot_ref[sl] = 1;
            page_table[b] = sl;
            slot_val[sl] = host[b];   // the copy kernel: host -> slot
        }
    }
    /// the value attention would read for block b after a resolve that included it
    bool read(int b, int64_t& v) const {
        const int sl = page_table[b];
        if (sl < 0) return false;
        v = slot_val[sl];
        return true;
    }
    void shrink(int new_live) {
        new_live = std::max(1, std::min(new_live, live));
        for (int sl = new_live; sl < live; ++sl) {
            if (slot_block[sl] >= 0) page_table[slot_block[sl]] = -1;
            slot_block[sl] = -1; slot_stamp[sl] = -1; slot_ref[sl] = 0; slot_val[sl] = -1;   // the page is gone
        }
        live = new_live;
        hand %= live;
    }
    void grow(int new_live) { live = std::max(live, std::min(new_live, n_slots)); }
};

}  // namespace strata::kernels::kvring
