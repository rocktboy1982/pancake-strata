// src/poly/stage_ahead_parity_test.cpp - Does Options::stage_ahead change what a prompt computes?  One engine reads
// fixed prompts of 500, 1000 and 1800 tokens with stage_ahead 1024 (a call of fewer than 1024 tokens copies only the
// experts it routes to) and with stage_ahead 128 (the default: the experts outside VRAM are copied before the layer
// routes), in the order 1024, 128, 1024, 128; after each read one more token is evaluated and its vocabulary's
// logits are compared with memcmp.  The setting only changes when and in what order the experts reach the card,
// not which expert computes which row, so all four must match bit for bit.
// One engine for all of them, because two loads of the same model in one process do not always agree with each other
// (TEST_LEDGER.md T-851: with stage_ahead 1024 on both sides, 2 of 5 loads differed from the rest by up to 0.5 in a
// logit; the expert cache the load fills is the suspect), so a comparison across loads cannot tell the setting from
// the load.  Expert swaps and the CPU's share are off: the CPU's share exists only for the arm that does not copy
// ahead, and sums in another order.
//
//   poly_stage_ahead_parity_test [MODEL.gguf]     (default: env STRATA_POLY_TEST_MODEL; none: SKIP)
//
// Needs the model file and a card.  Exit 0: bit for bit the same; 1: logits differ; 2: an error; 3: SKIP.
#include "engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using strata::poly::Engine;
using strata::poly::Options;

namespace {

const int kSizes[] = {500, 1000, 1800};
const int kArms[] = {1024, 128, 1024, 128};

// the number of logits that differ, and the largest difference
int compare(const std::vector<float>& a, const std::vector<float>& b, double* worst) {
    int bad = 0;
    *worst = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(float)) == 0) continue;
        ++bad;
        *worst = std::max(*worst, (double) std::fabs(a[i] - b[i]));
    }
    return bad;
}

}  // namespace

int main(int argc, char** argv) {
    const char* env = std::getenv("STRATA_POLY_TEST_MODEL");
    const std::string model = argc > 1 ? argv[1] : (env ? env : "");
    if (model.empty() || !std::ifstream(model).good()) {
        std::fprintf(stderr, "SKIP: no model file (argument or STRATA_POLY_TEST_MODEL)\n");
        return 3;
    }
    Options o;
    o.model = model;
    o.max_context = 4096;
    o.expert_swaps = 0;
    o.cpu_share = 0;
    try {
        auto e = Engine::open(o);
        int failed = 0;
        for (int n : kSizes) {
            std::vector<int32_t> text(n + 1);
            for (int i = 0; i <= n; ++i) text[(size_t) i] = 1000 + (53 * i + i / 9) % 100000;
            std::vector<std::vector<float>> got;
            for (int arm : kArms) {
                e->set_stage_ahead_for_test(arm);
                e->reset();
                if (!e->read(text.data(), n)) throw std::runtime_error("the read was given up");
                e->eval(text.data() + n, 1, true);
                got.emplace_back(e->logits(), e->logits() + e->n_vocab());
            }
            std::string line = std::to_string(n) + " tokens, " + std::to_string(got[0].size()) + " logits, differing against the first read (1024, 128, 1024, 128):";
            for (size_t k = 1; k < got.size(); ++k) {
                double worst = 0;
                const int bad = compare(got[0], got[k], &worst);
                char buf[96];
                std::snprintf(buf, sizeof buf, " %d (max %.3g)", bad, worst);
                line += buf;
                failed += bad != 0;
            }
            std::fprintf(stderr, "%s\n", line.c_str());
        }
        std::fprintf(stderr, "%s\n", failed ? "FAIL: stage_ahead changes the logits (or a read does not repeat)"
                                            : "PASS: stage_ahead 128 and 1024 give the same logits, bit for bit");
        return failed ? 1 : 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "ERROR: %s\n", ex.what());
        return 2;
    }
}
