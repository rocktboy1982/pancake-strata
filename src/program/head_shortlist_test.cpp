// MOD_PLAN.md section 6 item 38: STRATA_HEAD_SHORTLIST's table format and selection rule (include/strata/program/head_shortlist.hpp).
#include "strata/program/head_shortlist.hpp"

#include <cstdio>
#include <cstdlib>

using namespace strata::program;

static int checks = 0;
static void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

static std::vector<uint8_t> image(uint32_t v, uint32_t d, uint32_t k, const std::vector<int32_t>& cl, const std::vector<float>& ce) {
    std::vector<uint8_t> b(16 + 4 * cl.size() + 4 * ce.size());
    std::memcpy(b.data(), "SHL1", 4);
    std::memcpy(b.data() + 4, &v, 4); std::memcpy(b.data() + 8, &d, 4); std::memcpy(b.data() + 12, &k, 4);
    std::memcpy(b.data() + 16, cl.data(), 4 * cl.size());
    std::memcpy(b.data() + 16 + 4 * cl.size(), ce.data(), 4 * ce.size());
    return b;
}

int main() {
    // table: 6 rows, 2 dims, 3 clusters (the layout head_cluster.py save_table writes)
    const std::vector<int32_t> cl = {0, 0, 1, 1, 2, 2};
    const std::vector<float> ce = {1, 0, 0, 1, -1, -1};
    ShortlistTable t;
    std::string err;
    auto img = image(6, 2, 3, cl, ce);
    check(shortlist_parse(img.data(), img.size(), t, err), "a valid table parses");
    check(t.vocab == 6 && t.n_embd == 2 && t.k == 3 && t.cluster == cl && t.centre == ce, "fields round trip");
    auto bad = img; bad[0] = 'X';
    check(!shortlist_parse(bad.data(), bad.size(), t, err), "wrong magic refused");
    check(!shortlist_parse(img.data(), img.size() - 4, t, err), "short file refused");
    auto longer = img; longer.push_back(0);
    check(!shortlist_parse(longer.data(), longer.size(), t, err), "long file refused");
    auto oob = image(6, 2, 3, {0, 0, 1, 1, 2, 3}, ce);
    check(!shortlist_parse(oob.data(), oob.size(), t, err), "cluster id >= k refused");
    auto neg = image(6, 2, 3, {0, 0, 1, 1, 2, -1}, ce);
    check(!shortlist_parse(neg.data(), neg.size(), t, err), "negative cluster id refused");
    check(!shortlist_parse(img.data(), 8, t, err), "a truncated header is refused");
    check(shortlist_load("/no/such/file.shl", t, err) == false, "a missing file is refused");

    // how many clusters
    check(shortlist_clusters_scored(1024, 0) == 134, "13% of 1024 -> ceil(133.12) = 134");
    check(shortlist_clusters_scored(1024, 256) == 256, "an explicit count wins");
    check(shortlist_clusters_scored(1024, 5000) == 1024, "an explicit count clamps to k");
    check(shortlist_clusters_scored(3, 0, 0.0) == 1, "at least one cluster");
    check(shortlist_clusters_scored(100, 0, 0.25) == 25, "an exact product is not rounded up");
    check(shortlist_clusters_scored(0, 4) == 0, "no clusters");

    // selection: larger first, lower index on a tie, NaN last, n clamps
    const float sc[5] = {1.0f, 3.0f, 3.0f, -2.0f, 0.5f};
    auto k2 = shortlist_select(sc, 5, 2);
    check(!k2[0] && k2[1] && k2[2] && !k2[3] && !k2[4], "top 2 of a tie is the two 3.0s");
    auto k1 = shortlist_select(sc, 5, 1);
    check(k1[1] && !k1[2] && !k1[0], "a tie keeps the lower index");
    auto k3 = shortlist_select(sc, 5, 3);
    check(k3[0] && k3[1] && k3[2] && !k3[3] && !k3[4], "top 3");
    auto kall = shortlist_select(sc, 5, 99);
    check(kall[0] && kall[1] && kall[2] && kall[3] && kall[4], "n above k keeps all");
    auto kz = shortlist_select(sc, 5, 0);
    int kept = 0;
    for (auto x : kz) kept += x;
    check(kept == 1, "n below 1 keeps one");
    const float nn[3] = {std::nanf(""), -5.0f, std::nanf("")};
    auto kn = shortlist_select(nn, 3, 1);
    check(!kn[0] && kn[1] && !kn[2], "a NaN score loses to any number");
    auto kn2 = shortlist_select(nn, 3, 2);
    check(kn2[0] && kn2[1] && !kn2[2], "NaN scores tie among themselves by index");

    // the pick: argmax among kept rows only (the lossy case: the true top is outside)
    const float lg[6] = {0.1f, 0.2f, 0.9f, 0.3f, 5.0f, 0.0f};
    std::vector<uint8_t> keep01 = {1, 1, 0};
    check(shortlist_pick(lg, cl.data(), 6, keep01) == 2, "clusters 0 and 1 kept: row 2 (0.9) wins");
    std::vector<uint8_t> keep0 = {1, 0, 0};
    check(shortlist_pick(lg, cl.data(), 6, keep0) == 1, "only cluster 0: row 1");
    std::vector<uint8_t> keepall = {1, 1, 1};
    check(shortlist_pick(lg, cl.data(), 6, keepall) == 4, "all clusters kept = the full argmax");
    check(shortlist_pick(lg, cl.data(), 6, keep0) != 4, "the true top outside the kept clusters is lost (lossy)");
    const float tie[6] = {2, 2, 2, 2, 2, 2};
    check(shortlist_pick(tie, cl.data(), 6, keepall) == 0, "a tie goes to the lowest row");
    const float ninf[6] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY};
    check(shortlist_pick(ninf, cl.data(), 6, keepall) == 0, "nothing above -inf: 0");
    const float nanl[6] = {std::nanf(""), 1.0f, std::nanf(""), 0, 0, 0};
    check(shortlist_pick(nanl, cl.data(), 6, keepall) == 1, "NaN logits are never picked");
    check(shortlist_rows_scored(cl.data(), 6, keep01) == 4, "rows scored = rows of the kept clusters");

    // the centre score order: the same value as a plain sum on small exact inputs, and deterministic
    float a[300], c[300];
    for (int i = 0; i < 300; ++i) { a[i] = (float) (i % 7) - 3.0f; c[i] = (float) (i % 5) * 0.5f; }
    double ref = 0;
    for (int i = 0; i < 300; ++i) ref += (double) a[i] * c[i];
    check(std::fabs((double) shortlist_dot(a, c, 300) - ref) < 1e-3, "dot of 300 equals the plain sum");
    check(shortlist_dot(a, c, 300) == shortlist_dot(a, c, 300), "dot is deterministic");
    check(shortlist_dot(a, c, 0) == 0.0f, "empty dot is 0");

    // end to end on the toy: centre scores -> keep -> pick, head_cluster.py's rule
    const float h[2] = {1.0f, 0.2f};
    float cs[3];
    for (int i = 0; i < 3; ++i) cs[i] = shortlist_dot(h, ce.data() + 2 * i, 2);
    auto kk = shortlist_select(cs, 3, 1);
    check(kk[0] && !kk[1] && !kk[2], "centre 0 scores best for h = (1, .2)");

    std::printf("head_shortlist_test: %d checks passed\n", checks);
    return 0;
}
