"""reap_saliency.py on synthetic --dump-routing traces.  No model, no GPU.

Run: python -m pytest tools/pancake/test_reap_saliency.py   (or python tools/pancake/test_reap_saliency.py)"""
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reap_saliency as R  # noqa: E402

L, E, K = 48, 512, 10


def write_trace(path, windows=8, tokens=3, weights=True, seed=0):
    """--dump-routing records (layer, k, ids, weights), one per token and layer; expert 0..19 hot, the rest Zipf-cold;
    expert 500 of every layer is routed once per window with a large gate, expert 501 often with a small one."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(windows):
        for l in range(L):
            for t in range(tokens):
                ids = list(rng.choice(20, 6, replace=False)) + list(20 + rng.zipf(1.5, 4) % 470)
                w = [0.05] * K
                if t == 0:
                    ids[-1], w[-1] = 500, 0.6
                ids[-2], w[-2] = 501, 0.01
                ids = np.array(ids, np.int32)
                _, first = np.unique(ids, return_index=True)   # distinct ids, as a router's top-k
                for j in set(range(K)) - set(first):
                    ids[j] = 502 + j
                rec = np.array([l, K], np.int32).tobytes() + ids.tobytes()
                rec += np.array(w if weights else [0.0] * K, np.float32).tobytes()
                out.append(rec)
    Path(path).write_bytes(b"".join(out))


def trace(weights=True, **kw):
    d = tempfile.mkdtemp()
    p = Path(d) / "routing.bin"
    write_trace(p, weights=weights, **kw)
    return R.load(str(p))


def test_frequency_without_weights_reap_with_them():
    count, gate = R.stats(trace(weights=False))
    assert gate is None
    s, what = R.saliency(count, gate)
    assert "frequency" in what and np.array_equal(s, count)
    count, gate = R.stats(trace(weights=True))
    s, what = R.saliency(count, gate)
    assert what.startswith("REAP")
    # 500: rare, large gate -> high REAP score, low frequency; 501: frequent, tiny gate -> the reverse
    assert s[3, 500] > s[3, 501] and count[3, 500] < count[3, 501]
    assert abs(s[3, 500] - 0.6) < 1e-6 and abs(s[3, 501] - 0.01) < 1e-6


def test_act_norm_scales_the_score():
    count, gate = R.stats(trace())
    norm = np.ones((L, E))
    norm[:, 501] = 100.0
    s, what = R.saliency(count, gate, norm)
    assert "output norm" in what and s[3, 501] > s[3, 500]


def test_prune_mask_is_the_lowest_per_layer():
    count, gate = R.stats(trace())
    s, _ = R.saliency(count, gate)
    m = R.prune_mask(s, count, 0.25)
    assert (m.sum(axis=1) == 128).all()
    for l in (0, 47):
        assert s[l][m[l]].max() <= s[l][~m[l]].min()
    assert not m[:, 500].any()            # a high-gate expert survives REAP even though it is rare


def test_replay_bounds():
    tr = trace(windows=12)
    count, gate = R.stats(tr)
    s, _ = R.saliency(count, None)        # frequency
    m = R.prune_mask(s, count, 0.5)
    w = R.wtrace.to_replay_windows(tr)
    base, _ = R.replay(w, R.initial_residents(None, 2000, None, count))
    drop, _ = R.replay(w, R.initial_residents(None, 2000, m, count), m, "drop")
    miss, _ = R.replay(w, R.initial_residents(None, 2000, m, count), m, "miss")
    assert drop.sum() <= miss.sum() and drop.sum() <= base.sum()
    none, _ = R.replay(w, R.initial_residents(None, 2000, np.zeros((L, E), bool), count), np.zeros((L, E), bool), "miss")
    assert np.array_equal(none, base)     # nothing pruned: both bounds are the plain replay


def test_main_end_to_end_with_holdout():
    d = tempfile.mkdtemp()
    p = Path(d) / "routing.bin"
    write_trace(p, windows=10)
    out = R.main([str(p), "--slots", "3000", "--holdout", "--json", str(Path(d) / "o.json")])
    assert out["holdout"] and out["windows"] == [5, 5]
    assert set(out["ratios"]) == {"0.25", "0.5"}
    for row in out["ratios"].values():
        assert 0.0 <= row["entries_on_pruned"] <= 1.0 and "gate_mass_on_pruned" in row
        assert row["drop"]["misses_per_window"] <= row["miss"]["misses_per_window"]
    assert (Path(d) / "o.json").exists()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
