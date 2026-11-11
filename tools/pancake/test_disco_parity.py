"""A22: the features DiscoPolicy (C++, src/spec/disco.cpp) logs equal the ones disco_fit.py replays from a window trace.

The fitter trains on the replayed features and the engine scores with its own, so any drift between the two makes the
fitted weights wrong without an error.  This builds tools/pancake/disco_parity_driver.cpp with g++ (skipped when there is
no g++), feeds random MTP rounds to both and compares every draft's nine features.

    python -m pytest -q tools/pancake/test_disco_parity.py
"""
import csv
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import disco_fit  # noqa: E402

SOURCES = ["spec/disco.cpp", "spec/draft_policy.cpp", "spec/controller.cpp", "spec/suffix_drafter.cpp", "spec/draft_source.cpp"]


def random_rounds(n, seed=3):
    rng = random.Random(seed)
    rounds, pos = [], 100
    for _ in range(n):
        T = rng.randint(2, 7)
        probs = [round(rng.uniform(0.05, 1.0), 4) for _ in range(T - 1)]
        accepted = rng.randint(0, T - 1)
        rounds.append({"T": T, "a": accepted, "pos": pos, "match": rng.choice([0, 0, 3, 17]), "probs": probs,
                       "miss": [rng.randint(0, 40) for _ in range(T)]})
        pos += accepted + 1
    return rounds


def run_engine(rounds, tmp):
    gxx = shutil.which("g++")
    if gxx is None:
        pytest.skip("no g++")
    exe = tmp / "disco_parity_driver"
    cmd = [gxx, "-std=c++20", "-O1", f"-I{ROOT / 'include'}", str(HERE / "disco_parity_driver.cpp")] + \
          [str(ROOT / "src" / s) for s in SOURCES] + ["-o", str(exe)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    text = "".join(f"{r['T']} {r['a']} {r['pos']} {r['match']} | {' '.join(str(p) for p in r['probs'])} | "
                   f"{' '.join(str(m) for m in r['miss'])}\n" for r in rounds)
    out = tmp / "trace.csv"
    subprocess.run([str(exe), str(out)], input=text, check=True, text=True)
    with open(out) as f:
        return list(csv.DictReader(f))


def test_engine_features_equal_the_fitters_replay(tmp_path):
    rounds = random_rounds(400)
    lines = run_engine(rounds, tmp_path)
    st = disco_fit.DiscoState()
    k = 0
    for i, r in enumerate(rounds):
        chain = 1.0
        for j in range(r["T"] - 1):
            chain *= r["probs"][j]
            want = st.features(j, r["probs"][j], chain)
            row = lines[k]
            k += 1
            assert int(row["round"]) == i and int(row["idx"]) == j
            assert int(row["label"]) == (1 if j < r["a"] else 0 if j == r["a"] else -1)
            assert int(row["nmiss"]) == r["miss"][j + 1]
            for f in range(disco_fit.NF):
                # the engine's trace prints %.5g; feature 7 (DraftPolicy's row cost ratio) is a constant prior in both
                assert float(row[f"f{f}"]) == pytest.approx(want[f], rel=2e-4, abs=1e-6), (i, j, f)
        st.end_round(r["T"], r["a"], True, r["probs"], r["miss"], r["pos"], r["match"])
    assert k == len(lines)
