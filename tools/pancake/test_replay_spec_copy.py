"""Tests for replay_spec_copy.py on small synthetic routing (a 4-layer, 16-expert model patched into replay_cache_policy).
python -m pytest -q tools/pancake/test_replay_spec_copy.py"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_spec_copy as S  # noqa: E402

R = S.R
NL, NE, K = 4, 16, 2


@pytest.fixture(autouse=True)
def small_model(monkeypatch):
    monkeypatch.setattr(R, "NL", NL)
    monkeypatch.setattr(R, "NE", NE)


def drifting(n):
    """Each window routes two experts per layer, the hot pair moving one step every 5 windows: the tier keeps swapping."""
    out = []
    for wi in range(n):
        base = (wi // 5) % NE
        out.append([(l, np.array([[(base + l) % NE, (base + l + 1) % NE]] * 3, np.int64)) for l in range(NL)])
    return out


def init_slots(n):
    return np.array([(i // 4, i % 4) for i in range(n)], np.int64)


def run(names, **kw):
    pol = R.Policy(every=1, swaps=32, margin=0.5)
    return S.run(drifting(120), init_slots(8), pol, names, nl=NL, ne=NE, **kw)


def test_oracle_predicts_the_incoming_experts_with_enough_staging():
    out, bsw, stg = run(["or"], slots=8, max_copies=1000)
    assert bsw.sum() > 20
    hit = out["or"]["hit"].sum() / out["or"]["inc"].sum()
    assert hit >= 0.9      # the first round has nothing staged yet


def test_staging_never_exceeds_its_slots_and_copies_per_round_are_capped():
    out, bsw, stg = run(["la:0.6", "bl"], slots=2, max_copies=3)
    for nm in ("la:0.6", "bl"):
        assert out[nm]["made"].max() <= 3
        assert all(len(d) <= 2 for d in stg[nm].ent)
        slots = [sorted(v[1] for v in d.values()) for d in stg[nm].ent]
        assert all(len(s) == len(set(s)) for s in slots)     # no staging slot holds two experts


def test_hits_never_exceed_incoming_and_waste_is_accounted():
    out, bsw, stg = run(["ps", "blt", "bl"], slots=4, max_copies=8)
    for nm, o in out.items():
        assert (o["hit"] <= o["inc"]).all()
        assert stg[nm].consumed == o["hit"].sum()
        assert stg[nm].copies >= stg[nm].consumed


def test_staging_keeps_alive_a_repredicted_expert_without_a_second_copy():
    st = S.Staging(NL, 2, 3)
    assert st.stage(0, 5, 1) == 0
    assert st.stage(0, 5, 2) is None and st.copies == 1
    assert st.take(0, 5) == 0 and st.take(0, 5) is None
    st.stage(1, 1, 1)
    st.tick(10)
    assert st.wasted == 1 and st.leftover() == 0      # aged out


def test_schedule_file_lists_rounds_swaps_and_valid_staging_slots(tmp_path):
    pol = R.Policy(every=1, swaps=32, margin=0.5)
    sched = []
    S.run(drifting(60), init_slots(8), pol, ["ps"], nl=NL, ne=NE, slots=3, max_copies=4, sched=sched)
    path = tmp_path / "s.txt"
    S.write_schedule(path, sched, 3, NL, NE, init_slots(8))
    lines = path.read_text().splitlines()
    assert lines[0] == f"SCHED {NL} {NE} 3"
    assert sum(1 for x in lines if x.startswith("R ")) >= 10
    for x in lines:
        f = x.split()
        if f[0] == "I":
            assert -1 <= int(f[4]) < 3
        if f[0] == "S":
            assert 0 <= int(f[3]) < 3


def test_expert_bytes_are_the_derived_iq2_xs_sizes():
    assert S.EXPERT_BYTES["iq2_s"] == 1510400 and S.EXPERT_BYTES["iq2_xxs"] == 1305600 and S.EXPERT_BYTES["iq1_m"] == 1177600
    assert abs(S.mean_bytes() - 1442666.67) < 1.0
