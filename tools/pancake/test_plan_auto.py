"""Tests for tools/pancake/plan_auto.py on SYNTHETIC hardware and models (no GPU, no engine, nothing measured here)."""
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import plan_auto as pa  # noqa: E402

BASE = json.loads(pa.BASE_PRESET.read_text(encoding="utf-8"))


def knob(plan, name):
    return next(k for k in plan.knobs if k.name == name)


def box_plan():
    return pa.plan_auto(copy.deepcopy(pa.BOX), copy.deepcopy(pa.FLASHNEXT_IQ2XS))


def test_box_reproduces_daily_preset_except_pool_affinity():
    plan = box_plan()
    rows = pa.diff_vs_preset(BASE, plan)
    differ = [r["knob"] for r in rows if not r["same"]]
    assert differ == ["--pool-affinity"], differ


def test_slots_follow_vram_and_have_a_floor():
    hw = copy.deepcopy(pa.BOX)
    m = copy.deepcopy(pa.FLASHNEXT_IQ2XS)
    assert box_plan().ledger["vram"]["expert_tier_slots"] == 10909     # consistency with the anchor, not a prediction
    hw["vram_total_mib"] = 16368                                        # a 16 GB card: fewer slots, same rule
    plan = pa.plan_auto(hw, m)
    s16 = plan.ledger["vram"]["expert_tier_slots"]
    assert 7000 < s16 < 9000
    assert plan.ledger["prompt_loan"]["chunk_tokens"] < 32768           # the 90% loan rule shrinks the chunk
    hw["vram_total_mib"] = 5000                                         # under the token cycle: no tier
    plan = pa.plan_auto(hw, m)
    assert knob(plan, "--expert-cache").value == "0"


def test_pcie_frac_zero_on_the_box_and_taken_on_a_cpu_bound_fast_link():
    assert knob(box_plan(), "--pcie-frac").value == "0"
    hw = copy.deepcopy(pa.BOX)
    hw["copy_us"] = 28.0                                                # PCIe 5 class link
    hw["cpu_expert_us"] = 200.0                                         # a slow CPU
    hw["windows"] = [{"case": "edit", "gpu_ms": 10.0, "cpu_ms": 30.0, "weight": 1.0}]
    v = float(knob(pa.plan_auto(hw, copy.deepcopy(pa.FLASHNEXT_IQ2XS)), "--pcie-frac").value)
    assert 0.2 <= v <= 0.9


def test_margin_declines_a_small_gain():
    hw = copy.deepcopy(pa.BOX)
    hw["windows"] = [{"case": "edit", "gpu_ms": 16.0, "cpu_ms": 17.0, "weight": 1.0}]   # a 6% best case
    assert knob(pa.plan_auto(hw, copy.deepcopy(pa.FLASHNEXT_IQ2XS)), "--pcie-frac").value == "0"


def test_swaps_scale_with_break_even_and_clamp():
    base = int(knob(box_plan(), "--adapt-swaps").value)
    assert base == 32
    hw = copy.deepcopy(pa.BOX)
    hw["copy_us"] = 30.0                                                # promotions repay sooner
    assert int(knob(pa.plan_auto(hw, copy.deepcopy(pa.FLASHNEXT_IQ2XS)), "--adapt-swaps").value) > base
    hw["copy_us"] = 5000.0
    assert int(knob(pa.plan_auto(hw, copy.deepcopy(pa.FLASHNEXT_IQ2XS)), "--adapt-swaps").value) == 16


def test_hybrid_cpu_arms_affinity_measure_and_workers_follow_cores():
    hw = copy.deepcopy(pa.BOX)
    hw["hybrid_cpu"] = True
    assert knob(pa.plan_auto(hw, copy.deepcopy(pa.FLASHNEXT_IQ2XS)), "--pool-affinity").value == "measure"


def test_ram_ledger_limits_context():
    hw = copy.deepcopy(pa.BOX)
    hw["service_memory_max_gib"] = 40                                   # arena 32.9 GiB no longer leaves room for 131072
    plan = pa.plan_auto(hw, copy.deepcopy(pa.FLASHNEXT_IQ2XS))
    assert int(knob(plan, "--max-context").value) < 131072
    assert box_plan().ledger["ram"]["fits"]


def test_every_knob_is_a_real_engine_flag_or_env():
    src = (pa.ROOT / "src" / "program" / "generate.cpp").read_bytes().decode("utf-8", "replace")
    env_def = (pa.ROOT / "src" / "core" / "conversation_disk_env.def").read_text(encoding="utf-8")
    for k in box_plan().knobs:
        if k.kind == "arg":
            assert f'"{k.name}"' in src, k.name
        else:
            assert k.name in src or k.name in env_def, k.name


def test_lossy_knobs_marked_and_removable():
    plan = box_plan()
    lossy = {k.name for k in plan.knobs if not k.exact}
    assert {"--kv", "STRATA_PREFILL_AUTO_MAX", "STRATA_PREFILL_CPU_SHARE"} <= lossy
    exact_only = pa.plan_auto(copy.deepcopy(pa.BOX), copy.deepcopy(pa.FLASHNEXT_IQ2XS), allow_lossy=False)
    assert all(k.exact for k in exact_only.knobs)


def test_proposal_preset_roundtrips_and_trial_spec_shape():
    plan = box_plan()
    prop = pa.apply_plan(BASE, plan)
    names = [(k.name, k.kind, k.default) for k in plan.knobs]
    got = pa.effective(prop, names)
    for k in plan.knobs:
        assert str(got[k.name]) == str(k.value if k.value is not None else k.default), k.name
    assert "--pool-affinity" not in prop["args"]
    spec = pa.trial_spec(plan)
    assert spec["arms"]["base"] == {}
    assert spec["drift_check"] is True
    assert "pool-workers-6" in spec["arms"] and "pool-workers-8" in spec["arms"]
    assert spec["arms"]["adapt-swaps-16"]["args"] == {"--adapt-swaps": "16"}
    assert spec["arms"]["prefill-cpu-share-0.8"]["env"] == {"STRATA_PREFILL_CPU_SHARE": "0.8"}
    assert not any("spec-min-p" in a for a in spec["arms"])           # live key: calibrate.py, no restart


def test_cli_writes_files(tmp_path):
    out, ts = tmp_path / "p.json", tmp_path / "t.json"
    assert pa.main(["--out", str(out), "--trial-spec", str(ts)]) == 0
    assert json.loads(out.read_text())["args"]
    assert "arms" in json.loads(ts.read_text())
