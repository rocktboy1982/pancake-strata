"""abtest.py --drift-check (screen tier): the base arm again at the end, every arm judged against both base runs, and a warning when
the two base runs differ (TEST_LEDGER.md T-606, T-622: a base that ran in a slow window made every arm look like a gain).
No server, no GPU: screen_verdict / base_drift run on hand-made results.

Run: python3 -I -m pytest -q tools/pancake/test_screen_drift.py"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(scope="module")
def A():
    if not hasattr(os, "sysconf"):   # abtest reads the clock tick at import (Linux only)
        os.sysconf = lambda k: 100
    import abtest
    return abtest


def res(A, **per_case):
    return [{"rows": [{"case": c, A.metric(c): v} for c, v in per_case.items()]}]


def test_base_drift_is_late_against_early(A):
    results = {"base": res(A, short=100.0, edit=150.0), "base-late": res(A, short=116.0, edit=150.0)}
    d = A.base_drift(results, ["base", "x", "base-late"], ["short", "edit"])
    assert round(d["short"], 1) == 16.0 and round(d["edit"], 1) == 0.0
    assert A.base_drift(results, ["base", "x"], ["short"]) == {}            # no late base run: nothing


def test_verdict_uses_both_base_runs_and_flags_the_order_effect(A, capsys):
    # the early base ran slow (100 / 125), the late one at the usual level (116 / 160): arm x is at the usual level
    results = {"base": res(A, short=100.0, edit=125.0), "base-late": res(A, short=116.0, edit=160.0),
               "x": res(A, short=116.0, edit=160.0)}
    A.screen_verdict(results, ["base", "x", "base-late"], ["short", "edit"], drift=True)
    out = capsys.readouterr().out
    assert "ORDER EFFECT" in out
    line = [l for l in out.splitlines() if l.strip().startswith("x ")][0]
    # against the mean of both bases (108 / 142.5) x is +7.4% / +12.3%, not the +16% / +28% the early base alone gives
    assert "short +7.4%" in line and "edit +12.3%" in line
    assert not any(l.strip().startswith("base-late ") and "PROMOTE" in l for l in out.splitlines())


def test_default_verdict_is_unchanged_without_drift(A, capsys):
    results = {"base": res(A, short=100.0), "x": res(A, short=110.0)}
    A.screen_verdict(results, ["base", "x"], ["short"])
    out = capsys.readouterr().out
    assert "short +10.0%" in out and "base drift" not in out and "PROMOTE to confirm" in out


def test_no_warning_when_the_base_runs_agree(A, capsys):
    results = {"base": res(A, short=100.0), "base-late": res(A, short=101.0), "x": res(A, short=110.0)}
    A.screen_verdict(results, ["base", "x", "base-late"], ["short"], drift=True)
    out = capsys.readouterr().out
    assert "base drift (late base run vs early): short +1.0%" in out and "ORDER EFFECT" not in out
