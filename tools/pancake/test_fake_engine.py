"""probes/fake_engine.py: the model (profile, classify, speeds, log line), the Engine's request handling and the fake repo root layout.
No process is started but two tiny decoys; no GPU.  The end-to-end runs through abtest / queue_run.sh are in test_abtest_e2e.py.

Run: python -m pytest tools/pancake/test_fake_engine.py"""
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "probes"))
import abtest  # noqa: E402
import fake_engine as F  # noqa: E402

SHORT = "Write a detailed explanation of how a hash map works, with an example in Python."
EDIT = "Rewrite the following C++ file, renaming every variable to snake_case. Output only the full code.\n\n" + "int x;\n" * 900
QUIET = {"sd_start_pct": 0.0, "sd_request_pct": 0.0, "cold_pct": []}


def chat(eng, text, max_tokens=512, **body):
    return eng.chat({"messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, **body})


def speeds(log_path, n=None):
    """(prompt tok/s, decode tok/s) of the request lines of an engine log, as abtest.LINE reads them."""
    rows = [abtest.LINE.search(ln) for ln in Path(log_path).read_text().splitlines() if ln.startswith("strata serve: prompt")]
    return [(float(m[2]), float(m[4])) for m in rows][:n]


def test_profile_merges_defaults_the_fake_dict_and_the_env_keys():
    p = F.profile({"fake": {"seed": 5, "repeat_pct": {"edit": 10.0}}, "env": {"FAKE_GAIN_PCT": "3", "FAKE_GAIN_PCT_EDIT": "7", "FAKE_COLD_PCT": "30,5",
                                                                               "FAKE_DIE": "1", "STRATA_X": "1", "FAKE_SWITCH_SILENT": "STRATA_A,STRATA_B"}})
    assert p["seed"] == 5 and p["repeat_pct"] == {"short": 2.0, "edit": 10.0}      # one level deep: the other key stays
    assert p["gain_pct"] == 3.0 and p["gain_pct_edit"] == 7.0 and p["cold_pct"] == [30.0, 5.0] and p["die"] is True
    assert p["switch_silent"] == ["STRATA_A", "STRATA_B"] and "STRATA_X" not in p
    assert F.DEFAULTS["seed"] == 1 and F.DEFAULTS["repeat_pct"]["edit"] == 35.0    # DEFAULTS is not touched
    assert F.profile({}) == F.DEFAULTS


def test_classify_reads_the_case_through_abtest_s_first_lines():
    assert F.classify(SHORT) == "short" and F.classify(EDIT) == "edit"
    assert F.classify("[pass 3]\n" + EDIT) == "edit" and F.classify("[run abc-9]\n" + SHORT) == "short" and F.classify("hello") == "short"
    big = lambda tokens: F.SUMMARIZE + "x" * int(tokens * F.CPT)
    assert F.classify(big(11978)) == "long" and F.classify(big(1000)) == "long1k" and F.classify(big(3000)) == "long3k"
    assert F.classify(big(10000), 256) == "longdec10k" and F.classify(big(120000), 256) == "longdec128k"
    assert F.classify(big(32000), 32) == "long32k" and F.classify(big(32000), 256) == "longdec32k"
    assert F.classify(big(64000), 32) == "long64k" and F.classify(big(64000), 256) == "longdec64k"


def test_start_state_is_deterministic_and_slow_starts_are_forced():
    prof = F.profile({"fake": {"seed": 3, "outlier_prob": 0.5}})
    assert F.start_state(prof, 4) == F.start_state(prof, 4) and F.start_state(prof, 4) != F.start_state(prof, 5)
    assert 0.2 <= F.start_state(prof, 0)["factor"]
    forced = F.profile({"fake": {"sd_start_pct": 0.0, "slow_starts": [2], "slow_factor": 0.5}})
    assert F.start_state(forced, 2) == {"start": 2, "factor": 0.5, "outlier": True} and F.start_state(forced, 3)["factor"] == 1.0
    sure = F.profile({"fake": {"sd_start_pct": 0.0, "outlier_prob": 1.0, "outlier_range": [0.6, 0.7]}})
    assert all(0.6 <= F.start_state(sure, i)["factor"] <= 0.7 and F.start_state(sure, i)["outlier"] for i in range(20))


def test_speeds_gain_cold_and_repeat():
    prof = F.profile({"fake": QUIET, "env": {"FAKE_GAIN_PCT": "10", "FAKE_GAIN_PCT_LONG": "0"}})
    state = F.start_state(prof, 0)
    plain = F.profile({"fake": QUIET})
    p0, d0 = F.request_speeds(plain, state, "short", 5, 0)
    p1, d1 = F.request_speeds(prof, state, "short", 5, 0)
    assert d1 == pytest.approx(d0 * 1.10) and p1 == pytest.approx(p0)            # a decode case: the gain is on decode tok/s only
    lp0, ld0 = F.request_speeds(plain, state, "long3k", 5, 0)
    lp1, ld1 = F.request_speeds(prof, state, "long3k", 5, 0)
    assert lp1 == pytest.approx(lp0 * 1.10) and ld1 == pytest.approx(ld0)        # a prefill case: on prompt tok/s only
    assert F.request_speeds(prof, state, "long", 5, 0)[0] == pytest.approx(F.request_speeds(plain, state, "long", 5, 0)[0])   # per-case override
    cold = F.profile({"fake": {**QUIET, "cold_pct": [40.0, 10.0]}})
    assert F.request_speeds(cold, state, "edit", 0, 0)[1] == pytest.approx(0.6 * F.request_speeds(plain, state, "edit", 5, 0)[1])
    assert F.request_speeds(cold, state, "edit", 1, 0)[1] == pytest.approx(0.9 * F.request_speeds(plain, state, "edit", 5, 0)[1])
    assert F.request_speeds(plain, state, "edit", 5, 1)[1] == pytest.approx(1.35 * F.request_speeds(plain, state, "edit", 5, 0)[1])
    assert F.request_speeds(plain, state, "long", 5, 1) == F.request_speeds(plain, state, "long", 5, 0)      # no repeat gain for a cold read


def test_the_log_line_is_what_abtest_reads():
    line = F.log_line(2135, 650.2, 512, 141.0, 0.9)
    m = abtest.LINE.search(line)
    assert line.startswith("strata serve: prompt 2135 tokens")
    assert (int(m[1]), float(m[2]), int(m[3]), float(m[4])) == (2135, 650.2, 512, 141.0) and int(m[5]) / int(m[6]) == pytest.approx(0.9, abs=0.01)


def test_engine_repeat_is_by_exact_text_and_switch_lines_come_once(tmp_path):
    log = tmp_path / "e.log"
    prof = F.profile({"fake": {**QUIET, "switch_silent": ["STRATA_QUIET"], "switch_will_not": ["STRATA_NO"]}})
    eng = F.Engine(prof, str(log), 0, {"STRATA_RAM_KEEP_GIB": "12", "STRATA_ON": "1", "STRATA_QUIET": "1", "STRATA_NO": "1", "OTHER": "x"})
    first = chat(eng, EDIT)
    chat(eng, EDIT)                    # the same text again: the conversation cache
    chat(eng, "[pass 1]\n" + EDIT)     # a new first line: read cold again
    (_, d0), (_, d1), (_, d2) = speeds(log, 3)
    assert d1 == pytest.approx(1.35 * d0, rel=0.01) and d2 == pytest.approx(d0, rel=0.01)
    text = log.read_text()
    assert text.count("strata switch:") == 2 and "STRATA_ON acted:" in text and "STRATA_NO will not act:" in text
    assert "STRATA_QUIET" not in text and "RAM_KEEP" not in text and "OTHER" not in text
    assert first["choices"][0]["message"]["content"] == chat(eng, EDIT)["choices"][0]["message"]["content"]          # greedy: same text
    assert chat(eng, EDIT, temperature=0.8)["choices"][0]["message"]["content"] != first["choices"][0]["message"]["content"]   # sampled: never
    salted = F.Engine(F.profile({"fake": {"text_salt": "lossy"}}), None, 0)
    assert chat(salted, EDIT)["choices"][0]["message"]["content"] != first["choices"][0]["message"]["content"]
    assert first["usage"]["completion_tokens"] == 512 and chat(eng, EDIT, max_tokens=32)["usage"]["completion_tokens"] == 32


def test_real_run_active_looks_at_argv_not_at_text(tmp_path):
    if F.real_run_active():
        pytest.skip("a real run is active on this machine")
    mention = subprocess.Popen(["bash", "-c", "sleep 30; true  # tools/pancake/abtest.py serve/server.py --engine strata"])
    try:
        time.sleep(0.3)
        assert F.real_run_active() is None            # a shell that only mentions abtest does not count
        real = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", "x/tools/pancake/abtest.py", "spec.json"])
        time.sleep(0.3)
        try:
            assert F.real_run_active() and str(real.pid) in F.real_run_active()
        finally:
            real.kill()
    finally:
        mention.kill()


def test_fake_root_layout(tmp_path):
    root = F.make_fake_root(tmp_path / "r")
    clean_env = {"PATH": os.environ["PATH"]}
    assert (root / "tools/pancake/abtest.py").read_text() == (HERE / "abtest.py").read_text() and (root / "tools/pancake/queue_run.sh").exists()
    assert not list((root / "tools/pancake").glob("test_*"))
    assert (root / "serve/server.py").is_symlink() and (root / "serve/server.py").resolve() == Path(F.__file__).resolve()
    out = subprocess.run([str(root / ".venv/bin/python"), "-c", "import os, time; print(os.environ['FAKE_TIME_SCALE']); t = time.time(); time.sleep(2); "
                          "print(round(time.time() - t, 1))"], capture_output=True, text=True, timeout=30, env=clean_env).stdout.split()
    assert out == ["0.05", "0.1"]                     # the wrapper compresses every sleep by FAKE_TIME_SCALE
    capped = subprocess.run([str(root / "tools/pancake/capped.sh"), "echo", "ran"], capture_output=True, text=True, timeout=30, env=clean_env)
    assert capped.stdout.strip() == "ran" and "capped: tier=mmap keep_gib=12 big_free_gib=0" in capped.stderr
    assert (root / "src/spec/draft_policy.cpp").stat().st_size >= 6000
    assert sum((root / f"src/spec/{n}.cpp").stat().st_size for n in "abcd") == 34264                    # the real `long` prompt's size
    assert sum(f.stat().st_size for f in (root / "src/big").glob("*.cpp")) >= 120000 * abtest.LONG_CPT
