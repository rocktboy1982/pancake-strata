"""Tests for kfd_evict.py: a fake /sys/class/kfd tree, no GPU, no Linux needed (LEADS.md Part O O3)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kfd_evict as K  # noqa: E402


def tree(tmp_path, pid, **nodes):
    for gpu, ms in nodes.items():
        d = tmp_path / str(pid) / f"stats_{gpu}"
        d.mkdir(parents=True)
        (d / "evicted_ms").write_text(f"{ms}\n")
    return str(tmp_path)


def test_read_counters(tmp_path):
    root = tree(tmp_path, 42, **{"11": 5, "22": 7})
    (tmp_path / "42" / "queues").mkdir()          # not a stats directory: ignored
    assert K.read_counters(42, root) == {"stats_11": 5, "stats_22": 7}


def test_missing_process_or_tree_is_empty(tmp_path):
    assert K.read_counters(1, str(tmp_path)) == {}
    assert K.read_counters(1, str(tmp_path / "nope")) == {}


def test_rise_takes_the_largest_node_not_the_sum():
    assert K.rise({"a": 10, "b": 10}, {"a": 40, "b": 45}) == 35     # one eviction stops every node: 35, not 65
    assert K.rise({"a": 10}, {"a": 10}) == 0
    assert K.rise({}, {"a": 8}) == 8                                # a node new in `after` counts from 0
    assert K.rise({"a": 3}, {}) is None and K.rise({}, {}) is None  # not readable: None, never a fake 0


def test_a_counter_that_went_down_is_not_negative():
    assert K.rise({"a": 50}, {"a": 4}) == 0


def test_window_reads_only_when_enabled(tmp_path):
    root = tree(tmp_path, 7, **{"1": 100})
    with K.Window(7, enabled=False, root=root) as w:
        (tmp_path / "7" / "stats_1" / "evicted_ms").write_text("160\n")
    assert w.ms is None                                             # off: nothing read, nothing recorded
    with K.Window(7, enabled=True, root=root) as w:
        (tmp_path / "7" / "stats_1" / "evicted_ms").write_text("290\n")
    assert w.ms == 130
    with K.Window(8, enabled=True, root=root) as w:                 # no such process
        pass
    assert w.ms is None


def test_install_adds_kfd_evict_ms_to_each_run(tmp_path):
    import types
    root = tree(tmp_path, 99, **{"1": 10})
    srv = types.SimpleNamespace(pid=99)
    state = {"calls": 0}

    def start_server(*a, **k):
        return srv, None

    def stop_servers(*a, **k):
        state["stopped"] = True

    def _run_arm(*a, **k):
        ab.start_server()
        (tmp_path / "99" / "stats_1" / "evicted_ms").write_text("75")     # an eviction of 65 ms during the run
        ab.stop_servers()
        return {"rows": []}, None

    ab = types.SimpleNamespace(start_server=start_server, stop_servers=stop_servers, _run_arm=_run_arm)
    K.install(ab, root)
    res, err = ab._run_arm()
    assert err is None and res["kfd_evict_ms"] == 65 and state["stopped"]
    ab.start_server = lambda *a, **k: (None, "no engine")                     # a failed start: no pid, the result has None
    ab._run_arm = lambda *a, **k: ({"rows": []}, None)
    K.install(ab, root)
    assert ab._run_arm()[0]["kfd_evict_ms"] is None
