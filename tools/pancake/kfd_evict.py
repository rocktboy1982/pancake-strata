#!/usr/bin/env python3
"""kfd_evict.py - how long the kernel evicted a process's GPU queues (Linux, ROCm), to explain a slow run (LEADS.md Part O O3).

Idea from caminus' pass police (read as a description only; AGPL, no code taken): the amdkfd driver keeps, per process and per GPU node, the total time
it evicted that process's queues (`/sys/class/kfd/kfd/proc/<pid>/stats_<gpu_id>/evicted_ms`), for example while the host migrates or compacts pages.  A
decode window that was held for an eviction is slow for a reason the engine cannot fix, and it looks exactly like noise in an A/B.  Today abtest.py's
noise proxies are foreign CPU and kswapd/kcompactd time; this adds the direct counter.

The counter only grows, in milliseconds, and the driver adds an eviction's whole length when the queues come back: a rise between two samples is an
eviction that ENDED between them (it may have begun before the first sample).  One eviction stops every node at once, so the per-node rises are not
added: `rise()` takes the largest one.

    python tools/pancake/kfd_evict.py PID            print the counters of a running process (one line per node)
    python tools/pancake/abtest_kfd.py SPEC ...      abtest.py with every engine start's `kfd_evict_ms` added to its result (abtest.py itself is not edited; the
                                                     window is the whole engine run, start to stop, warm-up included; the plain abtest.py never reads the files)

Windows and any host without the sysfs files: every function returns None / {} and abtest records nothing.
Tests: python -m pytest -q tools/pancake/test_kfd_evict.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

KFD_PROC = "/sys/class/kfd/kfd/proc"


def read_counters(pid: int, root: str = KFD_PROC) -> dict:
    """{gpu_id directory name: evicted_ms} for one process; {} when the process or the sysfs tree is not there."""
    base = Path(root) / str(pid)
    out: dict = {}
    try:
        for d in sorted(base.iterdir()):
            if not re.fullmatch(r"stats_\d+", d.name):
                continue
            try:
                out[d.name] = int((d / "evicted_ms").read_text().strip())
            except (OSError, ValueError):
                continue
    except OSError:
        return {}
    return out


def rise(before: dict, after: dict) -> int | None:
    """Evicted ms between two samples: the LARGEST per-node rise (one eviction stops every node together, the nodes' values differ by a few ms as their
    queues are remapped in turn).  None when either sample is empty (no sysfs / the process was not there); a node missing from `before` counts from 0."""
    if not before and not after:
        return None
    if not after:
        return None
    return max(max(0, v - before.get(k, 0)) for k, v in after.items())


class Window:
    """with Window(pid) as w: ...work...; w.ms  -> evicted ms during the block, or None when the counters are not readable.  Reads only when `enabled`."""

    def __init__(self, pid: int, enabled: bool = True, root: str = KFD_PROC):
        self.pid, self.enabled, self.root, self.ms = pid, enabled, root, None
        self._b: dict = {}

    def __enter__(self):
        if self.enabled:
            self._b = read_counters(self.pid, self.root)
        return self

    def __exit__(self, *exc):
        if self.enabled:
            self.ms = rise(self._b, read_counters(self.pid, self.root))
        return False


def install(ab, root: str = KFD_PROC) -> None:
    """Wrap an abtest module so each `_run_arm` result carries `kfd_evict_ms` (None when the counters are unreadable).  Patches module attributes only:
    start_server notes the engine pid and its counters, stop_servers reads them again just before the process goes (the counters vanish with it)."""
    box: dict = {}
    start, stop, run = ab.start_server, ab.stop_servers, ab._run_arm

    def start_server(*a, **k):
        r = start(*a, **k)
        srv = r[0] if isinstance(r, tuple) else r
        pid = getattr(srv, "pid", None)
        if pid:
            box.update(pid=pid, before=read_counters(pid, root), ms=None)
        return r

    def stop_servers(*a, **k):
        if box.get("pid"):
            box["ms"] = rise(box["before"], read_counters(box["pid"], root))
        return stop(*a, **k)

    def _run_arm(*a, **k):
        box.clear()
        res, err = run(*a, **k)
        if isinstance(res, dict):
            res["kfd_evict_ms"] = box.get("ms")
        return res, err

    ab.start_server, ab.stop_servers, ab._run_arm = start_server, stop_servers, _run_arm


def main(argv=None) -> int:
    a = sys.argv[1:] if argv is None else argv
    if len(a) != 1 or not a[0].isdigit():
        print(__doc__.split("\n")[0])
        return 2
    c = read_counters(int(a[0]))
    if not c:
        print(f"no KFD counters for pid {a[0]} (not Linux/ROCm, or the process has no GPU queues)")
        return 1
    for k, v in c.items():
        print(f"{k}: evicted_ms {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
