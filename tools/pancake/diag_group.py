"""diag_group.py - run one step of a flag diagnostic in its own process group, with a hard time limit, and leave nothing behind.

    rc, timed_out, killed = diag_group.run(cmd, log_path, timeout_s)

The command (an abtest.py screen of one arm, or a ctest) starts in a NEW process group of the caller's session: a chain that ran
the diagnostic under `setsid` still finds every process of it by session id.  After `timeout_s` the whole group gets TERM
(abtest.py stops its own servers on TERM), KILL `kill_after` s later; the same happens when the caller itself is stopped (TERM ->
an exception in the wait).  Then it waits (at most `orphan_wait` s) until no process of that group is left (the server and the
engine below abtest.py are in the same group) and SIGKILLs only the leftovers of that group; `killed` lists them (pid, command).
Nothing outside the group is ever signalled.

Why: 2026-11-19 14:59 the chain's `timeout` ended flag_diag.py, but its abtest.py / serve/server.py / strata engine kept running,
and the next diagnostic's abtest refused to start (exit 3, "another test or engine is already running").  The group scan reads
/proc (Linux); elsewhere it finds nothing."""
import os
import signal
import subprocess
import time
from pathlib import Path

SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)


def members(pgid, proc="/proc"):
    """[(pid, command line)] of the live processes whose process group is `pgid` (zombies are over and left out)."""
    out = []
    try:
        names = os.listdir(proc)
    except OSError:
        return out
    for n in names:
        if not n.isdigit():
            continue
        try:
            st = Path(proc, n, "stat").read_text()
            f = st[st.rindex(")") + 2:].split()   # f[0] state, f[1] ppid, f[2] pgrp
            if f[0] in "ZX" or int(f[2]) != pgid:
                continue
            cmd = Path(proc, n, "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
            out.append((int(n), cmd))
        except (OSError, ValueError, IndexError):
            continue
    return out


def reap(pgid, wait=60.0, scan=members, kill=os.kill, sleep=time.sleep, clock=time.time):
    """Waits until no process of group `pgid` is left (at most `wait` s), then SIGKILLs the leftovers of that group only.
    Returns [(pid, command)] of the processes it killed (empty when the group ended by itself)."""
    t0 = clock()
    while clock() - t0 < wait:
        if not scan(pgid):
            return []
        sleep(2)
    killed = []
    for pid, cmd in scan(pgid):
        try:
            kill(pid, SIGKILL)
            killed.append((pid, cmd))
        except OSError:
            pass
    t1 = clock()
    while scan(pgid) and clock() - t1 < 10:
        sleep(1)
    return killed


def _signal_group(pgid, sig):
    try:
        os.killpg(pgid, sig)
    except (OSError, AttributeError):
        pass


def _stop(p, kill_after):
    _signal_group(p.pid, signal.SIGTERM)
    try:
        p.wait(timeout=kill_after)
    except subprocess.TimeoutExpired:
        _signal_group(p.pid, SIGKILL)
        p.wait()


def run(cmd, log_path, timeout_s, cwd=None, env=None, kill_after=30.0, orphan_wait=60.0, say=None):
    """-> (exit code or None after a timeout, timed_out, [(pid, command) killed as leftovers])."""
    kw = {}
    if os.name == "posix":
        kw["preexec_fn"] = os.setpgrp   # its own group, the caller's session
    with open(log_path, "w", encoding="utf-8") as fh:
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=cwd, env=env, **kw)
        timed_out, rc = False, None
        try:
            rc = p.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _stop(p, kill_after)
        except BaseException:   # the caller was stopped (TERM handler, Ctrl-C): stop this group too, then pass it on
            _stop(p, kill_after)
            killed = reap(p.pid, orphan_wait) if os.name == "posix" else []
            if killed and say:
                say(f"diag_group: killed leftovers of group {p.pid}: " + "; ".join(f"{a} {b[:160]}" for a, b in killed))
            raise
    killed = reap(p.pid, orphan_wait) if os.name == "posix" else []
    if killed and say:
        say(f"diag_group: killed leftovers of group {p.pid}: " + "; ".join(f"{a} {b[:160]}" for a, b in killed))
    return rc, timed_out, killed
