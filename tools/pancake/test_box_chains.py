"""Control-flow tests of the box chain scripts (tools/pancake/box_chains/) on a PC: every external command (ssh-side tools, the
engine, pgrep, ss, systemctl, cmake, abtest ...) is a stub, the clock runs fast, the paths are rewritten into a temp dir.  What
is proved: the marker waits, the busy() pattern (no chain waits for a LATER chain's idle waiter), the pancake.service stop only
after 3 polls without a client and the start at the end, START / END pairs, the marker after a failed step, the chainT-decide.py
winner rule, and that S's fallthrough grep matches the verdict wording replay_turns.py really prints.
Needs a POSIX sh (Git Bash on Windows).  python -m pytest -q tools/pancake/test_box_chains.py"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
CHAINS = HERE / "box_chains"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CHAINS))
SH = shutil.which("sh")
pytestmark = pytest.mark.skipif(not SH or not shutil.which("grep"), reason="needs a POSIX sh")

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("chainT_decide", CHAINS / "chainT-decide.py")
DECIDE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DECIDE)

SPEC_IDS = [128, 164, 165, 166, 167, 168, 169, 170, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205,
            206, 207, 208, 209, 210, 211, 212, 213, 214, 215]
ARMS = {192: ["base", "wmma", "wmma-rows"], 164: ["base", "lanecell", "armB"], 203: ["base", "defer", "armB"]}
PROBES = ["act_f16_probe", "link_touch_probe", "wmma_rate_probe", "int4_wmma_probe", "int4_moe_probe", "attn_fold_probe",
          "raw_buffer_probe", "persist_tape_probe", "persist_layer_parity", "retained_replay_probe", "expdec_bench"]

PYSTUB = r'''
import json, os, sys, runpy
H = os.environ["HWIN"]; REPO = os.environ["REPO_TOOLS"]
args = sys.argv[1:]
open(H + "/calls.log", "a").write("PY " + " ".join(args) + "\n")
fails = [f for f in os.environ.get("STUB_FAIL", "").split(",") if f]
joined = " ".join(args)
def fail():
    return any(f in joined for f in fails)
def opt(name, default=None):
    return args[args.index(name) + 1] if name in args else default
if not args:
    sys.exit(0)
a0 = args[0]
if a0.endswith("chainT-decide.py"):
    sys.argv = args; runpy.run_path(a0, run_name="__main__")
if a0 == "-m":
    sys.exit(1 if fail() else 0)
name = os.path.basename(a0)
if name == "abtest.py":
    spec = json.load(open(args[1]))
    arms = list(spec["arms"])
    out = opt("--out")
    print("abtest stub", joined)
    for case in ("c1", "c2"):
        for i, arm in enumerate(arms):
            mean = float(os.environ.get("STUB_GAIN", "105")) if (i == len(arms) - 1 and i > 0 and case == "c1") else 100.0
            print(f"  {case} {arm} {mean} +- 1.0 (n=5)")
    print("generated text vs base")
    for arm in arms[1:]:
        print(f"  {arm} identical on every case")
    print("  vs base: " + arms[-1] + " better 5.0%")
    sys.exit(1 if fail() else 0)
if name == "replay_turns.py":
    sys.path.insert(0, REPO)
    if not hasattr(os, "sysconf"):
        os.sysconf = lambda name: 100          # abtest.py reads SC_CLK_TCK at import (tools/pancake/conftest.py does the same)
    import replay_turns as RT
    arms = (opt("--arms") or "base,no-state").split(",")
    reps = int(opt("--reps", "1"))
    spread = os.environ.get("STUB_REPLAY", "worse")
    def tot(p):
        return {"prompt_s": p, "prompt_s_agent": p, "prompt_s_side": 0.0, "turn_s": p + 5, "cold_rereads": 0, "cold_reread_s": 0.0, "first_cold": 1,
                "first_cold_s": 1.0, "parked": 3, "restored": 3, "skip_park": {}, "min_avail_gib": 10.0, "peak_rss_gib": 5.0,
                "errors": 0}
    res = {}
    for arm in arms:
        res[arm] = []
        for r in range(reps):
            delta = 0.0 if arm == arms[0] else (-60.0 if arm in os.environ.get("STUB_BETTER", "").split(",") else (60.0 if spread == "worse" else (0.0 if r % 2 else 1.0)))
            jitter = (-1.0 if r % 2 else 1.0) * (0.2 if spread == "worse" else 30.0)
            res[arm].append({"rep": r, "totals": tot(100.0 + delta + jitter), "budget": "b", "clean": True})
    RT.recorded_totals = lambda plan: None
    RT.report(res, arms, [])
    sys.exit(1 if fail() else 0)
if name == "conv_trace.py":
    d = args[2]; tag = opt("--tag")
    for suffix in ("requests.json", "trace", "engine.log"):
        open(f"{d}/{tag}.{suffix}", "w").write("x\n" * 5)
    open(opt("--window-trace"), "w").write("x\n" * 5)
    sys.exit(1 if fail() else 0)
if name == "replay_fill_transition.py":
    print("replay header line")
    for k in ("base", "pm", "pmw", "ff:0.7", "tr:2.0"):
        print(f"{k} 1.0 2.0")
    open(opt("--json"), "w").write("{}")
    sys.exit(1 if fail() else 0)
if name == "quality_gate.py":
    print("PASS stub gate")
    sys.exit(1 if fail() else 0)
sys.exit(1 if fail() else 0)
'''

SH_STUBS = {
    "sleep": 'exec /usr/bin/sleep 0.01\n',
    "pgrep": 'echo "pgrep $*" >> "$H/calls.log" 2>/dev/null; [ "$1" = -f ] || exit 2; grep -Eq -- "$2" "$H/procs.txt"\n',
    "ss": ('n=$(head -n1 "$H/clients" 2>/dev/null); [ -n "$n" ] || n=0; sed -i 1d "$H/clients" 2>/dev/null; '
           'echo "SS $n" >> "$H/calls.log"; i=0; while [ $i -lt $n ]; do echo ESTAB; i=$((i+1)); done\n'),
    "systemctl": 'echo "SYSTEMCTL $*" >> "$H/calls.log"\n',
    "sudo": 'cat > /dev/null\n',
    "git": 'printf "abc123\\trefs/heads/port/rdna3-decode\\n"\n',
    "curl": 'while [ $# -gt 0 ]; do [ "$1" = -o ] && out=$2; shift; done; cp "$H/fake-src.tgz" "$out"\n',
    "timeout": 'shift; exec "$@"\n',
    "nice": 'shift 2; exec "$@"\n',
    "python3": 'exec "$PYREAL" "$H/stub/pystub.py" "$@"\n',
    "ctest": ('echo "ctest $*" >> "$H/calls.log"\n'
              'echo " 1/2 Test  #1: qsa_qmajor_parity ......   Passed    1.0 sec"\n'
              'echo " 2/2 Test  #2: hip_decode_attn_wmma_parity .....   Passed    1.0 sec"\n'
              'echo "100% tests passed, 0 tests failed out of 2"\n'),
    "hipcc": 'while [ $# -gt 0 ]; do [ "$1" = -o ] && out=$2; shift; done; printf "#!/bin/sh\nexit 0\n" > "$out"; chmod +x "$out"\n',
    "cmake": ('echo "cmake $*" >> "$H/calls.log"\n'
              'b=; t=; while [ $# -gt 0 ]; do case $1 in --build) b=$2;; -B) b=$2;; --target) t=$2;; esac; shift; done\n'
              '[ "$CMAKE_FAIL" = 1 ] && [ "$t" = strata ] && { rm -f "$b/strata"; exit 1; }\n'  # a failed link removes the binary (T-965 / T-968)
              '[ -n "$b" ] && mkdir -p "$b" && [ -n "$t" ] && [ ! -e "$b/$t" ] && { printf "#!/bin/sh\\nexit 0\\n" > "$b/$t"; chmod +x "$b/$t"; }\nexit 0\n'),
}


class Box:
    def __init__(self, tmp):
        self.h = tmp.as_posix()
        if os.name == "nt":       # msys tar reads "C:/x" as a remote host: hand the scripts /c/x paths
            self.h = subprocess.run(["cygpath", "-u", str(tmp)], capture_output=True, text=True).stdout.strip()
        self.tmp = tmp
        H = tmp
        (H / "stub").mkdir()
        for n, body in SH_STUBS.items():
            p = H / "stub" / n
            p.write_text("#!/bin/sh\n" + body, newline="\n")
        (H / "stub" / "pystub.py").write_text(PYSTUB)
        S = H / "clone-of-d" / "src"
        self.S = S
        R = S / "Strata-win"
        self.R = R
        (R / "bench" / "specs").mkdir(parents=True)
        (R / "tools" / "pancake").mkdir(parents=True)
        names = {209: "dq-fast-quality", 210: "route-budget-quality", 168: "quality-floor", 193: "wmma-rows-quality",
                 195: "prompt-lazy-quality", 211: "d4-suffix-index-replay"}
        for n in SPEC_IDS:
            arms = {"base": {}, "armA": {"env": {"X": "1"}}, "armB": {"env": {"Y": "1"}}}
            if n in ARMS:
                arms = {a: {"env": {"Z": a}} for a in ARMS[n]}
            (R / "bench" / "specs" / f"{n}-{names.get(n, 'spec')}.json").write_text(json.dumps({"arms": arms, "note": "n"}))
        shutil.copy(CHAINS / "chainT-decide.py", S / "chainT-decide.py")
        (R / "tools" / "pancake" / "replay_fill_transition.py").write_text("# --requests\n")
        (R / "tools" / "pancake" / "conv_trace.py").write_text("# --order\n")
        (S / "Strata").mkdir()
        (S / "Strata" / "pancake-flashnext-iq2_xs-fast-hip.json").write_text("{}")
        B = S / "build-win"
        B.mkdir()
        for t in ["strata", "vmm_test"] + PROBES:
            (B / t).write_text("#!/bin/sh\nexit 0\n", newline="\n")
        (H / "mnt").mkdir()
        (H / "clone-of-d" / "models" / "qwen36-UD-Q4_K_XL").mkdir(parents=True)
        (H / "clone-of-d" / "models" / "qwen36-UD-Q4_K_XL" / "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf").write_text("x")
        (H / "auto-runs" / "win16-conv").mkdir(parents=True)
        for f in ("real1.trace", "real1.requests.json"):
            (H / "auto-runs" / "win16-conv" / f).write_text("x\n")
        (R / "data").mkdir()
        (R / "data" / "expert-profile-pancake.bin").write_text("x")
        # a tarball that codeload would serve
        src = H / "srcdir" / "pancake-strata-abc"
        src.mkdir(parents=True)
        (src / "README").write_text("x")
        subprocess.run(["tar", "czf", (H / "fake-src.tgz").as_posix(), "-C", (H / "srcdir").as_posix(), "pancake-strata-abc"], check=True)
        (H / "calls.log").write_text("")
        (H / "procs.txt").write_text("")
        (H / "clients").write_text("")

    def log(self, name):
        p = self.S / name
        return p.read_text().splitlines() if p.exists() else []

    def calls(self):
        return (self.tmp / "calls.log").read_text().splitlines()

    def script_text(self, name, transform=None):
        t = (CHAINS / name).read_text()
        h = self.h
        t = t.replace("/home/linuxbrew/.linuxbrew/bin", h + "/stub").replace("/opt/rocm/core-10.0/bin", h + "/stub")
        t = t.replace("/opt/rocm/core-10.0", h + "/rocm").replace("/mnt/pancake", h + "/mnt").replace("/home/dan", h)
        if transform:
            t = transform(t)
        return t

    def start(self, name, transform=None, env=None):
        sp = self.tmp / ("run-" + name)
        sp.write_text(self.script_text(name, transform), newline="\n")
        e = dict(os.environ)
        e.update({"H": self.h, "HWIN": self.tmp.as_posix(), "PYREAL": sys.executable, "REPO_TOOLS": HERE.as_posix(),
                  "PATH": str(self.tmp / "stub") + os.pathsep + e["PATH"], "STUB_REPLAY": "worse"})
        e.update(env or {})
        return subprocess.Popen([SH, sp.as_posix()], env=e, cwd=self.tmp, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def finish(self, p, timeout=120):
        try:
            out, _ = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            out, _ = p.communicate()
            raise AssertionError("script hung:\n" + out[-2000:])
        return out


def append(box, logname, line):
    with open(box.S / logname, "a") as f:
        f.write(line + "\n")


def assert_start_end_pairs(lines):
    """After the header START line the START / END lines must alternate (steps are sequential) and end with END."""
    ev = [l.split(" ", 1)[0] for l in lines[1:] if re.match(r"^(START|END) ", l)]
    for i, e in enumerate(ev):
        assert e == ("START" if i % 2 == 0 else "END"), (i, ev, "START/END do not alternate")
    assert len(ev) % 2 == 0, "a START without END"


def assert_service_discipline(box, polls_zero=3):
    cl = [c for c in box.calls() if c.startswith(("SS ", "SYSTEMCTL "))]
    stops = [i for i, c in enumerate(cl) if c == "SYSTEMCTL --user stop pancake.service"]
    starts = [i for i, c in enumerate(cl) if c == "SYSTEMCTL --user start pancake.service"]
    assert len(stops) == 1 and len(starts) == 1 and stops[0] < starts[0], cl
    assert cl[-1] == "SYSTEMCTL --user start pancake.service"
    before = cl[:stops[0]]
    assert before[-polls_zero:] == ["SS 0"] * polls_zero, before[-5:]


# ------------------------------------------------------------------------------------------------ busy() pattern
def busy_fn(box, name):
    t = box.script_text(name)
    return next(l for l in t.splitlines() if l.startswith("busy()"))


def busy_result(box, name, procs):
    (box.tmp / "procs.txt").write_text("\n".join(procs) + "\n")
    fn = busy_fn(box, name)
    r = subprocess.run([SH, "-c", fn + "\nif busy; then echo BUSY; else echo IDLE; fi"], capture_output=True, text=True,
                       env={**os.environ, "H": box.h, "PATH": str(box.tmp / "stub") + os.pathsep + os.environ["PATH"]})
    return r.stdout.strip()


WAITERS = ["sh ./win-chainU.sh", "sh ./win-chainV.sh", "sh ./xtail-chainX.sh", "/bin/sh ./after7.sh"]


@pytest.mark.parametrize("name,own", [("win-chainT.sh", "win-chainT.sh"), ("win-chainV.sh", "win-chainV.sh"),
                                      ("xtail-chainX.sh", "xtail-chainX.sh")])
def test_busy_ignores_itself_and_later_waiters(tmp_path, name, own):
    box = Box(tmp_path)
    later = {"win-chainT.sh": WAITERS, "win-chainV.sh": ["sh ./xtail-chainX.sh", "/bin/sh ./after7.sh"],
             "xtail-chainX.sh": ["/bin/sh ./after7.sh"]}[name]
    assert busy_result(box, name, ["sh ./" + own] + later) == "IDLE"
    assert busy_result(box, name, ["sh ./win-chainH2.sh"]) == "BUSY"
    assert busy_result(box, name, ["sh ./win-chainS.sh"]) == "BUSY"
    assert busy_result(box, name, ["python3 tools/pancake/abtest.py x"]) == "BUSY"
    assert busy_result(box, name, ["cmake --build x"]) == "BUSY"
    assert busy_result(box, name, ["sh ./win-chainN.sh"]) == "BUSY"


def test_busy_s_sees_h2_only_of_the_chains(tmp_path):
    box = Box(tmp_path)
    assert busy_result(box, "win-chainS.sh", ["sh ./win-chainH2.sh"]) == "BUSY"
    assert busy_result(box, "win-chainS.sh", ["sh ./win-chainS.sh"] + WAITERS) == "IDLE"


def test_after7_busy_never_sees_the_x_waiter():
    # after7.sh on the box matches "[w]in-chain"; the tail chain's name must not contain it
    assert "win-chain" not in "xtail-chainX.sh"


# ------------------------------------------------------------------------------------------------ chains end to end
def run_chain(box, name, pred_log, marker, final_marker, clients=None, transform=None, env=None, timeout=120):
    """Start NAME, check that it only logs START while the predecessor marker is missing, then release it."""
    if clients is not None:
        (box.tmp / "clients").write_text("\n".join(map(str, clients)) + "\n")
    p = box.start(name, transform, env)
    log = {"win-chainS.sh": "win-chainS.log", "win-chainT.sh": "win-chainT.log", "win-chainU.sh": "win-chainU.log",
           "win-chainV.sh": "win-chainV.log", "xtail-chainX.sh": "xtail-chainX.log"}[name]
    time.sleep(1.5)
    assert p.poll() is None
    lines = box.log(log)
    assert len(lines) == 1 and lines[0].startswith("START "), lines
    assert not any(c.startswith("SYSTEMCTL") for c in box.calls())
    append(box, *pred_log) if pred_log[0] != "after7" else (box.tmp / "auto-runs" / "after7.log").open("a").write(marker + "\n")
    out = box.finish(p, timeout)
    lines = box.log(log)
    assert lines[-1] == final_marker, lines[-6:]
    assert_start_end_pairs(lines)
    return lines, out


def test_chain_t_full_flow(tmp_path):
    box = Box(tmp_path)
    (box.tmp / "procs.txt").write_text("\n".join(["sh ./win-chainT.sh"] + WAITERS) + "\n")   # U, V, X, after7 are waiting
    lines, out = run_chain(box, "win-chainT.sh", ("win-chainS.log", "WINCHAINS_DONE"), "WINCHAINS_DONE", "WINCHAINT_DONE",
                           clients=[2, 1, 0, 3, 0, 0, 0])
    assert_service_discipline(box)
    text = "\n".join(lines)
    for must in ("chain S done", "SYNC exit=0", "BUILD_END build-win exit=0", "END ctest new", "parity gates: qsa_qmajor_parity=1 "
                 "hip_decode_attn_wmma_parity=1", "START gate 209", "START gate 168", "START gate 210", "START replay 211 rep 1",
                 "START 199-screen", "START 197-screen", "START 192-screen", "START 164-screen", "START 169-screen",
                 "START 201-screen", "START 203-screen", "START 165-screen", "START 207-screen", "START 191-screen",
                 "START 170-screen", "START 205-screen", "END 199-confirm", "CONFIRM"):
        assert must in text, must
    # every step the owner wants kept: spec 192 confirm, replay 211, quality gates 209 / 168 / 210
    assert "START 192-confirm" in text
    # the tail items are NOT in T any more
    for gone in ("START 196", "START 167", "START 194", "START act_f16"):
        assert gone not in text
    # a screen is followed by its confirm only for winners: 164 has no sibling but its winner arm still confirms
    assert text.index("START gate 209") < text.index("START replay 211") < text.index("START 199-screen")
    assert text.index("START 199-screen") < text.index("START 197-screen") < text.index("START 192-screen")
    # the first wait is the full 3 idle minutes, the following ones the quick idle (observable through the number of pgrep polls)


def test_chain_t_old_busy_pattern_deadlocks(tmp_path):
    """The pattern chain T had before the fix ([A-SU-Z]: U and V are LATER chains) never goes idle beside their waiters."""
    box = Box(tmp_path)
    (box.tmp / "procs.txt").write_text("\n".join(["sh ./win-chainT.sh"] + WAITERS) + "\n")
    append(box, "win-chainS.log", "WINCHAINS_DONE")
    old = lambda t: t.replace(r"[w]in-chain[A-Sa-z0-9][A-Za-z0-9]*\.sh", r"[w]in-chain[A-SU-Za-z0-9]*\.sh")
    p = box.start("win-chainT.sh", old)
    time.sleep(6)
    still = p.poll() is None
    p.kill()
    p.communicate()
    assert still, "the old pattern should hang in wait_idle"
    assert not any("SYNC start" in l for l in box.log("win-chainT.log")), box.log("win-chainT.log")


def test_chain_t_failed_steps_still_write_the_marker(tmp_path):
    box = Box(tmp_path)
    lines, out = run_chain(box, "win-chainT.sh", ("win-chainS.log", "WINCHAINS_DONE"), "WINCHAINS_DONE", "WINCHAINT_DONE",
                           clients=[0, 0, 0], env={"STUB_FAIL": "199-screen,quality_gate,replay_turns.py"})
    text = "\n".join(lines)
    assert "END 199-screen exit=1" in text and "screen 199 failed or skipped: no confirm" in text
    assert "END gate 209 exit=1" in text and "END replay 211 rep 1" in text
    assert_service_discipline(box)


def test_chain_t_waits_for_clients(tmp_path):
    box = Box(tmp_path)
    run_chain(box, "win-chainT.sh", ("win-chainS.log", "WINCHAINS_DONE"), "WINCHAINS_DONE", "WINCHAINT_DONE",
              clients=[1, 1, 1, 1, 0, 1, 0, 0, 0])
    assert_service_discipline(box)
    assert sum("WAIT service has clients" in l for l in box.log("win-chainT.log")) == 5


def test_chain_x_full_flow(tmp_path):
    box = Box(tmp_path)
    (box.tmp / "auto-runs").mkdir(exist_ok=True)
    (box.tmp / "auto-runs" / "after7.log").write_text("QUEUED\n")
    (box.tmp / "procs.txt").write_text("sh ./xtail-chainX.sh\n")
    lines, out = run_chain(box, "xtail-chainX.sh", ("after7", ""), "AFTER7_DONE", "WINCHAINX_DONE", clients=[0, 0, 0])
    assert_service_discipline(box)
    text = "\n".join(lines)
    for must in ("after7 done", "ENGINE=1", "START act_f16", "START ablate_default", "BUILD_END", "START 194-screen",
                 "START 196-plain", "START 196-rotated", "START 167-soak", "START 212-screen", "START int4_moe_gateup",
                 "vmm_probe", "START vmm_test_build", "END vmm_test_zero exit=0"):
        assert must in text, must
    assert "vmm_probe skipped" in text            # chain T builds it, not present in this fixture
    # the queued-job order inside X: probes, ablation, 194 / 195, 196, 167, 212 / 213
    pos = [text.index(k) for k in ("START act_f16", "START ablate_default", "START 194-screen", "START 196-plain",
                                   "START 167-soak", "START 212-screen")]
    assert pos == sorted(pos)


def test_chain_x_vmm_probe_runs_when_chain_t_built_it(tmp_path):
    box = Box(tmp_path)
    (box.tmp / "mnt" / "chainT").mkdir(parents=True)
    (box.tmp / "mnt" / "chainT" / "vmm_probe").write_text("#!/bin/sh\nexit 0\n", newline="\n")
    (box.tmp / "auto-runs").mkdir(exist_ok=True)
    (box.tmp / "auto-runs" / "after7.log").write_text("QUEUED AFTER7_DONE\n")
    lines, _ = run_chain(box, "xtail-chainX.sh", ("after7", ""), "AFTER7_DONE", "WINCHAINX_DONE", clients=[0, 0, 0]) \
        if False else (None, None)
    p = box.start("xtail-chainX.sh")
    box.finish(p)
    text = "\n".join(box.log("xtail-chainX.log"))
    assert "START vmm_probe" in text and "END vmm_probe exit=0" in text
    assert text.index("END vmm_test_zero") < text.index("START vmm_probe")   # the ZERO_BACK diagnosis runs first


def test_chain_u_marker_chain(tmp_path):
    box = Box(tmp_path)
    run_chain(box, "win-chainU.sh", ("win-chainT.log", "WINCHAINT_DONE"), "WINCHAINT_DONE", "WINCHAINU_DONE")


def test_chain_v_full_flow(tmp_path):
    box = Box(tmp_path)
    lines, out = run_chain(box, "win-chainV.sh", ("win-chainU.log", "WINCHAINU_DONE"), "WINCHAINU_DONE", "WINCHAINV_DONE",
                           clients=[0, 0, 0])
    assert_service_discipline(box)
    text = "\n".join(lines)
    for must in ("chain U done", "START record real4", "START record real5", "START replay real4-main", "START replay real5-slots5000",
                 "START replay real1-chain16a", "COPY small outputs"):
        assert must in text, must


def test_chain_s_full_flow_and_fallthrough(tmp_path):
    box = Box(tmp_path)
    lines, out = run_chain(box, "win-chainS.sh", ("win-chainH2.log", "WINCHAINH2_DONE"), "WINCHAINH2_DONE", "WINCHAINS_DONE",
                           clients=[0, 0, 0], env={"STUB_REPLAY": "worse"})
    assert_service_discipline(box)
    text = "\n".join(lines)
    assert "START 128 replay3 single arms" in text and "128 replay3 skipped" not in text
    assert any(re.search(r"no-state .*-> worse", l) for l in lines)


def test_chain_s_skips_replay3_when_no_state_does_not_resolve(tmp_path):
    box = Box(tmp_path)
    lines, out = run_chain(box, "win-chainS.sh", ("win-chainH2.log", "WINCHAINH2_DONE"), "WINCHAINH2_DONE", "WINCHAINS_DONE",
                           clients=[0, 0, 0], env={"STUB_REPLAY": "none"})
    text = "\n".join(lines)
    assert "START 128 replay3" not in text and "128 replay3 skipped" in text
    assert any("no-state" in l and "no difference" in l for l in lines)


# ------------------------------------------------------------------------------------------------ S fallthrough wording
def fake_results(delta, reps=2):
    def tot(p):
        return {"prompt_s": p, "prompt_s_agent": p, "prompt_s_side": 0.0, "turn_s": p + 5, "cold_rereads": 0, "cold_reread_s": 0.0, "first_cold": 1,
                "first_cold_s": 1.0, "parked": 3, "restored": 3, "skip_park": {}, "min_avail_gib": 10.0, "peak_rss_gib": 5.0,
                "errors": 0}
    res = {"base": [], "no-state": []}
    for r in range(reps):
        j = (-1.0 if r % 2 else 1.0) * 0.2
        res["base"].append({"rep": r, "totals": tot(100.0 + j), "budget": "b", "clean": True})
        res["no-state"].append({"rep": r, "totals": tot(100.0 + delta + j), "budget": "b", "clean": True})
    return res


def replay_report(monkeypatch, capsys, res):
    import replay_turns as RT
    monkeypatch.setattr(RT, "recorded_totals", lambda plan: None)
    RT.report(res, list(res), [])
    return capsys.readouterr().out


def s_fallthrough_pattern():
    t = (CHAINS / "win-chainS.sh").read_text()
    m = re.search(r'if grep -Eq "([^"]+)" \$O/winS-128-replay2\.stdout', t)
    assert m, "the fallthrough grep is gone from win-chainS.sh"
    return m.group(1)


@pytest.mark.parametrize("delta,resolves", [(60.0, True), (-60.0, True), (0.0, False)])
def test_s_grep_matches_the_real_replay_verdict_wording(monkeypatch, capsys, tmp_path, delta, resolves):
    out = replay_report(monkeypatch, capsys, fake_results(delta))
    f = tmp_path / "replay2.stdout"
    f.write_text(out)
    rc = subprocess.run(["grep", "-Eq", s_fallthrough_pattern(), f.as_posix()]).returncode
    assert (rc == 0) == resolves, out
    verdict = [l for l in out.splitlines() if l.lstrip().startswith("no-state") and "->" in l]
    assert len(verdict) == 1
    word = re.search(r"-> (better \(less prompt time\)|worse|no difference);", verdict[0]).group(1)
    assert (word != "no difference") == resolves


def test_s_grep_matches_the_stored_run_162_wording(tmp_path):
    """The line the 162 run printed (bench/results/2026-11-18-box-auto-runs-night-chains/winR-162.stdout) with the arm name swapped."""
    line = "  no-state                 +25.9 s +-  24.0 (+15.8% +- 14.6%)  -> worse;  cold re-reads [1, 0, 1, 1, 0] vs [0, 0, 0, 0, 0]"
    f = tmp_path / "x"
    f.write_text(line + "\n")
    assert subprocess.run(["grep", "-Eq", s_fallthrough_pattern(), f.as_posix()]).returncode == 0
    f.write_text(line.replace("-> worse", "-> no difference") + "\n")
    assert subprocess.run(["grep", "-Eq", s_fallthrough_pattern(), f.as_posix()]).returncode != 0
    f.write_text(line.replace("-> worse", "-> better (less prompt time)") + "\n")
    assert subprocess.run(["grep", "-Eq", s_fallthrough_pattern(), f.as_posix()]).returncode == 0


# ------------------------------------------------------------------------------------------------ chainT-decide.py
def screen_text(rows, texts=None):
    out = []
    for case, arms in rows.items():
        for arm, mean in arms.items():
            out.append(f"  {case}  {arm}  {mean} +- 1.0 (n=5)")
    out.append("generated text vs base")
    for arm, same in (texts or {}).items():
        out.append(f"  {arm}  {'identical on every case' if same else 'DIFFERENT text'}")
    return "\n".join(out)


SPEC = {"arms": {"base": {}, "a": {"e": 1}, "b": {"e": 2}, "base-late": {}}}


def test_decide_winner_rule():
    rows = {"c1": {"base": 100.0, "base-late": 100.0, "a": 103.0, "b": 100.5}, "c2": {"base": 50.0, "base-late": 50.0, "a": 50.0, "b": 50.0}}
    winners, log = DECIDE.decide(screen_text(rows, {"a": True, "b": True}), SPEC)
    assert [w[0] for w in winners] == ["a"]                     # +3% >= 1% pass-2 bar, b +0.5% no
    # a case below the -3% floor vetoes the win
    rows["c2"]["a"] = 48.0
    assert DECIDE.decide(screen_text(rows, {"a": True, "b": True}), SPEC)[0] == []
    # text differs in an exact spec: lossy pass-2 bar 1.5% (a +1.2% fails, +1.6% passes)
    rows = {"c1": {"base": 100.0, "base-late": 100.0, "a": 101.2, "b": 101.6}}
    w, _ = DECIDE.decide(screen_text(rows, {"a": False, "b": False}), SPEC)
    assert [x[0] for x in w] == ["b"] and "quality" in w[0][2]
    # reference arm other than the first: only that arm is the reference mean
    rows = {"c1": {"base": 100.0, "a": 110.0, "b": 100.0}}
    w, _ = DECIDE.decide(screen_text(rows, {"a": True, "b": True}), {"arms": {"base": {}, "a": {}, "b": {}}}, ref="a")
    assert [x[0] for x in w] == []                               # b is -9% vs a; base is the first arm, skipped
    assert DECIDE.decide("nothing parsed", SPEC)[1] == ["no summary lines parsed"]


def test_decide_writes_the_confirm_spec(tmp_path):
    rows = {"c1": {"base": 100.0, "base-late": 100.0, "a": 104.0, "b": 100.0}}
    (tmp_path / "s.stdout").write_text(screen_text(rows, {"a": True, "b": True}))
    (tmp_path / "spec.json").write_text(json.dumps({"arms": {"base": {}, "a": {"e": 1}, "b": {"e": 2}}, "note": "n"}))
    (tmp_path / "sib.json").write_text(json.dumps({"arms": {"base": {}, "a": {"e": 1, "more": 2}}, "lossy": False, "note": "sib"}))
    out = tmp_path / "confirm.json"
    rc = DECIDE.main([str(tmp_path / "s.stdout"), str(tmp_path / "spec.json"), str(out), "--confirm-spec", str(tmp_path / "sib.json")])
    assert rc == 0
    conf = json.loads(out.read_text())
    assert list(conf["arms"]) == ["base", "a"] and conf["arms"]["a"] == {"e": 1, "more": 2} and "winners a" in conf["note"]
    out.unlink()
    rows["c1"]["a"] = 100.0
    (tmp_path / "s.stdout").write_text(screen_text(rows, {"a": True, "b": True}))
    assert DECIDE.main([str(tmp_path / "s.stdout"), str(tmp_path / "spec.json"), str(out)]) == 0 and not out.exists()
    assert DECIDE.main([str(tmp_path / "missing"), str(tmp_path / "spec.json"), str(out)]) == 2


def test_decide_parses_the_real_abtest_summary_format():
    """The LINE / TEXT regexes against a line shaped like abtest.py's summary (arm and case columns)."""
    import abtest
    src = Path(abtest.__file__).read_text()
    assert "generated text vs base" in src and "identical on every case" in src and "DIFFERENT text" in src


# ------------------------------------------------------------------------------------------------ two-pass rule (owner order 2026-11-18)
def test_two_pass_constants_are_half_the_promotion_bars():
    assert (DECIDE.PROMOTION_BAR, DECIDE.PROMOTION_LOSSY_BAR) == (2.0, 3.0)
    assert (DECIDE.PASS2_BAR, DECIDE.PASS2_LOSSY_BAR) == (1.0, 1.5)


def test_pass2_screen_arms_with_benefit_confirm_flat_and_worse_are_dropped(tmp_path):
    rows = {"c1": {"base": 100.0, "base-late": 100.0, "good": 101.5, "flat": 100.2, "worse": 96.0, "edge": 100.9},
            "c2": {"base": 50.0, "base-late": 50.0, "good": 50.0, "flat": 50.0, "worse": 49.5, "edge": 50.0}}
    spec = {"arms": {"base": {}, "good": {"e": 1}, "flat": {"e": 2}, "worse": {"e": 3}, "edge": {"e": 4}, "base-late": {}}}
    winners, log = DECIDE.decide(screen_text(rows, {a: True for a in ("good", "flat", "worse", "edge")}), spec)
    # (abtest means are rates, higher is better) good: +1.5%, edge: +0.9% (< 1% bar), flat and worse: no benefit
    assert [w[0] for w in winners] == ["good"]
    dropped = [l for l in log if "dropped after pass 1: no benefit observed (n=1 pass, not a refutation)" in l]
    assert sorted(l.split(":")[0] for l in dropped) == ["edge", "flat", "worse"]
    (tmp_path / "s.stdout").write_text(screen_text(rows, {a: True for a in ("good", "flat", "worse", "edge")}))
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    out = tmp_path / "confirm.json"
    assert DECIDE.main([str(tmp_path / "s.stdout"), str(tmp_path / "spec.json"), str(out)]) == 0
    assert list(json.loads(out.read_text())["arms"]) == ["base", "good"]       # pass 2 runs only the reference and the arm with a benefit


def test_pass2_replay_keeps_only_favourable_arms():
    text = """per arm run
verdict vs base (total prompt time; paired by rep, 95% interval when reps >= 2)
  a                       -10.1 s ( -5.3%) prompt time, turn time -71.6 s, cold re-reads 0 vs 1  (one rep: deltas only)
  b                        +1.0 s ( +0.5%) prompt time, turn time +1.0 s, cold re-reads 0 vs 1  (one rep: deltas only)
  c                       +51.4 s (+27.0%) prompt time, turn time +16.1 s, cold re-reads 0 vs 1  (one rep: deltas only)
  d                       -26.0 s +-  24.0 (-15.8% +- 14.6%)  -> better (less prompt time);  cold re-reads [1] vs [0]
  e                        -1.0 s ( -0.5%) prompt time, turn time +1.0 s, cold re-reads 0 vs 1  (one rep: deltas only)
"""
    keep, log = DECIDE.replay_pass2(text)
    assert keep == ["a", "d"]
    assert sum("dropped after pass 1" in l for l in log) == 3


def test_chain_t_replay_211_pass2_skipped_when_no_arm_benefits(tmp_path):
    box = Box(tmp_path)
    lines, out = run_chain(box, "win-chainT.sh", ("win-chainS.log", "WINCHAINS_DONE"), "WINCHAINS_DONE", "WINCHAINT_DONE", clients=[0, 0, 0, 0])
    text = "\n".join(lines)
    assert "START replay 211 rep 1 (base,sidx-4mib,sidx-16mib)" in text
    assert "START replay 211 rep 2" not in text and "START replay 211 rep 3" not in text      # stub: both sidx arms +60 s (worse)
    assert "replay 211 rep 2 skipped: no arm shows a benefit in rep 1" in text
    assert "dropped after pass 1: no benefit observed" in text


def test_chain_t_replay_211_pass2_runs_only_the_favourable_arm(tmp_path):
    box = Box(tmp_path)
    lines, out = run_chain(box, "win-chainT.sh", ("win-chainS.log", "WINCHAINS_DONE"), "WINCHAINS_DONE", "WINCHAINT_DONE", clients=[0, 0, 0, 0],
                           env={"STUB_BETTER": "sidx-16mib"})
    text = "\n".join(lines)
    assert "START replay 211 rep 1 (base,sidx-4mib,sidx-16mib)" in text
    assert "START replay 211 rep 2 (sidx-16mib,base)" in text and "START replay 211 rep 3 (base,sidx-16mib)" in text
    assert "sidx-4mib" not in "".join(l for l in lines if "rep 2" in l or "rep 3" in l)
    assert "sidx-4mib: " in text and "dropped after pass 1" in text


# ------------------------------------------------------------------------------------------------ tail chain Y (victim-unused / direct-swap)
def test_y_waiter_name_is_invisible_to_every_busy_pattern(tmp_path):
    """V's busy() ([A-UW-Z]) and X's ([A-Za-z0-9]*) match any win-chain<letter>.sh: a Y waiter named that way would deadlock both."""
    box = Box(tmp_path)
    assert "win-chain" not in "ytail-chainY.sh"
    for name in ("win-chainV.sh", "xtail-chainX.sh", "ytail-chainY.sh"):
        assert busy_result(box, name, ["sh ./ytail-chainY.sh"]) == "IDLE", name
    assert busy_result(box, "ytail-chainY.sh", ["sh ./win-chainH2.sh"]) == "BUSY"
    assert busy_result(box, "ytail-chainY.sh", ["sh ./win-chainT.sh"]) == "BUSY"
    assert busy_result(box, "ytail-chainY.sh", ["python3 tools/pancake/abtest.py x"]) == "BUSY"
    assert busy_result(box, "ytail-chainY.sh", ["sh ./xtail-chainX.sh"]) == "IDLE"      # X is reached through its marker, not through busy()


def prepare_y(box):
    R = box.R
    (R / "include" / "strata" / "program").mkdir(parents=True)
    (R / "include" / "strata" / "program" / "tier_victim.hpp").write_text("// x\n")
    (R / "src" / "core").mkdir(parents=True)
    (R / "src" / "core" / "conversation_disk_env.def").write_text('STRATA_ENV("STRATA_ADAPT_VICTIM_UNUSED", Speed, "x")\n')
    (R / "tools" / "pancake" / "replay_fill_transition.py").write_text("# --requests --victim\n")
    (R / "bench" / "prefetch").mkdir(parents=True)
    (R / "bench" / "prefetch" / "routing.bin").write_text("x\n")
    for n in (214, 215):
        for f in (R / "bench" / "specs").glob(f"{n}-*.json"):
            f.write_text(json.dumps({"arms": {"base": {}, "direct": {"env": {"A": "1"}}, "unused": {"env": {"B": "1"}},
                                                 "both": {"env": {"A": "1", "B": "1"}}, "both-ff70": {"env": {"A": "1", "B": "1", "C": "1"}}},
                                     "note": "n"}))
    (box.tmp / "auto-runs").mkdir(exist_ok=True)
    (box.tmp / "auto-runs" / "after7.log").write_text("QUEUED\n")
    (box.S / "win-chainV.log").write_text("START\n")
    (box.S / "xtail-chainX.log").write_text("START\n")
    (box.tmp / "auto-runs" / "winV-conv").mkdir()
    for t in ("real4", "real5"):
        for suf in ("trace", "requests.json", "stwt"):
            (box.tmp / "auto-runs" / "winV-conv" / f"{t}.{suf}").write_text("x\n")


def test_chain_y_waits_for_all_three_markers_then_runs_two_pass(tmp_path):
    box = Box(tmp_path)
    prepare_y(box)
    logs = box.tmp / "auto-runs" / "winY-214-screen"        # the abtest stub writes no engine logs: the proof lines are staged
    logs.mkdir()
    (logs / "unused-r1-a0.engine.log").write_text("strata serve: STRATA_ADAPT_VICTIM_UNUSED=1: a victim must have had no use\n")
    (logs / "direct-r1-a0.engine.log").write_text("strata serve: STRATA_ADAPT_DIRECT_SWAP=1: the victim's slot is rewritten\n")
    (box.tmp / "clients").write_text("0\n0\n0\n")
    p = box.start("ytail-chainY.sh", env={"STUB_GAIN": "103"})
    time.sleep(1.5)
    assert p.poll() is None and len(box.log("ytail-chainY.log")) == 1
    append(box, "win-chainV.log", "WINCHAINV_DONE")
    time.sleep(1.0)
    assert p.poll() is None and len(box.log("ytail-chainY.log")) == 1          # V alone is not enough
    (box.tmp / "auto-runs" / "after7.log").open("a").write("AFTER7_DONE\n")
    time.sleep(1.0)
    assert p.poll() is None and len(box.log("ytail-chainY.log")) == 1          # nor V + after7: it must not overlap X
    assert not any(c.startswith("SYSTEMCTL") for c in box.calls())
    append(box, "xtail-chainX.log", "WINCHAINX_DONE")
    box.finish(p)
    lines = box.log("ytail-chainY.log")
    text = "\n".join(lines)
    assert lines[-1] == "WINCHAINY_DONE", lines[-6:]
    assert_start_end_pairs(lines)
    assert_service_discipline(box)
    for must in ("chains V, after7 and X done", "SYNC exit=0", "HEADER tier_victim.hpp + env table entries present: 1; replay tool with --victim: 1",
                 "BUILD_END build-win exit=0", "END ctest tier_victim", "START replay routing-main", "START replay real4-main",
                 "START replay real5-slots3000", "START replay real1-chain16a-main", "START 214-screen", "CONFIRM", "START 214-confirm"):
        assert must in text, must
    assert text.index("START replay routing-main") < text.index("START 214-screen") < text.index("START 214-confirm")
    cmake = [c for c in box.calls() if c.startswith("cmake --build")]
    assert any("tier_victim_test" in c and "strata" in c for c in cmake), cmake


def test_chain_y_pass2_drops_arms_without_benefit_and_skips_missing_chain_v_traces(tmp_path):
    box = Box(tmp_path)
    prepare_y(box)
    for f in (box.tmp / "auto-runs" / "winV-conv").iterdir():
        f.unlink()
    (box.S / "win-chainV.log").write_text("WINCHAINV_DONE\n")
    (box.S / "xtail-chainX.log").write_text("WINCHAINX_DONE\n")
    (box.tmp / "auto-runs" / "after7.log").write_text("AFTER7_DONE\n")
    (box.tmp / "clients").write_text("0\n0\n0\n")
    p = box.start("ytail-chainY.sh", env={"STUB_GAIN": "100.2"})      # the last arm is only +0.2 %: below the pass-2 bar
    box.finish(p)
    lines = box.log("ytail-chainY.log")
    text = "\n".join(lines)
    assert lines[-1] == "WINCHAINY_DONE"
    assert "SKIP replays of real4" in text and "SKIP replays of real5" in text and "START replay real1-chain16a-main" in text
    assert "dropped after pass 1" in text or "NONE" in text
    assert "START 214-confirm" not in text


def test_chain_y_skips_the_engine_steps_when_the_synced_tree_lacks_the_code(tmp_path):
    box = Box(tmp_path)
    prepare_y(box)
    (box.R / "include" / "strata" / "program" / "tier_victim.hpp").unlink()
    (box.S / "win-chainV.log").write_text("WINCHAINV_DONE\n")
    (box.S / "xtail-chainX.log").write_text("WINCHAINX_DONE\n")
    (box.tmp / "auto-runs" / "after7.log").write_text("AFTER7_DONE\n")
    (box.tmp / "clients").write_text("0\n0\n0\n")
    box.finish(box.start("ytail-chainY.sh"))
    text = "\n".join(box.log("ytail-chainY.log"))
    assert "BUILD SKIPPED: the synced tree has no tier_victim.hpp" in text and "START 214-screen" not in text
    assert box.log("ytail-chainY.log")[-1] == "WINCHAINY_DONE"


# ------------------------------------------------------------------------------------------------ redo chain AA
AA = "redo-chainAA.sh"
SOAK60 = '{\n "reps": 60,\n "arms": {"base": {}, "base-again": {}}\n}\n'


def prepare_aa(box, markers=True):
    (box.tmp / "auto-runs").mkdir(exist_ok=True)
    (box.tmp / "auto-runs" / "after7.log").write_text("AFTER7_DONE\n" if markers else "QUEUED\n")
    for log, m in (("win-chainV.log", "WINCHAINV_DONE"), ("xtail-chainX.log", "WINCHAINX_DONE"), ("ytail-chainY.log", "WINCHAINY_DONE"),
                   ("ztail-chainZ.log", "WINCHAINZ_DONE")):
        (box.S / log).write_text(m + "\n" if markers else "START\n")
    (box.R / "tests" / "core").mkdir(parents=True, exist_ok=True)
    (box.R / "tests" / "core" / "vmm_test.cpp").write_text("// ZERO_BACK diag\n")
    (box.R / "tools" / "pancake" / "vmm_probe.cpp").write_text("int main(){}\n")
    (box.R / "bench" / "specs" / "167-race-soak.json").write_text(SOAK60)
    (box.tmp / "clients").write_text("0\n0\n0\n")
    il = box.S / "build-win-il"
    il.mkdir()
    for t in ("strata", "mmvq_il_parity"):
        (il / t).write_text("#!/bin/sh\nexit 0\n", newline="\n")
    d = box.tmp / "auto-runs" / "after7"
    d.mkdir()
    for n, arms in (("85-il-plain-diag", ["base", "il-build", "il-plain"]), ("88-idle-copy-diag", ["base", "adapt-idle-copy"]),
                    ("88-rerun", ["base", "idle-1m", "skip-w02"])):
        (d / f"{n}.json").write_text(json.dumps({"arms": {a: {"env": {"X": a}} for a in arms}, "lossy": True}))


def test_redo_aa_waits_for_all_five_markers_then_runs_in_the_ordered_flow(tmp_path):
    box = Box(tmp_path)
    prepare_aa(box, markers=False)
    p = box.start(AA)
    time.sleep(1.5)
    assert p.poll() is None and len(box.log("redo-chainAA.log")) == 1
    for log, m in (("win-chainV.log", "WINCHAINV_DONE"), ("xtail-chainX.log", "WINCHAINX_DONE"), ("ytail-chainY.log", "WINCHAINY_DONE")):
        append(box, log, m)
        time.sleep(0.8)
        assert p.poll() is None and len(box.log("redo-chainAA.log")) == 1, "must wait for every earlier chain"
    (box.tmp / "auto-runs" / "after7.log").open("a").write("AFTER7_DONE\n")
    time.sleep(0.8)
    assert p.poll() is None and len(box.log("redo-chainAA.log")) == 1
    assert not any(c.startswith("SYSTEMCTL") for c in box.calls())
    append(box, "ztail-chainZ.log", "WINCHAINZ_DONE")
    box.finish(p)
    lines = box.log("redo-chainAA.log")
    text = "\n".join(lines)
    assert lines[-1] == "WINCHAINAA_DONE", lines[-8:]
    assert "ERROR" not in text and "ABORTED" not in text
    assert_start_end_pairs(lines)
    assert_service_discipline(box)
    for must in ("SYNC exit=0", "HEADER spec_lint exit=", "BUILD backup of the previous build-win/strata", "BUILD_END build-win exit=0",
                 "ENGINE=1 (build-win/strata present: yes)", "BUILD_END build-win-il exit=0", "END vmm_test_zero exit=0", "END vmm_probe exit=0",
                 "START record real4", "START record real5", "START replay real4-main", "START replay real5-slots5000", "START replay real1-chain16a",
                 "COPY small outputs", "START 85-il-plain-diag", "START 88-idle-copy-diag", "START 88-rerun-screen", "CONFIRM", "START 88-rerun-confirm",
                 "START act_f16", "START ablate_default", "START 194-screen", "START 196-plain", "START 196-rotated", "START 212-screen", "START 167-soak",
                 "167 soak spec:"):
        assert must in text, must
    assert "pcie_link_json" not in text                     # the duplicate run was dropped
    assert "START 85-il-plain-diag-confirm" not in text and "START 88-idle-copy-diag-confirm" not in text    # diagnostics: screen only
    order = ["BUILD_END build-win exit=0", "START vmm_test_zero", "START vmm_probe", "START record real4", "START replay real1-chain16a",
             "START 85-il-plain-diag", "START 88-idle-copy-diag", "START 88-rerun-screen", "START act_f16", "START ablate_default",
             "START 194-screen", "START 196-plain", "START 212-screen", "START 167-soak"]
    pos = [text.index(k) for k in order]
    assert pos == sorted(pos), order
    assert any(l.startswith("PY") and "167-soak40.json" in l for l in box.calls()), "the shortened soak copy is used"
    assert '"reps": 60' in (box.R / "bench" / "specs" / "167-race-soak.json").read_text()      # the repo spec is untouched
    assert '"reps": 40' in (box.tmp / "auto-runs" / "chainAA" / "167-soak40.json").read_text()


def test_redo_aa_failed_build_restores_the_binary_logs_error_and_stops(tmp_path):
    box = Box(tmp_path)
    prepare_aa(box)
    (box.S / "build-win" / "strata").write_text("#!/bin/sh\n# the previous working binary\n", newline="\n")
    before = (box.S / "build-win" / "strata").read_text()
    box.finish(box.start(AA, env={"CMAKE_FAIL": "1"}))
    lines = box.log("redo-chainAA.log")
    text = "\n".join(lines)
    assert "BUILD_END build-win exit=1" in text
    assert "BUILD FAILED: the previous build-win/strata was restored" in text
    assert any(l.startswith("ERROR BUILD FAILED") for l in lines)
    assert lines[-1] == "WINCHAINAA_DONE" and lines[-3].startswith("WINCHAINAA_ABORTED") and lines[-2].startswith("service started")
    assert (box.S / "build-win" / "strata").read_text() == before         # the failed link removed it; the chain put the copy back
    for never in ("START record", "START vmm_test_zero", "START 85-", "START 88-", "START act_f16", "START 167-soak", "ENGINE="):
        assert never not in text, never
    assert_service_discipline(box)


def test_redo_aa_stops_when_the_source_download_fails(tmp_path):
    box = Box(tmp_path)
    prepare_aa(box)
    (box.tmp / "fake-src.tgz").unlink()
    box.finish(box.start(AA))
    lines = box.log("redo-chainAA.log")
    assert any(l.startswith("ERROR SYNC download failed") for l in lines) and lines[-1] == "WINCHAINAA_DONE"
    assert "BUILD" not in "\n".join(lines)
    assert_service_discipline(box)


def test_redo_aa_skips_85_without_the_il_build_but_runs_the_rest(tmp_path):
    box = Box(tmp_path)
    prepare_aa(box)
    shutil.rmtree(box.S / "build-win-il")
    box.finish(box.start(AA))
    lines = box.log("redo-chainAA.log")
    text = "\n".join(lines)
    assert "WARN" in text and "SKIP 85-il-plain-diag" in text and "START 88-idle-copy-diag" in text and "START 167-soak" in text
    assert lines[-1] == "WINCHAINAA_DONE"


def test_redo_aa_name_is_invisible_to_every_busy_pattern_and_sees_the_others(tmp_path):
    box = Box(tmp_path)
    assert "win-chain" not in AA
    for name in ("win-chainV.sh", "xtail-chainX.sh", "ytail-chainY.sh", "ztail-chainZ.sh"):
        assert busy_result(box, name, ["sh ./redo-chainAA.sh"]) == "IDLE", name
    assert busy_result(box, AA, ["sh ./redo-chainAA.sh"]) == "IDLE"
    for other in ("sh ./win-chainT.sh", "sh ./xtail-chainX.sh", "sh ./ytail-chainY.sh", "/bin/sh ./after7.sh", "python3 tools/pancake/abtest.py x",
                  "cmake --build x"):
        assert busy_result(box, AA, [other]) == "BUSY", other


# ------------------------------------------------------------------------------------------------ bench-updatedAA (updated-strata vs plain upstream)
BU = "bench-updatedAA.sh"
FLAG_DIAG_STUB = r'''#!/bin/sh
case "$1" in
  run) mkdir -p "$3"; echo "{\"verdicts\": []}" > "$3/verdicts.json"; echo "GAIN STRATA_RDNA_Q6K"; echo "15.0 minutes; verdicts in $3/verdicts.json" ;;
  hw) echo "{\"gfx\": \"gfx1100\"}" ;;
  --write-recommended) echo "{\"schema\": \"strata.tuned-flags/1\", \"profiles\": []}" > "$3"; echo "wrote $3" ;;
esac
exit 0
'''
BUILD_STUB = '#!/bin/sh\n[ "$BU_FAIL" = 1 ] && { echo "boom: error: undefined reference" >&2; exit 1; }\nmkdir -p "$2"; printf "#!/bin/sh\\nexit 0\\n" > "$2/strata"; chmod +x "$2/strata"; exit 0\n'


def prepare_bu(box, aa_done=True):
    (box.S / "redo-chainAA.log").write_text("START\nWINCHAINAA_DONE\n" if aa_done else "START\nrunning\n")
    (box.tmp / "clients").write_text("0\n0\n0\n")
    pkg = box.tmp / "bu-src" / "x"
    (pkg / "engine").mkdir(parents=True)
    (pkg / "engine" / "CMakeLists.txt").write_text("project(x)\n")
    for rel, body in (("tools/pancake/abtest.py", "# stub\n"), ("tools/pancake/flag_diag.py", "# stub\n"),
                      ("tools/pancake/flag_diag.sh", FLAG_DIAG_STUB), ("tools/pancake/build_updated.sh", BUILD_STUB)):
        (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
        (pkg / rel).write_text(body, newline="\n")
    root = HERE.parents[1]
    for rel in ("bench/specs/220-updated-vs-upstream-short.json", "bench/specs/221-updated-vs-upstream-long.json",
                "configs/pancake/flashnext-iq2_xs-plain-upstream.json", "configs/pancake/flashnext-iq2_xs-updated-hip.json"):
        (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(root / rel, pkg / rel)
    (box.tmp / "fake-src.tgz").unlink()
    subprocess.run(["tar", "czf", (box.tmp / "fake-src.tgz").as_posix(), "-C", (box.tmp / "bu-src").as_posix(), "x"], check=True)
    (box.tmp / "stub" / "df").write_text('#!/bin/sh\necho "Filesystem 1K-blocks Used Available Use% Mounted"\necho "x 1 1 ${DF_FREE:-99999999} 1% /"\n', newline="\n")


def test_bench_updated_waits_for_chain_aa_then_runs_the_ordered_flow(tmp_path):
    box = Box(tmp_path)
    prepare_bu(box, aa_done=False)
    p = box.start(BU)
    time.sleep(1.5)
    assert p.poll() is None and len(box.log("bench-updatedAA.log")) == 1
    assert not any(c.startswith("SYSTEMCTL") for c in box.calls())
    append(box, "redo-chainAA.log", "WINCHAINAA_DONE")
    box.finish(p)
    lines = box.log("bench-updatedAA.log")
    text = "\n".join(lines)
    assert lines[-1] == "WINBENCHUS_DONE", lines[-8:]
    assert "ERROR" not in text and "ABORTED" not in text, text
    assert_start_end_pairs(lines)
    assert_service_discipline(box)
    for must in ("SYNC harness: origin port/rdna3-decode abc123", "BUILD_END plain upstream exit=0", "BUILD_END updated-strata exit=0",
                 "START diag plain", "START diag updated", "START short-edit", "START long", "SUMMARY spec220 exit=0, spec221 exit=0"):
        assert must in text, must
    order = ["chain AA is done", "service stopped", "SYNC harness", "BUILD_END plain upstream", "BUILD_END updated-strata", "START diag plain",
             "START diag updated", "START short-edit", "START long", "SUMMARY"]
    pos = [text.index(k) for k in order]
    assert pos == sorted(pos), order
    calls = box.calls()
    cm = [c for c in calls if c.startswith("cmake") and "-S" in c]
    assert cm and all((box.h + "/mnt/benchus") in c for c in cm), cm        # nothing is built on /
    assert "-DCMAKE_HIP_ARCHITECTURES=gfx1100" in cm[0] and "STRATA_BUILD_TESTS" not in cm[0]   # the plain build: only the documented flags
    ab = [c for c in calls if c.startswith("PY") and "abtest.py" in c]
    assert len(ab) == 2 and all("--reps 5" in c and "--no-early-stop" in c and "--tier confirm" in c for c in ab), ab
    assert "220-updated-vs-upstream-short.json" in ab[0] and "221-updated-vs-upstream-long.json" in ab[1]
    assert "generated text vs base" in text                                  # the text-identity result reaches the log
    assert (box.tmp / "auto-runs" / "winbenchus" / "tuned-flags-recommended.json").exists()
    assert (box.tmp / "mnt" / "benchus" / "build-up" / "strata").exists() and (box.tmp / "mnt" / "benchus" / "build-us" / "strata").exists()


def test_bench_updated_failed_plain_build_restores_the_old_binary_and_stops(tmp_path):
    box = Box(tmp_path)
    prepare_bu(box)
    old = box.tmp / "mnt" / "benchus" / "build-up"
    old.mkdir(parents=True)
    (old / "strata").write_text("#!/bin/sh\n# previous\n", newline="\n")
    box.finish(box.start(BU, env={"CMAKE_FAIL": "1"}))
    lines = box.log("bench-updatedAA.log")
    text = "\n".join(lines)
    assert any(l.startswith("ERROR plain upstream build exit=1") for l in lines)
    assert "previous strata-up restored" in text
    assert (old / "strata").read_text() == "#!/bin/sh\n# previous\n"
    assert lines[-1] == "WINBENCHUS_DONE" and lines[-3].startswith("WINBENCHUS_ABORTED")
    for never in ("BUILD updated-strata", "START diag", "START short-edit"):
        assert never not in text, never
    assert_service_discipline(box)


def test_bench_updated_failed_updated_build_stops_before_any_measurement(tmp_path):
    box = Box(tmp_path)
    prepare_bu(box)
    box.finish(box.start(BU, env={"BU_FAIL": "1"}))
    lines = box.log("bench-updatedAA.log")
    text = "\n".join(lines)
    assert any(l.startswith("ERROR updated-strata build exit=1") for l in lines)
    assert "BUILD_END plain upstream exit=0" in text and lines[-1] == "WINBENCHUS_DONE"
    for never in ("START diag", "START short-edit", "START long"):
        assert never not in text, never
    assert_service_discipline(box)


def test_bench_updated_stops_with_error_when_the_data_disk_is_nearly_full(tmp_path):
    box = Box(tmp_path)
    prepare_bu(box)
    box.finish(box.start(BU, env={"DF_FREE": "1000"}))
    lines = box.log("bench-updatedAA.log")
    assert any((l.startswith("ERROR START: ") and "has 1000 KB free" in l) for l in lines)
    assert lines[-1] == "WINBENCHUS_DONE" and "BUILD" not in "\n".join(lines)


def test_bench_updated_stops_when_a_download_fails(tmp_path):
    box = Box(tmp_path)
    prepare_bu(box)
    (box.tmp / "stub" / "curl").write_text("#!/bin/sh\nexit 22\n", newline="\n")
    box.finish(box.start(BU))
    lines = box.log("bench-updatedAA.log")
    assert any(l.startswith("ERROR SYNC harness download failed") for l in lines) and lines[-1] == "WINBENCHUS_DONE"
    assert_service_discipline(box)


def test_bench_updated_name_is_invisible_to_every_busy_pattern_and_sees_the_others(tmp_path):
    box = Box(tmp_path)
    assert "win-chain" not in BU
    for name in ("win-chainV.sh", "xtail-chainX.sh", "ytail-chainY.sh", "ztail-chainZ.sh", AA):
        assert busy_result(box, name, ["sh ./bench-updatedAA.sh"]) == "IDLE", name
    assert busy_result(box, BU, ["sh ./bench-updatedAA.sh"]) == "IDLE"
    for other in ("sh ./redo-chainAA.sh", "sh ./win-chainT.sh", "python3 tools/pancake/abtest.py x", "cmake --build x", "strata --serve"):
        assert busy_result(box, BU, [other]) == "BUSY", other


def test_updated_vs_upstream_specs_match_the_preset():
    root = HERE.parents[1]
    plain = json.loads((root / "configs/pancake/flashnext-iq2_xs-plain-upstream.json").read_text())
    preset = json.loads((root / "configs/pancake/flashnext-iq2_xs-updated-hip.json").read_text())
    for n in ("220-updated-vs-upstream-short", "221-updated-vs-upstream-long"):
        spec = json.loads((root / f"bench/specs/{n}.json").read_text())
        arm = spec["arms"]["updated"]
        assert spec["arms"]["plain-upstream"]["exe"] != arm["exe"] and list(spec["arms"])[0] == "plain-upstream"
        assert arm["env"] == preset["env"]
        pa = preset["args"]
        got = {pa[i]: pa[i + 1] for i in range(0, len(pa), 2)}
        base = {plain["args"][i]: plain["args"][i + 1] for i in range(0, len(plain["args"]), 2)}
        assert arm["args"] == {k: v for k, v in got.items() if k not in base}
    assert plain["env"] == {}


# ------------------------------------------------------------------------------------------------ bench-updatedRD.sh
BD = "bench-updatedRD.sh"
CMAKE_ALL_TARGETS = ('echo "cmake $*" >> "$H/calls.log"\n'
                     'b=; mode=; bm=; ts=; while [ $# -gt 0 ]; do case $1 in --build) b=$2; bm=1;; -B) b=$2;; --target) mode=t;; '
                     '-*) ;; *) [ "$mode" = t ] && ts="$ts $1";; esac; shift; done\n'
                     '[ "$CMAKE_FAIL" = 1 ] && case "$ts" in *strata*) rm -f "$b/strata"; exit 1;; esac\n'
                     "[ \"$bm\" = 1 ] && mkdir -p \"$b\" && for t in $ts; do [ -e \"$b/$t\" ] || { printf '#!/bin/sh\nexit 0\n' > \"$b/$t\"; chmod +x \"$b/$t\"; }; done\nexit 0\n")


def prepare_bd(box, markers=True):
    """The chain waits for chain AA and the upstream benchmark; the fake codeload tarball carries what the three synced trees need."""
    log = box.S / "redo-chainAA.log"
    log.write_text("WINCHAINAA_DONE\n" if markers else "START\n")
    (box.S / "bench-updatedAA.log").write_text("WINBENCHUS_DONE\n" if markers else "START\n")
    (box.tmp / "clients").write_text("0\n0\n0\n")
    (box.tmp / "stub" / "cmake").write_text("#!/bin/sh\n" + CMAKE_ALL_TARGETS, newline="\n")
    tree = box.tmp / "srcdir" / "pancake-strata-abc"
    for rel in ("tools/pancake/box_chains", "tools/pancake", "bench/specs", "rdna3/configs"):
        (tree / rel).mkdir(parents=True, exist_ok=True)
    (tree / "tools/pancake/build_updated_rdna3.sh").write_text(
        "#!/bin/sh\ncmake -S $1 -B $2\ncmake --build $2 -j8 --target strata verify_ecap_test sampler_parity elementwise_parity\n", newline="\n")
    (tree / "tools/pancake/abtest.py").write_text("# stub\n")
    (tree / "tools/pancake/flag_diag.py").write_text("# stub\n")
    shutil.copy(CHAINS / "bench_updatedRD_conf.py", tree / "tools/pancake/box_chains/")
    shutil.copy(HERE.parent.parent / "bench" / "specs" / "250-rdna3-updated-vs-plain.json", tree / "bench/specs/")
    subprocess.run(["tar", "czf", (box.tmp / "fake-src.tgz").as_posix(), "-C", (box.tmp / "srcdir").as_posix(), "pancake-strata-abc"], check=True)


def test_bd_waits_for_both_markers_then_runs_in_the_ordered_flow(tmp_path):
    box = Box(tmp_path)
    prepare_bd(box, markers=False)
    p = box.start(BD)
    time.sleep(1.5)
    assert p.poll() is None and len(box.log("bench-updatedRD.log")) == 1
    append(box, "redo-chainAA.log", "WINCHAINAA_DONE")
    time.sleep(0.8)
    assert p.poll() is None and len(box.log("bench-updatedRD.log")) == 1, "chain AA alone is not enough: the upstream benchmark must be done too"
    assert not any(c.startswith("SYSTEMCTL") for c in box.calls())
    append(box, "bench-updatedAA.log", "WINBENCHUS_DONE")
    box.finish(p)
    lines = box.log("bench-updatedRD.log")
    text = "\n".join(lines)
    assert lines[-1] == "WINBENCHRD_DONE", lines[-8:]
    assert "ERROR" not in text and "ABORTED" not in text
    assert_start_end_pairs(lines)
    assert_service_discipline(box)
    for must in ("SYNC nicejunjie/Strata-rdna3 8b01b0c8512d388552e8601b3aff6753d05286e0", "SYNC rocktboy1982/pancake-strata updated-strata-rdna3",
                 "SYNC rocktboy1982/pancake-strata port/rdna3-decode", "BUILD_END plain exit=0", "BUILD_END updated exit=0", "ENGINE plain", "ENGINE updated",
                 "END ctest exit=0", "CONFIG written", "START diag-updated", "START diag-plain", "START 250-screen", "START 250-confirm"):
        assert must in text, must
    order = ["BUILD_END plain exit=0", "BUILD_END updated exit=0", "START ctest", "CONFIG written", "START diag-updated", "START diag-plain",
             "START 250-screen", "START 250-confirm"]
    pos = [text.index(k) for k in order]
    assert pos == sorted(pos), order
    calls = box.calls()
    assert any("--tier confirm" in c and "--block 3" in c and "--reps 5" in c for c in calls if c.startswith("PY")), "5-rep blocked confirm"
    assert any(c.startswith("cmake") and "benchRD/build-plain" in c for c in calls)
    spec = (box.tmp / "auto-runs" / "benchRD" / "250.json").read_text()
    assert "@PLAIN@" not in spec and "@UPDATED@" not in spec and "build-plain/strata" in spec and "build-updated/strata" in spec
    assert str(box.tmp / "mnt") in spec or "/mnt/" in spec, "arms point at /mnt/pancake builds"


def test_bd_failed_build_restores_the_binary_logs_error_and_stops(tmp_path):
    box = Box(tmp_path)
    prepare_bd(box)
    bp = box.tmp / "mnt" / "benchRD" / "build-plain"
    bp.mkdir(parents=True)
    (bp / "strata").write_text("#!/bin/sh\n# the previous working binary\n", newline="\n")
    before = (bp / "strata").read_text()
    (bp / "strata").chmod(0o755)
    box.finish(box.start(BD, env={"CMAKE_FAIL": "1"}))
    lines = box.log("bench-updatedRD.log")
    text = "\n".join(lines)
    assert "BUILD_END plain exit=1" in text and "BUILD FAILED: the previous" in text
    assert any(l.startswith("ERROR BUILD plain FAILED") for l in lines)
    assert lines[-1] == "WINBENCHRD_DONE" and lines[-3].startswith("WINBENCHRD_ABORTED") and lines[-2].startswith("service started")
    assert (bp / "strata").read_text() == before
    for never in ("BUILD_END updated", "START ctest", "START diag", "START 250", "CONFIG written"):
        assert never not in text, never
    assert_service_discipline(box)


def test_bd_stops_with_error_when_mnt_pancake_is_nearly_full(tmp_path):
    box = Box(tmp_path)
    prepare_bd(box)
    box.finish(box.start(BD, transform=lambda t: t.replace("-ge 2097152", "-ge 99999999999")))
    lines = box.log("bench-updatedRD.log")
    assert any(l.startswith("ERROR SYNC:") and "< 2 GB free" in l for l in lines) and lines[-1] == "WINBENCHRD_DONE"
    assert "BUILD" not in "\n".join(lines)
    assert_service_discipline(box)


def test_bd_stops_with_error_when_a_download_fails(tmp_path):
    box = Box(tmp_path)
    prepare_bd(box)
    (box.tmp / "fake-src.tgz").unlink()
    box.finish(box.start(BD))
    lines = box.log("bench-updatedRD.log")
    assert any(l.startswith("ERROR SYNC: download of") for l in lines) and lines[-1] == "WINBENCHRD_DONE"
    assert "BUILD" not in "\n".join(lines)
    assert_service_discipline(box)


def test_bd_name_is_invisible_to_every_busy_pattern_and_sees_the_others(tmp_path):
    box = Box(tmp_path)
    assert "win-chain" not in BD
    for name in ("win-chainV.sh", "xtail-chainX.sh", "ytail-chainY.sh", "ztail-chainZ.sh", AA):
        assert busy_result(box, name, ["sh ./bench-updatedRD.sh"]) == "IDLE", name
    assert busy_result(box, BD, ["sh ./bench-updatedRD.sh"]) == "IDLE"
    for other in ("sh ./win-chainT.sh", "sh ./xtail-chainX.sh", "sh ./redo-chainAA.sh", "/bin/sh ./after7.sh", "python3 tools/pancake/abtest.py x",
                  "cmake --build x"):
        assert busy_result(box, BD, [other]) == "BUSY", other


def test_bd_spec_and_conf(tmp_path):
    spec = json.loads((HERE.parent.parent / "bench" / "specs" / "250-rdna3-updated-vs-plain.json").read_text())
    assert set(spec["arms"]) == {"plain", "updated", "updated-ecap6"} and spec["cases"] == ["short", "edit", "long10k", "long32k", "long64k", "long100k"]
    spec_ = importlib.util.spec_from_file_location("bd_conf", CHAINS / "bench_updatedRD_conf.py")
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)

    class TF:   # a fake rdna3/tuned_flags.py
        @staticmethod
        def load_recommended(hw, d):
            return {"env": {"STRATA_VERIFY_ECAP": "0"}, "args": {}}, "gfx1100.json"

        @staticmethod
        def apply_to_config(cfg, rec, env, root):
            cfg["env"].update(rec["env"])

    (tmp_path / "rdna3" / "configs").mkdir(parents=True)
    (tmp_path / "rdna3" / "configs" / "iq3-1gpu.json").write_text(json.dumps({"env": {"STRATA_PF_F16": "2", "X": "@STRATA@/t"}}))
    preset = tmp_path / "preset.json"
    preset.write_text(json.dumps({"exe": "e", "args": ["--spec", "4", "--grammar", "on", "--lookup-max-t", "12", "--kv", "int8"],
                                  "env": {"STRATA_VERIFY_ECAP": "1"}, "grammar": True, "port": 8080}))
    plain, upd, path = mod.build(str(preset), str(tmp_path), "/u", "/b/plain", "/b/upd", TF)
    assert plain["args"] == ["--spec", "4", "--kv", "int8"] and "grammar" not in plain
    assert plain["env"] == {"STRATA_PF_F16": "2", "X": str(tmp_path) + "/t"}      # the plain build runs with the env of Strata-rdna3's own setup config
    assert plain["exe"] == "/b/plain/strata" and upd["exe"] == "/b/upd/strata"
    assert upd["env"]["STRATA_VERIFY_ECAP"] == "0" and "STRATA_VERIFY_ECAP" not in plain["env"]
    rec = HERE.parent.parent / "rdna3" / "recommended-flags" / "gfx1100.json"
    if rec.exists():       # on the updated-strata-rdna3 branch: the spec's updated arm is exactly the shipped file
        assert spec["arms"]["updated"]["env"] == json.loads(rec.read_text())["env"]
