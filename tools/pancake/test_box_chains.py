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
if args and args[0] == "-u":        # python3 -u (unbuffered: a killed diagnostic still leaves its stdout)
    args = args[1:]
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
if a0 == "-c":                      # inline python of the chains (spec edits, the recommended-flags writer): really run it
    sys.argv = ["-c"] + args[2:]
    exec(compile(args[1], "<c>", "exec"), {"__name__": "__main__"})
    sys.exit(0)
name = os.path.basename(a0)
if name == "flag_diag.py":
    if "run" in args:
        out = opt("--out"); os.makedirs(out, exist_ok=True)
        open(out + "/verdicts.json", "w").write("{}")
        print("POOL_FUSE_PHASES GAIN")
        if os.environ.get("STUB_DIAG_LEFTOVER"):   # the 2026-11-19 failure: the diagnostic ends, a child of its session keeps running
            open(H + "/sess.txt", "w").write(f"999991 @SID@ @SID@ python3 tools/pancake/abtest.py {out}/spec-base.json\n")
        if os.environ.get("STUB_DIAG_EXIT"):
            sys.exit(int(os.environ["STUB_DIAG_EXIT"]))
    elif "--hardware" in args:
        print('{"gfx": "gfx1100"}')
    sys.exit(0)
if name == "abtest.py":
    spec = json.load(open(args[1]))
    arms = list(spec["arms"])
    out = opt("--out")
    die = [d for d in os.environ.get("STUB_DIE", "").split(",") if d]
    if any(d in joined for d in die):      # the real abtest: "server died before ready", the cause only in the engine log of the attempt
        wd = out.rsplit(".", 1)[0]; os.makedirs(wd, exist_ok=True)
        open(wd + "/" + arms[0] + "-r0-a0.engine.log", "w").write("[strata] engine started: x --serve\nstrata generate: memory guard: ok\n"
            "strata generate: /p/iq2_xs is a native (IQ) pack: it needs --native SHARD1, --spec T (T >= 2) and --prefill CHUNK\n")
        print("rep 0 " + arms[0] + ": FAILED (server died before ready)")
        print("ABORT: the server died twice before ready (fix the machine first)")
        sys.exit(2)
    if out:
        open(out, "a").write(json.dumps({"arm": arms[0], "rows": [{"case": "short", "decode_tps": 12.5, "prompt_tps": 100.0}]}) + "\n")
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
    "pgrep": ('echo "pgrep $*" >> "$H/calls.log" 2>/dev/null; [ "$1" = -f ] || exit 2; m=$(grep -E -- "$2" "$H/procs.txt") || exit 1\n'
              'echo "$m" | awk \'$1 ~ /^[0-9]+$/ {print $1}\'; exit 0\n'),   # a line "PID ARGS" also prints its pid (aa_live, the scheduler)
    "ps": ('case "$*" in\n'
           '  "-eo pid=,args=") cat "$H/ps.txt" 2>/dev/null ;;\n'
           '  "-eo pid=,sid=,pgid=,args=") [ -f "$H/sess.txt" ] && sed "s/@SID@/$(cat "$H/sid.txt" 2>/dev/null)/g" "$H/sess.txt" ;;\n'
           '  "-o args= -p "*) grep -E "^ *$4 " "$H/ps.txt" | sed -E "s/^ *[0-9]+ +//" ;;\n'
           '  "-o stat= -p "*) cat "$H/stat.$4" 2>/dev/null ;;\n'
           'esac\nexit 0\n'),
    "sigcmd": ('echo "SIG $*" >> "$H/calls.log"\n'
               'case $1 in -STOP) echo T > "$H/stat.$2" ;; -CONT) echo S > "$H/stat.$2" ;; esac\n'),
    "ss": ('n=$(head -n1 "$H/clients" 2>/dev/null); [ -n "$n" ] || n=0; sed -i 1d "$H/clients" 2>/dev/null; '
           'echo "SS $n" >> "$H/calls.log"; i=0; while [ $i -lt $n ]; do echo ESTAB; i=$((i+1)); done\n'),
    "systemctl": 'echo "SYSTEMCTL $*" >> "$H/calls.log"\ncase "$*" in *is-active*) [ -n "$SVC_DOWN" ] && exit 3 ;; esac\nexit 0\n',
    "sudo": 'cat > /dev/null\n',
    "git": 'printf "abc123\\trefs/heads/port/rdna3-decode\\n"\n',
    "curl": 'while [ $# -gt 0 ]; do [ "$1" = -o ] && out=$2; shift; done; cp "$H/fake-src.tgz" "$out"\n',
    "timeout": ('echo "TIMEOUT $*" >> "$H/calls.log"\n'
                'while :; do case $1 in -k|-s) shift 2 ;; -*) shift ;; *) break ;; esac; done; shift; exec "$@"\n'),
    "setsid": 'echo "SETSID $$ $*" >> "$H/calls.log"; echo $$ > "$H/sid.txt"; exec "$@"\n',
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
        for rel in ("rocm/bin/rocminfo", "rocm/bin/hipcc", "rocm/lib/llvm/bin/clang++", "rocm/include/hip/hip_runtime.h",
                    "clone-of-d/src/Strata/.venv/bin/ninja", "clone-of-d/Strata-data/packs/iq2_xs/tokenizer",
                    "clone-of-d/Strata-data/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf",
                    "clone-of-d/Strata-data/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf",
                    "clone-of-d/Strata-data/mtp/rt", "clone-of-d/build-int/_deps/strata_llamacpp-src/x"):
            (H / rel).parent.mkdir(parents=True, exist_ok=True)
            (H / rel).write_text('#!/bin/sh\n[ "$ROCMINFO_EMPTY" = 1 ] || echo "  Name:                    gfx1100"\n' if rel.endswith("rocminfo") else "#!/bin/sh\nexit 0\n", newline="\n")
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
                  "PATH": str(self.tmp / "stub") + os.pathsep + e["PATH"], "STUB_REPLAY": "worse",
                  "SIGCMD": self.h + "/stub/sigcmd"})
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
  run) [ "$FD_EMPTY" = 1 ] && { echo "flag_diag: no AMD graphics card could be read (rocminfo / sysfs)"; exit 1; }
       mkdir -p "$3"; echo "{\"verdicts\": []}" > "$3/verdicts.json"; echo "GAIN STRATA_RDNA_Q6K"; echo "15.0 minutes; verdicts in $3/verdicts.json"
       echo "FLAG_DIAG $* abtest=$FLAG_DIAG_ABTEST" >> "$H/calls.log"
       case "$3" in *diag-plain) [ "$FD_LEFTOVER" = 1 ] && echo "999992 @SID@ @SID@ python3 serve/server.py --engine strata --config $3/x.json" > "$H/sess.txt" ;;
                    *) rm -f "$H/sess.txt" ;; esac
       [ -n "$FD_EXIT" ] && exit "$FD_EXIT" ;;
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
                      ("tools/pancake/flag_diag.sh", FLAG_DIAG_STUB), ("tools/pancake/build_updated.sh", BUILD_STUB),
                      ("tools/pancake/pcie_link.py", "# stub\n"), ("tools/pancake/host_tune.sh", "#!/bin/sh\necho \"host_tune $*\"\n"),
                      ("tools/pancake/host_tune_ab.sh", "#!/bin/sh\necho \"ab $*\"\n"), ("tools/pancake/os_tune.sh", "#!/bin/sh\necho \"os_tune $*\"\n")):
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
                 "START diag plain", "START diag updated", "START 220-screen", "START 220-confirm", "START 221-screen", "START 221-confirm",
                 "TUNEUP pcie_link", "TUNEUP host_tune --check", "TUNEUP host_tune_ab plan", "TUNEUP os_tune check", "SUMMARY spec220 exit=0, spec221 exit=0",
                 "PREREQ ok", "START smoke plain", "SMOKE plain ok", "START smoke updated", "SMOKE updated ok"):
        assert must in text, must
    order = ["chain AA is done", "service stopped", "PREREQ ok", "SYNC harness", "BUILD_END plain upstream", "START smoke plain", "SMOKE plain ok",
             "BUILD updated-strata", "BUILD_END updated-strata", "START smoke updated", "SMOKE updated ok", "START diag plain",
             "START diag updated", "START 220-screen", "START 220-confirm", "START 221-screen", "SUMMARY"]
    pos = [text.index(k) for k in order]
    assert pos == sorted(pos), order
    calls = box.calls()
    cm = [c for c in calls if c.startswith("cmake") and "-S" in c]
    assert cm and all((box.h + "/mnt/benchus") in c for c in cm), cm        # nothing is built on /
    assert "-DCMAKE_HIP_ARCHITECTURES=gfx1100" in cm[0] and "STRATA_BUILD_TESTS" not in cm[0]   # the plain build: only the documented flags
    smokes = [c for c in calls if c.startswith("PY") and "abtest.py" in c and "smoke-" in c]
    assert len(smokes) == 2 and all("--tier confirm" in c and "--reps 1" in c and "--retries 0" in c for c in smokes), smokes   # the prebuild smokes: one start, one request
    sp = json.loads((box.tmp / "auto-runs" / "winbenchus" / "smoke-updated" / "spec.json").read_text())
    assert sp["cases"] == ["short"] and sp["reps"] == 1 and list(sp["arms"]) == ["smoke"] and "STRATA_GR_RDNA" in sp["arms"]["smoke"]["env"]   # the benchmark's own updated arm
    assert json.loads((box.tmp / "auto-runs" / "winbenchus" / "smoke-plain" / "spec.json").read_text())["arms"]["smoke"]["exe"].endswith("build-up/strata")
    ab = [c for c in calls if c.startswith("PY") and "abtest.py" in c and "smoke-" not in c]
    assert len(ab) == 4, ab                                                   # 220 screen, 220 confirm, 221 screen, 221 confirm
    assert "220-updated-vs-upstream-short.json" in ab[0] and "--tier screen" in ab[0] and "--drift-check" in ab[0]
    assert "220-confirm.json" in ab[1] and "--tier confirm" in ab[1] and "--reps 3" in ab[1] and "--block 3" in ab[1] and "--no-early-stop" in ab[1]
    assert "221-screen.json" in ab[2] and "--tier screen" in ab[2] and "--drift-check" not in ab[2]
    assert "221-confirm.json" in ab[3] and "--reps 2" in ab[3]
    assert '"drift_check": false' in (box.tmp / "auto-runs" / "winbenchus" / "work" / "221-screen.json").read_text()
    assert any("-j8" in c for c in calls if c.startswith("cmake --build"))
    assert (box.tmp / "auto-runs" / "winbenchus" / "tuneup" / "pcie_link.txt").exists()
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
    assert lines[-1] == "WINBENCHUS_DONE" and any(l.startswith("WINBENCHUS_ABORTED") for l in lines[-4:])
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
    assert any(l.startswith("BUILD ERROR LINES (updated-strata") for l in lines)   # the compiler's error lines reach the chain log
    assert any("undefined reference" in l for l in lines)
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


def test_bench_updated_prereq_names_every_missing_file_before_anything_is_downloaded(tmp_path):
    box = Box(tmp_path)
    prepare_bu(box)
    (box.tmp / "clone-of-d" / "Strata-data" / "mtp" / "rt").unlink()
    (box.tmp / "rocm" / "bin" / "rocminfo").unlink()
    box.finish(box.start(BU))
    lines = box.log("bench-updatedAA.log")
    err = [l for l in lines if l.startswith("ERROR PREREQ missing files:")]
    assert err and "mtp/rt" in err[0] and "rocm/bin/rocminfo" in err[0], lines
    text = "\n".join(lines)
    assert lines[-1] == "WINBENCHUS_DONE" and "SYNC" not in text and "BUILD" not in text, text
    assert_service_discipline(box)


def test_bench_updated_prereq_stops_when_rocminfo_sees_no_gpu(tmp_path):
    box = Box(tmp_path)
    prepare_bu(box)
    box.finish(box.start(BU, env={"ROCMINFO_EMPTY": "1"}))
    lines = box.log("bench-updatedAA.log")
    assert any(l.startswith("ERROR PREREQ rocminfo lists no gfx device") for l in lines), lines
    assert lines[-1] == "WINBENCHUS_DONE" and "BUILD" not in "\n".join(lines)


def test_bench_updated_puts_rocm_bin_first_on_path(tmp_path):
    """The 13:33 failure: flag_diag.py runs `rocminfo` through PATH and the chain's PATH had no ROCm bin."""
    text = (CHAINS / BU).read_text()
    assert "export PATH=$ROCM/bin:" in text
    assert "export PATH=$ROCM/bin:" in (CHAINS / BD).read_text()


def test_bench_updated_smoke_failure_stops_within_the_prebuild_phase_with_the_engine_error(tmp_path):
    box = Box(tmp_path)
    prepare_bu(box)
    box.finish(box.start(BU, env={"STUB_DIE": "smoke-plain"}))
    lines = box.log("bench-updatedAA.log")
    text = "\n".join(lines)
    err = [l for l in lines if l.startswith("ERROR smoke plain FAILED exit=2")]
    assert err, lines
    assert "server died before ready" in err[0] and "it needs --native SHARD1, --spec T (T >= 2) and --prefill CHUNK" in err[0], err[0]   # stdout line + the engine's own line
    assert lines[-1] == "WINBENCHUS_DONE" and any(l.startswith("WINBENCHUS_ABORTED") for l in lines[-4:])
    for never in ("BUILD updated-strata", "START diag", "START 220", "START 221", "SMOKE plain ok"):
        assert never not in text, never
    assert_start_end_pairs(lines)
    assert_service_discipline(box)


def test_bench_updated_smoke_of_the_updated_build_stops_before_diagnostics(tmp_path):
    box = Box(tmp_path)
    prepare_bu(box)
    box.finish(box.start(BU, env={"STUB_DIE": "smoke-updated"}))
    lines = box.log("bench-updatedAA.log")
    text = "\n".join(lines)
    assert "SMOKE plain ok" in text and any(l.startswith("ERROR smoke updated FAILED") for l in lines), lines
    for never in ("START diag", "START 220", "BUILD sizes"):
        assert never not in text, never
    assert lines[-1] == "WINBENCHUS_DONE"


def test_bench_updated_diag_runs_in_its_own_session_with_limits_sized_from_the_per_flag_timeout(tmp_path):
    """2026-11-19 14:59: `timeout 1080` ended flag_diag after 18 min for 2 flags and left its abtest / server / engine running.  Now each
    diagnostic runs under setsid + timeout --signal=TERM --kill-after=100, its limit and budget come from the per-arm limit (240 s)."""
    box = Box(tmp_path)
    prepare_bu(box)
    box.finish(box.start(BU))
    lines = box.log("bench-updatedAA.log")
    text = "\n".join(lines)
    assert lines[-1] == "WINBENCHUS_DONE" and "DIAG-REAP" not in text and "ERROR" not in text, text
    to = diag_timeout_call(box, "flag_diag.sh")
    assert len(to) == 2 and all("--signal=TERM" in c and "--kill-after=100" in c for c in to), to
    assert [int(c.split()[3]) for c in to] == [3 * (240 + 90) + 180, 6 * (240 + 90) + 180]
    fd = [c for c in box.calls() if c.startswith("FLAG_DIAG run")]
    assert len(fd) == 2 and "--budget-min 13" in fd[0] and "--budget-min 25" in fd[1] and all("--flag-timeout 240" in c for c in fd), fd
    assert all("abtest=" in c and c.endswith("tools/pancake/abtest.py") for c in fd), fd     # FLAG_DIAG_ABTEST reaches flag_diag through the function
    assert sum(1 for c in box.calls() if c.startswith("SETSID") and "flag_diag.sh" in c) == 2


def test_bench_updated_diag_leftovers_are_reaped_before_the_next_diagnostic(tmp_path):
    """The exact 14:59 sequence: diag plain ends by its limit (124) with a server of its session still alive; the chain kills that one
    process (logged) before diag updated starts, so diag updated's abtest is not refused."""
    box = Box(tmp_path)
    prepare_bu(box)
    box.finish(box.start(BU, env={"FD_LEFTOVER": "1", "FD_EXIT": "124"}))
    lines = box.log("bench-updatedAA.log")
    text = "\n".join(lines)
    reap = [l for l in lines if l.startswith("DIAG-REAP")]
    assert len(reap) == 1 and "999992" in reap[0] and "serve/server.py" in reap[0], lines
    assert text.index("END diag plain exit=124") > text.index("START diag plain")
    assert text.index("DIAG-REAP") < text.index("END diag plain") < text.index("START diag updated") < text.index("START 220-screen")
    assert lines[-1] == "WINBENCHUS_DONE"


def test_bench_updated_empty_verdicts_is_an_error_with_the_underlying_message(tmp_path):
    box = Box(tmp_path)
    prepare_bu(box)
    box.finish(box.start(BU, env={"FD_EMPTY": "1"}))
    lines = box.log("bench-updatedAA.log")
    err = [l for l in lines if l.startswith("ERROR diag plain wrote no verdicts.json")]
    assert err and "flag_diag: no AMD graphics card could be read" in err[0], lines
    assert any(l.startswith("ERROR diag updated wrote no verdicts.json") for l in lines)
    assert lines[-1] == "WINBENCHUS_DONE" and "START 220-screen" in "\n".join(lines)     # the diagnostics are a cheap first step: the benchmark goes on


def test_bench_updated_screen_exit_2_logs_the_first_error_lines(tmp_path):
    box = Box(tmp_path)
    prepare_bu(box)
    box.finish(box.start(BU, env={"STUB_DIE": "220-updated-vs-upstream-short.json"}))
    lines = box.log("bench-updatedAA.log")
    err = [l for l in lines if l.startswith("ERROR 220-screen exit=2:")]
    assert err, lines
    assert "first error line of the stdout: [rep 0 plain-upstream: FAILED (server died before ready)]" in err[0], err[0]
    assert "it needs --native SHARD1" in err[0], err[0]                 # the engine log of the failed start says why
    assert any(l.startswith("ERROR spec 220 screen exit=2") for l in lines)
    assert "START 221-screen" in "\n".join(lines) and lines[-1] == "WINBENCHUS_DONE"


# ------------------------------------------------------------------------------------------------ bench-diagUS.sh (diag-only rerun of the US builds)
DU = "bench-diagUS.sh"


def prepare_du(box):
    prepare_bu(box)
    pkg = box.tmp / "bu-src" / "x"
    (pkg / "tools/pancake/flag_diag.py").write_text("# stub of the fixed version: --flag-timeout\n")
    (pkg / "tools/pancake/diag_group.py").write_text("# stub\n")
    (box.tmp / "fake-src.tgz").unlink()
    subprocess.run(["tar", "czf", (box.tmp / "fake-src.tgz").as_posix(), "-C", (box.tmp / "bu-src").as_posix(), "x"], check=True)
    for b in ("build-up", "build-us"):
        (box.tmp / "mnt" / "benchus" / b).mkdir(parents=True)
        (box.tmp / "mnt" / "benchus" / b / "strata").write_text("#!/bin/sh\nexit 0\n", newline="\n")
    (box.R / "configs" / "pancake").mkdir(parents=True)
    shutil.copy(HERE.parents[1] / "configs/pancake/flashnext-iq2_xs-plain-upstream.json", box.R / "configs/pancake/")
    (box.R / "tools" / "pancake" / "abtest.py").write_text("# stub\n")
    (box.S / "bench-updatedRD.log").write_text("START\nWINBENCHRD_DONE\n")


def test_diag_us_runs_only_the_two_fixed_diagnostics_on_the_existing_builds(tmp_path):
    box = Box(tmp_path)
    prepare_du(box)
    box.finish(box.start(DU, env={"BENCH_SCHED": "1", "BENCH_PREIDLED": "1", "BENCH_KEEP_SERVICE_DOWN": "1", "SVC_DOWN": "1"}))
    lines = box.log("bench-diagUS.log")
    text = "\n".join(lines)
    assert lines[-1] == "WINDIAGUS_DONE" and "ERROR" not in text and "ABORTED" not in text, lines
    assert_start_end_pairs(lines)
    calls = box.calls()
    assert not any(c.startswith("cmake") for c in calls), "builds nothing"
    assert not any(c.startswith("SIG") for c in calls), "never touches the (frozen) AA shell"
    assert not any(c.startswith("SYSTEMCTL") and "start" in c for c in calls), "service stays down"
    fd = [c for c in calls if c.startswith("FLAG_DIAG run")]
    assert len(fd) == 2 and "diag-rerun/diag-plain" in fd[0] and "diag-rerun/diag-updated" in fd[1], fd
    assert "--only STRATA_HIPBLASLT_TUNING,STRATA_PF_F16" in fd[0] and "STRATA_VERIFY_ECAP" in fd[1]
    assert all("--flag-timeout 240" in c for c in fd)
    cfgs = [json.loads((box.tmp / "auto-runs" / "winbenchus" / "diag-rerun" / f"diag-{n}-config.json").read_text())["exe"] for n in ("plain", "updated")]
    assert cfgs[0].endswith("benchus/build-up/strata") and cfgs[1].endswith("benchus/build-us/strata"), cfgs
    to = diag_timeout_call(box, "flag_diag.sh")
    assert [int(c.split()[3]) for c in to] == [3 * 330 + 180, 6 * 330 + 180] and all("--kill-after=100" in c for c in to)
    assert all("src-us-diag/tools/pancake/flag_diag.sh" in c for c in to), "the fixed flag_diag from its own folder (src-us untouched)"
    assert (box.tmp / "auto-runs" / "winbenchus" / "diag-rerun" / "tuned-flags-recommended.json").exists()
    assert text.index("START diag plain") < text.index("END diag plain") < text.index("START diag updated")


def test_diag_us_standalone_waits_for_the_rd_marker_and_reaps_leftovers(tmp_path):
    box = Box(tmp_path)
    prepare_du(box)
    (box.S / "bench-updatedRD.log").write_text("START\n")
    p = box.start(DU, env={"BENCH_PREIDLED": "1", "SVC_DOWN": "1", "FD_LEFTOVER": "1"})
    time.sleep(1.5)
    assert p.poll() is None and len(box.log("bench-diagUS.log")) == 1, "waits for WINBENCHRD_DONE"
    append(box, "bench-updatedRD.log", "WINBENCHRD_DONE")
    box.finish(p)
    lines = box.log("bench-diagUS.log")
    text = "\n".join(lines)
    assert lines[-1] == "WINDIAGUS_DONE"
    assert any(l.startswith("DIAG-REAP") and "999992" in l for l in lines), lines
    assert text.index("DIAG-REAP") < text.index("START diag updated")


def test_diag_us_stops_when_a_build_is_missing(tmp_path):
    box = Box(tmp_path)
    prepare_du(box)
    (box.tmp / "mnt" / "benchus" / "build-us" / "strata").unlink()
    box.finish(box.start(DU, env={"BENCH_SCHED": "1", "BENCH_PREIDLED": "1", "SVC_DOWN": "1"}))
    lines = box.log("bench-diagUS.log")
    ab = [l for l in lines if l.startswith("WINDIAGUS_ABORTED")]
    assert ab and "build-us/strata" in ab[0] and lines[-1] == "WINDIAGUS_DONE", lines
    assert not any(c.startswith("FLAG_DIAG") for c in box.calls())


def test_diag_us_name_is_invisible_to_every_busy_pattern(tmp_path):
    box = Box(tmp_path)
    for name in ("win-chainV.sh", "xtail-chainX.sh", "ytail-chainY.sh", "ztail-chainZ.sh", AA, BU, BD):
        assert busy_result(box, name, ["sh ./bench-diagUS.sh"]) == "IDLE", name
    assert busy_result(box, DU, ["sh ./bench-updatedRD.v4.sh"]) == "BUSY"      # it never overlaps a running benchmark chain
    assert busy_result(box, DU, ["4242 sh ./redo-chainAA.sh"]) == "IDLE"


def test_bench_updated_name_is_invisible_to_every_busy_pattern_and_sees_the_others(tmp_path):
    box = Box(tmp_path)
    assert "win-chain" not in BU
    for name in ("win-chainV.sh", "xtail-chainX.sh", "ytail-chainY.sh", "ztail-chainZ.sh", AA):
        assert busy_result(box, name, ["sh ./bench-updatedAA.sh"]) == "IDLE", name
    assert busy_result(box, BU, ["sh ./bench-updatedAA.sh"]) == "IDLE"
    assert busy_result(box, BU, ["4242 sh ./redo-chainAA.sh"]) == "IDLE"           # a (frozen) AA shell is not a job for busy(); aa_live() judges its state
    for other in ("sh ./win-chainT.sh", "python3 tools/pancake/abtest.py x", "cmake --build x", "strata --serve"):
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
        assert arm["args"] == {k: v for k, v in got.items() if k not in base or base[k] != v}   # extras, plus what differs (--expert-profile)
    assert plain["env"] == {}
    pa = plain["args"]
    flags = {pa[i]: pa[i + 1] for i in range(0, len(pa), 2)}
    # v0.2.16: "a native (IQ) pack needs --native SHARD1, --spec T (T >= 2) and --prefill CHUNK" (generate.cpp): without them the engine exits at once
    assert int(flags["--spec"]) >= 2 and flags["--prefill"] and flags["--native"].endswith(".gguf")
    # v0.2.16 refuses --spec verify without a profile-filled expert tier: the plain arm uses upstream's own profile
    assert flags["--expert-profile"].endswith("terminal/data/expert-profile.bin")


# ------------------------------------------------------------------------------------------------ bench-updatedRD.sh
BD = "bench-updatedRD.sh"
CMAKE_ALL_TARGETS = ('echo "cmake $*" >> "$H/calls.log"; echo "CPI $CPLUS_INCLUDE_PATH" >> "$H/calls.log"\n'
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
    for rel, body in (("tools/pancake/pcie_link.py", "# stub\n"), ("tools/pancake/host_tune.sh", "#!/bin/sh\necho host_tune\n"),
                      ("tools/pancake/host_tune_ab.sh", "#!/bin/sh\necho ab\n"), ("tools/pancake/os_tune.sh", "#!/bin/sh\necho os\n"),
                      ("rdna3/tuned_flags.py", "import os\ndef detect_hardware():\n    return {'gfx': 'gfx1100'}\n"
                       "def write_recommended(hw, v, d):\n    os.makedirs(d, exist_ok=True)\n    open(d + '/gfx1100.json', 'w').write('{}')\n    return d + '/gfx1100.json'\n")):
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_text(body, newline="\n")
    (tree / "tools/pancake/build_updated_rdna3.sh").write_text(
        "#!/bin/sh\ncmake -S $1 -B $2\ncmake --build $2 -j8 --target strata verify_ecap_test sampler_parity elementwise_parity\n", newline="\n")
    (tree / "tools/pancake/abtest.py").write_text("# stub\n")
    (tree / "tools/pancake/flag_diag.py").write_text("# stub\n")
    shutil.copy(CHAINS / "bench_updatedRD_conf.py", tree / "tools/pancake/box_chains/")
    shutil.copy(HERE.parent.parent / "bench" / "specs" / "250-rdna3-updated-vs-plain.json", tree / "bench/specs/")
    shutil.copy(HERE.parent.parent / "bench" / "specs" / "251-rdna3-hipblaslt-and-profile.json", tree / "bench/specs/")
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
                 "END ctest exit=0", "CONFIG written", "START diag-updated", "START 250-screen", "START 250-confirm", "START 251-screen", "START 251-confirm",
                 "TUNEUP pcie_link", "TUNEUP os_tune check", "wrote"):
        assert must in text, must
    order = ["PREREQ ok", "CONFIG written", "BUILD plain", "BUILD_END plain exit=0", "START smoke plain", "SMOKE plain ok", "BUILD updated", "BUILD_END updated exit=0",
             "START smoke updated", "SMOKE updated ok", "START ctest", "TUNEUP pcie_link", "START diag-updated",
             "START 250-screen", "START 250-confirm", "START 251-screen", "SUMMARY"]
    pos = [text.index(k) for k in order]
    assert pos == sorted(pos), order
    calls = box.calls()
    assert any("--tier confirm" in c and "--block 3" in c and "--reps 3" in c and "250-confirm.json" in c for c in calls if c.startswith("PY")), "3-rep blocked confirm"
    assert any("--tier screen" in c and "--drift-check" in c and "250.json" in c for c in calls if c.startswith("PY"))
    work = box.tmp / "auto-runs" / "benchRD"
    assert json.loads((work / "250-confirm.json").read_text())["cases"] == ["short", "edit"]          # the confirm judges the decode cases only
    assert json.loads((work / "251.json").read_text())["cases"] == ["short", "long32k"]
    assert (box.tmp / "auto-runs" / "benchRD-diag-updated" / "verdicts.json").exists() and (box.tmp / "auto-runs" / "benchRD-tuneup" / "pcie_link.txt").exists()
    assert (box.tmp / "auto-runs" / "benchRD-recommended" / "gfx1100.json").exists()
    assert not any("diag-plain" in c for c in calls)
    assert any(c.startswith("cmake") and "benchRD/build-plain" in c for c in calls)
    stub = box.h + "/mnt/benchRD/rccl-stub"
    assert "ncclSend" in (box.tmp / "mnt" / "benchRD" / "rccl-stub" / "rccl" / "rccl.h").read_text() and "PREREQ rccl/rccl.h is not in" in text
    cpi = [c for c in calls if c.startswith("CPI ")]
    assert cpi and all(c == "CPI " + stub for c in cpi), cpi            # every cmake of both builds saw the stub dir (the plain tree's source and CMake flags are untouched)
    assert any("--" in c and c.endswith("-k 0") for c in calls if c.startswith("cmake --build")), "ninja -k 0: one build shows every error"
    smk = [c for c in calls if c.startswith("PY") and "abtest.py" in c and "benchRD-smoke-" in c]
    assert len(smk) == 2 and all("--reps 1" in c and "--retries 0" in c for c in smk), smk
    spec = (box.tmp / "auto-runs" / "benchRD" / "250.json").read_text()
    assert "@PLAIN@" not in spec and "@UPDATED@" not in spec and "build-plain/strata" in spec and "build-updated/strata" in spec
    assert str(box.tmp / "mnt") in spec or "/mnt/" in spec, "arms point at /mnt/pancake builds"
    spec2 = (box.tmp / "auto-runs" / "benchRD" / "251.json").read_text()
    assert "@UPDATED_SRC@" not in spec2 and "@UPDATED@" not in spec2 and "src-updated/tools/hip/gfx1100-hipblaslt-100401.txt" in spec2


def diag_timeout_call(box, marker="flag_diag.py"):
    return [c for c in box.calls() if c.startswith("TIMEOUT") and marker in c]


def test_bd_diag_runs_in_its_own_session_with_a_limit_sized_from_the_per_flag_timeout(tmp_path):
    """2026-11-19 (US chain): `timeout` ended flag_diag but its abtest / server / engine kept running.  Now the diagnostic runs under
    setsid + timeout --signal=TERM --kill-after, the limit is (flags + 1) x (per-flag + 90) + ctest + margin, and no leftover means no reap."""
    box = Box(tmp_path)
    prepare_bd(box)
    box.finish(box.start(BD))
    lines = box.log("bench-updatedRD.log")
    text = "\n".join(lines)
    assert lines[-1] == "WINBENCHRD_DONE" and "DIAG-REAP" not in text and "ERROR" not in text
    calls = box.calls()
    to = diag_timeout_call(box)
    assert len(to) == 1 and "--signal=TERM" in to[0] and "--kill-after=100" in to[0], to
    assert int(to[0].split()[3]) == 6 * (240 + 90) + 300 + 180          # 5 flags + base, ctest of the exact flags, margin
    assert any(c.startswith("SETSID") and "flag_diag.py" in c for c in calls)
    py = [c for c in calls if c.startswith("PY") and "flag_diag.py" in c and " run " in c]
    assert py and "--flag-timeout 240" in py[0] and "--ctest-timeout 300" in py[0] and py[0].startswith("PY -u "), py
    assert "limit 2460 s" in text


def test_bd_diag_leftovers_of_its_session_are_killed_and_logged_and_the_chain_goes_on(tmp_path):
    box = Box(tmp_path)
    prepare_bd(box)
    box.finish(box.start(BD, env={"STUB_DIAG_LEFTOVER": "1", "STUB_DIAG_EXIT": "124"}))
    lines = box.log("bench-updatedRD.log")
    text = "\n".join(lines)
    reap = [l for l in lines if l.startswith("DIAG-REAP")]
    assert reap and "999991" in reap[0] and "abtest.py" in reap[0], lines
    assert "END diag-updated exit=124" in text
    assert text.index("DIAG-REAP") < text.index("END diag-updated") < text.index("START 250-screen")   # reaped before the next measurement
    assert lines[-1] == "WINBENCHRD_DONE"


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
    assert lines[-1] == "WINBENCHRD_DONE" and any(l.startswith("WINBENCHRD_ABORTED") for l in lines[-4:]) and any(l.startswith("service started") for l in lines[-3:])
    assert (bp / "strata").read_text() == before
    for never in ("BUILD_END updated", "START ctest", "START diag", "START 250", "START smoke"):
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


def test_bd_prereq_names_every_missing_file_before_anything_is_downloaded(tmp_path):
    box = Box(tmp_path)
    prepare_bd(box)
    (box.tmp / "clone-of-d" / "build-int" / "_deps" / "strata_llamacpp-src" / "x").unlink()
    (box.tmp / "clone-of-d" / "build-int" / "_deps" / "strata_llamacpp-src").rmdir()
    (box.tmp / "clone-of-d" / "Strata-data" / "packs" / "iq2_xs" / "tokenizer").unlink()
    box.finish(box.start(BD))
    lines = box.log("bench-updatedRD.log")
    err = [l for l in lines if l.startswith("ERROR PREREQ missing files:")]
    assert err and "strata_llamacpp-src" in err[0] and "iq2_xs/tokenizer" in err[0], lines
    text = "\n".join(lines)
    assert lines[-1] == "WINBENCHRD_DONE" and "SYNC nicejunjie" not in text and "BUILD" not in text
    assert_service_discipline(box)


def test_bd_no_rccl_stub_when_rocm_has_the_header(tmp_path):
    box = Box(tmp_path)
    prepare_bd(box)
    (box.tmp / "rocm" / "include" / "rccl").mkdir(parents=True)
    (box.tmp / "rocm" / "include" / "rccl" / "rccl.h").write_text("/* real */\n")
    box.finish(box.start(BD))
    lines = box.log("bench-updatedRD.log")
    assert "PREREQ rccl/rccl.h found in ROCm: no stub" in lines and lines[-1] == "WINBENCHRD_DONE" and "ERROR" not in "\n".join(lines)
    assert not (box.tmp / "mnt" / "benchRD" / "rccl-stub").exists()
    assert all(c == "CPI " for c in box.calls() if c.startswith("CPI ")), "CPLUS_INCLUDE_PATH stays empty when the header exists"


# what Strata-rdna3 8b01b0c85 src/core/peer_transport.inc (inside peer_experts.cpp, STRATA_USE_HIP + Linux) uses from rccl/rccl.h; the functions are
# taken with dlopen("librccl.so.1") + dlsym, so the stub only needs declarations (second RD attempt: the EMPTY stub gave 'ncclGroupEnd' was not declared)
RCCL_USED = ["ncclComm_t", "ncclResult_t", "ncclSuccess", "ncclUint8", "ncclCommInitAll", "ncclCommDestroy", "ncclGroupStart", "ncclGroupEnd",
             "ncclSend", "ncclRecv", "ncclGetErrorString"]


def test_bd_rccl_stub_declares_the_rccl_api_peer_transport_uses(tmp_path):
    box = Box(tmp_path)
    prepare_bd(box)
    box.finish(box.start(BD))
    h = (box.tmp / "mnt" / "benchRD" / "rccl-stub" / "rccl" / "rccl.h").read_text()
    code = re.sub(r"/\*.*?\*/", "", h, flags=re.S)
    for name in RCCL_USED:
        assert re.search(r"\b%s\b" % name, code), name
    assert re.search(r'extern "C"', code) and "#ifndef NCCL_H_" in code and code.count("#endif") >= 3
    assert "typedef struct ncclComm* ncclComm_t;" in code and "typedef struct ihipStream_t* hipStream_t;" in code
    # nccl.h enum values (the dlsym'd librccl.so.1 functions read them)
    assert re.search(r"\bncclSuccess = 0\b", code) and re.search(r"\bncclUint8 = 1\b", code) and re.search(r"\bncclFloat32 = 7\b", code)
    sig = {"ncclCommInitAll": "ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist);",
           "ncclCommDestroy": "ncclResult_t ncclCommDestroy(ncclComm_t comm);",
           "ncclGroupStart": "ncclResult_t ncclGroupStart(void);", "ncclGroupEnd": "ncclResult_t ncclGroupEnd(void);",
           "ncclSend": "ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, hipStream_t stream);",
           "ncclRecv": "ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, hipStream_t stream);",
           "ncclGetErrorString": "const char* ncclGetErrorString(ncclResult_t result);"}
    for name, s in sig.items():
        assert s in code, name
    calls = box.calls()
    assert not any("-lrccl" in c or "LINKER_FLAGS" in c for c in calls), "dlsym only: no link flag"
    lines = box.log("bench-updatedRD.log")
    assert any("PREREQ rccl/rccl.h is not in" in l and "no -lrccl" in l for l in lines)


def test_bd_rccl_stub_compiles_with_peer_transport_like_code(tmp_path):
    """Syntax check of the generated stub with a peer_transport.inc-like user (HIP typedef first, as hip_runtime_api.h has it) and as C99."""
    cxx = shutil.which("clang++") or shutil.which("g++") or (r"C:\Program Files\LLVM\bin\clang++.exe" if Path(r"C:\Program Files\LLVM\bin\clang++.exe").exists() else None)
    if not cxx or not SH:
        pytest.skip("no C++ compiler or sh")
    src = (CHAINS / BD).read_text(encoding="utf-8")
    fn = re.search(r"^rccl_stub\(\) \{.*?^EOF\n\}\n", src, flags=re.S | re.M).group(0)
    (tmp_path / "fn.sh").write_text(fn + "rccl_stub\n", newline="\n")
    (tmp_path / "inc" / "rccl").mkdir(parents=True)
    out = subprocess.run([SH, str(tmp_path / "fn.sh")], capture_output=True, text=True, check=True).stdout
    (tmp_path / "inc" / "rccl" / "rccl.h").write_text(out, newline="\n")
    (tmp_path / "t.cpp").write_text(r'''typedef struct ihipStream_t* hipStream_t;
typedef hipStream_t cudaStream_t;
#include <rccl/rccl.h>
extern "C" void* dlsym(void*, const char*);
struct T {
    void* library = nullptr;
    ncclComm_t comm[2]{};
    decltype(&ncclCommInitAll) init = nullptr; decltype(&ncclCommDestroy) destroy = nullptr;
    decltype(&ncclGroupStart) group_start = nullptr; decltype(&ncclGroupEnd) group_end = nullptr;
    decltype(&ncclSend) send = nullptr; decltype(&ncclRecv) recv = nullptr; decltype(&ncclGetErrorString) error_string = nullptr;
    cudaStream_t s = nullptr;
    int nc(ncclResult_t e) { return e != ncclSuccess ? error_string(e)[0] : 0; }
    template<class F> void symbol(F& f, const char* n) { f = reinterpret_cast<F>(dlsym(library, n)); }
    void run(float* d, const float* x, unsigned long b) {
        symbol(init, "ncclCommInitAll"); symbol(send, "ncclSend"); int dev[] = {0, 1}; nc(init(comm, 2, dev));
        nc(group_start()); nc(send(x, b, ncclUint8, 1, comm[0], s)); nc(recv(d, b, ncclUint8, 0, comm[1], s)); nc(group_end()); destroy(comm[0]); }
};
int main() { T t; t.run(nullptr, nullptr, 4); return 0; }
''')
    (tmp_path / "t.c").write_text("#include <rccl/rccl.h>\nint main(void){ ncclDataType_t t = ncclUint8; return (int)t + (int)ncclSuccess; }\n")
    tgt = ["-target", "x86_64-linux-gnu"] if "clang" in Path(cxx).name else []
    r = subprocess.run([cxx, *tgt, "-std=c++17", "-Wall", "-Werror", "-fsyntax-only", "-I", str(tmp_path / "inc"), str(tmp_path / "t.cpp")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    r = subprocess.run([cxx, *tgt, "-x", "c", "-std=c99", "-Wall", "-Werror", "-fsyntax-only", "-I", str(tmp_path / "inc"), str(tmp_path / "t.c")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_bd_smoke_failure_stops_before_the_updated_build_with_the_engine_error(tmp_path):
    box = Box(tmp_path)
    prepare_bd(box)
    box.finish(box.start(BD, env={"STUB_DIE": "benchRD-smoke-plain"}))
    lines = box.log("bench-updatedRD.log")
    text = "\n".join(lines)
    err = [l for l in lines if l.startswith("ERROR smoke plain FAILED exit=2")]
    assert err and "server died before ready" in err[0] and "it needs --native SHARD1" in err[0], lines
    assert lines[-1] == "WINBENCHRD_DONE" and any(l.startswith("WINBENCHRD_ABORTED") for l in lines[-4:])
    for never in ("BUILD updated", "START ctest", "START diag", "START 250", "SMOKE plain ok"):
        assert never not in text, never
    assert_start_end_pairs(lines)
    assert_service_discipline(box)


def test_bd_smoke_of_the_updated_build_stops_before_ctest(tmp_path):
    box = Box(tmp_path)
    prepare_bd(box)
    box.finish(box.start(BD, env={"STUB_DIE": "benchRD-smoke-updated"}))
    lines = box.log("bench-updatedRD.log")
    text = "\n".join(lines)
    assert "SMOKE plain ok" in text and any(l.startswith("ERROR smoke updated FAILED") for l in lines)
    for never in ("START ctest", "START diag", "START 250"):
        assert never not in text, never
    assert lines[-1] == "WINBENCHRD_DONE"


def test_bd_empty_verdicts_and_screen_exit_2_log_the_underlying_message(tmp_path):
    box = Box(tmp_path)
    prepare_bd(box)
    box.finish(box.start(BD, env={"STUB_DIE": "benchRD/250.json"}))
    lines = box.log("bench-updatedRD.log")
    err = [l for l in lines if l.startswith("ERROR 250-screen exit=2:")]
    assert err and "first error line of the stdout: [rep 0 plain: FAILED (server died before ready)]" in err[0] and "it needs --native SHARD1" in err[0], lines
    assert any(l.startswith("ERROR spec 250 screen exit=2") for l in lines) and lines[-1] == "WINBENCHRD_DONE"


def test_peer_experts_rccl_include_is_optional_when_the_file_has_it():
    """Strata-rdna3's peer_experts.cpp includes <rccl/rccl.h> for peer_transport.inc (rccl transport, dlsym); this box's ROCm has no rccl headers.
    When the header is optional (__has_include), every RCCL use in peer_transport.inc must sit under STRATA_PEER_RCCL."""
    f = HERE.parents[1] / "src" / "core" / "peer_experts.cpp"
    if not f.exists():
        pytest.skip("no src/core/peer_experts.cpp in this branch")
    txt = f.read_text()
    if "#include <rccl/rccl.h>" in txt:
        assert "__has_include(<rccl/rccl.h>)" in txt
    assert not re.search(r"\b(nccl|rccl)[A-Z]\w*\(", txt), "RCCL calls live in peer_transport.inc only"
    inc = f.parent / "peer_transport.inc"
    if "__has_include(<rccl/rccl.h>)" in txt and inc.exists():
        assert "#define STRATA_PEER_RCCL 1" in txt and "#define STRATA_PEER_RCCL 0" in txt
        depth, guarded = 0, []
        for line in inc.read_text().splitlines():
            s = line.strip()
            if s.startswith("#if"):
                guarded.append("STRATA_PEER_RCCL" in s); continue
            if s.startswith("#endif"):
                guarded.pop(); continue
            if s.startswith("#else"):
                guarded[-1] = False; continue
            if re.search(r"\bnccl[A-Z]\w*|\bdl(open|sym|close|error)\b", s):
                assert any(guarded), line


def test_bd_name_is_invisible_to_every_busy_pattern_and_sees_the_others(tmp_path):
    box = Box(tmp_path)
    assert "win-chain" not in BD
    for name in ("win-chainV.sh", "xtail-chainX.sh", "ytail-chainY.sh", "ztail-chainZ.sh", AA):
        assert busy_result(box, name, ["sh ./bench-updatedRD.sh"]) == "IDLE", name
    assert busy_result(box, BD, ["sh ./bench-updatedRD.sh"]) == "IDLE"
    assert busy_result(box, BD, ["4242 sh ./redo-chainAA.sh"]) == "IDLE"
    for other in ("sh ./win-chainT.sh", "sh ./xtail-chainX.sh", "/bin/sh ./after7.sh", "python3 tools/pancake/abtest.py x",
                  "cmake --build x"):
        assert busy_result(box, BD, [other]) == "BUSY", other


def test_bd_spec_251_arms_and_not_in_the_shipped_file():
    spec = json.loads((HERE.parent.parent / "bench" / "specs" / "251-rdna3-hipblaslt-and-profile.json").read_text())
    assert set(spec["arms"]) == {"updated", "hipblaslt-100401", "profile-pancake"}
    assert spec["arms"]["hipblaslt-100401"]["env"]["STRATA_HIPBLASLT_TUNING"].endswith("gfx1100-hipblaslt-100401.txt")
    assert spec["arms"]["profile-pancake"]["args"]["--expert-profile"].endswith("data/expert-profile-pancake.bin")
    rec = HERE.parent.parent / "rdna3" / "recommended-flags" / "gfx1100.json"
    if rec.exists():       # on the updated-strata-rdna3 branch: neither arm's value is in the shipped file until a measured GAIN
        shipped = json.loads(rec.read_text())
        assert "STRATA_HIPBLASLT_TUNING" not in shipped["env"] and "--expert-profile" not in shipped["args"]
        assert spec["arms"]["updated"]["env"] == shipped["env"]


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
    preset.write_text(json.dumps({"exe": "e", "args": ["--spec", "4", "--grammar", "on", "--lookup-max-t", "12", "--expert-profile", "/pancake/p.bin", "--kv", "int8"],
                                  "env": {"STRATA_VERIFY_ECAP": "1"}, "grammar": True, "port": 8080}))
    plain, upd, path = mod.build(str(preset), str(tmp_path), "/u", "/b/plain", "/b/upd", TF)
    assert plain["args"] == ["--spec", "4", "--expert-profile", str(tmp_path) + "/data/expert-profile.bin", "--kv", "int8"] and "grammar" not in plain
    assert plain["env"] == {"STRATA_PF_F16": "2", "X": str(tmp_path) + "/t"}      # the plain build runs with the env of Strata-rdna3's own setup config
    assert plain["exe"] == "/b/plain/strata" and upd["exe"] == "/b/upd/strata"
    assert upd["env"]["STRATA_VERIFY_ECAP"] == "0" and "STRATA_VERIFY_ECAP" not in plain["env"]
    rec = HERE.parent.parent / "rdna3" / "recommended-flags" / "gfx1100.json"
    if rec.exists():       # on the updated-strata-rdna3 branch: the spec's updated arm is exactly the shipped file
        assert spec["arms"]["updated"]["env"] == json.loads(rec.read_text())["env"]


# ------------------------------------------------------------------------------------------------ bench-sched.sh and the short plan (owner order 2026-11-19)
SCHED = "bench-sched.sh"
FAKE_CHAIN = ('#!/bin/sh\n'
              'echo "{name} env: BENCH_SCHED=$BENCH_SCHED BENCH_PREIDLED=$BENCH_PREIDLED KEEP=$BENCH_KEEP_SERVICE_DOWN stat=$(cat "$H/stat.4242" 2>/dev/null)" >> "$H/chains.log"\n'
              'echo "START $(date +%T)" >> "{log}"\n'
              '[ "$FAKE_FAIL" = "{name}" ] && exit 3\n'
              'echo "{marker}" >> "{log}"\n')


def prepare_sched(box, aa=("4242 sh ./redo-chainAA.sh",), busy=()):
    (box.tmp / "ps.txt").write_text("\n".join(list(aa) + ["1 /sbin/init", "77 sh ./bench-sched.sh"]) + "\n")
    (box.tmp / "procs.txt").write_text("\n".join(busy) + ("\n" if busy else ""))
    for pid in {l.split()[0] for l in aa}:
        (box.tmp / f"stat.{pid}").write_text("S\n")
    for name, log, marker in (("us", "bench-updatedAA.log", "WINBENCHUS_DONE"), ("rd", "bench-updatedRD.log", "WINBENCHRD_DONE")):
        (box.S / f"fake-{name}.sh").write_text(FAKE_CHAIN.format(name=name, log=(box.S / log).as_posix(), marker=marker), newline="\n")


def sched_env(**kw):
    return {"BENCH_US": "./fake-us.sh", "BENCH_RD": "./fake-rd.sh", **kw}


def chains_log(box):
    p = box.tmp / "chains.log"
    return p.read_text().splitlines() if p.exists() else []


def test_sched_freezes_aa_waits_for_the_running_step_runs_both_chains_then_resumes(tmp_path):
    box = Box(tmp_path)
    prepare_sched(box, busy=("5001 python3 tools/pancake/abtest.py x --tier screen",))   # the step of AA that runs now
    p = box.start(SCHED, env=sched_env())
    for _ in range(100):
        if any(c.startswith("SIG -STOP") for c in box.calls()):
            break
        time.sleep(0.1)
    time.sleep(0.5)
    assert "SIG -STOP 4242" in box.calls()
    assert p.poll() is None and not chains_log(box), "no chain starts while the AA step is still measuring"
    assert (box.tmp / "stat.4242").read_text().strip() == "T"
    (box.tmp / "procs.txt").write_text("")                                      # the abtest ends
    box.finish(p)
    lines = box.log("bench-sched.log")
    text = "\n".join(lines)
    assert lines[-1] == "BENCHSCHED_DONE" and "ERROR" not in text, lines
    order = ["AA_PAUSED pid 4242", "IDLE the box is idle", "START US", "END US exit=0", "START RD", "END RD exit=0", "AA_RESUMED pid 4242"]
    pos = [text.index(k) for k in order]
    assert pos == sorted(pos), order
    sig = [c for c in box.calls() if c.startswith("SIG")]
    assert sig == ["SIG -STOP 4242", "SIG -CONT 4242"], sig                      # only the AA shell, only those two signals
    assert chains_log(box) == ["us env: BENCH_SCHED=1 BENCH_PREIDLED=1 KEEP=1 stat=T", "rd env: BENCH_SCHED=1 BENCH_PREIDLED=1 KEEP=1 stat=T"]   # AA frozen during both
    assert (box.tmp / "stat.4242").read_text().strip() == "S"
    assert not any(c.startswith("SYSTEMCTL") and ("start" in c or "stop" in c) for c in box.calls())     # the scheduler never touches the service


def test_sched_a_chain_without_marker_is_an_error_the_next_chain_and_the_resume_still_happen(tmp_path):
    box = Box(tmp_path)
    prepare_sched(box)
    box.finish(box.start(SCHED, env=sched_env(FAKE_FAIL="us")))
    lines = box.log("bench-sched.log")
    text = "\n".join(lines)
    assert "ERROR US ended without its marker WINBENCHUS_DONE (exit 3)" in text and "END RD exit=0" in text
    assert [c for c in box.calls() if c.startswith("SIG")] == ["SIG -STOP 4242", "SIG -CONT 4242"] and lines[-1] == "BENCHSCHED_DONE"
    assert "ERROR RD" not in text


def test_sched_aborting_chain_marker_counts_as_done(tmp_path):
    box = Box(tmp_path)
    prepare_sched(box)
    (box.S / "fake-us.sh").write_text("#!/bin/sh\necho 'ERROR boom' >> " + (box.S / "bench-updatedAA.log").as_posix()
                                      + "\necho 'WINBENCHUS_ABORTED: boom' >> " + (box.S / "bench-updatedAA.log").as_posix()
                                      + "\necho WINBENCHUS_DONE >> " + (box.S / "bench-updatedAA.log").as_posix() + "\nexit 1\n", newline="\n")
    box.finish(box.start(SCHED, env=sched_env()))
    text = "\n".join(box.log("bench-sched.log"))
    assert "END US exit=1 last-log-line=WINBENCHUS_DONE" in text and "ended without its marker" not in text and "END RD exit=0" in text
    assert "SIG -CONT 4242" in box.calls()


def test_sched_only_the_exact_aa_cmdline_is_stopped(tmp_path):
    box = Box(tmp_path)
    prepare_sched(box, aa=("4242 sh ./redo-chainAA.sh --other", "4343 /bin/sh ./redo-chainAA.sh", "4444 vim redo-chainAA.sh"))
    box.finish(box.start(SCHED, env=sched_env()))
    text = "\n".join(box.log("bench-sched.log"))
    assert not [c for c in box.calls() if c.startswith("SIG")], box.calls()
    assert "AA shell not running" in text and "END US exit=0" in text and "END RD exit=0" in text


def test_sched_with_two_aa_shells_stops_nothing_and_runs_nothing(tmp_path):
    box = Box(tmp_path)
    prepare_sched(box, aa=("4242 sh ./redo-chainAA.sh", "4343 sh ./redo-chainAA.sh"))
    box.finish(box.start(SCHED, env=sched_env()))
    text = "\n".join(box.log("bench-sched.log"))
    assert "ERROR 2 AA shells" in text and not chains_log(box) and not [c for c in box.calls() if c.startswith("SIG")]


def test_sched_does_not_continue_a_pid_that_is_not_the_aa_shell_any_more(tmp_path):
    box = Box(tmp_path)
    prepare_sched(box)
    (box.S / "fake-us.sh").write_text("#!/bin/sh\necho '4242 sleep 5' > " + (box.tmp / "ps.txt").as_posix() + "\necho WINBENCHUS_DONE >> "
                                      + (box.S / "bench-updatedAA.log").as_posix() + "\n", newline="\n")      # pid 4242 reused by another program
    box.finish(box.start(SCHED, env=sched_env()))
    text = "\n".join(box.log("bench-sched.log"))
    assert "nothing to resume" in text and "SIG -CONT" not in "\n".join(box.calls())


@pytest.mark.skipif(os.name == "nt", reason="needs POSIX signals")
def test_sched_resumes_aa_when_it_is_terminated(tmp_path):
    box = Box(tmp_path)
    prepare_sched(box, busy=("5001 python3 tools/pancake/abtest.py x",))          # stays busy: the scheduler sits in its idle wait
    p = box.start(SCHED, env=sched_env())
    for _ in range(100):
        if "SIG -STOP 4242" in box.calls():
            break
        time.sleep(0.1)
    p.terminate()
    box.finish(p, timeout=30)
    assert box.calls()[-1] == "SIG -CONT 4242" and "signal received" in "\n".join(box.log("bench-sched.log"))


def test_sched_name_is_invisible_to_every_busy_pattern_and_aa_resumes_with_a_working_idle_check(tmp_path):
    box = Box(tmp_path)
    for bad in ("win-chain", "redo-chain", "tail-chain", "auto-chain", "after7"):
        assert bad not in SCHED
    mine = ["77 sh ./bench-sched.sh", "78 sh ./bench-updatedAA.sh", "79 sh ./bench-updatedRD.sh"]
    for name in ("win-chainT.sh", "win-chainV.sh", "xtail-chainX.sh", "ytail-chainY.sh", "ztail-chainZ.sh", AA, BU, BD, SCHED):
        assert busy_result(box, name, mine) == "IDLE", name
    for other in ("python3 tools/pancake/abtest.py x", "cmake --build x", "strata --serve", "sh ./win-chainT.sh"):      # AA's busy() still sees real jobs
        assert busy_result(box, AA, mine + [other]) == "BUSY", other
    # the scheduler's own busy(): the same job patterns, never the frozen AA shell
    assert busy_result(box, SCHED, ["4242 sh ./redo-chainAA.sh"]) == "IDLE"
    assert busy_result(box, SCHED, ["4242 sh ./redo-chainAA.sh", "5001 python3 tools/pancake/abtest.py x"]) == "BUSY"


@pytest.mark.parametrize("name", [BU, BD])
def test_bench_chains_treat_a_frozen_aa_shell_as_idle_and_a_live_one_as_busy(tmp_path, name):
    box = Box(tmp_path)
    t = box.script_text(name)
    fns = "\n".join(l for l in t.splitlines() if l.startswith(("busy()", "aa_live()", "bz()")))
    (box.tmp / "procs.txt").write_text("4242 sh ./redo-chainAA.sh\n")
    (box.tmp / "ps.txt").write_text("4242 sh ./redo-chainAA.sh\n")

    def run(stat):
        (box.tmp / "stat.4242").write_text(stat + "\n")
        r = subprocess.run([SH, "-c", fns + "\nif bz; then echo BUSY; else echo IDLE; fi"], capture_output=True, text=True,
                           env={**os.environ, "H": box.h, "PATH": str(box.tmp / "stub") + os.pathsep + os.environ["PATH"]})
        return r.stdout.strip()
    assert run("T") == "IDLE" and run("S") == "BUSY" and run("Ssl") == "BUSY" and run("Tl") == "IDLE"


def test_bench_us_under_the_scheduler_does_not_wait_for_aa_keeps_the_service_down_and_skips_the_long_idle(tmp_path):
    box = Box(tmp_path)
    prepare_bu(box, aa_done=False)
    box.finish(box.start(BU, env={"BENCH_SCHED": "1", "BENCH_PREIDLED": "1", "BENCH_KEEP_SERVICE_DOWN": "1", "SVC_DOWN": "1"}))
    lines = box.log("bench-updatedAA.log")
    text = "\n".join(lines)
    assert lines[-1] == "WINBENCHUS_DONE" and "ERROR" not in text, lines[-6:]
    assert "service already stopped" in text and "service stays down" in text
    cl = [c for c in box.calls() if c.startswith("SYSTEMCTL")]
    assert not any(" start " in c or " stop " in c for c in cl), cl                  # AA's busy() would see the service engine: it stays down
    assert not any(c.startswith("SS ") for c in box.calls()), "no client wait when the service is down"
    sleeps = [c for c in box.calls() if c.startswith("pgrep")]
    assert sleeps                                                                    # quick idle polls ran


def test_bench_rd_under_the_scheduler_does_not_wait_for_the_markers_and_keeps_the_service_down(tmp_path):
    box = Box(tmp_path)
    prepare_bd(box, markers=False)
    box.finish(box.start(BD, env={"BENCH_SCHED": "1", "BENCH_PREIDLED": "1", "BENCH_KEEP_SERVICE_DOWN": "1", "SVC_DOWN": "1"}))
    lines = box.log("bench-updatedRD.log")
    text = "\n".join(lines)
    assert lines[-1] == "WINBENCHRD_DONE" and "ERROR" not in text, lines[-6:]
    assert "service stays down" in text and not any(" start " in c or " stop " in c for c in box.calls() if c.startswith("SYSTEMCTL"))


def test_bench_aborts_under_the_scheduler_also_leave_the_service_down(tmp_path):
    box = Box(tmp_path)
    prepare_bu(box, aa_done=False)
    box.finish(box.start(BU, env={"BENCH_SCHED": "1", "BENCH_PREIDLED": "1", "BENCH_KEEP_SERVICE_DOWN": "1", "SVC_DOWN": "1", "BU_FAIL": "1"}))
    lines = box.log("bench-updatedAA.log")
    assert any(l.startswith("ERROR updated-strata build") for l in lines) and lines[-1] == "WINBENCHUS_DONE"
    assert not any(" start " in c for c in box.calls() if c.startswith("SYSTEMCTL")) and any("service stays down" in l for l in lines)


def test_bench_us_no_confirm_for_an_arm_without_a_screen_benefit(tmp_path):
    box = Box(tmp_path)
    prepare_bu(box)
    box.finish(box.start(BU, env={"STUB_GAIN": "100"}))
    lines = box.log("bench-updatedAA.log")
    text = "\n".join(lines)
    ab = [c for c in box.calls() if c.startswith("PY") and "abtest.py" in c and "smoke-" not in c]
    assert len(ab) == 2 and all("--tier screen" in c for c in ab), ab
    assert "220: no arm with a benefit in the screen: no confirm" in text and "221: no arm with a benefit in the screen: no confirm" in text
    assert lines[-1] == "WINBENCHUS_DONE" and "ERROR" not in text


def test_bench_us_budget_over_skips_the_optional_confirms(tmp_path):
    box = Box(tmp_path)
    prepare_bu(box)
    box.finish(box.start(BU, env={"BENCH_BUDGET_S": "100"}))
    lines = box.log("bench-updatedAA.log")
    text = "\n".join(lines)
    ab = [c for c in box.calls() if c.startswith("PY") and "abtest.py" in c and "smoke-" not in c]
    assert len(ab) == 2 and "SKIP 220 confirm: only" in text and "221 confirm not run: only" in text
    assert lines[-1] == "WINBENCHUS_DONE" and "SUMMARY spec220 exit=0, spec221 exit=0" in text


def test_bench_rd_no_confirm_without_benefit_and_budget_skips(tmp_path):
    box = Box(tmp_path)
    prepare_bd(box)
    box.finish(box.start(BD, env={"STUB_GAIN": "100"}))
    text = "\n".join(box.log("bench-updatedRD.log"))
    ab = [c for c in box.calls() if c.startswith("PY") and "abtest.py" in c and "--out" in c and "benchRD-2" in c]
    assert len(ab) == 2 and "250: no arm with a benefit in the screen: no confirm" in text, ab


def test_bench_rd_budget_over_skips_the_optional_confirms(tmp_path):
    box = Box(tmp_path)
    prepare_bd(box)
    box.finish(box.start(BD, env={"BENCH_BUDGET_S": "100"}))
    text = chr(10).join(box.log("bench-updatedRD.log"))
    ab = [c for c in box.calls() if c.startswith("PY") and "abtest.py" in c and "benchRD-2" in c]
    assert len(ab) == 2 and "SKIP 250 confirm: only" in text and "251 confirm not run: only" in text, ab
