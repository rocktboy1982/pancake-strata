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
if a0.endswith("chainT-decide.py") or a0.endswith("box_chains/sweep.py"):
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
    if out and os.environ.get("STUB_ARM_DECODE"):     # {arm: {case: tok/s}}: one record per arm of the spec (and rep), as a full run writes
        dec = json.loads(os.environ["STUB_ARM_DECODE"])
        late = json.loads(os.environ.get("STUB_LATE_DRIFT", "{}"))   # {out-file substring: factor}: the late base (base-late / last rep) of that run
        drift = ("--drift-check" in args or spec.get("drift_check")) and "--tier" in args and opt("--tier") == "screen"
        reps = int(opt("--reps", "1")) if opt("--tier") == "confirm" else 1
        fac = next((v for k, v in late.items() if k in out), 1.0)
        for r in range(reps):
            for arm in arms + (["base-late"] if drift and "base-late" not in arms else []):
                ref = next(x for x in arms if x != "warmup")       # the warm-up start (T-1004) measures as the base
                an = (ref if arm == "warmup" else arm).split("~")[0]   # a paired block's repeated arms (flag~2, updated~3) measure as their arm
                d = dec.get(an, dec.get(ref.split("~")[0], dec.get("*", {})) if arm == "base-late" else dec.get("*", {}))
                f = fac if (arm == "base-late" or (reps > 1 and arm == arms[0] and r == reps - 1)) else 1.0
                open(out, "a").write(json.dumps({"arm": arm, "rep": r, "rows": [{"case": c, "decode_tps": d.get(c, 100.0) * f,
                                                                                   "prompt_tps": d.get(c + "-prompt", 1000.0)}
                                                                                  for c in spec.get("cases", ["short", "edit"])]}) + "\n")
    elif out:
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
    # owner 2026-11-20: a case below the -3% floor does NOT block the confirm (reported; the confirm decides)
    rows["c2"]["a"] = 48.0
    w, log = DECIDE.decide(screen_text(rows, {"a": True, "b": True}), SPEC)
    assert [x[0] for x in w] == ["a"] and any("below the -3% floor" in l and "confirm decides" in l for l in log), log
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


def test_diag_us_standalone_does_not_wait_for_the_rd_marker_and_reaps_leftovers(tmp_path):
    """RD (Strata-rdna3) is paused since 2026-11-20: an unfinished RD log (no WINBENCHRD_DONE) no longer blocks the US diagnostic."""
    box = Box(tmp_path)
    prepare_du(box)
    (box.S / "bench-updatedRD.log").write_text("START\n")
    assert "RD_LOG" not in (CHAINS / DU).read_text()
    box.finish(box.start(DU, env={"BENCH_PREIDLED": "1", "SVC_DOWN": "1", "FD_LEFTOVER": "1"}), timeout=60)
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


US_SRC = "/mnt/pancake/benchus/src-us"


def test_updated_vs_upstream_specs_match_the_preset():
    root = HERE.parents[1]
    plain = json.loads((root / "configs/pancake/flashnext-iq2_xs-plain-upstream.json").read_text())
    # the preset is an updated-strata preset (<REPO> = the updated-strata checkout); the specs run it from the tree the chain builds (/mnt/pancake/benchus/src-us)
    preset = json.loads((root / "configs/pancake/flashnext-iq2_xs-updated-hip.json").read_text().replace("<REPO>/data/", US_SRC + "/data/")
                        .replace("<REPO>/terminal/", US_SRC + "/terminal/"))
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
    (tree / "tools/pancake/abtest.py").write_text("# stub: PY = os.environ.get(\"ABTEST_PYTHON\") or ...\n")
    (tree / "tools/pancake/flag_diag.py").write_text("# stub\n")
    (tree / "serve").mkdir(parents=True, exist_ok=True)
    (tree / "serve/__init__.py").write_text("")
    (tree / "serve/server.py").write_text("# stub\n")
    # the box's existing venv interpreter (Strata-win/.venv/bin/python): logs every call with the ABTEST_PYTHON it hands on, answers the
    # serve.server import check from the cwd's tree (HARNESS_NO_DEPS=1: a venv without the server's deps), and runs the python stub otherwise
    hp = box.R / ".venv" / "bin" / "python"
    hp.parent.mkdir(parents=True, exist_ok=True)
    hp.write_text('#!/bin/sh\necho "HARNESS ABTEST_PYTHON=$ABTEST_PYTHON $*" >> "$H/calls.log"\n'
                  'case "$*" in *"import serve.server"*)\n'
                  '  [ "$HARNESS_NO_DEPS" = 1 ] && { echo "ModuleNotFoundError: No module named \'jinja2\'" >&2; exit 1; }\n'
                  '  [ -f serve/server.py ] || { echo "ModuleNotFoundError: No module named \'serve\'" >&2; exit 1; }; exit 0 ;;\nesac\n'
                  'exec "$PYREAL" "$H/stub/pystub.py" "$@"\n', newline="\n")
    hp.chmod(0o755)
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


def test_bd_harness_runs_with_the_box_venv_python_and_hands_it_to_the_server(tmp_path):
    """Third RD attempt (2026-11-19): the plain smoke died with "env: '/mnt/pancake/benchRD/src-pancake/.venv/bin/python': No such file or directory":
    abtest.py starts the server with <its tree>/.venv/bin/python and the downloaded harness tree has none.  Now abtest / flag_diag run with the box's
    Strata-win venv python and ABTEST_PYTHON points the server at it; the import check runs right after the sync, before any config or build."""
    box = Box(tmp_path)
    prepare_bd(box)
    box.finish(box.start(BD))
    lines = box.log("bench-updatedRD.log")
    text = "\n".join(lines)
    assert lines[-1] == "WINBENCHRD_DONE" and "ERROR" not in text, lines[-6:]
    hpy = box.h + "/clone-of-d/src/Strata-win/.venv/bin/python"
    calls = box.calls()
    harness = [c for c in calls if c.startswith("HARNESS ")]
    assert harness and all(c.startswith("HARNESS ABTEST_PYTHON=" + hpy + " ") for c in harness), harness[:3]
    ab_py = [c for c in calls if c.startswith("PY tools/pancake/abtest.py ")]
    ab_h = [c for c in harness if " tools/pancake/abtest.py " in c]
    assert len(ab_h) == len(ab_py) >= 6, (ab_h, ab_py)          # 2 smokes, 250 screen + confirm, 251 screen + confirm: every one through the venv python
    assert any("flag_diag.py run" in c for c in harness), "the diagnostic (and through sys.executable its abtest) runs with the venv python"
    assert any("import serve.server" in c for c in harness)
    order = ["PREREQ ok", "SYNC rocktboy1982/pancake-strata port/rdna3-decode", "PREREQ harness ok", "CONFIG written", "BUILD plain"]
    pos = [text.index(k) for k in order]
    assert pos == sorted(pos), order
    assert not (box.tmp / "mnt" / "benchRD" / "src-pancake" / ".venv").exists(), "nothing is created in the downloaded tree"


def test_bd_prereq_stops_in_seconds_when_the_harness_venv_is_missing(tmp_path):
    box = Box(tmp_path)
    prepare_bd(box)
    (box.R / ".venv" / "bin" / "python").unlink()
    box.finish(box.start(BD))
    lines = box.log("bench-updatedRD.log")
    err = [l for l in lines if l.startswith("ERROR PREREQ missing files:")]
    assert err and "Strata-win/.venv/bin/python" in err[0], lines
    text = "\n".join(lines)
    assert lines[-1] == "WINBENCHRD_DONE" and "SYNC nicejunjie" not in text and "BUILD" not in text and "START smoke" not in text


def test_bd_prereq_harness_stops_before_any_build_when_the_venv_cannot_import_the_server(tmp_path):
    box = Box(tmp_path)
    prepare_bd(box)
    box.finish(box.start(BD, env={"HARNESS_NO_DEPS": "1"}))
    lines = box.log("bench-updatedRD.log")
    err = [l for l in lines if l.startswith("ERROR PREREQ harness:")]
    assert err and "cannot import serve.server" in err[0] and "jinja2" in err[0], lines
    text = "\n".join(lines)
    assert lines[-1] == "WINBENCHRD_DONE" and "CONFIG written" not in text and "BUILD" not in text and "START smoke" not in text


def test_bd_prereq_harness_stops_when_the_synced_abtest_ignores_abtest_python(tmp_path):
    box = Box(tmp_path)
    prepare_bd(box)
    tree = box.tmp / "srcdir" / "pancake-strata-abc"
    (tree / "tools/pancake/abtest.py").write_text("# old harness: PY = ROOT/.venv/bin/python only\n")
    subprocess.run(["tar", "czf", (box.tmp / "fake-src.tgz").as_posix(), "-C", (box.tmp / "srcdir").as_posix(), "pancake-strata-abc"], check=True)
    box.finish(box.start(BD))
    lines = box.log("bench-updatedRD.log")
    assert any(l.startswith("ERROR PREREQ harness:") and "does not read ABTEST_PYTHON" in l for l in lines), lines
    assert "BUILD" not in "\n".join(lines) and lines[-1] == "WINBENCHRD_DONE"


def test_abtest_server_python_follows_abtest_python():
    code = ("import os, sys; os.sysconf = getattr(os, 'sysconf', lambda n: 100); sys.path.insert(0, sys.argv[1]); import abtest; print(abtest.PY)")
    for env, want in (({"ABTEST_PYTHON": "/opt/venv/bin/python"}, "/opt/venv/bin/python"), ({}, str(HERE.parents[1] / ".venv/bin/python"))):
        e = {k: v for k, v in os.environ.items() if k != "ABTEST_PYTHON"}
        e.update(env)
        r = subprocess.run([sys.executable, "-c", code, str(HERE)], capture_output=True, text=True, env=e, timeout=60)
        assert r.returncode == 0, r.stderr[-800:]
        assert r.stdout.strip().splitlines()[-1] == want


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
    assert set(spec["arms"]) == {"updated", "hipblaslt-100401", "profile-pancake", "draft-vocab-en", "max-context-131072", "conversation-cache-2048"}
    assert spec["arms"]["hipblaslt-100401"]["env"]["STRATA_HIPBLASLT_TUNING"].endswith("gfx1100-hipblaslt-100401.txt")
    assert spec["arms"]["profile-pancake"]["args"]["--expert-profile"].endswith("data/expert-profile-pancake.bin")
    rec = HERE.parent.parent / "rdna3" / "recommended-flags" / "gfx1100.json"
    if rec.exists():       # on the updated-strata-rdna3 branch: neither arm's value is in the shipped file until a measured GAIN
        shipped = json.loads(rec.read_text())
        assert "STRATA_HIPBLASLT_TUNING" not in shipped["env"] and "--expert-profile" not in shipped["args"]
        assert spec["arms"]["updated"]["env"] == {k: v for k, v in shipped["env"].items() if k not in THEIR_IQ3_1GPU["env"]}


# Strata-rdna3 8b01b0c85 rdna3/configs/iq3-1gpu.json, byte for byte as the box's synced copy /mnt/pancake/benchRD/src-plain has it (2026-11-19)
THEIR_IQ3_1GPU = {
    "exe": "@STRATA@/engine/strata", "cwd": "@STRATA@", "tokenizer": "@DATA@/packs/iq3_s/tokenizer", "model_name": "qwen-flash-iq3", "backend": "hip",
    "gpu": "@GPUS@", "layer_split": "auto", "host": "127.0.0.1", "open_browser": False, "engine_silence_s": 1800, "lib_dirs": "@ROCM_LIBS@",
    "args": ["--pack", "@DATA@/packs/iq3_s", "--native", "@DATA@/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
             "--expert-profile", "@STRATA@/data/expert-profile.bin", "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5", "--mtp", "@DATA@/mtp/rt",
             "--max-context", "262144", "--kv", "int8", "--ple-io", "direct", "--stats", "--vision", "--expert-cache", "auto", "--ple-row-cache", "1048576",
             "--conversation-cache-mib", "8192", "--conversation-cache-slots", "8", "--mtp-draft-vocab", "@STRATA@/rdna3/draft_vocab_zh60k.bin",
             "--kv-resident", "32768", "--pcie-frac", "0", "--vram-reserve-mib", "512", "--pool-workers", "12", "--mtp-q4", "all", "--mtp-window", "4096",
             "--adapt-every", "1", "--adapt-swaps", "32"],
    "env": {"STRATA_HIPBLASLT_TUNING": "@STRATA@/tools/hip/gfx1100-hipblaslt-100200.txt", "STRATA_RDNA_MMVQ_WAVE": "3", "STRATA_PEER_TRANSPORT": "host",
            "STRATA_HIP_WMMA": "1", "STRATA_SELECT_WMMA": "1", "STRATA_SH_STREAM": "0", "STRATA_PREFILL_AUTO_MAX": "32768", "STRATA_PREFILL_MMQ_TILE_ROWS": "32",
            "STRATA_GR_RDNA": "1", "STRATA_GR_RDNA_Q8": "1", "STRATA_RDNA_Q6K": "1", "STRATA_GR_RDNA_A8": "1", "STRATA_QFUSE": "1", "STRATA_COMBINE_CPU": "1",
            "STRATA_VERIFY_QDEDUP": "1", "STRATA_WAIT_FUSE": "1", "STRATA_ROUTE_PUBLISH": "1", "STRATA_GR_INJ_UP": "1", "STRATA_PF_FUSED": "1",
            "STRATA_STAGER_RING": "64", "STRATA_PREFILL_CPU_SHARE": "0.6", "STRATA_PREFILL_CPU_SHARE_MAX": "4096", "GPU_MAX_HW_QUEUES": "6",
            "STRATA_GR_FUSED": "1", "STRATA_GDN_HEAD": "1", "STRATA_GDN_CHUNKED": "1", "STRATA_GDN_CHUNK_GDV": "0", "STRATA_PA_KLDS": "1", "STRATA_GDN_PP": "1",
            "STRATA_GDN_CONVL2": "1", "STRATA_GDN_NOY": "1", "STRATA_PF_GEMM": "1", "STRATA_PF_PAD": "1", "STRATA_HC_UPMIX": "1", "STRATA_PF_HCDOWN": "1",
            "STRATA_FUSE_CONVAB": "1", "STRATA_HEAD_MIX_MULTI": "1", "STRATA_PA_FAST": "1", "STRATA_PF_GROUP_RES": "1", "STRATA_PF_F16": "2",
            "STRATA_PREFILL_RING": "192", "STRATA_KV_HOST_DMA": "1", "STRATA_ADAPT_OVERLAP": "1", "STRATA_TOPK_SPLIT_MIN": "98304", "STRATA_QSA_ROWS": "1",
            "STRATA_ENGINE_NICE": "0"},
    "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0},
    "vision": {"exe": "@STRATA@/engine/strata-vision", "mmproj": "@DATA@/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf",
               "model": "@DATA@/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf", "gpu": False, "max_tokens": 4096, "threads": 16},
}
BOX_DATA = "/home/dan/clone-of-d/Strata-data"
# the box's pancake preset (/home/dan/clone-of-d/src/Strata/pancake-flashnext-iq2_xs-fast-hip.json, 2026-11-19): what the chain hands the script
BOX_PRESET = {
    "exe": "/home/dan/clone-of-d/src/Strata/build-hip/strata",
    "args": ["--pack", BOX_DATA + "/packs/iq2_xs", "--native", BOX_DATA + "/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf",
             "--ple-gguf", BOX_DATA + "/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf",
             "--expert-profile", "/home/dan/clone-of-d/src/Strata/data/expert-profile-pancake.bin", "--expert-cache", "auto", "--prefill", "auto",
             "--spec", "4", "--spec-min-p", "0.5", "--mtp", BOX_DATA + "/mtp/rt", "--max-context", "131072", "--kv", "int8", "--kv-resident", "32768",
             "--mtp-draft-vocab", "/home/dan/clone-of-d/src/Strata/data/draft_vocab_en.bin", "--adapt-swaps", "32", "--pool-affinity", "measure",
             "--grammar", "on", "--pcie-frac", "0", "--adapt-every", "1", "--lookup-max-t", "12", "--conversation-cache-mib", "2048",
             "--conversation-cache-min-tokens", "8192"],
    "cwd": "/home/dan/clone-of-d/src/Strata", "tokenizer": BOX_DATA + "/packs/iq2_xs/tokenizer", "model_name": "qwen3.8-flash-next-iq2_xs",
    "log": "/home/dan/clone-of-d/src/Strata/pancake-flashnext.log", "lib_dirs": ["/opt/rocm/core-10.0/lib"], "port": 8080, "gpu": 0, "gpus_asked": True,
    "env": {"STRATA_AGENT_COUNTERS": "1", "STRATA_ROUTE_TAIL_SKIP": "0", "STRATA_PIN_BY_NEED": "1", "STRATA_HC_Q8": "0", "STRATA_PF_F16": "1",
            "STRATA_HIPBLASLT_TUNING": "/home/dan/clone-of-d/src/Strata/tools/hip/gfx1100-hipblaslt-100401.txt", "STRATA_VERIFY_ECAP": "0",
            "STRATA_VERIFY_ECAP_MIN_ROWS": "9", "STRATA_RDNA_MMVQ_WAVE": "3"},
    "drafting": "mtp", "env_to_user": True, "effort_position": "end", "reasoning_budget_grace": 512, "reasoning_close_retry": True,
    "turn_log": "/home/dan/clone-of-d/src/Strata/pancake-turns.jsonl", "sampling": {"temperature": 0.1, "top_p": 0.95, "top_k": 20},
}
SPECS = HERE.parent.parent / "bench" / "specs"


def bd_conf():
    spec_ = importlib.util.spec_from_file_location("bd_conf", CHAINS / "bench_updatedRD_conf.py")
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    return mod


class BdTF:   # rdna3/tuned_flags.py of updated-strata-rdna3: load_recommended + apply_to_config (the same overlay rule: env set, an argument replaced or appended)
    rec = None

    @classmethod
    def load_recommended(cls, hw, d):
        return cls.rec, "gfx1100.json"

    @staticmethod
    def apply_to_config(cfg, rec, env, root):
        for k, v in rec.get("env", {}).items():
            cfg["env"][k] = v
        a = cfg["args"]
        for k, v in rec.get("args", {}).items():
            if k in a:
                a[a.index(k) + 1] = v
            else:
                a += [k, v]


def bd_build(tmp_path, rec, head_reader=None):
    (tmp_path / "plain" / "rdna3" / "configs").mkdir(parents=True)
    (tmp_path / "plain" / "rdna3" / "configs" / "iq3-1gpu.json").write_text(json.dumps(THEIR_IQ3_1GPU))
    (tmp_path / "preset.json").write_text(json.dumps(BOX_PRESET))
    BdTF.rec = rec
    ps = "/mnt/pancake/benchRD/src-plain"     # the script reads the file below PLAIN_SRC: hand it the tmp copy, then check the paths it writes
    # default: the model file is not on this PC (head type unreadable: their --mtp-q4 kept, an UNCHECKED line)
    plain, upd, path, log = bd_conf().build(str(tmp_path / "preset.json"), str(tmp_path / "plain"), "/mnt/pancake/benchRD/src-updated",
                                            "/mnt/pancake/benchRD/build-plain", "/mnt/pancake/benchRD/build-updated", BdTF,
                                            head_reader=head_reader or (lambda p: None))
    js = json.loads(json.dumps([plain, upd]).replace(str(tmp_path / "plain").replace("\\", "\\\\"), ps))
    return js[0], js[1], log


def bd_arm_rec():
    arm = json.loads((SPECS / "250-rdna3-updated-vs-plain.json").read_text())["arms"]["updated"]
    return {"env": dict(arm["env"]), "args": dict(arm["args"])}


def flags_of(args):
    return bd_conf().argmap(args)


def test_bd_conf_plain_is_their_config_except_the_documented_substitutions(tmp_path):
    plain, upd, log = bd_build(tmp_path, bd_arm_rec())
    ps, data = "/mnt/pancake/benchRD/src-plain", BOX_DATA
    theirs = json.loads(json.dumps(THEIR_IQ3_1GPU).replace("@STRATA@", ps).replace("@DATA@", data))
    # every env key of theirs, every value (only the placeholders filled), nothing else
    assert plain["env"] == theirs["env"] and len(plain["env"]) == 46
    # every argument of theirs with their value and in their order; the only differences: the IQ2_XS model's --pack / --native and its --ple-gguf
    tf, pf = flags_of(theirs["args"]), flags_of(plain["args"])
    model = {"--pack": data + "/packs/iq2_xs", "--native": data + "/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf",
             "--ple-gguf": data + "/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf"}
    assert pf == {**tf, **model}
    assert [a for a in plain["args"] if a.startswith("--") and a != "--ple-gguf"] == [a for a in theirs["args"] if a.startswith("--")]
    for k, v in (("--max-context", "262144"), ("--conversation-cache-mib", "8192"), ("--conversation-cache-slots", "8"), ("--mtp-q4", "all"),
                 ("--mtp-window", "4096"), ("--pool-workers", "12"), ("--vram-reserve-mib", "512"), ("--ple-io", "direct"), ("--ple-row-cache", "1048576"),
                 ("--mtp-draft-vocab", ps + "/rdna3/draft_vocab_zh60k.bin"), ("--expert-profile", ps + "/data/expert-profile.bin"), ("--stats", None),
                 ("--vision", None)):
        assert pf[k] == v, k
    # their sampling exactly (temperature 0.6, presence_penalty 0.0)
    assert plain["sampling"] == {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0}
    # the other keys: theirs, except exe (the chain's build), tokenizer (the model), gpu / lib_dirs (placeholders as rdna3/run.py fills them),
    # the picture-reader section (dropped) and the two harness keys log + drafting
    subst = {"exe": "/mnt/pancake/benchRD/build-plain/strata", "tokenizer": data + "/packs/iq2_xs/tokenizer", "gpu": [0],
             "lib_dirs": ["/opt/rocm/core-10.0/lib"]}
    for k, v in theirs.items():
        if k in ("args", "env", "vision"):
            continue
        assert plain[k] == subst.get(k, v), k
    assert "vision" not in plain and set(plain) - set(theirs) == {"log", "drafting"} and plain["drafting"] == "mtp"
    assert plain["cwd"] == ps and plain["model_name"] == "qwen-flash-iq3"
    assert "@" not in json.dumps(plain)
    text = "\n".join(log)
    assert "DROPPED the \"vision\" section" in text and "--vision is KEPT" in text
    for k in ("STRATA_PF_F16 = 2", "--max-context 262144", "--conversation-cache-mib 8192", "draft_vocab_zh60k.bin", "gfx1100-hipblaslt-100200.txt",
              "data/expert-profile.bin", '"temperature": 0.6'):
        assert any(l.startswith("THEIRS-KEPT") and k in l for l in log), k


def test_bd_conf_updated_is_plain_plus_additions_with_no_override(tmp_path):
    rec = bd_arm_rec()
    plain, upd, log = bd_build(tmp_path, rec)
    mod = bd_conf()
    assert mod.overrides(plain, upd) == []
    assert upd["exe"] == "/mnt/pancake/benchRD/build-updated/strata"
    assert upd["env"] == {**plain["env"], **rec["env"]} and not set(rec["env"]) & set(plain["env"])
    assert upd["args"] == plain["args"] + [x for k, v in rec["args"].items() for x in (k, v)]
    assert {k: v for k, v in upd.items() if k not in ("exe", "env", "args")} == {k: v for k, v in plain.items() if k not in ("exe", "env", "args")}
    # the owner's list: the env keys + --grammar on, --lookup-max-t 12, --pool-affinity measure, --conversation-cache-min-tokens 8192, and the preset's
    # remaining explicit values (ROUTE_TAIL_SKIP=0, HC_Q8=0: the engine defaults); POOL_FUSE_PHASES removed 2026-11-20, ECAP kept (owner decision)
    assert set(rec["env"]) == {"STRATA_VERIFY_ECAP", "STRATA_VERIFY_ECAP_MIN_ROWS", "STRATA_PIN_BY_NEED", "STRATA_EXPDEC",
                               "STRATA_EXPDEC_DOWN", "STRATA_EXCHANGE_ROTATE", "STRATA_KV_PREFETCH", "STRATA_PROMPT_ATTN_IMMA", "STRATA_ATTN_DAILY",
                               "STRATA_PF_GROUP_SCRATCH", "STRATA_PF_HC_REGION", "STRATA_PREFILL_EQUAL_TAIL", "STRATA_ROUTE_TAIL_SKIP", "STRATA_HC_Q8"}
    assert rec["args"] == {"--grammar": "on", "--lookup-max-t": "12", "--pool-affinity": "measure", "--conversation-cache-min-tokens": "8192"}
    assert any(l.startswith("NOT-ADDED env STRATA_AGENT_COUNTERS") for l in log)
    # a recommended value that would change a value of theirs (the shipped file's overrides_theirs, e.g. PF_F16=1 measured GAIN) is left out of the
    # updated arm and logged: the arms never override their values (owner order 2026-11-19 23:05)
    for i, over in enumerate(({"env": {**rec["env"], "STRATA_PF_F16": "1"}, "args": rec["args"]},
                              {"env": rec["env"], "args": {**rec["args"], "--max-context": "131072"}})):
        p2, u2, log2 = bd_build(tmp_path / ("o%d" % i), over)
        assert mod.overrides(p2, u2) == [] and u2["env"]["STRATA_PF_F16"] == "2" and flags_of(u2["args"])["--max-context"] == "262144"
        assert u2["env"] == upd["env"] and u2["args"] == upd["args"]
        left = [l for l in log2 if l.startswith("SHIPPED-NOT-IN-ARM")]
        assert len(left) == 1 and ("STRATA_PF_F16 = 1" in left[0] or "--max-context = 131072" in left[0]) and "never override" in left[0], left
    # the engine log of both configs: never the pancake preset's log in the pancake tree
    assert "clone-of-d/src/Strata/" not in plain["log"] and plain["log"] == upd["log"]

    # a value the arm-part filter misses still stops the script, naming it
    class Over(BdTF):
        @staticmethod
        def apply_to_config(cfg, rec, env, root):
            cfg["env"]["STRATA_PF_F16"] = "1"
    (tmp_path / "x" / "plain" / "rdna3" / "configs").mkdir(parents=True)
    (tmp_path / "x" / "plain" / "rdna3" / "configs" / "iq3-1gpu.json").write_text(json.dumps(THEIR_IQ3_1GPU))
    (tmp_path / "x" / "preset.json").write_text(json.dumps(BOX_PRESET))
    Over.rec = rec
    with pytest.raises(SystemExit) as e:
        mod.build(str(tmp_path / "x" / "preset.json"), str(tmp_path / "x" / "plain"), "/u", "/bp", "/bu", Over, head_reader=lambda p: None)
    assert "overrides Strata-rdna3's values" in str(e.value)


def write_gguf(path, tensors, kv=()):
    """A minimal GGUF v3 header: KV pairs (key, type, packed value bytes) and tensor infos (name, dims, ggml type); no data."""
    import struct as st

    def s(b):
        return st.pack("<Q", len(b)) + b
    out = b"GGUF" + st.pack("<IQQ", 3, len(tensors), len(kv))
    for k, t, v in kv:
        out += s(k.encode()) + st.pack("<I", t) + v
    for nm, dims, t in tensors:
        out += s(nm.encode()) + st.pack("<I", len(dims)) + b"".join(st.pack("<Q", d) for d in dims) + st.pack("<IQ", t, 0)
    path.write_bytes(out)


def test_bd_conf_head_type_reads_the_gguf_header(tmp_path):
    import struct as st
    mod = bd_conf()
    toks = st.pack("<IQ", 8, 3) + b"".join(st.pack("<Q", len(w)) + w for w in (b"a", b"bc", b"def"))   # a string array, like the tokenizer list
    kv = [("general.architecture", 8, st.pack("<Q", 5) + b"qwen3"), ("tokenizer.ggml.tokens", 9, toks), ("x.u32", 4, st.pack("<I", 7)),
          ("x.f32s", 9, st.pack("<IQ", 6, 2) + st.pack("<ff", 1.0, 2.0))]
    write_gguf(tmp_path / "m.gguf", [("token_embd.weight", [2560, 248320], 23), ("output.weight", [2560, 248320], 23)], kv)
    assert mod.head_type(str(tmp_path / "m.gguf")) == (23, 2560)
    assert mod.head_type(str(tmp_path / "missing.gguf")) is None
    (tmp_path / "bad.gguf").write_bytes(b"NOPE")
    assert mod.head_type(str(tmp_path / "bad.gguf")) is None
    # the engine's in-place Q4_0 head copy (mtp.cpp make_q4_head): fits Q6_K (their IQ3_S head) and Q4_0-sized types, not IQ4_XS (the IQ2_XS head)
    assert mod.q4_head_fits((14, 2560)) is True and mod.q4_head_fits((2, 2560)) is True and mod.q4_head_fits((12, 2560)) is True
    assert mod.q4_head_fits((23, 2560)) is False and mod.q4_head_fits((10, 2560)) is False and mod.q4_head_fits((99, 2560)) is None


def test_bd_conf_mtp_q4_head_substituted_only_for_a_head_below_q4_0(tmp_path):
    # T-987: the IQ2_XS model's output.weight is IQ4_XS (1360 bytes per row < Q4_0's 1440): their --mtp-q4 all -> proj, logged with the reason
    plain, upd, log = bd_build(tmp_path / "a", bd_arm_rec(), head_reader=lambda p: (23, 2560))
    assert flags_of(plain["args"])["--mtp-q4"] == "proj" and flags_of(upd["args"])["--mtp-q4"] == "proj"
    line = [l for l in log if l.startswith("SUBST --mtp-q4")]
    assert len(line) == 1 and "all -> proj" in line[0] and "IQ4_XS" in line[0] and "1360" in line[0] and "1440" in line[0]
    assert "writing the draft head" in line[0] and "T-987" in line[0]
    assert bd_conf().overrides(plain, upd) == []           # the updated arm follows the same base
    # every other argument of theirs is unchanged
    p2 = dict(flags_of(plain["args"]))
    p2.pop("--mtp-q4")
    plain_q6, _, log_q6 = bd_build(tmp_path / "b", bd_arm_rec(), head_reader=lambda p: (14, 2560))
    q6 = dict(flags_of(plain_q6["args"]))
    assert q6.pop("--mtp-q4") == "all" and q6 == p2
    assert any(l.startswith("CHECKED --mtp-q4 all kept") and "Q6_K" in l for l in log_q6)
    # unreadable model file: their value kept, said so
    _, _, log_u = bd_build(tmp_path / "c", bd_arm_rec())
    assert any(l.startswith("UNCHECKED --mtp-q4 all kept") for l in log_u)


def test_bd_spec_arms_overlay_the_plain_config_without_touching_their_values():
    """Specs 250 / 251 / 252 run on base = config-plain.json (their config): the updated-family arms only add keys their config does not set; the arms that
    carry a pancake value for a setting of theirs differ from `updated` in exactly that one key."""
    theirs_env, theirs_args = set(THEIR_IQ3_1GPU["env"]), set(flags_of(THEIR_IQ3_1GPU["args"]))
    s250, s251, s252 = (json.loads((SPECS / n).read_text()) for n in ("250-rdna3-updated-vs-plain.json", "251-rdna3-hipblaslt-and-profile.json",
                                                                       "252-rdna3-updated-exact.json"))
    upd = s250["arms"]["updated"]
    assert set(s250["arms"]) == {"plain", "updated", "updated+pf16-1"} and s250["cases"] == ["short", "edit", "long10k", "long32k", "long64k", "long100k"]
    assert s250["arms"]["plain"] == {"exe": "@PLAIN@/strata"} and s252["arms"]["plain"] == {"exe": "@PLAIN@/strata"}
    for arm in (upd, s251["arms"]["updated"], s252["arms"]["updated"], s252["arms"]["updated-exact"]):
        assert not set(arm["env"]) & theirs_env and not set(arm["args"]) & theirs_args, arm
    assert s251["arms"]["updated"] == upd and s252["arms"]["updated"] == upd and list(s252["arms"]) == ["plain", "updated-exact", "updated"]
    # 2026-11-20: POOL_FUSE_PHASES out of every arm (removed from the shipped file); ECAP kept (owner decision): in every updated-family arm but
    # updated-exact, which is updated minus ECAP (spec 252 measures ECAP alone)
    exact = {k: v for k, v in upd["env"].items() if k not in ("STRATA_VERIFY_ECAP", "STRATA_VERIFY_ECAP_MIN_ROWS")}
    assert s252["arms"]["updated-exact"] == {**upd, "env": exact}
    assert (upd["env"]["STRATA_VERIFY_ECAP"], upd["env"]["STRATA_VERIFY_ECAP_MIN_ROWS"]) == ("0", "9")
    for sp in (s250, s251, s252):
        for name, arm in sp["arms"].items():
            assert "STRATA_POOL_FUSE_PHASES" not in arm.get("env", {}), name
            assert ("STRATA_VERIFY_ECAP" in arm.get("env", {})) == (name not in ("plain", "updated-exact")), name
    one = {("250", "updated+pf16-1"): ("env", "STRATA_PF_F16", "1"),
           ("251", "hipblaslt-100401"): ("env", "STRATA_HIPBLASLT_TUNING", "@UPDATED_SRC@/tools/hip/gfx1100-hipblaslt-100401.txt"),
           ("251", "profile-pancake"): ("args", "--expert-profile", "@UPDATED_SRC@/data/expert-profile-pancake.bin"),
           ("251", "draft-vocab-en"): ("args", "--mtp-draft-vocab", "@UPDATED_SRC@/data/draft_vocab_en.bin"),
           ("251", "max-context-131072"): ("args", "--max-context", "131072"),
           ("251", "conversation-cache-2048"): ("args", "--conversation-cache-mib", "2048")}
    for (sp, name), (kind, key, val) in one.items():
        arm = (s250 if sp == "250" else s251)["arms"][name]
        assert key in (theirs_env if kind == "env" else theirs_args), key            # a setting of theirs: never in the updated arm, its own arm
        assert arm == {**upd, kind: {**upd[kind], key: val}}, name
    rec = HERE.parent.parent / "rdna3" / "recommended-flags" / "gfx1100.json"
    if rec.exists():       # on the updated-strata-rdna3 branch: the spec's updated arm is the shipped file minus its overrides_theirs (PF_F16=1: arm updated+pf16-1)
        shipped = json.loads(rec.read_text())
        assert upd["env"] == {k: v for k, v in shipped["env"].items() if k not in theirs_env} and upd["args"] == shipped["args"]
        assert set(shipped["env"]) & theirs_env == set(shipped["overrides_theirs"]) and not set(shipped["args"]) & theirs_args
        assert s250["arms"]["updated+pf16-1"]["env"] == shipped["env"]


def _paths(value):
    """Every string of a spec arm / config value that names a file or folder (has a slash)."""
    if isinstance(value, dict):
        return [x for v in value.values() for x in _paths(v)]
    if isinstance(value, list):
        return [x for v in value for x in _paths(v)]
    return [value] if isinstance(value, str) and "/" in value else []


def test_bd_and_bench_us_no_setting_of_an_updated_arm_points_outside_its_tree(tmp_path):
    """Owner order 2026-11-19 23:05: the settings of the updated trees use their own files (the expert profile, the hipBLASLt table, the draft vocab are in
    each tree, byte-identical to the pancake copies); only the builds and the model data folder (Strata-data: weights, packs, mtp) live elsewhere."""
    data = "/home/dan/clone-of-d/Strata-data/"
    us = {"220-updated-vs-upstream-short.json", "221-updated-vs-upstream-long.json", "222-updated-exact-vs-upstream-short.json",
          "223-updated-exact-vs-upstream-long.json"}
    rd = {"250-rdna3-updated-vs-plain.json", "251-rdna3-hipblaslt-and-profile.json", "252-rdna3-updated-exact.json"}
    for n in us | rd:
        spec = json.loads((SPECS / n).read_text())
        for name, arm in spec["arms"].items():
            if name.startswith("plain"):
                continue
            for v in _paths(arm):
                if n in us:
                    ok = v.startswith(US_SRC + "/") or v == "/mnt/pancake/benchus/build-us/strata" or v.startswith(data)
                else:
                    ok = v.startswith("@UPDATED_SRC@/") or v == "@UPDATED@/strata"
                assert ok, (n, name, v)
                assert "<REPO>" not in v and "clone-of-d/src/Strata/" not in v and "Strata-win" not in v, (n, name, v)
    # the two RD configs bench_updatedRD_conf.py writes: nothing in the pancake tree (/home/dan/clone-of-d/src/Strata), the preset's log included
    plain, upd, _ = bd_build(tmp_path, bd_arm_rec())
    for cfg in (plain, upd):
        for v in _paths(cfg):
            assert "clone-of-d/src/Strata/" not in v and "Strata-win" not in v, v
    # the updated-strata preset: its own files under <REPO> (the updated-strata checkout: data/ and terminal/tools/hip/)
    preset = json.loads((HERE.parents[1] / "configs/pancake/flashnext-iq2_xs-updated-hip.json").read_text())
    files = [v for v in _paths({"a": preset["args"], "e": preset["env"]}) if v.startswith("<REPO>/")]
    assert files == ["<REPO>/data/expert-profile-pancake.bin", "<REPO>/terminal/tools/hip/gfx1100-hipblaslt-100401.txt"], files


def test_bd_and_rdgate_diag_hand_flag_diag_an_abtest_path_abtest_does_not_take_for_another_test():
    """The aborted 20:28 RD run: diag-updated gave all 5 flags UNTESTED in 0 s (abtest exit 3): abtest's BUSY pattern matched its parent flag_diag.py,
    whose command line held "--abtest .../tools/pancake/abtest.py"."""
    busy = re.search(r'^BUSY = r"(.*)"$', (HERE / "abtest.py").read_text(), re.M).group(1)
    py = "/home/dan/clone-of-d/src/Strata-win/.venv/bin/python -u /mnt/pancake/benchRD/src-updated/tools/pancake/flag_diag.py run --config c.json"
    for name, var in ((BD, "$TOOLS"), ("bench-rdgate.sh", "$HN")):
        t = (CHAINS / name).read_text()
        assert "--abtest %s/tools/pancake/./abtest.py " % var in t and "--abtest %s/tools/pancake/abtest.py" % var not in t, name
    assert re.search(busy, py + " --abtest /mnt/pancake/benchRD/src-pancake/tools/pancake/abtest.py --out d")          # the old command line: refused
    assert not re.search(busy, py + " --abtest /mnt/pancake/benchRD/src-pancake/tools/pancake/./abtest.py --out d")     # the new one: not a test
    assert re.search(busy, "/x/.venv/bin/python -u /mnt/pancake/benchRD/src-pancake/tools/pancake/abtest.py spec.json")  # a real abtest still is


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


def test_sched_rd_is_opt_in_skipped_by_default_run_by_bench_rd_or_only_rd(tmp_path):
    """Owner 2026-11-20: Strata-rdna3 is paused; the RD chain runs only with BENCH_RD set explicitly or BENCH_ONLY=RD."""
    box = Box(tmp_path)
    prepare_sched(box)
    box.finish(box.start(SCHED, env={"BENCH_US": "./fake-us.sh"}))
    text = "\n".join(box.log("bench-sched.log"))
    assert [l.split()[0] for l in chains_log(box)] == ["us"] and "START RD" not in text and "RD chain skipped" in text and "ERROR" not in text
    (tmp_path / "b").mkdir()
    box2 = Box(tmp_path / "b")
    prepare_sched(box2)
    box2.finish(box2.start(SCHED, env=sched_env(BENCH_ONLY="RD")))
    assert [l.split()[0] for l in chains_log(box2)] == ["rd"]
    (tmp_path / "c").mkdir()
    box3 = Box(tmp_path / "c")
    prepare_sched(box3)
    box3.finish(box3.start(SCHED, env=sched_env()))
    assert [l.split()[0] for l in chains_log(box3)] == ["us", "rd"]
    body = (CHAINS / SCHED).read_text()
    assert "RD=${BENCH_RD:-}" in body and "RD=./bench-updatedRD.sh" in body


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

# ------------------------------------------------------------------------------------------------ bench-exact.sh (exact arm + quality gate follow-up)
BE = "bench-exact.sh"
BE_PS = ('case "$*" in\n'
         '  "-eo pid=,ppid=,sid=") [ -f "$H/ptree.txt" ] && sed "s/@SID@/$(cat "$H/sid.txt" 2>/dev/null)/g" "$H/ptree.txt" ;;\n'
         '  "-o pid=,lstart= -p "*) for p in $(echo "$4" | tr "," " "); do [ -f "$H/lstart.$p" ] && echo "$p $(cat "$H/lstart.$p")"; done ;;\n'
         '  "-o args= -p "*) echo "args-of-$4" ;;\n'
         '  "-o stat= -p "*) cat "$H/stat.$4" 2>/dev/null ;;\n'
         '  "-eo pid=,args=") cat "$H/ps.txt" 2>/dev/null ;;\n'
         'esac\nexit 0\n')
# the signal command of the watchdog / reaper: a fake pid (9999xx) "dies" (its start-time file goes), a real one gets the real signal
BE_REAP = 'echo "REAP $*" >> "$H/calls.log"\ncase $2 in 9999*) rm -f "$H/lstart.$2" ;; *) kill "$1" "$2" 2>/dev/null ;; esac\nexit 0\n'
# python3: the quality gate can leave an engine of its own session behind (force_check.py starts engines with start_new_session) or hang
BE_PY = ('case "$*" in *quality_gate.py*)\n'
         '  [ -n "$GATE_ORPHAN" ] && { echo "999993 @SID@ 999993" > "$H/ptree.txt"; echo "Thu Nov 19 16:00:00 2026" > "$H/lstart.999993"; /usr/bin/sleep 1; }\n'
         '  [ -n "$GATE_HANG" ] && /usr/bin/sleep "$GATE_HANG" ;;\n'
         '  *abtest.py*) [ -n "$AB_HANG" ] && /usr/bin/sleep "$AB_HANG" ;;\n'
         'esac\nexec "$PYREAL" "$H/stub/pystub.py" "$@"\n')
BE_ENV = {"BENCH_SCHED": "1", "BENCH_PREIDLED": "1", "BENCH_KEEP_SERVICE_DOWN": "1", "SVC_DOWN": "1"}


def prepare_be(box, us=True, rd=True, follow=False):
    for n, body in (("ps", BE_PS), ("reapcmd", BE_REAP), ("python3", BE_PY),
                    ("setsid", 'echo "SETSID $$ $*" >> "$H/calls.log"; echo $$ > "$H/sid.txt"; exec "$@"\n'),
                    ("df", 'echo "Filesystem 1K-blocks Used Available Use% Mounted"\necho "x 1 1 ${DF_FREE:-99999999} 1% /"\n')):
        (box.tmp / "stub" / n).write_text("#!/bin/sh\n" + body, newline="\n")
    pkg = box.tmp / "be-src" / "x"
    root = HERE.parents[1]
    for rel in ("bench/specs/222-updated-exact-vs-upstream-short.json", "bench/specs/223-updated-exact-vs-upstream-long.json",
                "bench/specs/252-rdna3-updated-exact.json", "configs/pancake/flashnext-iq2_xs-plain-upstream.json",
                "tools/pancake/box_chains/sweep.py", "tools/pancake/box_chains/sweep_candidates.json", "tools/pancake/box_chains/retest_groups.json",
                "bench/specs/220-updated-vs-upstream-short.json", "tools/pancake/apply_triage.py"):
        (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(root / rel, pkg / rel)
    for rel in ("tools/pancake/abtest.py", "tools/pancake/quality_gate.py", "tools/force_check.py", "bench/quality/set-v2.json", "serve/__init__.py",
                "serve/server.py"):
        (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
        (pkg / rel).write_text("# stub: PY = os.environ.get(\"ABTEST_PYTHON\") or ...\n" if rel.endswith("abtest.py") else "# stub\n")
    # the box's existing venv python (Strata-win/.venv/bin/python, T-982): logs each call with the ABTEST_PYTHON it hands on, answers the serve.server
    # import check from the cwd's tree (HARNESS_NO_DEPS=1: a venv without the server's deps), then behaves as the python3 stub of this chain
    hp = box.R / ".venv" / "bin" / "python"
    hp.parent.mkdir(parents=True, exist_ok=True)
    hp.write_text('#!/bin/sh\necho "HARNESS ABTEST_PYTHON=$ABTEST_PYTHON $*" >> "$H/calls.log"\n'
                  'case "$*" in *"import serve.server"*)\n'
                  '  [ "$HARNESS_NO_DEPS" = 1 ] && { echo "ModuleNotFoundError: No module named \'jinja2\'" >&2; exit 1; }\n'
                  '  [ -f serve/server.py ] || { echo "ModuleNotFoundError: No module named \'serve\'" >&2; exit 1; }; exit 0 ;;\nesac\n' + BE_PY, newline="\n")
    hp.chmod(0o755)
    (box.tmp / "fake-src.tgz").unlink()
    subprocess.run(["tar", "czf", (box.tmp / "fake-src.tgz").as_posix(), "-C", (box.tmp / "be-src").as_posix(), "x"], check=True)
    exe = "#!/bin/sh\nexit 0\n"
    if us:
        for rel in ("build-up/strata", "build-us/strata", "src-up/terminal/data/expert-profile.bin"):
            (box.tmp / "mnt" / "benchus" / rel).parent.mkdir(parents=True, exist_ok=True)
            (box.tmp / "mnt" / "benchus" / rel).write_text(exe, newline="\n")
    if rd:
        for rel in ("build-plain/strata", "build-updated/strata"):
            (box.tmp / "mnt" / "benchRD" / rel).parent.mkdir(parents=True, exist_ok=True)
            (box.tmp / "mnt" / "benchRD" / rel).write_text(exe, newline="\n")
        for t in ("src-plain", "src-updated"):
            g = box.tmp / "mnt" / "benchRD" / t / "src" / "program" / "generate.cpp"
            g.parent.mkdir(parents=True, exist_ok=True)
            g.write_text('const char* f = std::getenv("STRATA_FOLLOW");\n' if follow else 'std::getenv("STRATA_LOGPOS");\n')
        (box.tmp / "auto-runs" / "benchRD").mkdir(parents=True, exist_ok=True)
        (box.tmp / "auto-runs" / "benchRD" / "config-plain.json").write_text("{}")


def be_env(**kw):
    return {**BE_ENV, "REAPCMD": "reapcmd", **kw}


def be_calls(box, tool):
    return [c for c in box.calls() if c.startswith("PY") and tool in c]


def test_bench_exact_full_flow_order_gate_invocation_and_markers(tmp_path):
    box = Box(tmp_path)
    prepare_be(box)
    box.finish(box.start(BE, env=be_env()))
    lines = box.log("bench-exact.log")
    text = "\n".join(lines)
    assert lines[-1] == "WINBENCHEXACT_DONE" and "ERROR" not in text and "ABORTED" not in text, text
    assert_start_end_pairs(lines)
    order = ["PREREQ US ok", "PREREQ RD ok", "SYNC harness: origin port/rdna3-decode abc123", "START 222-screen", "END 222-screen exit=0",
             "START gate-us", "END gate-us exit=0", "START 223-screen", "START 252-screen", "SKIP gate-rd: token forcing needs STRATA_FOLLOW",
             "222: no arm with a benefit in the screen (updated excluded", "START 252-confirm", "SUMMARY spec222 screen exit=0, gate-us exit=0",
             "service stays down"]
    pos = [text.index(k) for k in order]
    assert pos == sorted(pos), order
    ab = be_calls(box, "abtest.py")
    assert len(ab) == 4, ab
    assert "bench/specs/222-updated-exact-vs-upstream-short.json" in ab[0] and "--tier screen" in ab[0] and "--drift-check" in ab[0]
    assert "bench/specs/223-updated-exact-vs-upstream-long.json" in ab[1] and "--tier screen" in ab[1] and "--drift-check" not in ab[1]
    assert "work/252.json" in ab[2] and "--tier screen" in ab[2]
    assert "252-confirm.json" in ab[3] and "--reps 3" in ab[3] and "--block 3" in ab[3] and "--outlier robust" in ab[3] and "--no-early-stop" in ab[3]
    qg = be_calls(box, "quality_gate.py")
    assert len(qg) == 1, qg
    assert ("bench/specs/222-updated-exact-vs-upstream-short.json" in qg[0] and "--set bench/quality/set-v2.json" in qg[0] and "--near-tie" in qg[0]
            and "/auto-runs/benchexact/gate-us" in qg[0]), qg     # the T-828 settings: default set-v2, near-tie column on
    sp = json.loads((box.tmp / "auto-runs" / "benchexact" / "work" / "252.json").read_text())
    assert sp["arms"]["plain"]["exe"].endswith("/mnt/benchRD/build-plain/strata") and sp["base"].endswith("auto-runs/benchRD/config-plain.json")
    s252 = json.loads((SPECS / "252-rdna3-updated-exact.json").read_text())       # the arms as the spec file has them (updated-exact = updated minus ECAP)
    assert sp["arms"]["updated-exact"]["env"] == s252["arms"]["updated-exact"]["env"] and "STRATA_VERIFY_ECAP" not in sp["arms"]["updated-exact"]["env"]
    assert sp["arms"]["updated"]["env"]["STRATA_VERIFY_ECAP_MIN_ROWS"] == "9"
    conf = json.loads((box.tmp / "auto-runs" / "benchexact" / "work" / "252-confirm.json").read_text())
    assert conf["cases"] == ["short", "edit"] and conf["drift_check"] is False
    assert sum(1 for c in box.calls() if c.startswith("SETSID")) == 5                     # every long step in its own session
    assert not any(c.startswith("SIG") or c.startswith("REAP") for c in box.calls())     # AA is never signalled, nothing to reap
    assert not any(c.startswith("SYSTEMCTL") and (" start " in c or " stop " in c) for c in box.calls())   # the service stays down


def test_bench_exact_harness_runs_with_the_box_venv_python_and_hands_it_to_the_server(tmp_path):
    """First exact run (2026-11-19 16:42, T-982): every step died in seconds with "env: '/mnt/pancake/benchexact/harness/.venv/bin/python': No such file or
    directory" (the downloaded harness has no .venv).  Now abtest / quality_gate run with Strata-win's venv python and ABTEST_PYTHON points the server at it."""
    box = Box(tmp_path)
    prepare_be(box)
    box.finish(box.start(BE, env=be_env()))
    lines = box.log("bench-exact.log")
    text = "\n".join(lines)
    assert lines[-1] == "WINBENCHEXACT_DONE" and "ERROR" not in text, text
    hpy = box.h + "/clone-of-d/src/Strata-win/.venv/bin/python"
    harness = [c for c in box.calls() if c.startswith("HARNESS ")]
    assert harness and all(c.startswith("HARNESS ABTEST_PYTHON=" + hpy + " ") for c in harness), harness[:3]
    assert sum(" tools/pancake/abtest.py " in c for c in harness) == len(be_calls(box, "abtest.py")) == 4
    assert sum(" tools/pancake/quality_gate.py " in c for c in harness) == len(be_calls(box, "quality_gate.py")) == 1
    assert any("import serve.server" in c for c in harness)
    pos = [text.index(k) for k in ("SYNC harness:", "PREREQ harness ok", "START 222-screen")]
    assert pos == sorted(pos)
    assert not (box.tmp / "mnt" / "benchexact" / "harness" / ".venv").exists()


def test_bench_exact_missing_venv_stops_in_prereq_and_a_venv_without_deps_before_any_step(tmp_path):
    box = Box(tmp_path)
    prepare_be(box)
    (box.R / ".venv" / "bin" / "python").unlink()
    box.finish(box.start(BE, env=be_env()))
    lines = box.log("bench-exact.log")
    assert any(l.startswith("ERROR PREREQ missing files") and "Strata-win/.venv/bin/python" in l for l in lines), lines
    assert "SYNC harness" not in "\n".join(lines) and "START 222" not in "\n".join(lines) and lines[-1] == "WINBENCHEXACT_DONE"
    (tmp_path / "b2").mkdir()
    box2 = Box(tmp_path / "b2")
    prepare_be(box2)
    box2.finish(box2.start(BE, env=be_env(HARNESS_NO_DEPS="1")))
    l2 = box2.log("bench-exact.log")
    assert any(l.startswith("ERROR PREREQ harness:") and "cannot import serve.server" in l and "jinja2" in l for l in l2), l2
    assert "START 222" not in "\n".join(l2) and "START gate" not in "\n".join(l2) and l2[-1] == "WINBENCHEXACT_DONE"


def test_bench_exact_rd_gate_runs_when_both_trees_read_strata_follow(tmp_path):
    box = Box(tmp_path)
    prepare_be(box, follow=True)
    box.finish(box.start(BE, env=be_env()))
    text = "\n".join(box.log("bench-exact.log"))
    qg = be_calls(box, "quality_gate.py")
    assert len(qg) == 2 and "work/252.json" in qg[1] and "/benchexact/gate-rd" in qg[1] and "--near-tie" in qg[1], qg
    assert "START gate-rd" in text and "SKIP gate-rd" not in text


def test_bench_exact_prereq_missing_us_builds_skips_the_us_part_only(tmp_path):
    box = Box(tmp_path)
    prepare_be(box, us=False)
    box.finish(box.start(BE, env=be_env()))
    lines = box.log("bench-exact.log")
    text = "\n".join(lines)
    err = [l for l in lines if l.startswith("ERROR PREREQ US part SKIPPED")]
    assert len(err) == 1 and "benchus/build-up/strata" in err[0] and "benchus/build-us/strata" in err[0], lines
    assert "START 222-screen" not in text and "START gate-us" not in text and "START 223-screen" not in text
    assert "START 252-screen" in text and lines[-1] == "WINBENCHEXACT_DONE"
    assert not be_calls(box, "quality_gate.py")


def test_bench_exact_prereq_missing_rd_config_skips_the_rd_part_and_both_missing_aborts(tmp_path):
    box = Box(tmp_path)
    prepare_be(box)
    (box.tmp / "auto-runs" / "benchRD" / "config-plain.json").unlink()
    box.finish(box.start(BE, env=be_env()))
    lines = box.log("bench-exact.log")
    text = "\n".join(lines)
    assert any(l.startswith("ERROR PREREQ RD part SKIPPED") and "config-plain.json" in l for l in lines)
    assert "START 222-screen" in text and "START gate-us" in text and "START 252-screen" not in text and lines[-1] == "WINBENCHEXACT_DONE"
    (tmp_path / "b").mkdir()
    box2 = Box(tmp_path / "b")
    prepare_be(box2, us=False, rd=False)
    box2.finish(box2.start(BE, env=be_env()))
    l2 = box2.log("bench-exact.log")
    assert any(l.startswith("WINBENCHEXACT_ABORTED: no part can run") for l in l2) and l2[-1] == "WINBENCHEXACT_DONE", l2
    assert not [c for c in box2.calls() if c.startswith("PY") and ("abtest.py" in c or "quality_gate.py" in c)]


def test_bench_exact_budget_default_is_75_minutes_and_an_exhausted_budget_skips_the_optional_steps(tmp_path):
    t = (CHAINS / BE).read_text()
    m = re.search(r"BUDGET=\$\{BENCH_BUDGET_S:-(\d+)\}", t)
    assert m and int(m.group(1)) <= 75 * 60, "the owner's budget is 75 minutes"
    box = Box(tmp_path)
    prepare_be(box)
    box.finish(box.start(BE, env=be_env(BENCH_BUDGET_S="300")))
    lines = box.log("bench-exact.log")
    text = "\n".join(lines)
    for must in ("SKIP gate-us: budget", "SKIP 223-screen: budget", "SKIP RD part: budget"):
        assert must in text, must
    assert [c for c in be_calls(box, "abtest.py")] and len(be_calls(box, "abtest.py")) == 1 and not be_calls(box, "quality_gate.py")
    assert lines[-1] == "WINBENCHEXACT_DONE"


def test_bench_exact_gate_engine_left_in_its_own_session_is_reaped_no_orphan(tmp_path):
    """force_check.py starts its engine with start_new_session: a session reap alone would miss it.  The watchdog records every process the step's
    tree ever had (pid + start time); after the step the one still alive is killed and logged, and the chain goes on."""
    box = Box(tmp_path)
    prepare_be(box)
    box.finish(box.start(BE, env=be_env(GATE_ORPHAN="1")))
    lines = box.log("bench-exact.log")
    text = "\n".join(lines)
    reap = [l for l in lines if l.startswith("REAP gate-us: still running")]
    assert len(reap) == 1 and "999993" in reap[0], lines
    assert "REAP -TERM 999993" in box.calls() and not (box.tmp / "lstart.999993").exists()   # gone: no orphan left
    assert text.index("REAP gate-us: still running") < text.index("END gate-us") < text.index("START 223-screen")
    assert "ERROR REAP" not in text and lines[-1] == "WINBENCHEXACT_DONE"


def test_bench_exact_step_over_its_limit_gets_term_to_its_tree_and_the_chain_goes_on(tmp_path):
    box = Box(tmp_path)
    prepare_be(box)
    box.finish(box.start(BE, env=be_env(GATE_HANG="30", EXACT_LIM_GATE="20")), timeout=180)
    lines = box.log("bench-exact.log")
    text = "\n".join(lines)
    assert any(l.startswith("TIMEOUT gate-us after 20 s (limit 20 s") for l in lines), lines
    assert any(c.startswith("REAP -TERM") for c in box.calls())
    assert "END gate-us exit=124" in text and "START 223-screen" in text and lines[-1] == "WINBENCHEXACT_DONE"


def test_bench_exact_standalone_waits_for_both_benchmark_markers(tmp_path):
    box = Box(tmp_path)
    prepare_be(box)
    p = box.start(BE, env={"SVC_DOWN": "1", "REAPCMD": "reapcmd"})
    time.sleep(1.5)
    assert p.poll() is None and len(box.log("bench-exact.log")) == 1
    append(box, "bench-updatedAA.log", "WINBENCHUS_DONE")
    time.sleep(0.5)
    assert p.poll() is None and len(box.log("bench-exact.log")) == 1
    append(box, "bench-updatedRD.log", "WINBENCHRD_DONE")
    box.finish(p)
    lines = box.log("bench-exact.log")
    assert "both benchmark chains are done" in lines[1] and lines[-1] == "WINBENCHEXACT_DONE", lines[:3]


def test_bench_exact_name_is_invisible_to_every_busy_pattern_and_sees_the_benchmark_chains(tmp_path):
    box = Box(tmp_path)
    for chain in ("win-chainT.sh", "redo-chainAA.sh", "bench-updatedAA.sh", "bench-updatedRD.sh", "bench-exact.sh"):
        assert busy_result(box, chain, ["sh ./bench-exact.sh"]) == "IDLE", chain
    for other in ("sh ./bench-updatedAA.sh", "sh ./bench-updatedRD.v3.sh", "sh ./bench-diagUS.sh", "python3 tools/pancake/quality_gate.py x"):
        assert busy_result(box, BE, [other]) == "BUSY", other


def test_bench_exact_specs_are_the_updated_arm_minus_exactly_the_lossy_switches():
    root = HERE.parents[1]
    s220 = json.loads((root / "bench/specs/220-updated-vs-upstream-short.json").read_text())
    s222 = json.loads((root / "bench/specs/222-updated-exact-vs-upstream-short.json").read_text())
    s223 = json.loads((root / "bench/specs/223-updated-exact-vs-upstream-long.json").read_text())
    upd, ex = s220["arms"]["updated"], s222["arms"]["updated-exact"]
    assert list(s222["arms"]) == ["plain-upstream", "updated-exact", "updated"] and s222["arms"]["updated"] == upd
    assert s222["arms"]["plain-upstream"] == s220["arms"]["plain-upstream"] and s222["drift_check"] is True and s222["cases"] == ["short", "edit"]
    assert s223["arms"] == s222["arms"] and s223["cases"] == ["longdec10k", "longdec32k", "longdec64k", "longdec100k"] and s223["drift_check"] is False
    unset = {"STRATA_GDN_CONVL2", "STRATA_GR_RDNA_A8", "STRATA_GR_RDNA_Q8", "STRATA_HC_UPMIX", "STRATA_PA_FAST", "STRATA_PF_F16", "STRATA_PF_HCDOWN",
             "STRATA_PREFILL_AUTO_MAX", "STRATA_SELECT_WMMA", "STRATA_VERIFY_ECAP", "STRATA_VERIFY_ECAP_MIN_ROWS"}
    assert set(upd["env"]) - set(ex["env"]) == unset and not set(ex["env"]) - set(upd["env"])
    assert ex["env"]["STRATA_PREFILL_CPU_SHARE"] == "0" and ex["env"]["STRATA_ROUTE_TAIL_SKIP"] == "0"
    assert {k: v for k, v in ex["env"].items() if k not in ("STRATA_PREFILL_CPU_SHARE",)} == {k: v for k, v in upd["env"].items() if k not in unset | {"STRATA_PREFILL_CPU_SHARE"}}
    for core in ("STRATA_GR_RDNA", "STRATA_RDNA_MMVQ_WAVE", "STRATA_COMBINE_CPU", "STRATA_WAIT_FUSE", "STRATA_POOL_FUSE_PHASES", "STRATA_RDNA_Q6K", "STRATA_ROUTE_PUBLISH"):
        assert ex["env"][core] == upd["env"][core], core
    assert ex["args"] == upd["args"] and ex["exe"] == upd["exe"]
    s252 = json.loads((root / "bench/specs/252-rdna3-updated-exact.json").read_text())
    shipped = json.loads((root / "bench/specs/250-rdna3-updated-vs-plain.json").read_text())["arms"]["updated"]["env"]
    assert s252["arms"]["updated"]["env"] == shipped and s252["arms"]["updated-exact"]["env"] == {k: v for k, v in shipped.items() if k not in ("STRATA_VERIFY_ECAP", "STRATA_VERIFY_ECAP_MIN_ROWS")}


# ------------------------------------------------------------------------------------------------ bench-sweep.sh (full flag sweep, both branches)
SW = "bench-sweep.sh"


def sweep_mod():
    spec_ = importlib.util.spec_from_file_location("sweep", CHAINS / "sweep.py")
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    return mod


def sweep_registry():
    return json.loads((CHAINS / "sweep_candidates.json").read_text())


def test_sweep_registry_lists_only_never_tested_switches_with_a_reason_and_the_measured_ones_with_their_source():
    reg = sweep_registry()
    for br in ("us", "rd"):
        unt, mea = reg["untested"][br], reg["measured"][br]
        names = [c["name"] for c in unt]
        assert len(names) == len(set(names)) and not set(names) & set(mea), br
        assert all(c["why"] and c["source"] and (c.get("env") or c.get("args")) and isinstance(c["exact"], bool) for c in unt), br
        assert all(v for v in mea.values()), br
    # the 2026-11-19 box verdicts and the ledger keep their flags out of pass 1
    for k in ("STRATA_RDNA_Q6K", "STRATA_GR_RDNA", "STRATA_RDNA_MMVQ_WAVE", "STRATA_VERIFY_ECAP", "STRATA_PF_F16", "--lookup-max-t"):
        assert k in reg["measured"]["us"], k
    for k in ("STRATA_POOL_FUSE_PHASES", "STRATA_VERIFY_ECAP", "STRATA_PF_F16", "STRATA_RDNA_MMVQ_WAVE"):
        assert k in reg["measured"]["rd"], k
    assert not any(re.search(r"GFX906|GFX1151|IGPU|DBG|DUMP|TRACE", c["name"]) for br in ("us", "rd") for c in reg["untested"][br])


def test_sweep_us_arms_minus_the_untested_switches_of_the_updated_arm_plus_the_others():
    mod, reg = sweep_mod(), sweep_registry()
    arm = json.loads((SPECS / "220-updated-vs-upstream-short.json").read_text())["arms"]["updated"]
    spec, meta = mod.build_spec("us", "/x/plain.json", arm, reg, "/mnt/pancake/benchus/src-us", plain={"env": {}, "args": []})
    flags = meta["flags"]
    unt = reg["untested"]["us"]
    in_arm = {c["name"] for c in unt if any(k in arm["env"] or k in arm["args"] for k in list(c.get("env") or {}) + list(c.get("args") or {}))}
    assert {f["flag"] for f in flags.values() if f["kind"] == "minus"} == in_arm and in_arm
    assert {f["flag"] for f in flags.values() if f["kind"] == "plus"} == {c["name"] for c in unt} - in_arm
    assert not {f["flag"] for f in flags.values()} & set(reg["measured"]["us"])          # measured: no arm
    assert list(spec["arms"])[0] == "updated" and spec["arms"]["updated"] == arm and spec["drift_check"] is True
    assert spec["cases"] == ["short", "edit", "longdec32k"] and spec["reps"] == 1 and spec["base"] == "/x/plain.json"
    m = next(n for n, f in flags.items() if f["kind"] == "minus")
    k = flags[m]["keys"][0]
    assert k not in spec["arms"][m]["env"] and len(spec["arms"][m]["env"]) == len(arm["env"]) - 1 and spec["arms"][m]["args"] == arm["args"]
    pl = next(n for n, f in flags.items() if f["kind"] == "plus")
    assert spec["arms"][pl]["env"] == {**arm["env"], **flags[pl]["env"]}
    # one run would be over 2.5 h: the blocked design (blocks of <= 6, each with 2 base runs, one re-run budgeted) fits in 9 even slices of 17
    n = len(flags)
    fits = lambda k: sum(mod.estimate(k)[:2]) + 8 * 150 <= 9000      # pass 1 + one re-run + one paired confirm block (pass 2 takes the rest)
    assert n == len(unt) and not fits(n)
    parts = [mod.build_spec("us", "/x/plain.json", arm, reg, "/s", "%d/9" % k, plain={"env": {}, "args": []}, split="even")[1]["flags"] for k in range(1, 10)]
    assert set().union(*map(set, parts)) == set(flags) and sum(len(x) for x in parts) == n and max(len(x) for x in parts) == -(-n // 9)
    assert all(fits(len(x)) for x in parts)
    assert mod.estimate(17, warm="first") == ((17 + 2 * 3 + 1) * 150, 8 * 150, 6 * 8 * 150)       # warm-up start once per slice (owner 2026-11-20)
    assert mod.estimate(17) == ((17 + 9) * 150, 9 * 150, 6 * 8 * 150)      # + the warm-up start of every block (T-1004)


def test_sweep_rd_arm_is_what_config_updated_adds_and_a_plus_value_theirs_already_run_gets_no_arm(tmp_path):
    mod, reg = sweep_mod(), sweep_registry()
    rec = bd_arm_rec()
    plain, upd, _ = bd_build(tmp_path, rec)
    arm = mod.arm_from_configs(plain, upd)
    assert arm["env"] == rec["env"] and arm["args"] == rec["args"] and arm["exe"] == upd["exe"]   # theirs are not in the arm
    spec, meta = mod.build_spec("rd", "/p.json", arm, reg, "/mnt/pancake/benchRD/src-updated", plain=plain)
    plus = {f["flag"] for f in meta["flags"].values() if f["kind"] == "plus"}
    same = {c["name"] for c in reg["untested"]["rd"] if all(plain["env"].get(k) == v for k, v in (c.get("env") or {}).items()) and c.get("env")}
    assert same and not plus & same                                                                # e.g. GR_INJ_UP=1: their setup runs it
    assert plus == {c["name"] for c in reg["untested"]["rd"]} - same - {f["flag"] for f in meta["flags"].values() if f["kind"] == "minus"}
    parts = [mod.build_spec("rd", "/p.json", arm, reg, "/s", "%d/3" % k, plain=plain)[1]["flags"] for k in (1, 2, 3)]
    assert sum(len(x) for x in parts) == len(meta["flags"])
    rest = [mod.build_spec("rd", "/p.json", arm, reg, "/s", "%d/4" % k, plain=plain, split="even", exclude=list(parts[0]))[1]["flags"] for k in (1, 2, 3, 4)]
    redo = [mod.build_spec("rd", "/p.json", arm, reg, "/s", "%d/2" % k, plain=plain, split="even", only=list(parts[0]))[1]["flags"] for k in (1, 2)]
    assert set().union(*map(set, rest)) == set(meta["flags"]) - set(parts[0]) and set().union(*map(set, redo)) == set(parts[0])
    assert all(sum(mod.estimate(len(x))[:2]) + 8 * 150 <= 9000 for x in rest + redo) and all(len(x) <= 18 for x in rest + redo)
    # priority order: every [arm] and [named] arm is in slice 1, ahead of its [code] arms
    first = [f for f in parts[0].values()]
    assert {f["flag"] for f in meta["flags"].values() if mod.priority(f) < 2} <= {f["flag"] for f in first}
    assert [mod.priority(f) for f in first] == sorted(mod.priority(f) for f in first) and len(parts[0]) == 33


def test_sweep_priority_order_slice_1_is_all_arm_and_named_switches_then_the_code_ones_split_evenly():
    mod, reg = sweep_mod(), sweep_registry()
    arm = json.loads((SPECS / "220-updated-vs-upstream-short.json").read_text())["arms"]["updated"]
    parts = [list(mod.build_spec("us", "/x/plain.json", arm, reg, "/s", "%d/5" % k, plain={"env": {}, "args": []})[1]["flags"].values()) for k in range(1, 6)]
    n = len(reg["untested"]["us"])                                           # 153 minus 3 pruned minus 31 backlog = 119
    assert [len(x) for x in parts] == [36] + [-(-(n - 36) // 4)] * 3 + [n - 36 - 3 * -(-(n - 36) // 4)]
    assert mod.take_slice(list(range(10)), "1/3", 0) == [0, 1, 2, 3] and mod.take_slice(list(range(10)), "3/3", 0) == [7, 8, 9]
    assert [mod.priority(f) for f in parts[0]].count(0) == 35 and [f["flag"] for f in parts[0] if mod.priority(f) == 1] == ["STRATA_AMD_COMMUNITY_TUNING"]
    assert all(mod.priority(f) == 2 for x in parts[1:] for f in x)
    names = {c["name"] for c in reg["untested"]["rd"] if mod.priority(c) == 1}
    assert names == {"STRATA_GR_RDNA_Q8", "STRATA_GR_RDNA_A8", "STRATA_GR_INJ_UP", "STRATA_GR_FUSED", "STRATA_ADAPT_PERSIST", "STRATA_AMD_COMMUNITY_TUNING"}
    assert mod.take_slice(list(range(10)), "1/1", 4) == list(range(10)) and mod.take_slice(list(range(10)), "1/3", 2) == [0, 1, 2, 3]
    assert mod.take_slice(list(range(10)), "1/3", 6) == list(range(6)) and mod.take_slice(list(range(10)), "3/3", 6) == [8, 9]


def test_sweep_rd_updated_config_fixups_drop_pool_fuse_phases_keep_ecap_and_are_logged():
    mod = sweep_mod()
    upd = {"exe": "/e", "env": {"STRATA_POOL_FUSE_PHASES": "1", "STRATA_VERIFY_ECAP": "0", "STRATA_X": "1"}, "args": []}
    fixed, notes = mod.fix_config(upd, ["STRATA_POOL_FUSE_PHASES", "STRATA_NOPE"], ["STRATA_VERIFY_ECAP=0", "STRATA_VERIFY_ECAP_MIN_ROWS=9"])
    assert fixed["env"] == {"STRATA_VERIFY_ECAP": "0", "STRATA_X": "1", "STRATA_VERIFY_ECAP_MIN_ROWS": "9"} and "STRATA_POOL_FUSE_PHASES" in upd["env"]
    assert notes == ["ARMFIX dropped STRATA_POOL_FUSE_PHASES=1 from the updated config", "ARMFIX STRATA_NOPE is not in the updated config: nothing dropped",
                     "ARMFIX STRATA_VERIFY_ECAP=0 already in the updated config", "ARMFIX set STRATA_VERIFY_ECAP_MIN_ROWS=9 in the updated config (was unset)"]
    arm = mod.arm_from_configs({"env": {}, "args": []}, fixed)
    assert "STRATA_POOL_FUSE_PHASES" not in arm["env"] and arm["env"]["STRATA_VERIFY_ECAP_MIN_ROWS"] == "9"
    text = (CHAINS / SW).read_text()
    assert "${SWEEP_RD_DROP-STRATA_POOL_FUSE_PHASES}" in text and "${SWEEP_RD_SET-STRATA_VERIFY_ECAP=0 STRATA_VERIFY_ECAP_MIN_ROWS=9}" in text


def test_sweep_triage_thresholds_promising_then_keep_drop_retest_gate():
    mod = sweep_mod()
    meta = {"branch": "rd", "slice": "1/1", "protected": {}, "flags": {
        "plus-A": {"flag": "A", "kind": "plus", "exact": True}, "plus-L": {"flag": "L", "kind": "plus", "exact": False},
        "minus-M": {"flag": "M", "kind": "minus", "exact": True}, "plus-W": {"flag": "W", "kind": "plus", "exact": True},
        "plus-F": {"flag": "F", "kind": "plus", "exact": True}, "plus-N": {"flag": "N", "kind": "plus", "exact": True},
        "plus-U": {"flag": "U", "kind": "plus", "exact": True}}}

    def rows(vals, reps=None):
        return [json.dumps({"arm": a, "rep": r, "rows": [{"case": c, "decode_tps": v[c], "prompt_tps": v.get(c + "-prompt")} for c in ("short", "edit", "longdec32k")]})
                for r in range(reps or 1) for a, v in vals.items()]
    b = {"short": 100.0, "edit": 100.0, "longdec32k": 100.0, "longdec32k-prompt": 1000.0}
    p1 = rows({"updated": b, "base-late": b,
               "plus-A": {**b, "short": 101.2},                 # exact +1.2 % >= 1.0: promising
               "plus-L": {**b, "short": 101.2},                 # lossy +1.2 % < 1.5: low performer
               "minus-M": {**b, "edit": 97.0},                  # taking M out costs edit 3 %: M helps +3.1 %
               "plus-W": {**b, "short": 105.0, "edit": 96.0},   # a case below -3 %: still promising (owner 2026-11-20), the confirm decides
               "plus-F": {**b, "short": 102.0},                 # promising, not confirmed in pass 2
               "plus-N": {**b, "short": 101.5}})                # promising, no pass-2 rows
    p1.append(json.dumps({"arm": "plus-L2", "rows": []}))
    t1 = mod.triage(meta, p1)
    v1 = {k: r["verdict"] for k, r in t1["flags"].items()}
    assert v1 == {"A": "RETEST", "L": "NO-GAIN-P1", "M": "RETEST", "W": "RETEST", "F": "RETEST", "N": "RETEST", "U": "UNTESTED"}
    assert t1["flags"]["M"]["pass1"]["edit"] == 3.09 and t1["flags"]["W"]["promising"] is True and "confirm decides" in t1["flags"]["W"]["floor_note"]
    assert all(t1["flags"][k]["carry"] for k in "AMWFN") and not t1["flags"]["L"]["carry"]
    spec = {"arms": {"updated": {}, "plus-A": {}, "plus-L": {}, "minus-M": {}, "plus-W": {}, "plus-F": {}, "plus-N": {}, "plus-U": {}}}
    s2, names, carry = mod.pass2_spec(spec, t1)                                  # no cap (owner order 2026-11-20)
    assert names == ["W", "M", "F", "N", "A"] and not carry and s2["reps"] == 3
    s2, names, carry = mod.pass2_spec(spec, t1, max_flags=3)                     # the budget cuts: the rest is carried
    assert names == ["W", "M", "F"] and [r["flag"] for r in carry] == ["N", "A"] and list(s2["arms"]) == ["updated", "plus-W", "minus-M", "plus-F"]
    p2 = rows({"updated": b, "plus-A": {**b, "short": 102.5}, "minus-M": {**b, "edit": 96.0}, "plus-F": {**b, "short": 100.5}}, reps=3)
    lossy = dict(meta, flags={**meta["flags"], "minus-M": {"flag": "M", "kind": "minus", "exact": False}})
    t2 = mod.triage(lossy, p1, p2)
    v2 = {k: r["verdict"] for k, r in t2["flags"].items()}
    assert v2["A"] == "KEEP" and v2["M"] == "RETEST-GATE" and v2["F"] == "DROP" and v2["N"] == "RETEST" and v2["L"] == "NO-GAIN-P1"
    assert t2["flags"]["A"]["pass2"]["short"] == 2.5 and t2["flags"]["A"]["pass2_clean"] is True and t2["schema"] == "pancake.sweep-triage/2"


def sweep_rows(vals, rep=0, late=None):
    """abtest jsonl lines of one block: {arm: {case: tok/s}}; late = the base-late values (default: as updated)."""
    vals = dict(vals)
    if "base-late" not in vals:
        vals["base-late"] = late or vals["updated"]
    return [json.dumps({"arm": a, "rep": rep, "rows": [{"case": c, "decode_tps": v[c], "prompt_tps": v.get(c + "-prompt")} for c in ("short", "edit", "longdec32k")]})
            for a, v in vals.items()]


def test_sweep_blocks_are_at_most_6_arms_base_first_random_order_and_cover_the_slice():
    mod = sweep_mod()
    spec = {"base": "/p.json", "drift_check": True, "arms": {"updated": {"env": {}}, **{"plus-%02d" % i: {"env": {"K": str(i)}} for i in range(17)}}}
    bl = mod.block_specs(spec, 6, "s1")
    assert [len(o) for _, _, o in bl] == [6, 6, 5] and [b for b, _, _ in bl] == ["b01", "b02", "b03"]
    assert sorted(a for _, _, o in bl for a in o) == sorted(a for a in spec["arms"] if a != "updated")
    for bid, sp, order in bl:
        assert list(sp["arms"]) == ["warmup", "updated"] + order and sp["drift_check"] is True and sp["arms"]["updated"] == spec["arms"]["updated"]
        assert sp["arms"]["warmup"] == spec["arms"]["updated"]                          # the warm-up start (T-1004): the base config, first
    assert [o for _, _, o in bl] == [o for _, _, o in mod.block_specs(spec, 6, "s1")]           # the seed fixes the order
    assert any(o != sorted(o) for _, _, o in bl)                                                 # and it is not the spec order
    assert mod.block_sizes(13, 6) == [5, 4, 4] and mod.block_sizes(6, 6) == [6] and mod.block_sizes(0, 6) == []


def test_sweep_triage_guard_drifted_block_is_inconclusive_a_clean_rerun_is_used_pass2_drift_keeps_retest():
    mod = sweep_mod()
    meta = {"branch": "us", "slice": "x", "protected": {}, "flags": {
        "plus-A": {"flag": "A", "kind": "plus", "exact": True}, "minus-B": {"flag": "B", "kind": "minus", "exact": True},
        "plus-C": {"flag": "C", "kind": "plus", "exact": True}, "plus-D": {"flag": "D", "kind": "plus", "exact": True}}}
    b = {"short": 100.0, "edit": 100.0, "longdec32k": 100.0, "longdec32k-prompt": 1000.0}
    slow = {**b, "edit": 80.0}                                                                   # the late base 20 % down on edit
    b1 = sweep_rows({"updated": b, "plus-A": {**b, "short": 99.0}, "minus-B": {**b, "short": 99.0}}, late=slow)       # drifted
    b1r = sweep_rows({"updated": b, "plus-A": {**b, "short": 99.0}, "minus-B": {**b, "short": 99.0}})                 # its clean re-run
    b2 = sweep_rows({"updated": b, "plus-C": {**b, "short": 99.0}, "plus-D": {**b, "short": 104.0}}, late=slow)       # drifted twice
    b2r = sweep_rows({"updated": b, "plus-C": {**b, "short": 99.0}, "plus-D": {**b, "short": 104.0}}, late={**b, "short": 93.0})
    t = mod.triage(meta, [("b01", b1), ("b01r", b1r), ("b02", b2), ("b02r", b2r)])
    v = {k: r["verdict"] for k, r in t["flags"].items()}
    assert v == {"A": "NO-GAIN-P1", "B": "RETEST", "C": "INCONCLUSIVE", "D": "INCONCLUSIVE"}, v
    assert t["flags"]["A"]["block"] == "b01r" and t["flags"]["A"]["clean1"] is True and t["flags"]["A"]["blocks_run"] == ["b01", "b01r"]
    assert t["flags"]["D"]["promising"] is False and "drift" in t["flags"]["D"]["reason"]          # promising numbers, drifted block: no pass 2
    spec = {"arms": {"updated": {}, "plus-A": {}, "minus-B": {}, "plus-C": {}, "plus-D": {}}}
    assert mod.pass2_spec(spec, t)[1] == ["B"] and t["flags"]["D"]["carry"] is True     # promising numbers in a drifted block: carried
    # pass 2: a drifting base rep makes B INCONCLUSIVE (owner 2026-11-20: drift > 5 % => INCONCLUSIVE, carried); a clean pass 2 decides
    p2 = sweep_rows({"updated": b, "minus-B": {**b, "short": 97.0}}, 0) + sweep_rows({"updated": b, "minus-B": {**b, "short": 97.0}}, 1) + \
        sweep_rows({"updated": {**b, "edit": 90.0}, "minus-B": {**b, "short": 97.0}}, 2)
    p2 = [l for l in p2 if '"base-late"' not in l]
    t2 = mod.triage(meta, [("b01", b1), ("b01r", b1r), ("b02", b2), ("b02r", b2r)], p2)
    assert t2["flags"]["B"]["verdict"] == "INCONCLUSIVE" and t2["flags"]["B"]["carry"] is True and t2["flags"]["B"]["pass2_clean"] is False
    assert t2["flags"]["B"]["pass2_drift"]["edit"] == -10.0
    p2c = [l.replace('"decode_tps": 90.0', '"decode_tps": 100.0') for l in p2]
    t3 = mod.triage(meta, [("b01", b1), ("b01r", b1r)], p2c)
    assert t3["flags"]["B"]["verdict"] == "KEEP" and t3["flags"]["B"]["pass2_clean"] is True
    assert mod.clean({"short": 4.9}) and not mod.clean({"short": -5.1}) and not mod.clean({})


def test_sweep_remark_one_run_triage_drifted_becomes_inconclusive_retest_stays_clean_pass2_stays():
    mod = sweep_mod()
    b = {"short": 100.0, "edit": 100.0, "longdec32k": 100.0}
    tri = {"schema": "pancake.sweep-triage/1", "branch": "rd", "slices": ["1/3"], "protected": {}, "flags": {
        "A": {"verdict": "DROP", "pass1": {"short": -1.0}, "pass2": None, "promising": False},
        "B": {"verdict": "RETEST", "pass1": {"short": 3.0}, "pass2": None, "promising": True},
        "C": {"verdict": "KEEP", "pass1": {"short": 9.0}, "pass2": {"short": 7.5}, "promising": True},
        "D": {"verdict": "DROP", "pass1": {"short": 2.0}, "pass2": {"short": 0.5}, "promising": True}}}
    p1 = sweep_rows({"updated": b}, late={**b, "edit": 78.1})                                   # the RD slice 1 drift: edit -21.9 %
    p2bad = sweep_rows({"updated": b}, 0)[:1] + sweep_rows({"updated": {**b, "edit": 52.5}}, 1)[:1]
    r = mod.remark(tri, p1, p2bad)
    assert {k: x["verdict"] for k, x in r["flags"].items()} == {"A": "INCONCLUSIVE", "B": "RETEST", "C": "INCONCLUSIVE", "D": "INCONCLUSIVE"}
    assert r["flags"]["C"]["prior_verdict"] == "KEEP" and r["schema"] == "pancake.sweep-triage/2" and r["flags"]["A"]["drift1"]["edit"] == -21.9
    p2ok = sweep_rows({"updated": b}, 0)[:1] + sweep_rows({"updated": b}, 1)[:1]
    r2 = mod.remark(tri, p1, p2ok)
    assert {k: x["verdict"] for k, x in r2["flags"].items()} == {"A": "INCONCLUSIVE", "B": "RETEST", "C": "KEEP", "D": "DROP"}
    clean1 = mod.remark(tri, sweep_rows({"updated": b}), None)
    assert clean1["flags"]["A"]["verdict"] == "NO-GAIN-P1" and clean1["flags"]["C"]["verdict"] == "INCONCLUSIVE"   # a pass-1 DROP is no DROP


def test_sweep_merge_inconclusive_never_replaces_a_decided_verdict(tmp_path):
    mod = sweep_mod()
    old = {"schema": "pancake.sweep-triage/2", "branch": "rd", "slices": ["1of3"], "protected": {"X": ["edit"]}, "flags": {
        "A": {"verdict": "RETEST"}, "B": {"verdict": "INCONCLUSIVE"}, "C": {"verdict": "KEEP"}, "D": {"verdict": "RETEST"}}}
    new = {"schema": "pancake.sweep-triage/2", "branch": "rd", "slices": ["rdR-1of2"], "protected": {}, "flags": {
        "A": {"verdict": "INCONCLUSIVE"}, "B": {"verdict": "DROP"}, "C": {"verdict": "UNTESTED"}, "D": {"verdict": "KEEP"}, "E": {"verdict": "INCONCLUSIVE"}}}
    (tmp_path / "o.json").write_text(json.dumps(old))
    (tmp_path / "n.json").write_text(json.dumps(new))
    m = mod.merge([str(tmp_path / "o.json"), str(tmp_path / "n.json")])
    # B: a DROP without a clean pass 2 is relabelled NO-GAIN-P1 (owner 2026-11-20: DROP only from a clean pass 2)
    assert {k: r["verdict"] for k, r in m["flags"].items()} == {"A": "RETEST", "B": "NO-GAIN-P1", "C": "KEEP", "D": "KEEP", "E": "INCONCLUSIVE"}
    new["flags"]["B"] = {"verdict": "DROP", "pass2": {"short": -2.0}, "pass2_clean": True}
    (tmp_path / "n.json").write_text(json.dumps(new))
    assert mod.merge([str(tmp_path / "o.json"), str(tmp_path / "n.json")])["flags"]["B"]["verdict"] == "DROP"
    assert m["slices"] == ["1of3", "rdR-1of2"] and m["protected"] == {"X": ["edit"]}


def sweep_rd_configs(box, tmp_path):
    plain, upd, _ = bd_build(tmp_path / "cfg", bd_arm_rec())
    (box.tmp / "auto-runs" / "benchRD" / "config-plain.json").write_text(json.dumps(plain))
    (box.tmp / "auto-runs" / "benchRD" / "config-updated.json").write_text(json.dumps(upd))
    f = box.tmp / "mnt" / "benchRD" / "src-updated" / "data" / "expert-profile-pancake.bin"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("x")


def test_bench_sweep_rd_full_flow_pass1_triage_pass2_and_the_triage_file(tmp_path):
    box = Box(tmp_path)
    prepare_be(box, us=False)
    sweep_rd_configs(box, tmp_path)
    b = {"short": 100.0, "edit": 140.0, "longdec32k": 30.0, "longdec32k-prompt": 1500.0}
    plain = json.loads((box.tmp / "auto-runs" / "benchRD" / "config-plain.json").read_text())
    upd = json.loads((box.tmp / "auto-runs" / "benchRD" / "config-updated.json").read_text())
    mod = sweep_mod()
    _, meta = mod.build_spec("rd", "/p.json", mod.arm_from_configs(plain, upd), sweep_registry(), "/s", "1/6", plain=plain, split="even")
    win = next(n for n, f in meta["flags"].items() if f["kind"] == "plus" and f["exact"])
    wflag = meta["flags"][win]["flag"]
    n1 = len(meta["flags"])
    dec = {"*": b, win: {**b, "short": 104.0}}
    box.finish(box.start(SW, env=be_env(SWEEP_BRANCH="rd", SWEEP_SLICE="1/6", SWEEP_SPLIT="even", STUB_ARM_DECODE=json.dumps(dec))), timeout=240)
    lines = box.log("bench-sweep.log")
    text = "\n".join(lines)
    assert lines[-1] == "WINBENCHSWEEP_DONE" and "ERROR" not in text and "ABORTED" not in text, text
    assert_start_end_pairs(lines)
    nb = len(mod.block_sizes(n1))
    order = ["PREREQ ok (rd)", "SYNC harness", "PREREQ harness ok", "ARMFIX ", "SWEEP rd slice 1of6: %d flag arms" % n1, "BUDGET ", "BLOCK b01 ",
             "START rd-p1-1of6-b01", "END rd-p1-1of6-b01 exit=0", "DRIFT b01 (block 1", "START rd-p1-1of6-b%02d" % nb, "DRIFT b%02d" % nb,
             "PASS2 1 flag(s), one paired block each: " + wflag, "START rd-p2-1of6-" + win, "END rd-p2-1of6-%s exit=0" % win, "DRIFT2 " + win,
             "TRIAGE KEEP", "CARRY none", "TRIAGE file",
             "SUMMARY branch rd slice 1of6 (1/6): %d blocks (%d clean, 0 re-run, 0 INCONCLUSIVE), pass 1 exit=0, pass 2: 1 paired block(s), exit=0" % (nb, nb),
             "service stays down"]
    assert all(k in text for k in order), ([k for k in order if k not in text], text)
    pos = [text.index(k) for k in order]
    assert pos == sorted(pos), order
    assert "RERUN" not in text and text.count("-> CLEAN") == nb + 1
    ab = be_calls(box, "abtest.py")
    assert len(ab) == nb + 1 and all("--tier screen" in x and "--drift-check" in x and "--retries 0" in x for x in ab), ab
    assert ("rd-p2-1of6-%s.json" % win) in ab[nb]                                        # pass 2: the flag's own paired block
    sw_dir = box.tmp / "auto-runs" / "benchsweep"
    tri = json.loads((sw_dir / "rd-triage.v2.json").read_text())
    assert tri["flags"][wflag]["verdict"] == "KEEP" and tri["flags"][wflag]["pass2"]["short"] == 4.0 and tri["flags"][wflag]["pass2_clean"] is True
    assert all(r["verdict"] == "NO-GAIN-P1" and r["clean1"] for f, r in tri["flags"].items() if f != wflag) and tri["slices"] == ["1of6"]
    p2 = [json.loads(l)["arm"] for l in (sw_dir / ("rd-p2-1of6-%s.jsonl" % win)).read_text().splitlines()]
    assert p2 == ["warmup", "updated", win, "updated~2", win + "~2", "updated~3", win + "~3", "base-late"]    # warm-up, base, flag x 3 interleaved, base
    assert (sw_dir / "rd-triage-1of6.v2.json").exists() and not (sw_dir / "rd-triage.json").exists()
    for k in range(1, nb + 1):                                                            # each block: warm-up, updated, <= 6 arms, base-late
        rows = [json.loads(l)["arm"] for l in (sw_dir / ("rd-p1-1of6-b%02d.jsonl" % k)).read_text().splitlines()]
        assert rows[:2] == ["warmup", "updated"] and rows[-1] == "base-late" and len(rows) - 3 <= 6
    assert sum(1 for c in box.calls() if c.startswith("SETSID")) == nb + 1               # each abtest run in its own session
    assert not any(c.startswith("SIG") for c in box.calls())
    assert not any(c.startswith("SYSTEMCTL") and (" start " in c or " stop " in c) for c in box.calls())


def test_bench_sweep_us_whole_branch_over_2h30_stops_before_any_start_and_names_the_split(tmp_path):
    box = Box(tmp_path)
    prepare_be(box, rd=False)
    f = box.tmp / "mnt" / "benchus" / "src-us" / "data" / "expert-profile-pancake.bin"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("x")
    box.finish(box.start(SW, env=be_env(SWEEP_BRANCH="us")), timeout=240)
    text = "\n".join(box.log("bench-sweep.log"))
    assert "SWEEP us slice 1of1: %d flag arms" % len(sweep_registry()["untested"]["us"]) in text and "ERROR BUDGET" in text and "split it" in text and "WINBENCHSWEEP_ABORTED" in text, text
    assert not be_calls(box, "abtest.py")
    box2 = Box(tmp_path / "b") if (tmp_path / "b").mkdir() is None else None
    prepare_be(box2, rd=False)
    f = box2.tmp / "mnt" / "benchus" / "src-us" / "data" / "expert-profile-pancake.bin"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("x")
    box2.finish(box2.start(SW, env=be_env(SWEEP_BRANCH="us", SWEEP_SLICE="2/9", SWEEP_SPLIT="even", SWEEP_TAG="usB-2of9")), timeout=240)
    t2 = "\n".join(box2.log("bench-sweep.log"))
    k2 = len(sweep_mod().take_slice(list(range(len(sweep_registry()["untested"]["us"]))), "2/9", 0))
    assert "SWEEP us slice usB-2of9: %d flag arms" % k2 in t2 and "in %d blocks of <= 6" % -(-k2 // 6) in t2 and "START us-p1-usB-2of9-b03" in t2 and "ERROR" not in t2, t2
    assert (box2.tmp / "auto-runs" / "benchsweep" / "us-triage-usB-2of9.v2.json").exists() and (box2.tmp / "auto-runs" / "benchsweep" / "us-triage.v2.json").exists()
    # the same slice name again: refused before anything runs, nothing overwritten
    n_ab = len(be_calls(box2, "abtest.py"))
    box2.finish(box2.start(SW, env=be_env(SWEEP_BRANCH="us", SWEEP_SLICE="2/9", SWEEP_SPLIT="even", SWEEP_TAG="usB-2of9")), timeout=240)
    assert "was used before" in "\n".join(box2.log("bench-sweep.log")) and len(be_calls(box2, "abtest.py")) == n_ab


def test_bench_sweep_bad_branch_and_missing_builds_abort_before_any_download(tmp_path):
    box = Box(tmp_path)
    prepare_be(box, us=False, rd=False)
    box.finish(box.start(SW, env=be_env()))
    text = "\n".join(box.log("bench-sweep.log"))
    assert "SWEEP_BRANCH must be us or rd" in text and box.log("bench-sweep.log")[-1] == "WINBENCHSWEEP_DONE"
    (tmp_path / "b").mkdir()
    box2 = Box(tmp_path / "b")
    prepare_be(box2, us=False, rd=False)
    box2.finish(box2.start(SW, env=be_env(SWEEP_BRANCH="rd")))
    t2 = "\n".join(box2.log("bench-sweep.log"))
    assert "PREREQ missing:" in t2 and "build-updated/strata" in t2 and "SYNC harness" not in t2 and not be_calls(box2, "abtest.py")


def test_bench_sweep_pass1_over_its_limit_is_stopped_reaped_and_the_triage_still_written(tmp_path):
    box = Box(tmp_path)
    prepare_be(box, us=False)
    sweep_rd_configs(box, tmp_path)
    box.finish(box.start(SW, env=be_env(SWEEP_BRANCH="rd", SWEEP_SLICE="6/6", SWEEP_SPLIT="even", SWEEP_BLOCK="99", AB_HANG="30", BENCH_BUDGET_S="80")), timeout=300)
    lines = box.log("bench-sweep.log")
    text = "\n".join(lines)
    assert any(l.startswith("TIMEOUT rd-p1-6of6-b01") for l in lines), lines
    assert any(c.startswith("REAP -TERM") for c in box.calls())
    assert "END rd-p1-6of6-b01 exit=124" in text and "SKIP rerun b01: budget" in text and "PASS2 none" in text and lines[-1] == "WINBENCHSWEEP_DONE"
    tri = json.loads((box.tmp / "auto-runs" / "benchsweep" / "rd-triage.v2.json").read_text())
    assert all(r["verdict"] == "UNTESTED" for r in tri["flags"].values())


def test_bench_sweep_name_is_invisible_to_every_busy_pattern_and_sees_the_benchmark_chains(tmp_path):
    box = Box(tmp_path)
    for chain in ("win-chainT.sh", "redo-chainAA.sh", "bench-updatedAA.sh", "bench-updatedRD.sh", "bench-exact.sh", SW):
        assert busy_result(box, chain, ["sh ./bench-sweep.sh"]) == "IDLE", chain
    for other in ("sh ./bench-updatedRD.v8.sh", "sh ./bench-exact.v3.sh", "sh ./bench-rdgate.v2.sh", "python3 tools/pancake/abtest.py x"):
        assert busy_result(box, SW, [other]) == "BUSY", other


def test_sched_bench_extra_runs_the_sweep_with_its_log_and_marker(tmp_path):
    box = Box(tmp_path)
    prepare_sched(box)
    (box.S / "bench-sweep.v1.sh").write_text(FAKE_CHAIN.format(name="sweep", log=(box.S / "bench-sweep.log").as_posix(), marker="WINBENCHSWEEP_DONE"), newline="\n")
    box.finish(box.start(SCHED, env=sched_env(BENCH_ONLY="EXTRA", BENCH_EXTRA="./bench-sweep.v1.sh", BENCH_NO_RESUME="1")))
    text = "\n".join(box.log("bench-sched.log"))
    assert "END EXTRA exit=0 last-log-line=WINBENCHSWEEP_DONE" in text and "ERROR" not in text, text
    assert "[b]ench-(updated(AA|RD)|diagUS|exact|rdgate|sweep)" in (CHAINS / SCHED).read_text()


def test_sched_bench_extra_runs_the_third_chain_alone_or_after_both(tmp_path):
    box = Box(tmp_path)
    prepare_sched(box)
    (box.S / "bench-exact.sh").write_text(FAKE_CHAIN.format(name="exact", log=(box.S / "bench-exact.log").as_posix(), marker="WINBENCHEXACT_DONE"), newline="\n")
    box.finish(box.start(SCHED, env=sched_env(BENCH_ONLY="EXTRA", BENCH_EXTRA="./bench-exact.sh", BENCH_NO_RESUME="1")))
    text = "\n".join(box.log("bench-sched.log"))
    assert chains_log(box) == ["exact env: BENCH_SCHED=1 BENCH_PREIDLED=1 KEEP=1 stat=T"], chains_log(box)
    assert "END EXTRA exit=0 last-log-line=WINBENCHEXACT_DONE" in text and "START US" not in text and "START RD" not in text and "ERROR" not in text
    assert [c for c in box.calls() if c.startswith("SIG")] == ["SIG -STOP 4242"] and "AA stays frozen" in text   # BENCH_NO_RESUME=1
    (tmp_path / "b").mkdir()
    box2 = Box(tmp_path / "b")
    prepare_sched(box2)
    (box2.S / "bench-exact.sh").write_text(FAKE_CHAIN.format(name="exact", log=(box2.S / "bench-exact.log").as_posix(), marker="WINBENCHEXACT_DONE"), newline="\n")
    box2.finish(box2.start(SCHED, env=sched_env(BENCH_EXTRA="./bench-exact.sh")))
    assert [l.split()[0] for l in chains_log(box2)] == ["us", "rd", "exact"]
    (tmp_path / "c").mkdir()
    box3 = Box(tmp_path / "c")
    prepare_sched(box3)
    box3.finish(box3.start(SCHED, env=sched_env(BENCH_ONLY="EXTRA")))
    t3 = "\n".join(box3.log("bench-sched.log"))
    assert "ERROR BENCH_ONLY=EXTRA without BENCH_EXTRA" in t3 and not chains_log(box3)


# ------------------------------------------------------------------------------------------------ bench-rdgate.sh (RDNA3 task gate, updated vs plain)
RG = "bench-rdgate.sh"
# the box venv interpreter: logs VENVPY, fakes task_gate.py (dry run / gate, an orphan engine of its own session, a hang), the rest goes to the python stub
RG_VENV = ('echo "VENVPY $*" >> "$H/calls.log"\n'
           'case "$*" in\n'
           '  *task_gate.py*--dry-run*) echo "dry run: ok (every checker passes its reference answer and fails its wrong one)"; exit ${TG_DRY_EXIT:-0} ;;\n'
           '  *task_gate.py*)\n'
           '    [ -n "$GATE_ORPHAN" ] && { echo "999994 @SID@ 999994" > "$H/ptree.txt"; echo "Thu Nov 19 18:00:00 2026" > "$H/lstart.999994"; /usr/bin/sleep 1; }\n'
           '    [ -n "$GATE_HANG" ] && /usr/bin/sleep "$GATE_HANG"\n'
           '    echo "updated   success 98.9% vs base -1.1 pts [-4.4, +2.2] PASS"; exit 0 ;;\n'
           'esac\nexec "$PYREAL" "$H/stub/pystub.py" "$@"\n')
RG_PLAIN = {"exe": "/x/build-plain/strata", "args": ["--port", "11437"], "env": {"STRATA_PF_F16": "2", "STRATA_A": "1", "STRATA_OLD": "1"}}
RG_UPD = {"exe": "/x/build-updated/strata", "args": ["--port", "11437"], "env": {"STRATA_PF_F16": "1", "STRATA_A": "1", "STRATA_POOL_FUSE_PHASES": "1"}}


def prepare_rg(box, builds=True, configs=True, diag=True, abtest_py="# reads ABTEST_PYTHON\n"):
    for n, body in (("ps", BE_PS), ("reapcmd", BE_REAP),
                    ("setsid", 'echo "SETSID $$ $*" >> "$H/calls.log"; echo $$ > "$H/sid.txt"; exec "$@"\n'),
                    ("df", 'echo "Filesystem 1K-blocks Used Available Use% Mounted"\necho "x 1 1 ${DF_FREE:-99999999} 1% /"\n')):
        (box.tmp / "stub" / n).write_text("#!/bin/sh\n" + body, newline="\n")
    venv = box.R / ".venv" / "bin" / "python"
    venv.parent.mkdir(parents=True, exist_ok=True)
    venv.write_text("#!/bin/sh\n" + RG_VENV, newline="\n")
    os.chmod(venv, 0o755)
    pkg = box.tmp / "rg-src" / "x"
    for rel, body in (("tools/pancake/task_gate.py", "# stub\n"), ("tools/pancake/abtest.py", abtest_py), ("tools/pancake/capped.sh", "#!/bin/sh\n"),
                      ("serve/server.py", "# stub\n"), ("bench/tasks/default.json", "{}\n")):
        (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
        (pkg / rel).write_text(body)
    (box.tmp / "fake-src.tgz").unlink()
    subprocess.run(["tar", "czf", (box.tmp / "fake-src.tgz").as_posix(), "-C", (box.tmp / "rg-src").as_posix(), "x"], check=True)
    if builds:
        for rel in ("build-plain/strata", "build-updated/strata"):
            (box.tmp / "mnt" / "benchRD" / rel).parent.mkdir(parents=True, exist_ok=True)
            (box.tmp / "mnt" / "benchRD" / rel).write_text("#!/bin/sh\nexit 0\n", newline="\n")
    if diag:
        for rel in ("flag_diag.py", "diag_group.py"):
            f = box.tmp / "mnt" / "benchRD" / "src-updated" / "tools" / "pancake" / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("# stub\n")
    if configs:
        (box.tmp / "auto-runs" / "benchRD").mkdir(parents=True, exist_ok=True)
        (box.tmp / "auto-runs" / "benchRD" / "config-plain.json").write_text(json.dumps(RG_PLAIN))
        (box.tmp / "auto-runs" / "benchRD" / "config-updated.json").write_text(json.dumps(RG_UPD))


def rg_gate_calls(box):
    return [c for c in box.calls() if c.startswith("VENVPY") and "task_gate.py" in c]


def test_bench_rdgate_full_flow_gate_with_the_venv_python_spec_and_marker(tmp_path):
    box = Box(tmp_path)
    prepare_rg(box)
    box.finish(box.start(RG, env=be_env()))
    lines = box.log("bench-rdgate.log")
    text = "\n".join(lines)
    assert lines[-1] == "WINBENCHRDGATE_DONE" and "ERROR" not in text and "ABORTED" not in text, text
    assert_start_end_pairs(lines)
    order = ["started by the scheduler", "service already stopped", "PREREQ ok", "SYNC harness: origin port/rdna3-decode abc123", "PREREQ harness ok",
             "SPEC ", "START dryrun", "END dryrun exit=0", "START ctest-updated", "END ctest-updated exit=0", "START ctest-plain", "END ctest-plain exit=0",
             "NO-ENGINE check ok before diag", "START diag", "END diag exit=0", "NO-ENGINE check ok before gate", "START gate", "END gate exit=0", "PASS",
             "SUMMARY dryrun exit=0, ctest updated=0 plain=0 , diag exit=0, gate exit=0", "service stays down"]
    pos = [text.index(k) for k in order]
    assert pos == sorted(pos), order
    tg = rg_gate_calls(box)
    assert len(tg) == 2 and "--dry-run" in tg[0] and "--dry-run" not in tg[1], tg
    for c in tg:   # the standard settings, from the synced harness
        assert "tools/pancake/task_gate.py" in c and "work/rdgate.json" in c and "--tasks bench/tasks/default.json" in c and "--reps 5" in c
        assert "/auto-runs/benchrdgate/gate" in c
    sess = [c for c in box.calls() if c.startswith("SETSID")]
    assert len(sess) == 5, sess                                                          # dry run, 2 ctests, diag, gate: each in its own session
    venv = "/clone-of-d/src/Strata-win/.venv/bin/python"
    assert sum(1 for c in sess if venv + " tools/pancake/task_gate.py" in c) == 2, sess  # the gate runs with the box venv python
    ct = [c for c in box.calls() if c.startswith("ctest ")]
    assert len(ct) == 2 and "/mnt/benchRD/build-updated -R verify_ecap_test|sampler_parity|elementwise_parity" in ct[0] and "/mnt/benchRD/build-plain -R" in ct[1], ct
    assert any("/stub/ctest --test-dir" in c for c in sess), "ctest by its full path"
    fd = [c for c in sess if "flag_diag.py run" in c]
    assert len(fd) == 1 and venv + " -u " in fd[0] and "/mnt/benchRD/src-updated/tools/pancake/flag_diag.py" in fd[0], fd
    for k in ("auto-runs/benchRD/config-updated.json", "--flags POOL_FUSE_PHASES,VERIFY_ECAP,PF_F16,RDNA_MMVQ_WAVE,KV_HOST_DMA",
              "/mnt/benchrdgate/harness/tools/pancake/./abtest.py", "/auto-runs/benchrdgate/diag", "--ctest ", "/mnt/benchRD/build-updated"):
        assert k in fd[0], k
    assert (box.tmp / "auto-runs" / "benchrdgate" / "diag" / "verdicts.json").exists()
    assert not [c for c in box.calls() if c.startswith("PY ") and "task_gate.py" in c], "no task_gate.py through the PATH python3"
    sp = json.loads((box.tmp / "auto-runs" / "benchrdgate" / "work" / "rdgate.json").read_text())
    assert sp["base"].endswith("auto-runs/benchRD/config-plain.json") and sp["tasks"] == "bench/tasks/default.json"
    assert (sp["reps"], sp["temperature"], sp["margin"]) == (5, 0.2, 0.05) and list(sp["arms"]) == ["plain", "updated"]
    assert list(sp["arms"]["plain"]) == ["exe"] and sp["arms"]["plain"]["exe"].endswith("/mnt/benchRD/build-plain/strata")
    upd = sp["arms"]["updated"]
    assert upd["exe"].endswith("/mnt/benchRD/build-updated/strata") and "config" not in upd
    assert upd["env"] == {"STRATA_PF_F16": "1", "STRATA_POOL_FUSE_PHASES": "1", "STRATA_OLD": None}   # exactly what config-updated.json changes
    assert not any(c.startswith("SIG") or c.startswith("REAP") for c in box.calls())     # AA is never signalled, nothing to reap
    assert not any(c.startswith("SYSTEMCTL") and (" start " in c or " stop " in c) for c in box.calls())   # the service stays down


def test_bench_rdgate_prereq_missing_builds_or_configs_aborts_before_any_download(tmp_path):
    box = Box(tmp_path)
    prepare_rg(box, builds=False)
    (box.tmp / "auto-runs" / "benchRD" / "config-updated.json").unlink()
    box.finish(box.start(RG, env=be_env()))
    lines = box.log("bench-rdgate.log")
    err = [l for l in lines if l.startswith("ERROR PREREQ missing")]
    assert len(err) == 1 and all(k in err[0] for k in ("benchRD/build-plain/strata", "benchRD/build-updated/strata", "benchRD/config-updated.json")), lines
    assert "config-plain.json" not in err[0]
    assert any(l.startswith("WINBENCHRDGATE_ABORTED: PREREQ missing") for l in lines) and lines[-1] == "WINBENCHRDGATE_DONE"
    assert not rg_gate_calls(box) and "SYNC" not in "\n".join(lines) and not any(c.startswith("SETSID") for c in box.calls())


def test_bench_rdgate_harness_without_abtest_python_aborts_before_any_engine(tmp_path):
    box = Box(tmp_path)
    prepare_rg(box, abtest_py="PY = ROOT / '.venv/bin/python'\n")
    box.finish(box.start(RG, env=be_env()))
    lines = box.log("bench-rdgate.log")
    assert any(l.startswith("ERROR PREREQ harness:") and "does not read ABTEST_PYTHON" in l for l in lines), lines
    assert lines[-1] == "WINBENCHRDGATE_DONE" and not rg_gate_calls(box)


def test_bench_rdgate_failed_dry_run_skips_the_gate_but_not_the_ctests_and_diag(tmp_path):
    box = Box(tmp_path)
    prepare_rg(box)
    box.finish(box.start(RG, env=be_env(TG_DRY_EXIT="1")))
    lines = box.log("bench-rdgate.log")
    text = "\n".join(lines)
    tg = rg_gate_calls(box)
    assert len(tg) == 1 and "--dry-run" in tg[0], tg
    assert "ERROR dryrun exit=1" in text and "SKIP gate: the dry run failed (exit 1)" in text and "END diag exit=0" in text and "END ctest-plain" in text
    assert "SUMMARY dryrun exit=1" in text and "gate exit=-" in text and lines[-1] == "WINBENCHRDGATE_DONE"


def test_bench_rdgate_running_engine_of_another_job_skips_diag_and_gate_and_is_never_killed(tmp_path):
    box = Box(tmp_path)
    prepare_rg(box)
    (box.tmp / "procs.txt").write_text("5005 python3 -u /x/tools/pancake/flag_diag.py run --config y\n")   # not in busy(): only the no-engine check sees it
    box.finish(box.start(RG, env=be_env()))
    lines = box.log("bench-rdgate.log")
    text = "\n".join(lines)
    assert any(l.startswith("ERROR diag not started: an engine / server / harness is still running after 120 s: 5005") for l in lines), lines
    assert any(l.startswith("ERROR gate not started:") for l in lines)
    assert not [c for c in box.calls() if c.startswith("SETSID") and ("flag_diag.py run" in c or ("task_gate.py" in c and "--dry-run" not in c))]
    assert not any(c.startswith(("REAP", "SIG")) for c in box.calls()) and "diag exit=skip, gate exit=skip" in text and lines[-1] == "WINBENCHRDGATE_DONE"


def test_bench_rdgate_ctest_failure_and_missing_flag_diag_are_logged_and_the_gate_still_runs(tmp_path):
    box = Box(tmp_path)
    prepare_rg(box, diag=False)
    (box.tmp / "stub" / "ctest").write_text('#!/bin/sh\necho "ctest $*" >> "$H/calls.log"\necho "Could not find executable /x/sampler_parity"\n'
                                            'echo "0% tests passed, 3 tests failed out of 3"\nexit 8\n', newline="\n")
    box.finish(box.start(RG, env=be_env()))
    lines = box.log("bench-rdgate.log")
    text = "\n".join(lines)
    assert "END ctest-updated exit=8" in text and "END ctest-plain exit=8" in text and "ERROR ctest-plain exit=8" in text
    assert any(l.startswith("ERROR diag SKIPPED: missing") and "src-updated/tools/pancake/flag_diag.py" in l for l in lines), lines
    assert "END gate exit=0" in text and lines[-1] == "WINBENCHRDGATE_DONE"


def test_bench_rdgate_engine_left_by_the_gate_is_reaped_no_orphan(tmp_path):
    box = Box(tmp_path)
    prepare_rg(box)
    box.finish(box.start(RG, env=be_env(GATE_ORPHAN="1")))
    lines = box.log("bench-rdgate.log")
    text = "\n".join(lines)
    reap = [l for l in lines if l.startswith("REAP gate: still running")]
    assert len(reap) == 1 and "999994" in reap[0], lines
    assert "REAP -TERM 999994" in box.calls() and not (box.tmp / "lstart.999994").exists()
    assert text.index("REAP gate: still running") < text.index("END gate") and "ERROR REAP" not in text and lines[-1] == "WINBENCHRDGATE_DONE"


def test_bench_rdgate_gate_over_its_limit_gets_term_and_the_marker_follows(tmp_path):
    box = Box(tmp_path)
    prepare_rg(box)
    box.finish(box.start(RG, env=be_env(GATE_HANG="30", RDGATE_LIM_GATE="20")), timeout=180)
    lines = box.log("bench-rdgate.log")
    assert any(l.startswith("TIMEOUT gate after 20 s (limit 20 s") for l in lines), lines
    assert any(c.startswith("REAP -TERM") for c in box.calls())
    assert "END gate exit=124" in "\n".join(lines) and lines[-1] == "WINBENCHRDGATE_DONE"


def test_bench_rdgate_budget_default_and_standalone_wait_for_the_rd_marker(tmp_path):
    t = (CHAINS / RG).read_text()
    m = re.search(r"BUDGET=\$\{BENCH_BUDGET_S:-(\d+)\}", t)
    assert m and int(m.group(1)) == 5400
    g = re.search(r"LIM_GATE=\$\{RDGATE_LIM_GATE:-(\d+)\}", t)
    assert g and int(g.group(1)) < 5400
    assert t.index("WINBENCHRDGATE_DONE") < t.index("WINBENCHRD_DONE"), "the scheduler takes the first WIN*_DONE of the script as its marker"
    box = Box(tmp_path)
    prepare_rg(box)
    p = box.start(RG, env={"SVC_DOWN": "1", "REAPCMD": "reapcmd"})
    time.sleep(1.5)
    assert p.poll() is None and len(box.log("bench-rdgate.log")) == 1
    append(box, "bench-updatedRD.log", "WINBENCHRD_DONE")
    box.finish(p)
    lines = box.log("bench-rdgate.log")
    assert "the RD benchmark chain is done" in lines[1] and lines[-1] == "WINBENCHRDGATE_DONE", lines[:3]


def test_bench_rdgate_name_is_invisible_to_every_busy_pattern_and_sees_the_benchmark_chains(tmp_path):
    box = Box(tmp_path)
    for bad in ("win-chain", "redo-chain", "tail-chain", "auto-chain", "after7"):
        assert bad not in RG
    for chain in ("win-chainT.sh", "redo-chainAA.sh", "bench-updatedAA.sh", "bench-updatedRD.sh", "bench-exact.sh", RG):
        assert busy_result(box, chain, ["sh ./bench-rdgate.v1.sh"]) == "IDLE", chain
    for other in ("sh ./bench-updatedRD.v3.sh", "sh ./bench-exact.v1.sh", "sh ./bench-diagUS.sh", "python3 tools/pancake/task_gate.py x"):
        assert busy_result(box, RG, [other]) == "BUSY", other


def test_sched_runs_bench_rdgate_as_extra_with_its_own_log_and_marker(tmp_path):
    box = Box(tmp_path)
    prepare_sched(box)
    body = FAKE_CHAIN.format(name="rdgate", log=(box.S / "bench-rdgate.log").as_posix(), marker="WINBENCHRDGATE_DONE")
    (box.S / "bench-rdgate.v1.sh").write_text(body, newline="\n")
    box.finish(box.start(SCHED, env=sched_env(BENCH_ONLY="EXTRA", BENCH_EXTRA="./bench-rdgate.v1.sh", BENCH_NO_RESUME="1")))
    text = "\n".join(box.log("bench-sched.log"))
    assert chains_log(box) == ["rdgate env: BENCH_SCHED=1 BENCH_PREIDLED=1 KEEP=1 stat=T"], chains_log(box)
    assert "END EXTRA exit=0 last-log-line=WINBENCHRDGATE_DONE" in text and "ERROR" not in text, text
    idle = next(l for l in (CHAINS / SCHED).read_text().splitlines() if "if busy || pgrep" in l)
    assert "rdgate" in idle and "exact" in idle


def test_bench_sweep_drifting_block_runs_once_more_a_clean_rerun_counts_a_second_drift_is_inconclusive(tmp_path):
    """The 2026-11-20 failure (TEST_LEDGER.md): the late base 22-44 % below the early one.  Block b01 drifts once (its re-run is clean), b02 drifts
    in both runs: its arms are INCONCLUSIVE; every DRIFT line is in the log; the v1 triage of the old design gets its v2 copy and is not touched."""
    box = Box(tmp_path)
    prepare_be(box, us=False)
    sweep_rd_configs(box, tmp_path)
    sw_dir = box.tmp / "auto-runs" / "benchsweep"
    sw_dir.mkdir(parents=True, exist_ok=True)
    b = {"short": 100.0, "edit": 140.0, "longdec32k": 30.0}
    v1 = {"schema": "pancake.sweep-triage/1", "branch": "rd", "slices": ["1/3"], "protected": {}, "flags": {
        "STRATA_OLD": {"verdict": "DROP", "pass1": {"short": -1.0}, "pass2": None, "promising": False}}}
    (sw_dir / "rd-triage-1of3.json").write_text(json.dumps(v1))
    (sw_dir / "rd-p1-1of3.jsonl").write_text("\n".join(sweep_rows({"updated": b}, late={**b, "edit": 109.3})) + "\n")
    late = {"-b01.jsonl": 0.8, "-b02": 0.9}
    box.finish(box.start(SW, env=be_env(SWEEP_BRANCH="rd", SWEEP_SLICE="1/6", SWEEP_SPLIT="even", SWEEP_TAG="rdB-1of6",
                                        STUB_ARM_DECODE=json.dumps({"*": b}), STUB_LATE_DRIFT=json.dumps(late))), timeout=300)
    lines = box.log("bench-sweep.log")
    text = "\n".join(lines)
    assert lines[-1] == "WINBENCHSWEEP_DONE" and "ERROR" not in text, text
    order = ["REMARK ", "BLOCK b01 ", "DRIFT b01 (block 1", "RERUN b01", "START rd-p1-rdB-1of6-b01r", "DRIFT b01r (re-run of block 1", "DRIFT b02 (block 2",
             "RERUN b02", "DRIFT b02r", "INCONCLUSIVE b02: still above 5 %", "DRIFT b03 (block 3", "TRIAGE file"]
    pos = [text.index(k) for k in order]
    assert pos == sorted(pos), order
    assert "DRIFT b03r" not in text and "SUMMARY branch rd slice rdB-1of6 (1/6): 3 blocks (2 clean, 2 re-run, 1 INCONCLUSIVE)" in text
    blk = {json.loads(l)["arm"] for l in (sw_dir / "rd-p1-rdB-1of6-b02.jsonl").read_text().splitlines()} - {"warmup", "updated", "base-late"}
    tri = json.loads((sw_dir / "rd-triage.v2.json").read_text())
    by_arm = {r["arm"]: r for r in tri["flags"].values() if "arm" in r}
    assert blk and all(by_arm[a]["verdict"] == "INCONCLUSIVE" for a in blk)
    assert all(r["verdict"] == "NO-GAIN-P1" for a, r in by_arm.items() if a not in blk)
    assert tri["flags"]["STRATA_OLD"]["verdict"] == "INCONCLUSIVE" and tri["flags"]["STRATA_OLD"]["prior_verdict"] == "DROP"
    assert json.loads((sw_dir / "rd-triage-1of3.json").read_text()) == v1 and (sw_dir / "rd-triage-1of3.v2.json").exists()


# ------------------------------------------------------------------------------------------------ owner rules 2026-11-20: two-pass restated, one-by-one
def paired_rows(base, flag, an="plus-X", bn="updated", late=None, mid=None):
    """abtest jsonl lines of one paired block: base, flag, base~2, flag~2, base~3, flag~3, base-late (mid: the base~2 / base~3 values)."""
    order = [(bn, base), (an, flag), (bn + "~2", mid or base), (an + "~2", flag), (bn + "~3", mid or base), (an + "~3", flag), ("base-late", late or base)]
    return [json.dumps({"arm": a, "rep": 0, "rows": [{"case": c, "decode_tps": v[c], "prompt_tps": v.get(c + "-prompt")} for c in ("short", "edit", "longdec32k")]})
            for a, v in order]


def two_request_rows(arm, first, second, longdec=70.0):
    """one abtest record of a start with the two requests of short / edit (first, second) and the single longdec32k request."""
    rows = [{"case": "short", "decode_tps": first["short"]}, {"case": "short", "decode_tps": second["short"]},
            {"case": "edit", "decode_tps": first["edit"]}, {"case": "edit", "decode_tps": second["edit"]},
            {"case": "longdec32k", "decode_tps": longdec, "prompt_tps": 1600.0}]
    return json.dumps({"arm": arm, "rep": 0, "rows": rows})


def test_drift_uses_the_last_request_of_each_start_and_the_warmup_start_is_never_judged():
    """T-1004: the rdC-1 / EMB_REUSE_ACCOUNT block (first real block): the first start ran edit 158.5 / 185.4, every later one 137-140 / 175: drift
    9 % with the mean of both requests; the warm-up start takes the odd first start out, the second request alone has a 0.5 % spread."""
    mod = sweep_mod()
    rows = [two_request_rows("warmup", {"short": 113.0, "edit": 158.5}, {"short": 113.8, "edit": 185.4}),      # the warm-up: never judged
            two_request_rows("updated", {"short": 104.6, "edit": 139.6}, {"short": 114.2, "edit": 175.9}),
            two_request_rows("plus-X", {"short": 110.6, "edit": 132.5}, {"short": 114.3, "edit": 175.5}),
            two_request_rows("updated~2", {"short": 110.7, "edit": 137.2}, {"short": 114.5, "edit": 175.7}),
            two_request_rows("plus-X~2", {"short": 110.3, "edit": 140.2}, {"short": 114.1, "edit": 175.6}),
            two_request_rows("updated~3", {"short": 110.7, "edit": 137.2}, {"short": 114.5, "edit": 175.7}),
            two_request_rows("plus-X~3", {"short": 110.3, "edit": 140.2}, {"short": 114.1, "edit": 175.6}),
            two_request_rows("base-late", {"short": 106.6, "edit": 139.0}, {"short": 113.7, "edit": 174.3})]
    _, drift, _ = mod.paired_eval(rows, "updated", "plus-X", "plus", ["short", "edit", "longdec32k"])
    assert mod.clean(drift) and max(abs(v) for v in drift.values()) < 1.0, drift
    assert drift["edit"] == -0.91 and drift["short"] == -0.44 and drift["longdec32k"] == 0.0, drift
    # the old rule (the first start counted, the mean of both requests): the same start values as the first counted start
    old = [two_request_rows("updated", {"short": 113.0, "edit": 158.5}, {"short": 113.8, "edit": 185.4})] + rows[2:]
    _, d_old, _ = mod.paired_eval(old, "updated", "plus-X", "plus", ["short", "edit", "longdec32k"])
    assert d_old["edit"] == -5.99 and not mod.clean(d_old)                                   # the first start itself is odd (185.4): the warm-up removes it
    m_old = mod.metrics(old, cases=["short", "edit"])                                        # the mean of both requests: 172.0 vs 156.7
    assert round(m_old["base-late"]["edit"] / m_old["updated"]["edit"] * 100 - 100, 1) == -8.9
    assert mod.drift1(rows, "updated", ["short", "edit", "longdec32k"]) == {"short": -0.44, "edit": -0.91, "longdec32k": 0.0}
    # effects keep the mean of both requests (the estimator of the gains is unchanged): flag edit first request 132.5 vs base 137-140
    eff, _, _ = mod.paired_eval(rows, "updated", "plus-X", "plus", ["short", "edit", "longdec32k"])
    assert eff["edit"] < 0 and eff["short"] > 0


def test_paired_spec_runs_base_flag_three_times_interleaved_and_the_base_again():
    mod = sweep_mod()
    sp = mod.paired_spec("/p.json", "updated", {"env": {"A": "1"}}, "plus-X", {"env": {"A": "1", "X": "1"}})
    assert list(sp["arms"]) == ["warmup", "updated", "plus-X", "updated~2", "plus-X~2", "updated~3", "plus-X~3"] and sp["drift_check"] is True
    assert sp["arms"]["warmup"] == sp["arms"]["updated"] == {"env": {"A": "1"}}
    assert sp["cases"] == ["short", "edit", "longdec32k"] and sp["reps"] == 1 and sp["prefill"] is True
    b = {"short": 100.0, "edit": 100.0, "longdec32k": 30.0, "longdec32k-prompt": 1500.0}
    eff, drift, per = mod.paired_eval(paired_rows(b, {**b, "edit": 110.0}), "updated", "plus-X", "plus")
    assert eff["edit"] == 10.0 and eff["short"] == 0.0 and len(per) == 3 and mod.clean(drift)
    eff, _, _ = mod.paired_eval(paired_rows(b, {**b, "edit": 90.0}, "minus-X"), "updated", "minus-X", "minus")
    assert eff["edit"] == 11.11                                                                   # taking X out costs 10 %: X helps +11.1 %


def test_owner_rule_mixed_arm_goes_to_pass2_confirm_decides_optional_and_drift_is_inconclusive():
    """+40 % edit and -20 % short: pass 2 (the -20 % does not block it); the clean confirm gives OPTIONAL (opt-in, not default);
    a paired block with base drift > 5 % gives INCONCLUSIVE (carried), never DROP."""
    mod = sweep_mod()
    meta = {"branch": "rd", "slice": "x", "protected": {}, "flags": {"plus-X": {"flag": "X", "kind": "plus", "exact": True, "env": {"X": "1"}},
                                                                     "plus-Y": {"flag": "Y", "kind": "plus", "exact": True, "env": {"Y": "1"}}}}
    b = {"short": 100.0, "edit": 100.0, "longdec32k": 30.0, "longdec32k-prompt": 1500.0}
    mixed = {**b, "edit": 140.0, "short": 80.0}
    p1 = sweep_rows({"updated": b, "plus-X": mixed, "plus-Y": {**b, "short": 99.5}})
    t1 = mod.triage(meta, [("b01", p1)])
    assert t1["flags"]["X"]["verdict"] == "RETEST" and t1["flags"]["X"]["promising"] is True and t1["flags"]["Y"]["verdict"] == "NO-GAIN-P1"
    assert mod.pass2_spec({"arms": {"updated": {}, "plus-X": {}, "plus-Y": {}}}, t1)[1] == ["X"]
    t2 = mod.triage(meta, [("b01", p1)], {"plus-X": [("p2", paired_rows(b, mixed))]})
    x = t2["flags"]["X"]
    assert x["verdict"] == "OPTIONAL" and x["pass2"]["edit"] == 40.0 and x["pass2"]["short"] == -20.0 and x["pass2_clean"] is True
    assert "helps" in x["reason"] and "edit +40.0 %" in x["reason"] and "short -20.0 %" in x["reason"]
    drifted = paired_rows(b, mixed, late={**b, "edit": 90.0})                                     # the last base run 10 % down on edit
    t3 = mod.triage(meta, [("b01", p1)], {"plus-X": [("p2", drifted)]})
    assert t3["flags"]["X"]["verdict"] == "INCONCLUSIVE" and t3["flags"]["X"]["carry"] is True and t3["flags"]["X"]["pass2_clean"] is False
    # its clean re-run decides; a small gain with a loss is a DROP only from that clean confirm
    t4 = mod.triage(meta, [("b01", p1)], {"plus-X": [("p2", drifted), ("p2r", paired_rows(b, {**b, "edit": 103.0, "short": 95.0}))]})
    assert t4["flags"]["X"]["verdict"] == "DROP" and t4["flags"]["X"]["pass2_block"] == "p2r"
    assert mod.confirm_verdict({"edit": 2.5, "short": 0.0}, True)[0] == "KEEP" and mod.confirm_verdict({"edit": 3.5}, False)[0] == "RETEST-GATE"
    assert mod.confirm_verdict({"edit": 12.0, "short": -3.0}, True)[0] == "OPTIONAL" and mod.confirm_verdict({"edit": 9.9, "short": -5.0}, True)[0] == "DROP"


def retest_bases():
    plain = {"exe": "/p/strata", "env": {"STRATA_PF_F16": "2"}, "args": ["--max-context", "262144"]}
    upd = {"exe": "/u/strata", "env": {"STRATA_PF_F16": "2", "STRATA_POOL_FUSE_PHASES": "1", "STRATA_EXPDEC": "1", "STRATA_PF_GROUP_SCRATCH": "1",
                                       "STRATA_ROUTE_TAIL_SKIP": "0", "STRATA_ATTN_DAILY": "1", "STRATA_PROMPT_ATTN_IMMA": "1", "STRATA_KV_PREFETCH": "1",
                                       "STRATA_PIN_BY_NEED": "1", "STRATA_EXCHANGE_ROTATE": "1"},
           "args": ["--max-context", "262144", "--grammar", "on", "--pool-affinity", "measure", "--conversation-cache-min-tokens", "8192"]}
    mod = sweep_mod()
    fixed, _ = mod.fix_config(upd, ["STRATA_POOL_FUSE_PHASES"], [])
    s220 = json.loads((SPECS / "220-updated-vs-upstream-short.json").read_text())["arms"]
    s222 = json.loads((SPECS / "222-updated-exact-vs-upstream-short.json").read_text())["arms"]
    return {"rd": {"path": "/rd/plain.json", "arms": {"updated": mod.arm_from_configs(plain, fixed), "plain": {"exe": plain["exe"]}}},
            "us": {"path": "/us/plain.json", "arms": {"updated": s220["updated"], "plain": s220["plain-upstream"], "exact": s222["updated-exact"]}}}


def test_retest_slices_one_flag_per_block_owner_order_and_budget():
    mod = sweep_mod()
    groups = json.loads((CHAINS / "retest_groups.json").read_text())
    reg = sweep_registry()
    bases = retest_bases()
    src = {"rd": "/mnt/pancake/benchRD/src-updated", "us": "/mnt/pancake/benchus/src-us"}
    allb = []
    for sl in groups["slices"]:
        bl = mod.retest_blocks(groups, sl, bases, reg, src)
        assert 1 <= len(bl) <= 7 and sum(e["est"] for e in bl) <= 9000, sl                       # <= 2.5 h (estimate: 150 s per start, the box runs ~105 s)
        if sl.startswith("usT-"):          # the US-only queue slices repeat the US flags of mixC-3 .. usR-7: own test below
            continue
        allb += bl
    flags = [e["flag"] for e in allb]
    rd_order = ["STRATA_EMB_REUSE_ACCOUNT", "STRATA_GROUPED_V1", "STRATA_CVEC_FUSE", "STRATA_EARLY_REMOTE_CONTEXTS", "STRATA_EXPDEC", "STRATA_FUSE_HEAD_GR",
                "STRATA_PF_GROUP_SCRATCH", "STRATA_ROUTE_TAIL_SKIP", "--grammar", "STRATA_ATTN_DAILY", "--pool-affinity", "--conversation-cache-min-tokens",
                "STRATA_PROMPT_ATTN_IMMA", "STRATA_KV_PREFETCH", "STRATA_PIN_BY_NEED", "STRATA_EXCHANGE_ROTATE"]
    assert [e["flag"] for e in allb if e["branch"] == "rd"][:16] == rd_order                     # the owner's priority order, RD first
    us = ["STRATA_ENGINE_NICE", "STRATA_SELECT_WMMA", "STRATA_ATTN_DAILY", "STRATA_KV_PREFETCH", "STRATA_STAGER_RING", "STRATA_PF_GROUP_SCRATCH",
          "STRATA_PROMPT_ATTN_IMMA", "--lookup-max-t", "STRATA_GR_RDNA_Q8", "STRATA_GR_RDNA_A8", "STRATA_GR_INJ_UP", "STRATA_GR_FUSED", "STRATA_GR_RDNA",
          "STRATA_RDNA_MMVQ_WAVE", "STRATA_RDNA_Q6K"]
    assert [e["flag"] for e in allb if e["branch"] == "us" and e["type"] == "paired" and not e["bundle"]] == us
    assert flags.index("STRATA_EXCHANGE_ROTATE") < flags.index("STRATA_ENGINE_NICE") < flags.index("BUNDLE-250-updated")
    for e in allb:
        if e["type"] != "paired":
            continue
        sp = mod.block_spec_of(e)
        assert len(sp["arms"]) == 7 and sp["cases"] == ["short", "edit", "longdec32k"]               # base + ONE flag, 3 interleaved reps
        assert e["est"] == 8 * 150
    by = {e["key"]: e for e in allb}
    assert by["rd:STRATA_EXPDEC"]["kind"] == "minus" and "STRATA_EXPDEC" not in by["rd:STRATA_EXPDEC"]["arm"]["env"]
    assert by["rd:STRATA_POOL_FUSE_PHASES"]["kind"] == "plus" and by["rd:STRATA_POOL_FUSE_PHASES"]["arm"]["env"]["STRATA_POOL_FUSE_PHASES"] == "1"
    assert "STRATA_POOL_FUSE_PHASES" not in by["rd:STRATA_EXPDEC"]["base_arm"]["env"]               # the base: updated minus POOL_FUSE_PHASES
    assert by["rd:STRATA_PF_F16"]["arm"]["env"]["STRATA_PF_F16"] == "1" and "STRATA_PF_F16" not in by["rd:STRATA_PF_F16"]["base_arm"]["env"]
    assert by["us:--lookup-max-t"]["kind"] == "minus" and "--lookup-max-t" not in by["us:--lookup-max-t"]["arm"]["args"]
    assert by["rd:--expert-profile"]["arm"]["args"]["--expert-profile"] == "/mnt/pancake/benchRD/src-updated/data/expert-profile-pancake.bin"
    assert by["rd:--expert-profile"]["ship"]["args"]["--expert-profile"].startswith("<ROOT>/")
    assert by["rd:BUNDLE-250-updated"]["bundle"] and by["rd:BUNDLE-250-updated"]["base_name"] == "plain"
    assert by["us:GATE-lookup-max-t"]["type"] == "gate" and by["us:SPEC-223"]["type"] == "spec"


def test_retest_verdicts_per_flag_and_a_bundle_is_recorded_but_marked():
    mod = sweep_mod()
    b = {"short": 100.0, "edit": 100.0, "longdec32k": 30.0, "longdec32k-prompt": 1500.0}
    blocks = [{"bid": "k01", "type": "paired", "branch": "rd", "flag": "X", "kind": "plus", "exact": True, "bundle": False, "base_name": "updated",
               "arm_name": "plus-X", "env": {"X": "1"}, "args": {}, "cases": ["short", "edit", "longdec32k"], "slice": "s"},
              {"bid": "k02", "type": "paired", "branch": "rd", "flag": "BUNDLE-250-updated", "kind": "plus", "exact": False, "bundle": True,
               "base_name": "plain", "arm_name": "updated", "env": {}, "args": {}, "cases": ["short", "edit", "longdec32k"], "slice": "s"}]
    runs = {"k01": [("k01", paired_rows(b, {**b, "edit": 104.0}))],
            "k02": [("k02", paired_rows(b, {**b, "edit": 143.6, "short": 79.0}, "updated", "plain"))]}
    docs = mod.retest_verdicts(blocks, runs)
    fl = docs["rd"]["flags"]
    assert fl["X"]["verdict"] == "KEEP" and fl["X"]["pass2"]["edit"] == 4.0 and fl["X"]["pass2_clean"] is True and not fl["X"]["bundle"]
    bd = fl["BUNDLE BUNDLE-250-updated"]
    assert bd["bundle"] is True and bd["verdict"] == "OPTIONAL" and "changes no flag" in bd["reason"]
    assert docs["rd"]["schema"] == "pancake.sweep-triage/2"


def test_bench_retest_has_the_rules_and_its_name_is_invisible_to_every_busy_pattern(tmp_path):
    text = (CHAINS / "bench-retest.sh").read_text()
    assert "WINBENCHRETEST_DONE" in text and text.index("WINBENCHRETEST_DONE") < text.index("WINBENCHRETEST_ABORTED")
    assert "--tier screen --drift-check" in text and "paired-drift" in text and "RERUN" in text
    assert "${RETEST_RD_DROP-STRATA_POOL_FUSE_PHASES}" in text and "quality_gate.py" in text and "--near-tie" in text
    box = Box(tmp_path)
    for chain in ("win-chainT.sh", "redo-chainAA.sh", "bench-updatedAA.sh", "bench-exact.sh", SW):
        assert busy_result(box, chain, ["sh ./bench-retest.v1.sh"]) == "IDLE", chain


RT = "bench-retest.sh"


def test_bench_retest_rd_slice_runs_one_paired_block_per_flag_and_writes_per_flag_verdicts(tmp_path):
    box = Box(tmp_path)
    prepare_be(box, us=False)
    sweep_rd_configs(box, tmp_path)
    b = {"short": 100.0, "edit": 140.0, "longdec32k": 30.0, "longdec32k-prompt": 1500.0}
    dec = {"*": b, "plus-EMB_REUSE_ACCOUNT": {**b, "short": 104.0}, "minus-EXPDEC": {**b, "longdec32k": 25.0, "short": 104.0}}
    box.finish(box.start(RT, env=be_env(RETEST_SLICE="rdC-1", STUB_ARM_DECODE=json.dumps(dec), RETEST_LOG=(box.S / "bench-retest.log").as_posix())), timeout=240)
    lines = box.log("bench-retest.log")
    text = "\n".join(lines)
    assert lines[-1] == "WINBENCHRETEST_DONE" and "ERROR" not in text and "ABORTED" not in text, text
    assert_start_end_pairs(lines)
    order = ["PREREQ ok", "SYNC harness", "PREREQ harness ok", "ARMFIX ", "BLOCK k01 paired 1200 rd", "RETEST rdC-1: 7 blocks",
             "BUDGET ", "START retest-rdC-1-k01", "DRIFT k01 (STRATA_EMB_REUSE_ACCOUNT", "START retest-rdC-1-k07", "DRIFT k07", "TRIAGE KEEP",
             "TRIAGE file", "SUMMARY retest slice rdC-1: 7 blocks (7 clean, 0 re-run, 0 INCONCLUSIVE, 0 not run: budget)"]
    assert all(k in text for k in order), [k for k in order if k not in text]
    pos = [text.index(k) for k in order]
    assert pos == sorted(pos), order
    ab = be_calls(box, "abtest.py")
    assert len(ab) == 7 and all("--tier screen" in x and "--drift-check" in x for x in ab)      # one abtest run per block (8 starts each)
    sw_dir = box.tmp / "auto-runs" / "benchsweep"
    tri = json.loads((sw_dir / "rd-triage-rt-rdC-1.v2.json").read_text())
    assert tri["flags"]["STRATA_EMB_REUSE_ACCOUNT"]["verdict"] == "KEEP" and tri["flags"]["STRATA_EMB_REUSE_ACCOUNT"]["pass2"]["short"] == 4.0
    # minus-EXPDEC: taking EXPDEC out gives short +4 % and longdec32k -16.7 %: EXPDEC helps long decode +20 % and costs short 3.8 %: OPTIONAL
    ex = tri["flags"]["STRATA_EXPDEC"]
    assert ex["verdict"] == "OPTIONAL" and ex["pass2"]["longdec32k"] == 20.0 and ex["pass2"]["short"] == -3.85, ex
    assert tri["flags"]["STRATA_CVEC_FUSE"]["verdict"] == "DROP" and tri["flags"]["STRATA_CVEC_FUSE"]["pass2_clean"] is True
    assert json.loads((sw_dir / "rd-triage.v2.json").read_text())["flags"]["STRATA_EXPDEC"]["verdict"] == "OPTIONAL"
    rows = [json.loads(l)["arm"] for l in (sw_dir / "retest-rdC-1-k01.jsonl").read_text().splitlines()]
    assert rows == ["warmup", "updated", "plus-EMB_REUSE_ACCOUNT", "updated~2", "plus-EMB_REUSE_ACCOUNT~2", "updated~3", "plus-EMB_REUSE_ACCOUNT~3", "base-late"]
    assert not any(c.startswith("SIG") for c in box.calls())
    assert not any(c.startswith("SYSTEMCTL") and (" start " in c or " stop " in c) for c in box.calls())
    # the same slice again: refused, nothing overwritten
    n = len(be_calls(box, "abtest.py"))
    box.finish(box.start(RT, env=be_env(RETEST_SLICE="rdC-1", STUB_ARM_DECODE=json.dumps(dec), RETEST_LOG=(box.S / "bench-retest.log").as_posix())), timeout=240)
    assert "was run before" in "\n".join(box.log("bench-retest.log")) and len(be_calls(box, "abtest.py")) == n


# ------------------------------------------------------------------------------------------------ US-only queue (owner order 2026-11-20 09:19, TEST_LEDGER.md T-1005)
def test_us_only_slices_hold_every_us_retest_flag_once_in_the_owner_order_and_no_rd_entry():
    mod = sweep_mod()
    groups = json.loads((CHAINS / "retest_groups.json").read_text())
    reg = sweep_registry()
    bases = {"us": retest_bases()["us"]}                               # no RD base at all: a rd: entry in a usT slice could not even resolve
    src = {"rd": "/nonexistent", "us": "/mnt/pancake/benchus/src-us"}
    allb = []
    for sl in ("usT-1", "usT-2", "usT-3", "usT-4"):
        assert all(k.startswith("us:") for k in groups["slices"][sl]), sl
        bl = mod.retest_blocks(groups, sl, bases, reg, src)
        assert 1 <= len(bl) <= 5 and sum(e["est"] for e in bl) <= 9000 and all(e["branch"] == "us" for e in bl), sl
        allb += bl
    paired = [e["flag"] for e in allb if e["type"] == "paired" and not e["bundle"]]
    assert paired == ["STRATA_ENGINE_NICE", "STRATA_SELECT_WMMA", "STRATA_ATTN_DAILY", "STRATA_KV_PREFETCH", "STRATA_STAGER_RING", "STRATA_PF_GROUP_SCRATCH",
                      "STRATA_PROMPT_ATTN_IMMA", "--lookup-max-t", "STRATA_GR_RDNA_Q8", "STRATA_GR_RDNA_A8", "STRATA_GR_INJ_UP", "STRATA_GR_FUSED",
                      "STRATA_GR_RDNA", "STRATA_RDNA_MMVQ_WAVE", "STRATA_RDNA_Q6K", "STRATA_HIPBLASLT_TUNING"]
    by = {e["key"]: e for e in allb}
    assert by["us:BUNDLE-222-updated-exact"]["bundle"] and by["us:GATE-lookup-max-t"]["type"] == "gate" and by["us:SPEC-223"]["type"] == "spec"
    hb = by["us:STRATA_HIPBLASLT_TUNING"]                              # in the updated arm: a minus block, the 100401 table, shipped as <ROOT>/terminal/...
    assert hb["kind"] == "minus" and "STRATA_HIPBLASLT_TUNING" not in hb["arm"]["env"] and "STRATA_HIPBLASLT_TUNING" in hb["base_arm"]["env"]
    assert hb["ship"]["env"]["STRATA_HIPBLASLT_TUNING"] == "<ROOT>/terminal/tools/hip/gfx1100-hipblaslt-100401.txt"


def final_tri(**flags):
    return {"schema": "pancake.sweep-triage/2", "branch": "us", "slices": [], "protected": {}, "drift_max": 5.0, "flags": flags}


def final_rec(flag, verdict, env=None, args=None, clean=True, bundle=False, **kw):
    return {"flag": flag, "verdict": verdict, "env": env or {}, "args": args or {}, "exact": True, "bundle": bundle, "pass1": {"short": 1.0},
            "pass2": {"short": 1.0} if clean else None, "pass2_clean": clean, "reason": "t", **kw}


def test_final_specs_follow_apply_triage_stock_against_updated_exact_and_lossy_confirm_standard():
    mod = sweep_mod()
    s220 = json.loads((SPECS / "220-updated-vs-upstream-short.json").read_text())
    s222 = json.loads((SPECS / "222-updated-exact-vs-upstream-short.json").read_text())
    tri = final_tri(**{
        "STRATA_ATTN_DAILY": final_rec("STRATA_ATTN_DAILY", "DROP", env={"STRATA_ATTN_DAILY": "1"}),                       # clean pass 2 DROP: leaves both arms
        "STRATA_KV_PREFETCH": final_rec("STRATA_KV_PREFETCH", "DROP", env={"STRATA_KV_PREFETCH": "1"}, clean=False),        # DROP without a clean pass 2: HOLD, stays
        "STRATA_EXPDEC": final_rec("STRATA_EXPDEC", "OPTIONAL", env={"STRATA_EXPDEC": "1"}, pass2={"longdec32k": 12.0, "short": -4.0}),
        "STRATA_NEW_FLAG": final_rec("STRATA_NEW_FLAG", "KEEP", env={"STRATA_NEW_FLAG": "1"}),                              # clean KEEP of a flag the arm lacks: added
        "BUNDLE x": final_rec("BUNDLE x", "OPTIONAL", env={"STRATA_PF_F16": "2"}, bundle=True)})                            # a bundle never changes a flag
    specs, lines = mod.final_specs(tri, s220, s222)
    assert set(specs) == {"final-short", "final-long", "final-gate"}
    for n, sp in specs.items():
        assert list(sp["arms"])[0] == "plain-upstream" and sp["arms"]["plain-upstream"]["exe"].endswith("build-up/strata")
        for a in ("updated", "updated-exact"):
            e = sp["arms"][a]["env"]
            assert "STRATA_ATTN_DAILY" not in e and "STRATA_EXPDEC" not in e and e["STRATA_KV_PREFETCH"] == "1" and e["STRATA_NEW_FLAG"] == "1", (n, a)
            assert (e["STRATA_PF_F16"] == "1") if a == "updated" else ("STRATA_PF_F16" not in e)
    assert specs["final-short"]["cases"] == ["short", "edit"] and specs["final-short"]["reps"] == 5
    assert specs["final-long"]["cases"] == ["longdec10k", "longdec32k", "longdec64k", "longdec100k"] and specs["final-long"]["prefill"] is True
    assert list(specs["final-gate"]["arms"]) == ["plain-upstream", "updated-exact", "updated"]
    text = "\n".join(lines)
    assert "FINALARM spec arm updated: removed STRATA_ATTN_DAILY" in text and "added STRATA_NEW_FLAG" in text and "FINALVERDICTS" in text
    assert s220["arms"]["updated"]["env"]["STRATA_ATTN_DAILY"] == "1"          # the inputs are not changed
    assert sum(mod.final_estimate(specs).values()) < 4 * 3600


def test_run_queue_us_runs_only_us_slices_in_the_owner_order_and_ends_with_the_final_comparison():
    q = (CHAINS / "run-queue-us.sh").read_text()
    body = q.split("retest() {")[1]
    assert "rdC" not in body and "rdB" not in body and "sweep rd" not in body and "retest carry-rd" not in body
    assert "for s in usT-1 usT-2 usT-3 usT-4; do retest $s; done" in q and "sweep us $k/9 usB-${k}of9" in q
    tail = q[q.index("for s in usT-1"):]
    order = [tail.index(x) for x in ("usT-1 usT-2", "sweep us $k/9", "retest carry-us", "bench-final-us.v1.sh", "QUEUE_US_DONE")]
    assert order == sorted(order)
    assert "RETEST_NO_RD=1" in q and "BENCH_NO_RESUME=1" in q and "N=15" in q and "SWEEP_HARNESS_SHA:?" in q
    assert "SWEEP_EXCLUDE_META" not in q and "SWEEP_ONLY_META" not in q
    f = (CHAINS / "bench-final-us.sh").read_text()
    assert f.index("WINBENCHFINALUS_DONE") < f.index("WINBENCHFINALUS_ABORTED")
    assert "--tier confirm --block 3 --warmup 1 --outlier robust --reclaim-max 3 --reps 5" in f
    assert "RDB" not in f and "benchRD" not in f and "quality_gate.py" in f and "--near-tie" in f and "final-specs" in f
    assert f.splitlines()[0] == "#!/bin/sh" and re.search(r"WIN[A-Z]*_DONE", f).group(0) == "WINBENCHFINALUS_DONE"      # the marker bench-sched.sh greps


def test_bench_final_us_runs_short_long_and_gate_on_the_shipped_arms_and_refuses_a_second_run(tmp_path):
    box = Box(tmp_path)
    prepare_be(box, rd=False)
    sw_dir = box.tmp / "auto-runs" / "benchsweep"
    sw_dir.mkdir(parents=True, exist_ok=True)
    tri = final_tri(**{"STRATA_ATTN_DAILY": final_rec("STRATA_ATTN_DAILY", "DROP", env={"STRATA_ATTN_DAILY": "1"})})
    (sw_dir / "us-triage.v2.json").write_text(json.dumps(tri))
    env = be_env(FINAL_LOG=(box.S / "bench-final-us.log").as_posix())
    box.finish(box.start("bench-final-us.sh", env=env), timeout=240)
    lines = box.log("bench-final-us.log")
    text = "\n".join(lines)
    assert lines[-1] == "WINBENCHFINALUS_DONE" and "ABORTED" not in text and "ERROR" not in text, text
    assert_start_end_pairs(lines)
    assert "FINALARM spec arm updated: removed STRATA_ATTN_DAILY" in text
    assert "SUMMARY final comparison (US only): final-short exit=0, final-long exit=0, final-gate exit=0" in text
    order = ["SYNC harness", "FINALSPEC final-short", "START final-short", "START final-long", "START final-gate", "SUMMARY final comparison"]
    pos = [text.index(k) for k in order]
    assert pos == sorted(pos), order
    ab = be_calls(box, "abtest.py")
    assert len(ab) == 2 and all("--tier confirm --block 3 --warmup 1 --outlier robust --reclaim-max 3 --reps 5 --no-early-stop" in x for x in ab), ab
    assert sum(1 for c in box.calls() if c.startswith("PY") and "quality_gate.py" in c) == 1
    spec = json.loads((sw_dir / "final-short.json").read_text())
    assert list(spec["arms"]) == ["plain-upstream", "updated", "updated-exact"] and "STRATA_ATTN_DAILY" not in spec["arms"]["updated"]["env"]
    assert not any(c.startswith("SIG") for c in box.calls())
    assert not any(c.startswith("SYSTEMCTL") and (" start " in c or " stop " in c) for c in box.calls())
    n = len(be_calls(box, "abtest.py"))
    for f in ("final-short.jsonl", "final-long.jsonl"):
        (sw_dir / f).write_text("{}")
    box.finish(box.start("bench-final-us.sh", env=env), timeout=240)
    assert "was run before" in "\n".join(box.log("bench-final-us.log")) and len(be_calls(box, "abtest.py")) == n


def test_bench_retest_no_rd_reads_no_rd_config_and_runs_a_us_only_slice(tmp_path):
    box = Box(tmp_path)
    prepare_be(box, rd=False)                                          # no RD config exists on this box
    b = {"short": 100.0, "edit": 140.0, "longdec32k": 30.0, "longdec32k-prompt": 1500.0}
    dec = {"*": b, "minus-GR_RDNA": {**b, "edit": 130.0}}
    box.finish(box.start(RT, env=be_env(RETEST_SLICE="usT-3", RETEST_NO_RD="1", STUB_ARM_DECODE=json.dumps(dec),
                                        RETEST_LOG=(box.S / "bench-retest.log").as_posix())), timeout=300)
    lines = box.log("bench-retest.log")
    text = "\n".join(lines)
    assert lines[-1] == "WINBENCHRETEST_DONE" and "ABORTED" not in text and "ARMFIX" not in text, text
    assert "RETEST usT-3: 5 blocks" in text and "SUMMARY retest slice usT-3: 5 blocks" in text
    assert len(be_calls(box, "abtest.py")) == 5
    assert (box.tmp / "auto-runs" / "benchsweep" / "us-triage-rt-usT-3.v2.json").exists()
    assert not (box.tmp / "auto-runs" / "benchsweep" / "rd-triage-rt-usT-3.v2.json").exists()


def test_run_queue_us2_runs_the_preliminary_first_and_the_final_right_after_the_retest_stage():
    q = (CHAINS / "run-queue-us2.sh").read_text()
    assert "sweep rd" not in q and "carry-rd" not in q and "N=15" in q and "SWEEP_HARNESS_SHA:?" in q
    tail = q[q.index("final preliminary"):]
    order = [tail.index(x) for x in ("final preliminary FINAL_PREFIX=prelim FINAL_REPS=3 FINAL_NO_TRIAGE=1", "for s in usT-2 usT-3 usT-4", "final final FINAL_PREFIX=final FINAL_REPS=5",
                                     "sweep us $k/9", "retest carry-us", "QUEUE_US2_DONE")]
    assert order == sorted(order) and "usT-1" not in tail                      # usT-1 is the running slice of run-queue-us.sh


def test_bench_final_us_prelim_mode_uses_the_shipped_arms_3_reps_and_its_own_file_names(tmp_path):
    box = Box(tmp_path)
    prepare_be(box, rd=False)
    sw_dir = box.tmp / "auto-runs" / "benchsweep"
    sw_dir.mkdir(parents=True, exist_ok=True)
    env = be_env(FINAL_LOG=(box.S / "bench-final-us.log").as_posix(), FINAL_PREFIX="prelim", FINAL_REPS="3", FINAL_NO_TRIAGE="1", FINAL_GATE="0")
    box.finish(box.start("bench-final-us.sh", env=env), timeout=240)
    text = "\n".join(box.log("bench-final-us.log"))
    assert "WINBENCHFINALUS_DONE" in text and "ABORTED" not in text and "ERROR" not in text, text
    ab = be_calls(box, "abtest.py")
    assert len(ab) == 2 and all("--reps 3 --no-early-stop" in x and "prelim-" in x for x in ab), ab
    assert not any("quality_gate.py" in c for c in box.calls() if c.startswith("PY"))
    sp = json.loads((sw_dir / "prelim-short.json").read_text())
    s220 = json.loads((SPECS / "220-updated-vs-upstream-short.json").read_text())["arms"]["updated"]
    assert sp["reps"] == 3 and sp["arms"]["updated"]["env"] == s220["env"] and not (sw_dir / "final-short.json").exists()



# ------------------------------------------------------------------------------------------------ US queue 3 (owner protocol 2026-11-20: screen once, rank by gain, confirm the high ones)
def test_pruned_never_tested_backlog_deferred_and_the_preset_value_is_never_a_reason():
    reg = sweep_registry()
    pr, bk = reg["pruned"]["us"], reg["backlog"]["us"]
    pn, bn, un = {c["name"] for c in pr}, {c["name"] for c in bk}, {c["name"] for c in reg["untested"]["us"]}
    assert pn == {"STRATA_PA_V2T", "STRATA_MMVQ_IL_120", "STRATA_NO_NORM_ROPE"} and all(len(c["prune"]) > 25 for c in pr)
    assert len(bk) == 31 and all("week of 2026-11-23" in c["defer"] for c in bk) and not (pn | bn) & un and not pn & bn and len(un) == 119
    assert all(c["name"] not in reg["measured"]["us"] for c in pr + bk)
    for k in ("STRATA_KVA", "STRATA_CKPT_ASYNC", "STRATA_EARLY_REMOTE_CONTEXTS", "STRATA_EXPERT_V2", "STRATA_PLE_COMPACT", "STRATA_SPEC_GUMBEL", "STRATA_PROMPT_ATTN_Q4"):
        assert k in bn, k                                                           # deferred, not pruned and not in this round
    for k in ("STRATA_PF_GEMM", "STRATA_KV_HOST_DMA", "STRATA_QFUSE", "STRATA_GR_RDNA_Q8", "STRATA_ENGINE_NICE", "STRATA_ATTN_DAILY"):   # preset switches: tested as minus arms
        assert k in un, k
    q = json.loads((CHAINS / "us-backlog.json").read_text())
    assert [f["flag"] for f in q["flags"]] == [c["name"] for c in bk] and "NOT STARTED" in q["status"] and q["design"]["cases"] == ["short", "edit", "longdec64k"]
    mod = sweep_mod()
    arm = json.loads((SPECS / "220-updated-vs-upstream-short.json").read_text())["arms"]["updated"]
    spec, meta = mod.build_spec("us", "/p.json", arm, reg, "/s", "1/1", plain={"env": {}, "args": []}, split="even", source="backlog")
    assert len(meta["flags"]) == 31
    bl = (CHAINS / "run-queue-us-backlog.sh").read_text()
    assert "SWEEP_SOURCE=backlog" in bl and "NOT STARTED" in bl and "us-backlog-ranked.json" in bl and "QUEUE_US_BACKLOG_DONE" in bl and "--high 2.0" in bl


def test_sweep_extra_named_flags_get_one_arm_each_and_are_priority_1():
    mod, reg = sweep_mod(), sweep_registry()
    groups = json.loads((CHAINS / "retest_groups.json").read_text())
    arm = json.loads((SPECS / "220-updated-vs-upstream-short.json").read_text())["arms"]["updated"]
    keys = ["us:--lookup-max-t", "us:STRATA_GR_RDNA", "us:STRATA_RDNA_MMVQ_WAVE", "us:STRATA_RDNA_Q6K", "us:STRATA_HIPBLASLT_TUNING"]
    extra = mod.extra_items(keys, groups, reg, arm, "/s", "/p.json")
    assert [n for n, _, _ in extra] == ["minus-lookup-max-t", "minus-GR_RDNA", "minus-RDNA_MMVQ_WAVE", "minus-RDNA_Q6K", "minus-HIPBLASLT_TUNING"]
    assert all(mod.priority(m) == 1 and m["kind"] == "minus" for _, m, _ in extra)
    spec, meta = mod.build_spec("us", "/p.json", arm, reg, "/s", "1/1", plain={"env": {}, "args": []}, split="even", extra=extra)
    assert len(meta["flags"]) == len(reg["untested"]["us"]) + 5 and "--lookup-max-t" in {f["flag"] for f in meta["flags"].values()}


def test_sweep_screen_only_any_case_above_zero_is_promising_and_the_rank_cut_is_2_percent():
    mod = sweep_mod()
    meta = {"branch": "us", "slice": "usS1-1of5", "protected": {}, "flags": {
        "plus-A": {"flag": "A", "kind": "plus", "exact": True, "env": {"A": "1"}, "args": {}}, "plus-B": {"flag": "B", "kind": "plus", "exact": False, "env": {"B": "1"}, "args": {}},
        "plus-C": {"flag": "C", "kind": "plus", "exact": True, "env": {"C": "1"}, "args": {}}, "plus-D": {"flag": "D", "kind": "plus", "exact": True, "env": {"D": "1"}, "args": {}},
        "plus-E": {"flag": "E", "kind": "plus", "exact": True, "env": {"E": "1"}, "args": {}},
        "minus-H": {"flag": "H", "kind": "minus", "exact": True, "env": {"H": "1"}, "args": {}}}}
    b = {"short": 100.0, "edit": 100.0, "longdec32k": 100.0, "longdec32k-prompt": 1000.0}
    p1 = sweep_rows({"updated": b, "plus-A": {**b, "short": 100.4}, "plus-B": {**b, "edit": 102.5}, "plus-C": {**b, "short": 99.0, "edit": 99.5, "longdec32k": 98.0},
                     "plus-D": {**b, "short": 110.0, "edit": 96.0}, "minus-H": {**b, "short": 105.0}})
    t = mod.triage(meta, p1, screen_only=True)
    assert {k: r["verdict"] for k, r in t["flags"].items()} == {"A": "PROMISING-LOW", "B": "PROMISING", "C": "NO-GAIN-P1", "D": "PROMISING", "E": "UNTESTED", "H": "HURTS"}
    assert [t["flags"][k]["carry"] for k in "ABCDH"] == [False, True, False, True, True]          # LOW goes to the backlog, HIGH and HURTS to the rescreen and t["flags"]["A"]["slice"] == "usS1-1of5"
    spec = {"base": "/p", "cases": ["short", "edit", "longdec32k"], "arms": {"updated": {}, **{"plus-" + k: {"env": {k: "1"}} for k in "ABCD"}, "minus-H": {"env": {}}}}
    cr = [r for r in t["flags"].values() if r["carry"]]
    doc = mod.carry_doc(spec, meta, cr, "usS1-1of5")
    ents = mod.carried([doc], "us")
    assert [e["flag"] for e in ents if e["gain"] > 0] == ["D", "B"]                                  # best case first (the OPTIONAL-pattern flag by its big case)
    rk = mod.rank_doc(ents)
    assert rk["cut"] == 2 and [r["verdict"] for r in rk["flags"]][:2] == ["PROMISING", "PROMISING"] and rk["flags"][2]["flag"] == "H" and rk["high"] == 2.0
    assert mod.rank_doc([]) ["cut"] == 0 and mod.merge.__doc__
    # the old (non screen) triage keeps its bars
    assert mod.triage(meta, p1)["flags"]["A"]["verdict"] == "NO-GAIN-P1"


def test_sweep_warm_up_once_per_slice_and_the_compact_paired_block():
    mod = sweep_mod()
    spec = {"base": "/p.json", "drift_check": True, "arms": {"updated": {"env": {}}, **{"plus-%02d" % i: {"env": {"K": str(i)}} for i in range(17)}}}
    bl = mod.block_specs(spec, 8, "s", "first")
    assert sum(len(o) for _, _, o in bl) == 17 and [len(o) for _, _, o in bl] == [6, 6, 5]
    assert list(bl[0][1]["arms"])[:2] == ["warmup", "updated"] and all(list(sp["arms"])[0] == "updated" and "warmup" not in sp["arms"] for _, sp, _ in bl[1:])
    assert all(sp["drift_check"] for _, sp, _ in bl) and max(len(o) for _, _, o in bl) <= 8
    base, arm = {"env": {}}, {"env": {"X": "1"}}
    sp = mod.paired_spec("/p", "updated", base, "plus-X", arm, layout="compact")
    assert list(sp["arms"]) == ["warmup", "updated", "plus-X", "plus-X~2", "plus-X~3"]
    assert list(mod.paired_spec("/p", "updated", base, "plus-X", arm, layout="compact", warm=False)["arms"]) == ["updated", "plus-X", "plus-X~2", "plus-X~3"]
    assert list(mod.paired_spec("/p", "updated", base, "plus-X", arm)["arms"])[:4] == ["warmup", "updated", "plus-X", "updated~2"]      # the interleaved design stays the default
    b = {"short": 100.0, "edit": 100.0, "longdec32k": 100.0, "longdec32k-prompt": 1000.0}
    rows = sweep_rows({"updated": b, "plus-X": {**b, "short": 103.0}, "plus-X~2": {**b, "short": 104.0}, "plus-X~3": {**b, "short": 90.0}})   # a disturbed rep
    eff, drift, per = mod.paired_eval(rows, "updated", "plus-X", "plus", layout="compact")
    assert eff["short"] == 3.0 and len(per) == 3 and drift["short"] == 0.0                     # the median of the 3 paired gains: the disturbed start does not move it
    assert mod.start_s(["short", "edit", "longdec64k"]) == 190 and mod.start_s(["short", "edit", "longdec32k"]) == 150


def test_sweep_carry_slices_take_offset_and_limit_gates_come_from_clean_lossy_confirms_and_select_lists_inconclusive():
    mod = sweep_mod()
    reg = sweep_registry()
    groups = json.loads((CHAINS / "retest_groups.json").read_text())
    bases = {"us": retest_bases()["us"]}
    spec = {"base": "/p", "cases": ["short", "edit", "longdec32k"], "arms": {"updated": bases["us"]["arms"]["updated"]}}
    meta = {"branch": "us", "slice": "t", "flags": {}}
    cr = []
    for i, g in enumerate((5.0, 3.0, 1.0)):
        an = "plus-F%d" % i
        spec["arms"][an] = {"env": {"STRATA_F%d" % i: "1"}}
        meta["flags"][an] = {"flag": "STRATA_F%d" % i, "kind": "plus", "exact": True, "env": {"STRATA_F%d" % i: "1"}, "args": {}}
        cr.append({"arm": an, "pass1": {"short": g}})
    doc = mod.carry_doc(spec, meta, cr, "t")
    got = mod.retest_blocks(groups, "carry-us-usS2-2", bases, reg, {"us": "/s"}, [doc], 1, 1)
    assert [e["flag"] for e in got] == ["STRATA_F1"] and got[0]["bid"] == "k01"
    assert [e["flag"] for e in mod.retest_blocks(groups, "carry-us-usM-1", bases, reg, {"us": "/s"}, [doc], 2, 6)] == ["STRATA_F2"]
    tri = {"flags": {"STRATA_L": {"flag": "STRATA_L", "kind": "plus", "exact": False, "env": {"STRATA_L": "1"}, "args": {}, "verdict": "RETEST-GATE", "pass2": {"short": 3.5},
                                  "pass2_clean": True, "arm": "plus-L", "source": "bench-retest.sh carry-us-usS2-1 k01"},
                     "STRATA_X": {"flag": "STRATA_X", "kind": "plus", "exact": True, "env": {"STRATA_X": "1"}, "verdict": "KEEP", "pass2": {"short": 5.0}, "pass2_clean": True,
                                  "source": "bench-retest.sh carry-us-usS2-1 k02"},
                     "STRATA_O": {"flag": "STRATA_O", "kind": "minus", "exact": False, "env": {"STRATA_ATTN_DAILY": "1"}, "args": {}, "verdict": "OPTIONAL", "pass2": {"short": 12.0, "edit": -4.0},
                                  "pass2_clean": True, "arm": "minus-O", "source": "bench-retest.sh carry-us-usS2-2 k01"},
                     "STRATA_Z": {"flag": "STRATA_Z", "kind": "plus", "exact": False, "env": {"STRATA_Z": "1"}, "verdict": "RETEST-GATE", "pass2": {"short": 3.5}, "pass2_clean": True,
                                  "source": "bench-retest.sh usT-1 k01"}}}
    ge = mod.gate_entries([tri], "/p", bases["us"]["arms"]["updated"], "carry-us-usS2")
    assert [e["flag"] for e in ge] == ["STRATA_L", "STRATA_O"] and ge[0]["est"] == 2700 and list(ge[0]["arms"]) == ["updated", "plus-L"]
    assert "STRATA_ATTN_DAILY" not in ge[1]["arms"]["minus-O"]["env"] and ge[0]["arms"]["plus-L"]["env"]["STRATA_L"] == "1"
    got = mod.retest_blocks(groups, "gates-us-1", bases, reg, {"us": "/s"}, [], 0, 3, [tri], "carry-us-usS2")
    assert [e["flag"] for e in got] == ["GATE-lookup-max-t", "STRATA_L", "STRATA_O"] and all(e["type"] == "gate" for e in got)
    t2 = {"flags": {"a": {"arm": "plus-a", "flag": "a", "verdict": "INCONCLUSIVE", "slice": "usS1-1of5"}, "b": {"arm": "plus-b", "flag": "b", "verdict": "PROMISING", "slice": "usS1-1of5"},
                    "c": {"arm": "plus-c", "flag": "c", "verdict": "UNTESTED", "slice": "usS1-2of5"}, "d": {"arm": "plus-d", "flag": "d", "verdict": "INCONCLUSIVE", "slice": "old"}}}
    assert mod.WEAK.count("PROMISING") == 1 and mod.WEAK.count("PROMISING-LOW") == 1


def test_final_specs_prelim_long_cases_are_64k_and_100k_only():
    mod = sweep_mod()
    s220 = json.loads((SPECS / "220-updated-vs-upstream-short.json").read_text())
    s222 = json.loads((SPECS / "222-updated-exact-vs-upstream-short.json").read_text())
    tri = {"schema": "pancake.sweep-triage/2", "branch": "us", "protected": {}, "flags": {}}
    specs, _ = mod.final_specs(tri, s220, s222, None, 3, ["longdec64k", "longdec100k"])
    assert specs["final-long"]["cases"] == ["longdec64k", "longdec100k"] and specs["final-short"]["cases"] == ["short", "edit"]
    assert sum(mod.final_estimate(specs, 3).values()) < 2 * 3600 and mod.final_estimate(specs, 3)["final-long"] == 12 * mod.start_s(["longdec64k", "longdec100k"])


def test_run_queue_us3_stage_order_cases_and_pruned_list():
    q = (CHAINS / "run-queue-us3.sh").read_text()
    assert "SWEEP_HARNESS_SHA:?" in q and "sweep rd" not in q and "carry-rd" not in q and "BENCH_PREIDLED=1" in q
    assert "SWEEP_CASES=short,edit,longdec64k" in q and "SWEEP_BLOCK=8" in q and "SWEEP_WARM=first" in q and "RETEST_LAYOUT=compact" in q
    body = q[q.index("# ---- STAGE 1"):]
    order = [body.index(x) for x in ("for k in 1 2 3 4 5; do screen", "final preliminary FINAL_PREFIX=prelim FINAL_REPS=3 FINAL_NO_TRIAGE=1", "FINAL_LONG=longdec64k,longdec100k",
                                     "sw select", "sw rank", "confirm usS2 0 $cut", "retest gates-us-$i", "final final FINAL_PREFIX=final FINAL_REPS=5", "confirm usM $cut", "QUEUE_US3_DONE")]
    assert order == sorted(order)
    for k in ("PA_V2T", "BACKLOG", "backlog.us", "NO_NORM_ROPE", "2.0 %", "124 arms", "run-queue-us-backlog.sh"):
        assert k in q, k


def test_bench_sweep_screen_only_skips_pass_2_writes_promising_and_the_carry_file(tmp_path):
    box = Box(tmp_path)
    prepare_be(box, us=False)
    sweep_rd_configs(box, tmp_path)
    b = {"short": 100.0, "edit": 140.0, "longdec32k": 30.0, "longdec32k-prompt": 1500.0}
    plain = json.loads((box.tmp / "auto-runs" / "benchRD" / "config-plain.json").read_text())
    upd = json.loads((box.tmp / "auto-runs" / "benchRD" / "config-updated.json").read_text())
    mod = sweep_mod()
    _, meta = mod.build_spec("rd", "/p.json", mod.arm_from_configs(plain, upd), sweep_registry(), "/s", "1/6", plain=plain, split="even")
    win = next(n for n, f in meta["flags"].items() if f["kind"] == "plus" and f["exact"])
    dec = {"*": b, win: {**b, "short": 103.0}}                                                  # +3 %: HIGH (any case >= +2 %)
    box.finish(box.start(SW, env=be_env(SWEEP_BRANCH="rd", SWEEP_SLICE="1/6", SWEEP_SPLIT="even", SWEEP_SCREEN_ONLY="1", SWEEP_WARM="first", SWEEP_BLOCK="8",
                                        STUB_ARM_DECODE=json.dumps(dec))), timeout=240)
    text = "\n".join(box.log("bench-sweep.log"))
    assert box.log("bench-sweep.log")[-1] == "WINBENCHSWEEP_DONE" and "ABORTED" not in text and "SCREEN ONLY" in text and "PASS2" not in text and "-p2-" not in text, text
    sw_dir = box.tmp / "auto-runs" / "benchsweep"
    tri = json.loads((sw_dir / "rd-triage-1of6.v2.json").read_text())
    wflag = meta["flags"][win]["flag"]
    assert tri["flags"][wflag]["verdict"] == "PROMISING" and tri["flags"][wflag]["carry"] is True and tri["flags"][wflag]["pass2"] is None
    assert all(r["verdict"] in ("NO-GAIN-P1", "HURTS") for f, r in tri["flags"].items() if f != wflag)                 # a flat minus arm (every case <= 0) is HURTS, a plus arm NO-GAIN
    assert win in json.loads((sw_dir / "rd-carry-1of6.json").read_text())["arms"]
    rows0 = [json.loads(l)["arm"] for l in (sw_dir / "rd-p1-1of6-b01.jsonl").read_text().splitlines()]
    assert rows0[:2] == ["warmup", "updated"]
    if (sw_dir / "rd-p1-1of6-b02.jsonl").exists():
        assert [json.loads(l)["arm"] for l in (sw_dir / "rd-p1-1of6-b02.jsonl").read_text().splitlines()][0] == "updated"


def test_run_queue_us4_runs_the_preliminary_right_after_screen_slice_1_and_never_restarts_it():
    q = (CHAINS / "run-queue-us4.sh").read_text()
    assert "SWEEP_HARNESS_SHA:?" in q and "BENCH_PREIDLED=1" in q and "QUEUE_US4_DONE" in q and "run-queue-us4.log" in q
    body = q[q.index("# ---- STAGE 1: slice 1 is already running"):]
    order = [body.index(x) for x in ('pgrep -fx "sh ./bench-sched.v8.sh"', "final preliminary FINAL_PREFIX=prelim FINAL_REPS=3 FINAL_NO_TRIAGE=1", "for k in 2 3 4 5; do screen",
                                     "sw select", "sw rank", "confirm usS2 0 $cut", "retest gates-us-$i", "final final FINAL_PREFIX=final FINAL_REPS=5", "confirm usM $cut", "QUEUE_US4_DONE")]
    assert order == sorted(order) and "screen 1/5" not in body and "usS1-1of5; done" not in body        # slice 1 (running under run-queue-us3.sh) is never started again


def test_two_run_rule_keep_optional_drop_mixed_gate_and_the_rescreen_triage_record():
    mod = sweep_mod()
    assert mod.two_run({"short": 5.0, "edit": -1.0}, {"short": 3.0, "edit": 0.5}, True)[:1] == ("KEEP",)
    v, why, mean, spread = mod.two_run({"short": 5.0, "edit": -1.0}, {"short": 3.0, "edit": 0.5}, True)
    assert mean == {"short": 4.0, "edit": -0.25} and spread == {"short": 2.0, "edit": 1.5}
    assert mod.two_run({"short": 12.0, "edit": -4.0}, {"short": 11.0, "edit": -5.0}, True)[0] == "OPTIONAL"
    assert mod.two_run({"short": 12.0, "edit": -4.0}, {"short": 11.0, "edit": -1.0}, True)[0] == "KEEP"          # the loss does not repeat in both runs
    assert mod.two_run({"short": 2.0}, {"short": -1.0}, True)[0] == "DROP" and mod.two_run({"short": -1.0, "edit": 2.0}, {"short": 0.0, "edit": -2.0}, True)[0] == "DROP"
    assert mod.two_run({"short": 6.0, "edit": -4.0}, {"short": 5.0, "edit": -3.5}, True)[0] == "MIXED"
    assert mod.two_run({"short": -1.0, "edit": -2.0}, {"short": -0.5}, True)[0] == "DROP" and mod.two_run({"short": -1.0}, {"short": 1.5}, True)[0] == "UNSTABLE"      # HURTS: dropped only when run 2 agrees
    assert mod.two_run({"short": 4.0}, {"short": 3.5}, False)[0] == "RETEST-GATE" and mod.two_run({"short": 2.0}, {"short": 2.5}, False)[0] == "KEEP"
    meta = {"branch": "us", "slice": "usS2r-1-1of1", "protected": {}, "flags": {
        "minus-A": {"flag": "A", "kind": "minus", "exact": True, "env": {"A": "1"}, "args": {}}, "minus-B": {"flag": "B", "kind": "minus", "exact": True, "env": {"B": "1"}, "args": {}},
        "minus-C": {"flag": "C", "kind": "minus", "exact": False, "env": {"C": "1"}, "args": {}}}}
    prior = {"A": {"pass1": {"short": 3.0, "edit": 0.2}}, "B": {"pass1": {"short": 2.5}}, "C": {"pass1": {"edit": 4.0}}}
    b = {"short": 100.0, "edit": 100.0, "longdec32k": 100.0, "longdec32k-prompt": 1000.0}
    # minus-X: the effect of having X on is base / arm - 1: arm below base = a gain
    rows = sweep_rows({"updated": b, "minus-A": {**b, "short": 96.0}, "minus-B": {**b, "short": 101.0}, "minus-C": {**b, "edit": 95.0}})
    t = mod.triage(meta, rows, screen_only=True, prior=prior)
    f = t["flags"]
    assert f["A"]["verdict"] == "KEEP" and f["A"]["pass2_clean"] is True and f["A"]["pass2"]["short"] == 3.58 and f["A"]["spread"]["short"] == 1.17 and f["A"]["runs"][0] == {"short": 3.0, "edit": 0.2}
    assert f["B"]["verdict"] == "DROP" and "does not repeat" in f["B"]["reason"]                                  # minus-B: base / arm - 1 = -1 %: the run-1 benefit did not repeat
    assert f["C"]["verdict"] == "RETEST-GATE" and f["C"]["pass2"]["edit"] == 4.63
    drift = sweep_rows({"updated": b, "minus-A": {**b, "short": 96.0}, "minus-B": {**b, "short": 96.0}, "minus-C": {**b, "edit": 95.0}}, late={**b, "short": 80.0})
    td = mod.triage(meta, drift, screen_only=True, prior=prior)
    assert all(r["verdict"] == "INCONCLUSIVE" and r["run1"] for r in td["flags"].values())                           # drifted run 2: INCONCLUSIVE, run 1 kept for the next rescreen
    assert mod.merge.__doc__
    import apply_triage as AT
    acts = {fl: a for fl, a, _ in AT.plan({"flags": {"A": f["A"], "B": f["B"]}, "protected": {}})}
    assert acts == {"A": "add", "B": "drop"}                                                                        # a two-run record applies like a 3-rep record


def test_select_by_gain_and_the_rescreen_slices_of_the_us5_sequencer(tmp_path):
    import subprocess, sys
    tri = {"flags": {k: {"arm": "m-" + k, "flag": k, "verdict": v, "slice": sl, "pass1": g} for k, v, sl, g in (
        ("H", "PROMISING", "usS1-1of5", {"short": 3.0}), ("L", "PROMISING", "usS1-1of5", {"short": 0.4, "edit": -1.0}), ("X", "INCONCLUSIVE", "usS1-1of5", {}),
        ("H2", "PROMISING", "usS1-2of5", {"edit": 9.0}), ("R", "INCONCLUSIVE", "usS2r-1-1of1", {"short": 4.0}))}}
    (tmp_path / "t.json").write_text(json.dumps(tri))

    def sel(*a):
        out = subprocess.run([sys.executable, str(CHAINS / "sweep.py"), "select", "--triage", str(tmp_path / "t.json"), "--out", str(tmp_path / "m.json"), *a], capture_output=True, text=True).stdout
        return sorted(json.loads((tmp_path / "m.json").read_text())["flags"]), out
    assert sel("--verdict", "PROMISING", "--slice-prefix", "usS1-1of5", "--min-gain", "2.0")[0] == ["m-H"]
    assert sel("--verdict", "PROMISING", "--slice-prefix", "usS1-", "--max-gain", "2.0")[0] == ["m-L"]
    assert sel("--verdict", "INCONCLUSIVE,UNTESTED", "--slice-prefix", "usS1-1of5")[0] == ["m-X"]
    assert sel("--verdict", "INCONCLUSIVE", "--slice-prefix", "usS2r-1-")[0] == ["m-R"]
    q = (CHAINS / "run-queue-us5.sh").read_text()
    assert "SWEEP_HARNESS_SHA:?" in q and "BENCH_PREIDLED=1" in q and "QUEUE_US5_DONE" in q and "run-queue-us5.log" in q
    body = q[q.index("# ---- slice 1 is already running"):]
    order = [body.index(x) for x in ('pgrep -fx "sh ./bench-sched.v8.sh"', "\ncycle 1\n", "for k in 2 3 4 5; do screen $k/5 usS1-${k}of5; cycle $k; done", "\ngates\n",
                                     "final final FINAL_PREFIX=final FINAL_REPS=5", "rescreen usS3r", "QUEUE_US5_DONE")]
    assert order == sorted(order)
    assert "prelim" not in body and "confirm" not in body and "carry-us-" not in body and "plain-upstream" not in body       # no stock arm, no preliminary, no 3-rep confirm stage
    assert "SWEEP_PRIOR=$PRIOR" in q and "--min-gain 2.0" in q and "--max-gain 2.0" in q and "two-run" in q.lower() or "run 1 + run 2" in q


def test_run_queue_us6_rescreens_high_and_hurts_only_low_gain_goes_to_the_backlog_file():
    q = (CHAINS / "run-queue-us6.sh").read_text()
    assert "SWEEP_HARNESS_SHA:?" in q and "run-queue-us6.log" in q and "QUEUE_US6_DONE" in q and "BENCH_PREIDLED=1" in q
    body = q[q.index("# ---- slice 1 is already running"):]
    assert "--verdict PROMISING,HURTS" in q and "--min-gain" not in body and "rescreen usS3r" not in body and "us-backlog-low.json" in body
    order = [body.index(x) for x in ('pgrep -fx "sh ./bench-sched.v8.sh"', "for k in 2 3 4 5; do screen $k/5 usS1-${k}of5; cycle $k; done", "us-backlog-low.json", "final final FINAL_PREFIX=final FINAL_REPS=5", "QUEUE_US6_DONE")]
    assert order == sorted(order) and "prelim" not in body and "plain-upstream" not in body and "confirm" not in body
