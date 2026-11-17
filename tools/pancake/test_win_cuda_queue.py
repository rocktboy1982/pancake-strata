"""Self test of tools/pancake/win_cuda_queue.ps1 and its plan (no GPU, no build, no queue run).
Checks: the plan parses; every script, spec and repo file an item references exists; every {build} exe is listed under
built_by_build (the build step makes it); every external token is declared; the specs lint; bench_win_ab parses them;
`powershell -NoProfile -File win_cuda_queue.ps1 -Plan` lists every item, exits 0 and creates no result folder."""
import json, os, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLAN = HERE / "win_cuda_queue_plan.json"
PS1 = HERE / "win_cuda_queue.ps1"
sys.path.insert(0, str(HERE))


def load():
    return json.loads(PLAN.read_text(encoding="utf-8"))


def all_strings(item):
    out = []
    for s in item.get("steps", []):
        out.append(s["exe"])
        out.extend(s.get("args", []))
        out.extend((s.get("env") or {}).values())
    out.extend(item.get("needs", []))
    return out


def test_plan_parses_and_ids_are_unique_and_ordered():
    p = load()
    ids = [i["id"] for i in p["items"]]
    assert len(ids) == len(set(ids)) and ids == sorted(ids)
    assert ids[0].endswith("build") and p["items"][0].get("kind") == "build"
    for i in p["items"]:
        assert i["group"] in ("A", "B"), i["id"]
        assert i["timeout_min"] > 0 and i["est_min"] > 0 and i["ram_gb"] >= 0 and i["vram_gb"] >= 0
        assert i.get("kind") == "build" or i["steps"], i["id"]
        assert i["ledger"].startswith("T-") or i["ledger"] == "none", i["id"]


def test_every_referenced_path_exists_or_is_built_or_external():
    p = load()
    ext = set(p["external"])
    built = set(p["built_by_build"])
    for i in p["items"]:
        for s in all_strings(i):
            if not s.startswith("{"):
                continue
            tok = s[: s.index("}") + 1]
            rest = s[len(tok):].lstrip("/\\")
            if tok in ext:
                continue
            if tok == "{repo}":
                assert (ROOT / rest).exists(), (i["id"], s)
            elif tok == "{build}" and rest == "":
                continue                                   # the build directory itself (ctest --test-dir)
            elif tok == "{build}":
                assert rest.endswith(".exe") and rest[:-4] in built, (i["id"], s, "not built by the build step")
            elif tok == "{out}":
                continue
            else:
                raise AssertionError((i["id"], s, "unknown token"))


def test_build_targets_exist_in_cmake():
    cm = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8", errors="replace")
    for t in load()["built_by_build"]:
        assert ("add_executable(" + t) in cm, t


def test_specs_lint_and_runner_reads_them():
    import bench_win_ab
    tmp = ROOT / "bench" / "results" / ".tmp_lint_win_queue"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        for n in sorted((ROOT / "bench" / "specs").glob("1[56]?-gtx1660ti-*.json")):
            shutil.copy(n, tmp / n.name)
            spec = json.loads(n.read_text(encoding="utf-8"))
            names = list(spec["arms"])
            assert names[0] == "base" and names[-1] == "base-again", n.name
            assert spec["runner"] == "bench_win_ab"
            args = bench_win_ab.argparse.Namespace(runs=None, warmup=None, measured=None, max_new=None, only=None)
            steps = bench_win_ab.plan(spec, args)
            assert [s[0] for s in steps] == names
        r = subprocess.run([sys.executable, str(HERE / "spec_lint.py"), "--dir", str(tmp)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_bench_summary_parser():
    import bench_win_ab
    line = "short  decode  34.30 +/- 14.10 tok/s (95%, n=3 runs)  per-run ['39.8']  time-to-first-token 0.62 s"
    assert bench_win_ab.parse_summary(line)["short"]["decode_tps"] == 34.3


def test_plan_mode_lists_every_item_and_creates_nothing():
    exe = shutil.which("powershell") or shutil.which("powershell.exe")
    if not exe or os.name != "nt":
        print("skip: no Windows PowerShell")
        return
    before = set((ROOT / "bench" / "results").iterdir())
    r = subprocess.run([exe, "-NoProfile", "-File", str(PS1), "-Plan"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    for i in load()["items"]:
        if i.get("only_with"):
            continue
        assert i["id"] in r.stdout, i["id"]
    assert "MISSING" not in r.stdout, r.stdout
    assert "estimated total" in r.stdout
    assert set((ROOT / "bench" / "results").iterdir()) == before


def test_script_has_no_forbidden_constructs():
    s = PS1.read_text(encoding="utf-8")
    code = "\n".join(l for l in s.splitlines() if not l.lstrip().startswith("#"))
    assert " && " not in code            # PowerShell 5.1 (the cmd line of the build step has a cmd ||)
    assert "Stop-Process" not in code and "Register-ScheduledTask" not in code and "powercfg /s" not in code
    assert "Set-ExecutionPolicy" not in code and "taskkill.exe /IM" not in code


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
