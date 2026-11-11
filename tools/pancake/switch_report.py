"""switch_report.py - which of an arm's env switches acted, read from the engine's own log (project feature 1, MOD_PLAN.md).

Why: COPY256, IC_WARM_MB and ROUTER_FUSE were silent no-ops under deferral, so earlier arms measured nothing and nobody could tell.
Every switch now prints one line the first time it acts, or says why it will not (include/strata/core/switch_report.hpp):

    strata switch: NAME acted: what it did
    strata switch: NAME will not act: why

SWITCH_LINE below is the one place that knows this format.  NAME is matched without its STRATA_ prefix and without a "=value" suffix.

An arm's switches are the STRATA_* keys of its env overlay in the spec, except a value that turns the key off (OFF) and the names of
the exemption list.  The exemption list (bench/promotion/switch-exempt.json, plus an item's own "switch_exempt"; both
{"reason": ["STRATA_NAME", ...]}) is for switches that print no such line yet: remove a name from it when its switch starts to log.

Status of one switch over the runs of one arm (one engine start each):
  ACTED    every run logged "acted"
  PARTIAL  some runs logged "acted", the others did not, or have no engine log: those runs measured nothing or are unverified
  VOID     no run logged "acted" (a "will not act" line gives the reason): the arm measured nothing
  NO_LOG   no run has an engine log at all (a result from before abtest kept it): the switch is unverified
  EXEMPT   on the exemption list

Where the logs are: abtest keeps, per attempt, <results>/<stem>/<arm>-r<rep>-a<attempt>.engine.log: the engine's log (stderr) of that
engine start.  The .out file beside it is only the server's narration of the start; the engine's lines are not in it (checked on
bench/results/2026-11-11-gpu/25-kern-hip-confirm), so it is not read.  The engine's own log file (the "log" key of the preset) is shared by
every start and arm, so it cannot say which run a line belongs to.  This tool only reads.

    python tools/pancake/switch_report.py bench/results/<day>/<stem> [--spec bench/specs/<stem>.json]
        per arm and switch: the status over every engine start in the folder (promote.py judges only the attempts abtest used);
        exit 1 when a switch is not ACTED or EXEMPT
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SWITCH_LINE = re.compile(r"strata switch: (?P<name>\w+?)(?:=\S*)? (?P<state>acted|will not act):[ \t]*(?P<detail>.*)")
ACTED_WORD, WILL_NOT = "acted", "will not act"
LOG_SUFFIX = ".engine.log"
LOG_NAME = re.compile(r"(?P<arm>.+)-r(?P<rep>\d+)-a(?P<attempt>\d+)" + re.escape(LOG_SUFFIX))
OFF = {"", "0", "off", "false", "no"}      # a value that turns the key off: that arm does not ask the switch to act
EXEMPT_FILE = "bench/promotion/switch-exempt.json"
ACTED, PARTIAL, VOID, NO_LOG, EXEMPT = "ACTED", "PARTIAL", "VOID", "NO_LOG", "EXEMPT"


def short(name):
    return name[len("STRATA_"):] if name.startswith("STRATA_") else name


def parse(text):
    """Engine log text -> {short name: (state, detail)}.  "acted" beats "will not act" (a switch can be asked again after a reload)."""
    found = {}
    for m in SWITCH_LINE.finditer(text):
        name = short(m["name"])
        if found.get(name, ("",))[0] != ACTED_WORD:
            found[name] = (m["state"], m["detail"].strip())
    return found


def switches(env):
    """The env overlay of an arm -> the engine switches it turns on (STRATA_* keys that are not removed or set to an off value)."""
    return [k for k, v in (env or {}).items() if k.startswith("STRATA_") and v is not None and str(v).strip().lower() not in OFF]


def load_exempt(root, item=None):
    """{short name: reason} from the repo's exemption file and the item's "switch_exempt" (both {"reason": [names]})."""
    path = Path(root) / EXEMPT_FILE
    groups = [json.loads(path.read_text()).get("exempt", {})] if path.is_file() else []
    groups.append((item or {}).get("switch_exempt", {}))
    return {short(n): reason for g in groups for reason, names in g.items() for n in names}


def read_log(folder, arm, rep, attempt):
    """The engine log text kept for one attempt, or None when abtest kept none."""
    path = Path(folder) / f"{arm}-r{rep}-a{attempt}{LOG_SUFFIX}"
    return path.read_text(errors="replace") if path.is_file() else None


def arm_status(env, logs, exempt):
    """env: the arm's env overlay; logs: the engine log text of every run of the arm (None where none was kept).
    -> {name: (status, text)} for every switch of the arm."""
    have = [parse(t) for t in logs if t is not None]
    out = {}
    for name in switches(env):
        key = short(name)
        if key in exempt:
            out[name] = (EXEMPT, exempt[key])
        elif not have:
            out[name] = (NO_LOG, f"no <arm>-r<rep>-a<attempt>{LOG_SUFFIX} kept" + (f" for any of the {len(logs)} runs" if logs else ""))
        else:
            acted = [p[key][1] for p in have if p.get(key, ("",))[0] == ACTED_WORD]
            refused = [p[key][1] for p in have if p.get(key, ("",))[0] == WILL_NOT]
            if len(acted) == len(logs):
                out[name] = (ACTED, acted[0])
            elif acted:
                out[name] = (PARTIAL, f"acted in {len(acted)} of {len(logs)} runs" + (f", {len(logs) - len(have)} without an engine log"
                                                                                  if len(have) < len(logs) else ""))
            else:
                out[name] = (VOID, ("will not act: " + refused[0]) if refused else f"no line in {len(have)} engine logs (a switch that prints none yet belongs in {EXEMPT_FILE})")
    return out


# ------------------------------------------------------------------------------------------------ command line
def folder_logs(folder):
    """{arm: [log text per attempt]} for every <arm>-r<rep>-a<attempt> log in the folder."""
    keys = sorted({(m["arm"], int(m["rep"]), int(m["attempt"])) for p in Path(folder).iterdir() if (m := LOG_NAME.fullmatch(p.name))})
    logs = {}
    for arm, rep, attempt in keys:
        logs.setdefault(arm, []).append(read_log(folder, arm, rep, attempt))
    return logs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", help="bench/results/<day>/<stem>: the folder with the per-attempt logs")
    ap.add_argument("--spec", help="the spec of the run (default: bench/specs/<folder name>.json)")
    a = ap.parse_args(argv)
    spec_path = Path(a.spec or ROOT / "bench/specs" / f"{Path(a.folder).name}.json")
    if not spec_path.is_file() or not Path(a.folder).is_dir():
        print(f"switch_report: need the results folder and its spec ({spec_path})", file=sys.stderr)
        return 2
    spec = json.loads(spec_path.read_text())
    exempt = load_exempt(ROOT)
    logs = folder_logs(a.folder)
    worst = 0
    for arm, v in spec["arms"].items():
        env = v.get("env")
        runs = logs.get(arm, [])
        print(f"arm {arm}: {len(runs)} engine starts, {len(switches(env))} switches")
        for name, (status, text) in arm_status(env, runs, exempt).items():
            print(f"  {status:8s} {name}: {text}")
            worst = max(worst, int(status not in (ACTED, EXEMPT)))
    return worst


if __name__ == "__main__":
    sys.exit(main())
