"""task_gate.py - pancake-strata task gate (MOD_PLAN A55, phase 16.13): a fixed task set decides between close variants.

Token forcing (quality_gate.py) only rejects arms that deviate a lot; "Displacement Is Not Direction" (Toronto 2026,
arXiv 2606.19558) shows KLD / top-1 agreement carry no information about the task score among near-baseline variants
(rho = 0.00 on Qwen3.6-A3B).  So the lossy arms that survive token forcing (turbo, AcceptMoE depth prior,
VERIFY_ECAP, deferral rank guard, a different quant) are judged here, by pass rate on a fixed task set, repeated.

Same spec as tools/pancake/abtest.py: {"base": PRESET.json, "arms": {"base": {}, "turbo": {"env": {...}, "args": {...}}}}.
The first arm is the base.  Optional spec keys: "tasks" (task set path), "reps" (default 5), "temperature" (0.2),
"max_tokens" (1024), "thinking" (false), "margin" (0.05), "validity_margin" (0.03).
Per arm (A18, A17, A20): "request": {"thinking": true, "reasoning_effort": "xhigh", "max_tokens": 8192, "temperature": 0.2}
overrides those request settings for that arm only (an arm per effort level = time to success per level), and
"config": {"reasoning_budget_by_effort": {...}, "rumination_close": 0.4, ...} sets top-level keys of the server config for
that arm (abtest.make_config; a null removes a key).  "time_curve": true (spec key) adds, per arm, the share of runs that
succeeded within each of "time_budgets" seconds (default 10 30 60 120 300) and the area under that curve up to
"time_horizon_s" (default 120): mean over runs of max(0, 1 - t / T) for a success, 0 for a failure (SWE-Effi's idea, A57).
The default report is unchanged without it.

    python tools/pancake/task_gate.py SPEC.json --out bench/results/<day>-taskgate [--tasks bench/tasks/default.json]
                                      [--reps 5] [--only code_edit,tool_call] [--max-load 1.0]
    python tools/pancake/task_gate.py SPEC.json --out ... --dry-run     validate spec + task set + every checker; no server
    python tools/pancake/task_gate.py --from-runs DIR [--regrade]       recompute the report from DIR/runs-*.jsonl

Per arm, one fresh server (abtest.start_server); the base arm runs twice (first and last: the run-to-run noise and
the drift check) unless --no-base-repeat.  Within an arm: rep-major order (every task once per rep), seed = 1000 + rep
(the same seed in every arm: paired).  A task is code_edit (the answer's python block is run with the task's test, no
network, CPU / memory limits), tool_call (tools with JSON schemas: the call must parse, validate against the schema
and match the expected name and arguments; multi-turn and parallel calls supported; an expected "no call" too), or
qa (regex on the answer).  Checkers need no network and no external service.

Per arm the report has: success rate with a 95% Wilson interval (pooled runs), per-kind rates, tool-call validity
rate, time to success (median, and seconds per success = all request seconds / successes), the area under the
time-to-success curve (A57: time_auc(); one common time axis, the longest run of the batch or --auc-horizon SECONDS; a
paired bootstrap interval of the difference vs base; information only, it never changes a verdict), and the verdict vs
base (see verdict()).  A server that dies mid-arm discards that attempt and retries the arm once (like abtest.py); a
request that times out or returns an HTTP error is a failed run, a refused connection is a dead server.
"""
import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abtest  # noqa: E402

ROOT = abtest.ROOT
DEFAULT_TASKS = "bench/tasks/default.json"
KINDS = ("code_edit", "tool_call", "qa")
SEED0 = 1000
DEFAULTS = {"reps": 5, "temperature": 0.2, "max_tokens": 1024, "thinking": False, "margin": 0.05, "validity_margin": 0.03,
            "time_curve": False, "time_horizon_s": 120.0, "time_budgets": [10, 30, 60, 120, 300]}
REQUEST_KEYS = ("thinking", "reasoning_effort", "max_tokens", "temperature")     # an arm's "request" may set these


# ---- JSON-schema subset (tool parameters): no dependency, no network
def validate_schema(value, schema, path="$"):
    """Errors (a list of strings) of `value` against the schema subset used for tool parameters: type (name or list),
    enum, const, required, properties, additionalProperties (false), items, minItems, maxItems, minimum, maximum,
    minLength, maxLength, pattern, anyOf.  Empty list = valid."""
    errs = []
    if not isinstance(schema, dict):
        return errs
    if "anyOf" in schema:
        if not any(not validate_schema(value, s, path) for s in schema["anyOf"]):
            errs.append(f"{path}: matches none of anyOf")
    t = schema.get("type")
    if t is not None:
        ok = False
        for name in (t if isinstance(t, list) else [t]):
            ok = ok or {
                "string": isinstance(value, str),
                "integer": isinstance(value, int) and not isinstance(value, bool)
                or (isinstance(value, float) and value.is_integer()),
                "number": isinstance(value, (int, float)) and not isinstance(value, bool),
                "boolean": isinstance(value, bool), "null": value is None,
                "array": isinstance(value, list), "object": isinstance(value, dict)}.get(name, False)
        if not ok:
            return errs + [f"{path}: expected {t}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r} not in enum")
    if "const" in schema and value != schema["const"]:
        errs.append(f"{path}: not {schema['const']!r}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errs.append(f"{path}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errs.append(f"{path}: longer than {schema['maxLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errs.append(f"{path}: does not match {schema['pattern']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{path}: below {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{path}: above {schema['maximum']}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errs.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errs.append(f"{path}: more than {schema['maxItems']} items")
        if isinstance(schema.get("items"), dict):
            for i, v in enumerate(value):
                errs += validate_schema(v, schema["items"], f"{path}[{i}]")
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        for k in schema.get("required") or []:
            if k not in value:
                errs.append(f"{path}: missing {k!r}")
        if schema.get("additionalProperties") is False:
            for k in value:
                if k not in props:
                    errs.append(f"{path}: unexpected {k!r}")
        for k, sub in props.items():
            if k in value:
                errs += validate_schema(value[k], sub, f"{path}.{k}")
    return errs


def schema_problems(schema, path="$"):
    """Problems in a task's own schema (dry-run): unknown type names, required not in properties, bad bounds."""
    out = []
    if not isinstance(schema, dict):
        return [f"{path}: schema is not an object"]
    t = schema.get("type")
    for name in ([] if t is None else t if isinstance(t, list) else [t]):
        if name not in ("string", "integer", "number", "boolean", "null", "array", "object"):
            out.append(f"{path}: unknown type {name!r}")
    props = schema.get("properties") or {}
    for k in schema.get("required") or []:
        if k not in props:
            out.append(f"{path}: required {k!r} is not in properties")
    if "minimum" in schema and "maximum" in schema and schema["minimum"] > schema["maximum"]:
        out.append(f"{path}: minimum > maximum")
    if "pattern" in schema:
        try:
            re.compile(schema["pattern"])
        except re.error as e:
            out.append(f"{path}: pattern: {e}")
    for k, sub in props.items():
        out += schema_problems(sub, f"{path}.{k}")
    if isinstance(schema.get("items"), dict):
        out += schema_problems(schema["items"], path + "[]")
    return out


# ---- checkers
OPS = {"any_of", "re", "min", "max", "set"}
THINK = re.compile(r"<think>.*?</think>", re.S)


def arg_match(want, got):
    """Does the argument value `got` satisfy `want`?  want: a literal (strings compare case-insensitive, trimmed),
    {"any_of": [...]}, {"re": PATTERN} (search, case-insensitive), {"min": x, "max": y}, {"set": [...]} (a list in
    any order), or a nested object (every key listed must match; extra keys in got are fine)."""
    if isinstance(want, dict):
        if want and set(want) <= OPS:
            if "any_of" in want and not any(arg_match(w, got) for w in want["any_of"]):
                return False
            if "re" in want and not (isinstance(got, str) and re.search(want["re"], got, re.I)):
                return False
            if "min" in want and not (isinstance(got, (int, float)) and not isinstance(got, bool) and got >= want["min"]):
                return False
            if "max" in want and not (isinstance(got, (int, float)) and not isinstance(got, bool) and got <= want["max"]):
                return False
            if "set" in want:
                if not isinstance(got, list) or len(got) != len(want["set"]):
                    return False
                rest = list(got)
                for w in want["set"]:
                    i = next((i for i, g in enumerate(rest) if arg_match(w, g)), None)
                    if i is None:
                        return False
                    rest.pop(i)
            return True
        return isinstance(got, dict) and all(k in got and arg_match(v, got[k]) for k, v in want.items())
    if isinstance(want, str):
        return isinstance(got, str) and got.strip().lower() == want.strip().lower()
    if isinstance(want, bool) or isinstance(got, bool):
        return want is got
    if isinstance(want, (int, float)):
        return isinstance(got, (int, float)) and abs(got - want) < 1e-9
    return want == got


def parse_calls(message, tools):
    """The message's tool calls as [{"name", "args", "problems": [...]}].  A problem makes the call invalid: unknown
    tool, arguments that do not parse as a JSON object, schema errors.  <tool_call> text with no tool_calls is one
    invalid call (a client cannot use it)."""
    by_name = {(t.get("function") or t).get("name"): (t.get("function") or t) for t in tools or []}
    raw = message.get("tool_calls") or []
    calls = []
    for c in raw:
        fn = (c or {}).get("function") or {}
        name, args, probs = fn.get("name"), fn.get("arguments"), []
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except ValueError:
                probs.append("arguments are not JSON")
                args = None
        if args is not None and not isinstance(args, dict):
            probs.append("arguments are not an object")
            args = None
        if name not in by_name:
            probs.append(f"unknown tool {name!r}")
        elif args is not None:
            probs += validate_schema(args, by_name[name].get("parameters") or {})
        calls.append({"name": name, "args": args, "problems": probs})
    if not raw and "<tool_call>" in (message.get("content") or ""):
        calls.append({"name": None, "args": None, "problems": ["tool call returned as text"]})
    return calls


def grade_tool(task, message):
    want = task["expect"].get("calls", [])
    calls = parse_calls(message, task.get("tools"))
    bad = [p for c in calls for p in c["problems"]]
    valid = not bad and (bool(calls) or not want)
    if not calls and want:
        bad = ["no tool call"]
    if bad:
        return {"ok": False, "valid": False, "reason": "; ".join(bad)[:300]}
    if len(calls) != len(want):
        return {"ok": False, "valid": True, "reason": f"{len(calls)} calls, wanted {len(want)}"}
    rest = list(calls)
    for w in want:
        i = next((i for i, c in enumerate(rest) if c["name"] == w["name"] and arg_match(w.get("args") or {}, c["args"])), None)
        if i is None:
            return {"ok": False, "valid": valid, "reason": f"no call matches {w['name']} {json.dumps(w.get('args'))[:120]}"}
        rest.pop(i)
    if not want and task["expect"].get("answer_re"):
        if not re.search(task["expect"]["answer_re"], THINK.sub("", message.get("content") or ""), re.I | re.S):
            return {"ok": False, "valid": True, "reason": "answer does not match"}
    return {"ok": True, "valid": True, "reason": ""}


def grade_qa(task, message):
    text = THINK.sub("", message.get("content") or "").strip()
    if not re.search(task["expect"]["answer_re"], text, re.I | re.S):
        return {"ok": False, "valid": None, "reason": f"no match for {task['expect']['answer_re']}: {text[:80]!r}"}
    if task["expect"].get("forbid_re") and re.search(task["expect"]["forbid_re"], text, re.I | re.S):
        return {"ok": False, "valid": None, "reason": "forbidden text"}
    return {"ok": True, "valid": None, "reason": ""}


FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.S)


def extract_code(text):
    """The code of an answer: the last fenced block that defines something, else the last block, else the whole text
    when it looks like code, else None."""
    text = THINK.sub("", text or "")
    blocks = [(lang.lower(), body) for lang, body in FENCE.findall(text)]
    py = [b for lang, b in blocks if lang in ("", "python", "py", "python3")]
    pick = [b for b in py if re.search(r"^\s*(def|class)\s", b, re.M)] or py
    if pick:
        return pick[-1]
    if re.search(r"^\s*(def|class)\s", text, re.M):
        return text
    return None


_RUNNER = ("import resource,runpy,sys;resource.setrlimit(resource.RLIMIT_AS,(1<<30,1<<30));"
           "resource.setrlimit(resource.RLIMIT_CPU,({cpu},{cpu}));resource.setrlimit(resource.RLIMIT_FSIZE,(1<<20,1<<20));"
           "runpy.run_path(sys.argv[1],run_name='__main__')")
_NETLESS = None


def _netless_prefix():
    """["unshare", "-rn"] when unprivileged network namespaces work here (the code then has no network), else []."""
    global _NETLESS
    if _NETLESS is None:
        try:
            r = subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=10)
            _NETLESS = ["unshare", "-rn"] if r.returncode == 0 else []
        except (OSError, subprocess.SubprocessError):
            _NETLESS = []
    return _NETLESS


def run_python(code, timeout=10):
    """Runs code in a throw-away directory, `python -I`, empty environment, 1 GiB / CPU / file-size limits, and no network
    namespace where available.  Returns (returncode or None on timeout, output tail)."""
    with tempfile.TemporaryDirectory(prefix="taskgate-") as d:
        (Path(d) / "main.py").write_text(code)
        cmd = _netless_prefix() + [sys.executable, "-I", "-c", _RUNNER.format(cpu=timeout), "main.py"]
        try:
            r = subprocess.run(cmd, cwd=d, env={"PATH": "/usr/bin:/bin"}, capture_output=True, text=True, timeout=timeout + 5)
        except subprocess.TimeoutExpired:
            return None, "timeout"
        return r.returncode, (r.stdout + r.stderr)[-300:]


def grade_code(task, message):
    code = extract_code(message.get("content") or "")
    if code is None:
        return {"ok": False, "valid": None, "reason": "no code in the answer"}
    rc, out = run_python(code + "\n\n" + task["expect"]["test"], task["expect"].get("timeout", 10))
    if rc == 0:
        return {"ok": True, "valid": None, "reason": ""}
    return {"ok": False, "valid": None, "reason": ("timeout" if rc is None else out.strip().splitlines()[-1] if out.strip()
                                                  else f"exit {rc}")[:300]}


def grade(task, message):
    """message: the assistant message of the response (content, tool_calls).  -> {"ok", "valid", "reason"}"""
    return {"code_edit": grade_code, "tool_call": grade_tool, "qa": grade_qa}[task["kind"]](task, message)


def as_message(ref):
    """A task's reference / wrong answer ({"content": str, "calls": [{"name", "args"}]}) in the API message shape."""
    calls = [{"id": f"call_{i}", "type": "function",
              "function": {"name": c["name"], "arguments": c["args"] if isinstance(c["args"], str) else json.dumps(c["args"])}}
             for i, c in enumerate(ref.get("calls") or [])]
    return {"role": "assistant", "content": ref.get("content"), **({"tool_calls": calls} if calls else {})}


# ---- task set
def load_tasks(path, only=None):
    ts = json.load(open(path))
    tasks = ts["tasks"]
    if only:
        keep = set(only.split(","))
        tasks = [t for t in tasks if t["kind"] in keep or t["id"] in keep]
    return ts, tasks


def validate_tasks(tasks):
    """Problems of the task set as a list of strings (empty = fine); does not run anything."""
    probs, seen = [], set()
    for t in tasks:
        i = t.get("id", "?")
        if i in seen or not t.get("id"):
            probs.append(f"{i}: id missing or duplicated")
        seen.add(i)
        k = t.get("kind")
        if k not in KINDS:
            probs.append(f"{i}: kind {k!r} not in {KINDS}")
            continue
        if not (t.get("prompt") or t.get("messages")):
            probs.append(f"{i}: needs prompt or messages")
        for m in t.get("messages") or []:
            if m.get("role") not in ("system", "user", "assistant", "tool"):
                probs.append(f"{i}: message role {m.get('role')!r}")
        ex = t.get("expect")
        if not isinstance(ex, dict):
            probs.append(f"{i}: expect missing")
            continue
        if not isinstance(t.get("reference"), dict) or not isinstance(t.get("wrong"), dict):
            probs.append(f"{i}: reference and wrong answers are required (checker self-test)")
        if k == "code_edit" and not ex.get("test"):
            probs.append(f"{i}: expect.test missing")
        if k == "qa" or (k == "tool_call" and not ex.get("calls") and "answer_re" in ex):
            try:
                re.compile(ex["answer_re"])
            except (KeyError, re.error) as e:
                probs.append(f"{i}: expect.answer_re: {e}")
        if k == "tool_call":
            names = set()
            if not t.get("tools"):
                probs.append(f"{i}: tools missing")
            for tool in t.get("tools") or []:
                fn = tool.get("function") or {}
                if not fn.get("name") or not isinstance(fn.get("parameters"), dict):
                    probs.append(f"{i}: tool without name / parameters")
                    continue
                names.add(fn["name"])
                probs += [f"{i}: {fn['name']}: {p}" for p in schema_problems(fn["parameters"])]
            for c in ex.get("calls") or []:
                if c.get("name") not in names:
                    probs.append(f"{i}: expected call to unknown tool {c.get('name')!r}")
                else:
                    sch = next(x["function"] for x in t["tools"] if x["function"]["name"] == c["name"])["parameters"]
                    for key in c.get("args") or {}:
                        if key not in (sch.get("properties") or {}):
                            probs.append(f"{i}: expected argument {key!r} is not a parameter of {c['name']}")
    return probs


def selftest(tasks):
    """Runs each checker on the task's reference answer (must pass) and wrong answer (must fail)."""
    out = []
    for t in tasks:
        try:
            g, w = grade(t, as_message(t["reference"])), grade(t, as_message(t["wrong"]))
        except Exception as e:  # a checker bug is a finding, not a crash
            out.append(f"{t['id']}: checker raised {type(e).__name__}: {e}")
            continue
        if not g["ok"]:
            out.append(f"{t['id']}: reference answer fails its checker ({g['reason']})")
        if w["ok"]:
            out.append(f"{t['id']}: wrong answer passes its checker")
        if t["kind"] == "tool_call" and t["expect"].get("calls") and not g["valid"]:
            out.append(f"{t['id']}: reference answer is not a valid call ({g['reason']})")
    return out


# ---- statistics
def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def task_rates(runs, key="ok"):
    """{task: [0/1, ...]} from run records (records whose `key` is None are left out)."""
    out = {}
    for r in runs:
        if r.get(key) is not None:
            out.setdefault(r["task"], []).append(1 if r[key] else 0)
    return out


def boot_diff(arm, base, boots=2000, seed=0, resample_tasks=True, resample_reps=True):
    """Paired difference of the mean per-task rate, arm minus base, with a 95% bootstrap interval.  arm / base:
    {task: [0/1,...]}.  Only tasks present in both count.  Each draw resamples the tasks (with replacement; off with
    resample_tasks=False: then the interval is the repetition noise on this fixed task set only) and, inside a task,
    the repetitions of each arm (off with resample_reps=False, tasks only: a task's rate already carries its repetition
    noise, so resampling both levels counts it twice - in a simulation of 18 tasks x 5 repetitions the default interval
    was ~35% wider than the tasks-only one and covered the true difference 97-100% of the time for a nominal 95%).
    Returns (diff, lo, hi, n_tasks) or None when no common task."""
    tasks = [t for t in arm if t in base and arm[t] and base[t]]
    if not tasks:
        return None
    rng = random.Random(seed)
    mean = lambda xs: sum(xs) / len(xs)
    per_task = {t: mean(arm[t]) - mean(base[t]) for t in tasks}
    d = mean(list(per_task.values()))
    ds = []
    for _ in range(boots):
        pick = [rng.choice(tasks) for _ in tasks] if resample_tasks else tasks
        if resample_reps or not resample_tasks:     # (tasks fixed and reps fixed would not be an interval at all)
            s = 0.0
            for t in pick:
                a, b = arm[t], base[t]
                s += mean(rng.choices(a, k=len(a))) - mean(rng.choices(b, k=len(b)))
        else:
            s = sum(per_task[t] for t in pick)
        ds.append(s / len(pick))
    ds.sort()
    return d, ds[int(0.025 * boots)], ds[min(boots - 1, int(0.975 * boots))], len(tasks)


ORDER = ["BETTER", "PASS", "INCONCLUSIVE", "REJECT"]


def verdict(lo, hi, margin):
    """Non-inferiority on the paired difference arm - base (95% interval [lo, hi], in rate points, margin > 0):
      BETTER        lo > 0                    the arm is detectably better
      PASS          lo >= -margin             not worse than the margin (a detectable loss inside the margin still passes;
                                              the report shows the interval, the call on the speed gain is the owner's)
      REJECT        hi < -margin              detectably worse than the margin allows
      INCONCLUSIVE  otherwise                 the interval straddles the margin: more repetitions / tasks, not a retry
    Direction, not displacement: the arm is judged by what it does to the task score, never by how far its tokens moved."""
    if lo > 0:
        return "BETTER"
    if lo >= -margin:
        return "PASS"
    if hi < -margin:
        return "REJECT"
    return "INCONCLUSIVE"


def worst(*vs):
    return max(vs, key=ORDER.index)


def med(xs):
    xs = sorted(xs)
    return float("nan") if not xs else xs[len(xs) // 2] if len(xs) % 2 else (xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2


def time_curve(runs, horizon, budgets):
    """A18 / A57: the share of runs that succeeded within each budget (seconds), and the area under that curve to `horizon`:
    the mean over all runs of max(0, 1 - elapsed / horizon) for a success and 0 for a failure (so a slow success counts
    less than a fast one, a failure nothing)."""
    if not runs:
        return None
    within = {str(b): success_within(runs, b) for b in budgets}
    return {"within": within, "auc": time_auc(runs, horizon), "horizon": horizon}


def arm_stats(runs, curve=None):
    ok = [r for r in runs if r["ok"]]
    tc = [r for r in runs if r.get("valid") is not None]
    lo, hi = wilson(len(ok), len(runs))
    kinds = {}
    for k in KINDS:
        kr = [r for r in runs if r["kind"] == k]
        if kr:
            kinds[k] = {"n": len(kr), "rate": sum(1 for r in kr if r["ok"]) / len(kr)}
    return {"n": len(runs), "ok": len(ok), "rate": len(ok) / len(runs) if runs else float("nan"), "lo": lo, "hi": hi,
            "kinds": kinds, "valid_n": len(tc), "valid_rate": sum(1 for r in tc if r["valid"]) / len(tc) if tc else None,
            "t_success_median": med([r["elapsed"] for r in ok]),
            "s_per_success": sum(r["elapsed"] for r in runs) / len(ok) if ok else float("nan"),
            "tokens_per_run": sum(r.get("completion_tokens") or 0 for r in runs) / len(runs) if runs else float("nan"),
            "errors": sum(1 for r in runs if r.get("error")),
            **({"time_curve": time_curve(runs, *curve)} if curve and runs else {})}


def time_auc(runs, horizon):
    """A57: the area under the time-to-success curve, normalised to 0..1.  S(t) = the share of runs that succeeded within t seconds
    (a failed or errored run never counts); AUC = (1 / horizon) * integral of S over 0..horizon = the mean over ALL runs of
    ok * (horizon - elapsed) / horizon.  A fast success counts nearly 1, a success at the horizon 0, a failure 0, so one number
    holds both the success rate and how fast the successes came (SWE-Effi's effectiveness-under-a-budget idea).  The horizon must be
    the same for the arms that are compared (build_report: the longest run of all arms, or --auc-horizon).  Returns None for no runs."""
    if not runs or horizon <= 0:
        return None
    return sum(max(0.0, horizon - min(r["elapsed"], horizon)) for r in runs if r["ok"]) / (len(runs) * horizon)


def success_within(runs, t):
    return sum(1 for r in runs if r["ok"] and r["elapsed"] <= t) / len(runs) if runs else None


def task_auc_scores(runs, horizon):
    """{task: [ok * (horizon - elapsed) / horizon, ...]}, for boot_diff (a paired interval on the AUC difference)."""
    out = {}
    for r in runs:
        v = max(0.0, horizon - min(r["elapsed"], horizon)) / horizon if r["ok"] else 0.0
        out.setdefault(r["task"], []).append(v)
    return out


def build_report(arms, runs_by_arm, margin, vmargin, boots=2000, seed=0, resample_tasks=True, noisy=None, base_drift=None, auc_horizon=None,
                 curve=None, resample_reps=True):
    """arms: arm names, arms[0] = base.  runs_by_arm: {name: [run records]} (base pooled over its repeats; base_drift:
    (runs_a, runs_b) of the two base runs).  Returns a JSON-able dict."""
    base = arms[0]
    rep = {"arms": {}, "margin": margin, "validity_margin": vmargin, "resample_tasks": resample_tasks, "resample_reps": resample_reps, "warnings": []}
    for a in arms:
        rep["arms"][a] = {"stats": arm_stats(runs_by_arm[a], curve) if runs_by_arm.get(a) else None}
    # A57: area under the time-to-success curve, one horizon for every arm (information, never part of the verdict)
    horizon = auc_horizon or max((r["elapsed"] for rs in runs_by_arm.values() for r in rs or []), default=0.0)
    if horizon > 0:
        rep["auc_horizon_s"] = round(horizon, 2)
        for a in arms:
            if rep["arms"][a]["stats"]:
                rs = runs_by_arm[a]
                rep["arms"][a]["auc"] = {"auc": time_auc(rs, horizon),
                                         "within": {f"{f:g}": success_within(rs, f * horizon) for f in (0.25, 0.5, 0.75, 1.0)}}
    bs = rep["arms"][base]["stats"]
    if bs and bs["rate"] < 0.3:
        rep["warnings"].append(f"the base passes only {bs['rate'] * 100:.0f}% of its runs: a gate that fails everything cannot "
                               "show a loss; check the checkers (--dry-run) and the preset before reading the verdicts")
    if base_drift:
        r = boot_diff(task_rates(base_drift[1]), task_rates(base_drift[0]), boots, seed, resample_tasks, resample_reps)
        if r:
            rep["base_noise"] = {"diff": r[0], "lo": r[1], "hi": r[2]}
            if r[1] > 0 or r[2] < 0:
                rep["warnings"].append(f"the two base runs differ ({r[0] * 100:+.1f} pts [{r[1] * 100:+.1f}, {r[2] * 100:+.1f}]): "
                                       "the server or the machine is not stable; treat every verdict as provisional")
    for a in arms[1:]:
        e = rep["arms"][a]
        if not e["stats"] or not runs_by_arm.get(base):
            e["verdict"] = "not run"
            continue
        ra = boot_diff(task_rates(runs_by_arm[a]), task_rates(runs_by_arm[base]), boots, seed, resample_tasks, resample_reps)
        rv = boot_diff(task_rates(runs_by_arm[a], "valid"), task_rates(runs_by_arm[base], "valid"), boots, seed, resample_tasks, resample_reps)
        if not ra:
            e["verdict"] = "not run"
            continue
        e["success_diff"] = {"diff": ra[0], "lo": ra[1], "hi": ra[2], "tasks": ra[3]}
        if horizon > 0 and runs_by_arm.get(base):
            rauc = boot_diff(task_auc_scores(runs_by_arm[a], horizon), task_auc_scores(runs_by_arm[base], horizon), boots, seed, resample_tasks, resample_reps)
            if rauc:
                e["auc_diff"] = {"diff": rauc[0], "lo": rauc[1], "hi": rauc[2]}
        v = verdict(ra[1], ra[2], margin)
        e["success_verdict"] = v
        if rv:
            e["validity_diff"] = {"diff": rv[0], "lo": rv[1], "hi": rv[2]}
            e["validity_verdict"] = verdict(rv[1], rv[2], vmargin)
            v = worst(v, e["validity_verdict"] if e["validity_verdict"] != "BETTER" else "PASS")
        e["verdict"] = v
        ta, tb = task_rates(runs_by_arm[a]), task_rates(runs_by_arm[base])
        e["task_flips"] = sorted(({"task": t, "arm": sum(ta[t]) / len(ta[t]), "base": sum(tb[t]) / len(tb[t])}
                                  for t in ta if t in tb and abs(sum(ta[t]) / len(ta[t]) - sum(tb[t]) / len(tb[t])) >= 0.4),
                                 key=lambda x: x["arm"] - x["base"])
        if noisy and noisy.get(a):
            rep["warnings"].append(f"{a}: foreign CPU load during the run: its times are unreliable (success rates are not affected)")
    return rep


def format_report(rep):
    L = []
    pc = lambda x: "  n/a " if x is None or x != x else f"{x * 100:5.1f}%"
    L.append(f"{'arm':22s} {'runs':>5s} {'success (95% Wilson)':>24s} {'tool valid':>10s} {'t_success med':>13s} {'s/success':>9s}  vs base")
    for a, e in rep["arms"].items():
        s = e["stats"]
        if not s:
            L.append(f"{a:22s} no runs")
            continue
        line = (f"{a:22s} {s['n']:5d} {pc(s['rate'])} [{s['lo'] * 100:4.1f}, {s['hi'] * 100:5.1f}]"
                f" {pc(s['valid_rate']):>10s} {s['t_success_median']:12.1f}s {s['s_per_success']:8.1f}s")
        if "verdict" in e and "success_diff" in e:
            d, v = e["success_diff"], e.get("validity_diff")
            line += (f"  success {d['diff'] * 100:+5.1f} pts [{d['lo'] * 100:+5.1f}, {d['hi'] * 100:+5.1f}] {e['success_verdict']}")
            if v:
                line += f"; validity {v['diff'] * 100:+5.1f} [{v['lo'] * 100:+5.1f}, {v['hi'] * 100:+5.1f}] {e['validity_verdict']}"
            line += f"  => {e['verdict']}"
        elif "verdict" in e:
            line += f"  => {e['verdict']}"
        L.append(line)
        if e.get("auc") and e["auc"]["auc"] is not None:
            w = e["auc"]["within"]
            L.append(f"{'':22s} time-AUC {e['auc']['auc']:.3f} (horizon {rep['auc_horizon_s']:.0f} s; success within 25% / 50% / 75% / 100% of it: "
                     + " / ".join(pc(w[k]).strip() for k in ("0.25", "0.5", "0.75", "1")) + ")"
                     + (f"  vs base {e['auc_diff']['diff']:+.3f} [{e['auc_diff']['lo']:+.3f}, {e['auc_diff']['hi']:+.3f}]" if "auc_diff" in e else ""))
        L.append(f"{'':22s} by kind: " + "  ".join(f"{k} {v['rate'] * 100:.0f}% (n={v['n']})" for k, v in s["kinds"].items())
                 + (f"  request errors {s['errors']}" if s["errors"] else ""))
        if s.get("time_curve"):
            c = s["time_curve"]
            L.append(f"{'':22s} succeeded within: " + "  ".join(f"{b}s {v * 100:.0f}%" for b, v in c["within"].items())
                     + f"  | area under the curve to {c['horizon']:g} s: {c['auc']:.3f}  | tokens/run {s['tokens_per_run']:.0f}")
        for f in e.get("task_flips", []):
            L.append(f"{'':22s}   {f['task']:22s} arm {f['arm'] * 100:3.0f}%  base {f['base'] * 100:3.0f}%")
    if "base_noise" in rep:
        n = rep["base_noise"]
        L.append(f"base run 2 vs run 1: {n['diff'] * 100:+.1f} pts [{n['lo'] * 100:+.1f}, {n['hi'] * 100:+.1f}]")
    L.append(f"verdict: success non-inferior within {rep['margin'] * 100:.0f} pts, tool-call validity within "
             f"{rep['validity_margin'] * 100:.0f} pts; paired by task, bootstrap {'over repetitions only' if not rep['resample_tasks'] else 'over tasks and repetitions' if rep.get('resample_reps', True) else 'over tasks only'}")
    L += [f"WARNING: {w}" for w in rep["warnings"]]
    return "\n".join(L)


# ---- running
def messages_of(task, spec):
    if task.get("messages"):
        msgs = list(task["messages"])
    else:
        msgs = [{"role": "user", "content": task["prompt"]}]
    if task.get("system") and not (msgs and msgs[0]["role"] == "system"):
        msgs.insert(0, {"role": "system", "content": task["system"]})
    return msgs


def chat(url, task, spec, rep, timeout):
    """One request.  Returns a run record's response part; raises ConnectionError when the server is gone."""
    body = {"model": "x", "messages": messages_of(task, spec), "max_tokens": task.get("max_tokens", spec["max_tokens"]),
            "temperature": spec["temperature"], "seed": SEED0 + rep,
            "chat_template_kwargs": {"enable_thinking": bool(spec["thinking"])}}
    if spec.get("reasoning_effort"):                 # A18: the level this arm asks for (low, medium, high, xhigh ...)
        body["reasoning_effort"] = spec["reasoning_effort"]
    if task.get("tools"):
        body["tools"] = task["tools"]
        body["tool_choice"] = "auto"
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=timeout))
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read()[:200].decode('utf-8', 'replace')}", "elapsed": time.time() - t0, "message": {}}
    except (urllib.error.URLError, ConnectionError, OSError) as e:
        reason = getattr(e, "reason", e)
        if isinstance(reason, (ConnectionRefusedError, ConnectionResetError, BrokenPipeError)) or isinstance(e, ConnectionError):
            raise ConnectionError(str(reason)) from e
        return {"error": f"{type(reason).__name__}: {reason}"[:200], "elapsed": time.time() - t0, "message": {}}
    except ValueError as e:
        return {"error": f"bad JSON from the server: {e}", "elapsed": time.time() - t0, "message": {}}
    el = time.time() - t0
    try:
        ch = r["choices"][0]
        return {"message": ch["message"], "finish": ch.get("finish_reason"), "elapsed": el,
                "completion_tokens": (r.get("usage") or {}).get("completion_tokens")}
    except (KeyError, IndexError, TypeError):
        return {"error": "no choices in the response", "elapsed": el, "message": {}}


def run_tasks(url, tasks, spec, label, reps, timeout, sink, srv=None):
    """All tasks x reps against a ready server; one JSON line per run to `sink`.  Returns the records, or None when
    the server died."""
    recs = []
    for rep in range(reps):
        for t in tasks:
            try:
                r = chat(url, t, spec, rep, timeout)
            except ConnectionError as e:
                print(f"  {label}: server gone ({e})", flush=True)
                return None
            if srv is not None and srv.poll() is not None:
                print(f"  {label}: server process exited", flush=True)
                return None
            g = {"ok": False, "valid": False if t["kind"] == "tool_call" else None, "reason": r["error"]} if r.get("error") \
                else grade(t, r["message"])
            rec = {"arm": label, "rep": rep, "task": t["id"], "kind": t["kind"], "ok": g["ok"], "valid": g["valid"],
                   "reason": g["reason"], "elapsed": round(r["elapsed"], 3), "completion_tokens": r.get("completion_tokens"),
                   "finish": r.get("finish"), "error": r.get("error"),
                   "message": {k: v for k, v in (r.get("message") or {}).items() if k in ("content", "tool_calls")}}
            recs.append(rec)
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
        done = [r for r in recs if r["rep"] == rep]
        print(f"  {label} rep {rep + 1}/{reps}: {sum(r['ok'] for r in done)}/{len(done)} ok, "
              f"{sum(r['elapsed'] for r in done):.0f} s", flush=True)
    return recs


def load_runs(out):
    """{label: [records]} from DIR/runs-<label>.jsonl."""
    runs = {}
    for p in sorted(Path(out).glob("runs-*.jsonl")):
        runs[p.stem[len("runs-"):]] = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return runs


def arrange(arms, runs):
    """Maps run labels to arms: label "base#a" / "base#b" are the base repeats (pooled), other labels are arm names."""
    by = {}
    for a in arms:
        by[a] = list(runs.get(a, [])) if a != arms[0] else []
    by[arms[0]] = [r for lab, rs in runs.items() if lab == arms[0] or lab.startswith(arms[0] + "#") for r in rs]
    drift = (runs.get(arms[0] + "#a"), runs.get(arms[0] + "#b"))
    return by, drift if all(drift) else None


def regrade(runs, tasks):
    by_id = {t["id"]: t for t in tasks}
    for rs in runs.values():
        for r in rs:
            t = by_id.get(r["task"])
            if t:
                g = r["error"] and {"ok": False, "valid": False if t["kind"] == "tool_call" else None} or grade(t, r["message"])
                r["ok"], r["valid"] = g["ok"], g["valid"]


def conf(spec_path):
    spec = json.load(open(spec_path))
    s = {**DEFAULTS, **{k: spec[k] for k in DEFAULTS if k in spec}}
    return spec, s


def arm_settings(s, arm):
    """The run settings of one arm: the spec's, with the arm's "request" overrides on top (A18)."""
    return {**s, **{k: v for k, v in (arm.get("request") or {}).items() if k in REQUEST_KEYS}}


def curve_of(s):
    """(horizon, budgets) for build_report when the spec asks for the time curve, else None (the report stays as it was)."""
    return (float(s["time_horizon_s"]), list(s["time_budgets"])) if s.get("time_curve") else None


def label_name(arm, tag):
    return f"{arm}#{tag}" if tag else arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--tasks")
    ap.add_argument("--only", help="comma list of kinds and / or task ids")
    ap.add_argument("--reps", type=int)
    ap.add_argument("--dry-run", action="store_true", help="validate the spec, the task set and every checker; no server")
    ap.add_argument("--from-runs", help="a finished --out directory: print the report again (no server)")
    ap.add_argument("--regrade", action="store_true", help="with --from-runs: grade the stored answers again with the current checkers")
    ap.add_argument("--no-base-repeat", action="store_true", help="run the base arm once (no drift check)")
    ap.add_argument("--no-task-resample", action="store_true", help="interval from repetition noise only (tasks held fixed)")
    ap.add_argument("--tasks-only", action="store_true", help="interval from resampling the tasks alone (a task's rate already holds its repetition "
                    "noise; the default resamples the repetitions too, which counts it twice and is ~35%% wider)")
    ap.add_argument("--boots", type=int, default=2000)
    ap.add_argument("--auc-horizon", type=float, help="seconds: the time axis of the time-to-success AUC (A57); default the longest run of the batch")
    ap.add_argument("--timeout", type=int, default=600, help="seconds per request")
    ap.add_argument("--max-load", type=float, default=0.0)
    ap.add_argument("--foreign-cpu", type=float, default=25.0)
    ap.add_argument("--max-wait", type=int, default=600)
    ap.add_argument("--retries", type=int, default=1, help="repeats of an arm whose server died")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    if not a.spec and not a.from_runs:
        ap.error("a SPEC.json (or --from-runs DIR) is required")
    if a.spec:
        spec, s = conf(a.spec)
        if a.reps:
            s["reps"] = a.reps
    tpath = a.tasks or (spec.get("tasks") if a.spec else None) or DEFAULT_TASKS
    tpath = tpath if os.path.isabs(tpath) else str(ROOT / tpath)
    ts, tasks = load_tasks(tpath, a.only)

    if a.from_runs:
        runs = load_runs(a.from_runs)
        sj = Path(a.from_runs) / "spec.json"
        sp = json.load(open(sj)) if sj.exists() else (spec if a.spec else {})
        arms = list(sp["arms"])
        s = {**DEFAULTS, **{k: sp[k] for k in DEFAULTS if k in sp}}
        if a.regrade:
            regrade(runs, tasks)
        by, drift = arrange(arms, runs)
        print(format_report(build_report(arms, by, s["margin"], s["validity_margin"], a.boots, 0, not a.no_task_resample,
                                         base_drift=drift, auc_horizon=a.auc_horizon, curve=curve_of(s),
                                         resample_reps=not a.tasks_only)))
        return 0

    arms = list(spec["arms"])
    problems = []
    if not isinstance(spec.get("base"), str) or not arms:
        problems.append("spec needs \"base\" (preset json) and a non-empty \"arms\"")
    base = None
    if isinstance(spec.get("base"), str):
        base = spec["base"] if os.path.isabs(spec["base"]) else str(ROOT / spec["base"])
        if not Path(base).exists():
            problems.append(f"base preset {base} not found")
    for n, arm in spec["arms"].items():
        for k in ("env", "args", "request", "config"):
            if arm.get(k) is not None and not isinstance(arm[k], dict):
                problems.append(f"arm {n}: {k} must be an object")
        bad = sorted(set(arm.get("request") or {}) - set(REQUEST_KEYS)) if isinstance(arm.get("request"), dict) else []
        if bad:
            problems.append(f"arm {n}: request has unknown keys {bad} (allowed: {', '.join(REQUEST_KEYS)})")
    if s["time_curve"] and (not isinstance(s["time_horizon_s"], (int, float)) or s["time_horizon_s"] <= 0
                            or not isinstance(s["time_budgets"], list)
                            or not all(isinstance(b, (int, float)) and b > 0 for b in s["time_budgets"])):
        problems.append("time_curve: time_horizon_s must be > 0 and time_budgets a list of seconds > 0")
    if base and not problems:   # the arm configs themselves (env / args edits applied to the preset), in a temp dir
        with tempfile.TemporaryDirectory(prefix="taskgate-cfg-") as d:
            for n, arm in spec["arms"].items():
                try:
                    abtest.make_config(base, arm, str(Path(d) / f"{n}.json"))
                except Exception as e:  # noqa: BLE001 - report, do not crash
                    problems.append(f"arm {n}: config does not build: {type(e).__name__}: {e}")
    problems += validate_tasks(tasks)
    if not a.dry_run and problems:
        print("\n".join("ERROR: " + p for p in problems))
        return 2
    runs_per_arm = len(tasks) * s["reps"]
    n_servers = len(arms) + (0 if a.no_base_repeat else 1)
    if a.dry_run:
        probs = problems + (selftest(tasks) if not problems else [])
        kinds = {k: sum(1 for t in tasks if t["kind"] == k) for k in KINDS}
        print(f"spec: base {spec.get('base')}, arms {arms}, reps {s['reps']}, temperature {s['temperature']}, "
              f"margin {s['margin']}, validity margin {s['validity_margin']}")
        print(f"tasks: {len(tasks)} from {tpath} ({', '.join(f'{k} {n}' for k, n in kinds.items())})")
        for n, arm in spec["arms"].items():
            if arm.get("request") or arm.get("config"):
                print(f"arm {n}: request {arm.get('request') or {}}, server config {arm.get('config') or {}}")
        print(f"plan: {n_servers} server starts ({'base twice, ' if not a.no_base_repeat else ''}{len(arms) - 1} other arm(s)), "
              f"{runs_per_arm} requests per start, {runs_per_arm * n_servers} in all; order "
              + " -> ".join([label_name(arms[0], "a" if not a.no_base_repeat else "")] + arms[1:]
                            + ([label_name(arms[0], "b")] if not a.no_base_repeat else [])))
        if s["temperature"] == 0:
            print("note: temperature 0 gives identical repeats; the task-resampling interval still works, the repetitions add nothing")
        print(f"checkers: network namespace for model code {'available' if _netless_prefix() else 'not available here (limits only)'}")
        for p in probs:
            print("ERROR: " + p)
        print("dry run: " + ("FAILED" if probs else "ok (every checker passes its reference answer and fails its wrong one)"))
        return 1 if probs else 0

    out = Path(a.out or "")
    if not a.out:
        ap.error("--out is required")
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.json").write_text(json.dumps({**spec, "reps": s["reps"]}, indent=1))
    plan = ([(arms[0], "a")] if not a.no_base_repeat else [(arms[0], "")]) + [(x, "") for x in arms[1:]] \
        + ([(arms[0], "b")] if not a.no_base_repeat else [])
    abtest.stop_servers()
    noisy, deaths = {}, 0
    for arm, tag in plan:
        label = label_name(arm, tag)
        for attempt in range(a.retries + 1):
            abtest.quiet_gate(a.max_load, a.foreign_cpu, [], a.max_wait)
            cfg_path = str(out / f"{arm}.json")
            abtest.make_config(base, spec["arms"][arm], cfg_path)
            print(f"{label}: starting server (attempt {attempt + 1})", flush=True)
            srv, err = abtest.start_server(cfg_path, str(out / f"{label.replace('#', '-')}-a{attempt}.out"), a.port)
            if err:
                print(f"{label}: FAILED ({err})", flush=True)
                abtest.stop_servers()
                deaths += 1
                if deaths >= 2:
                    print("ABORT: the server died twice before ready (free RAM in 2 MiB blocks: "
                          f"{abtest.big_free_gib():.0f} GiB; 'sudo tools/pancake/os_tune.sh compact'). Partial report below.", flush=True)
                    plan = []
                    break
                continue
            url = f"http://127.0.0.1:{a.port}"
            sa = arm_settings(s, spec["arms"][arm])
            try:
                chat(url, {"id": "warm", "kind": "qa", "prompt": "Say hello."}, sa, 0, a.timeout)   # warm-up, not counted
            except ConnectionError:
                pass
            mon = abtest.NoiseMonitor([srv.pid])
            mon.start()
            with open(out / f"runs-{label}.jsonl", "w") as sink:
                recs = run_tasks(url, tasks, sa, label, s["reps"], a.timeout, sink, srv)
            noise, top = mon.result()
            abtest.stop_servers()
            if recs is None:
                print(f"{label}: server died during the run; discarding and {'retrying' if attempt < a.retries else 'giving up'}",
                      flush=True)
                (out / f"runs-{label}.jsonl").unlink(missing_ok=True)
                continue
            noisy[arm] = noisy.get(arm) or noise > a.foreign_cpu
            if noise > a.foreign_cpu:
                print(f"  {label}: foreign CPU {noise:.0f}% {top}: times unreliable", flush=True)
            break
        if not plan:
            break
    runs = load_runs(out)
    by, drift = arrange(arms, runs)
    rep = build_report(arms, by, s["margin"], s["validity_margin"], a.boots, 0, not a.no_task_resample, noisy, drift, a.auc_horizon,
                       curve_of(s), resample_reps=not a.tasks_only)
    text = format_report(rep)
    print("\n" + text)
    (out / "summary.json").write_text(json.dumps(rep, indent=1))
    (out / "REPORT.txt").write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
