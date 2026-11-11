"""spec_lint.py - static lint of the A/B specs in bench/specs/*.json against what the harness reads.

The spec format is defined by the harness: tools/pancake/abtest.py (quality_gate.py, task_gate.py and replay_turns.py read
the same file shape).  This tool reads that source, the specs, the configs they name and the engine / server source that
reads the switches.  It never starts the engine, a server or a model, and never imports the harness.

Per spec:
  - JSON parses, no duplicate key (JSON would silently keep the last one); "base" and "arms" are present;
  - every key is one the harness reads (top level, arm, "request", "replay"); values have the right type;
  - every case is one abtest.py builds a prompt for; every arm name is a file name (abtest writes files named after it);
  - "base" exists (a generated pancake-X.json that is not here is linted through configs/pancake/X.json);
    the base config has "args" and "env", its exe is an engine target, its flags and env names are known;
  - per arm: each env name is read by engine or serve source (a quoted name in ENV_READER_DIRS), each args flag is accepted
    by the parser of the arm's exe (FLAG_PARSERS), each config key is read by serve/ or strata_cli/ (CONFIG_READER_DIRS);
  - "tasks", the harness's default files and the replay turn log exist (the turn log is written by a run: a warning);
  - warnings only: an arm that changes nothing against the base config or against an earlier arm; a long case whose prompt
    does not fit the arm's --max-context; an exe that is not built here or lies outside this checkout.

Errors make the exit status 1; warnings do not.

    python3 tools/pancake/spec_lint.py                 every bench/specs/*.json
    python3 tools/pancake/spec_lint.py SPEC.json ...   the given specs
    -v                                                  also print the notes (a generated config that is not here)
"""
import argparse
import ast
import copy
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ERROR, WARN, INFO = "error", "warn", "info"

HARNESS_SPEC_FILES = ("tools/pancake/abtest.py", "tools/pancake/quality_gate.py", "tools/pancake/task_gate.py",
                      "tools/pancake/replay_turns.py")
SAMPLING_KEYS = ("temperature", "top_p", "seed")          # copied from the spec by abtest.main and replay_turns.main
DOC_KEYS = ("note", "_note", "lossy")                     # documentation only: no harness reads them
REPLAY_DOC_KEYS = ("note", "_note")
LONG_SLACK = 1.08                                         # abtest warns when a long prompt is more than 8% off its target
DEFAULT_CASES = ["short", "edit", "long"]                 # abtest.main's default

# Where the engine and the server read their switches: quoted names in these places.  Test files are not readers.
ENV_READER_DIRS = ("src", "include", "metal", "sycl", "serve", "strata_cli", "strata_control")
CONFIG_READER_DIRS = ("serve", "strata_cli", "strata_control")
CODE_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hh", ".hip", ".hpp", ".inl", ".mm", ".py"}
# The flag parser of each engine executable (its CMake target).  strata also runs the Qwen3.5/3.6 engine (src/qwen35).
FLAG_PARSERS = {"strata": ("src/program/generate.cpp", "src/qwen35"),
                "strata-q36": ("src/qwen35",),
                "strata-poly": ("src/poly",)}


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_scalar_or_none(v):
    return v is None or (isinstance(v, (str, int, float)) and not isinstance(v, bool))


def _module_literal(path, name):
    """The value of the module-level literal `name = ...` of a harness file (read with ast, never imported)."""
    tree = ast.parse(path.read_text(), str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"{path}: no module-level literal {name}")


def _prompt_cases(path):
    """The case names abtest.prompts() builds: the keys of the P table in its body."""
    tree = ast.parse(path.read_text(), str(path))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "prompts")
    for node in ast.walk(fn):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                and any(isinstance(t, ast.Name) and t.id == "P" for t in node.targets)):
            return {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    raise ValueError(f"{path}: prompts() has no P table")


def _read_keys(root, rels, var):
    """The keys the harness reads from a dict called `var`: var.get("k") and var["k"] in the source."""
    keys = set()
    for rel in rels:
        text = (root / rel).read_text()
        keys |= set(re.findall(rf'\b{var}(?:\.get)?\(\s*"(\w+)"', text))
        keys |= set(re.findall(rf'\b{var}\[\s*"(\w+)"\s*\]', text))
    return keys


def _quoted(root, places, pattern, suffixes):
    """Every quoted literal matching `pattern` in the code under `places` (directories or files), tests excluded."""
    names = set()
    for place in places:
        base = root / place
        files = [base] if base.is_file() else [p for p in base.rglob("*") if p.is_file()]
        for p in files:
            if p.suffix in suffixes and not p.name.startswith("test_") and not p.name.endswith("_test.cpp"):
                names |= set(re.findall(pattern, p.read_text(errors="replace")))
    return names


class Schema:
    """What the harness, the engine and the server accept, read from their source."""

    def __init__(self, root=ROOT):
        self.root = root
        ab = root / "tools/pancake/abtest.py"
        tg = root / "tools/pancake/task_gate.py"
        self.long_target = _module_literal(ab, "LONG_TARGET")
        self.maxtok = _module_literal(ab, "MAXTOK")
        self.cases = _prompt_cases(ab) | set(self.long_target)
        self.prompt_files = sorted(set(re.findall(r'ROOT / "(src/[\w./-]+)"', ab.read_text())))
        self.defaults = _module_literal(tg, "DEFAULTS")
        self.request_keys = set(_module_literal(tg, "REQUEST_KEYS"))
        self.default_tasks = _module_literal(tg, "DEFAULT_TASKS")
        self.default_quality_set = re.search(r'"--set",\s*default="([^"]+)"',
                                             (root / "tools/pancake/quality_gate.py").read_text()).group(1)
        self.top_keys = (_read_keys(root, HARNESS_SPEC_FILES, "spec") | set(self.defaults) | set(SAMPLING_KEYS)
                         | set(DOC_KEYS))
        self.arm_keys = _read_keys(root, ("tools/pancake/abtest.py", "tools/pancake/task_gate.py"), "arm")
        self.replay_keys = _read_keys(root, ("tools/pancake/replay_turns.py",), "rp") | set(REPLAY_DOC_KEYS)
        self.targets = set(re.findall(r"add_executable\(\s*([\w.+-]+)", (root / "CMakeLists.txt").read_text()))
        self._env = self._config = None
        self._flags = {}

    def env_known(self, name):
        if self._env is None:
            self._env = _quoted(self.root, ENV_READER_DIRS, r'["\']([A-Z][A-Z0-9_]*)["\']', CODE_SUFFIXES)
        return name in self._env

    def config_key_known(self, key):
        if self._config is None:
            self._config = _quoted(self.root, CONFIG_READER_DIRS, r'["\']([a-z][a-z0-9_]*)["\']', {".py"})
        return key in self._config

    def flags_of(self, exe):
        """The flags the engine parser for executable `exe` accepts, or None when no parser is known for it."""
        if exe not in FLAG_PARSERS:
            return None
        if exe not in self._flags:
            self._flags[exe] = _quoted(self.root, FLAG_PARSERS[exe], r'"(--[A-Za-z0-9][A-Za-z0-9_-]*)"', CODE_SUFFIXES)
        return self._flags[exe]


def merged(base, arm):
    """The config an arm runs: the base with the arm's edits, applied the way abtest.make_config applies them."""
    cfg = copy.deepcopy(base)
    env = cfg.setdefault("env", {})
    for k, v in (arm.get("env") or {}).items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = str(v)
    a = cfg.setdefault("args", [])
    for k, v in (arm.get("args") or {}).items():
        if k in a:
            i = a.index(k)
            n = 2 if i + 1 < len(a) and not str(a[i + 1]).startswith("--") else 1
            del a[i:i + n]
        if v is not None:
            a += [k] + ([str(v)] if v != "" else [])
    if arm.get("drafting"):
        cfg["drafting"] = arm["drafting"]
    for k, v in (arm.get("config") or {}).items():
        if v is None:
            cfg.pop(k, None)
        else:
            cfg[k] = v
    if arm.get("exe"):
        cfg["exe"] = arm["exe"]
    return cfg


def _flag_value(args, flag):
    """The integer value of `flag` in an args list, or None."""
    if flag not in args or args.index(flag) + 1 >= len(args):
        return None
    v = str(args[args.index(flag) + 1])
    return int(v) if v.isdigit() else None


def load_base(schema, rel, add):
    """The base config a spec runs: the file, or the configs/pancake template a generated pancake-*.json comes from."""
    p = Path(rel) if Path(rel).is_absolute() else schema.root / rel
    if not p.exists():
        m = re.fullmatch(r"pancake-(.+\.json)", Path(rel).name)
        tpl = schema.root / "configs/pancake" / m.group(1) if m else None
        if tpl is None or not tpl.exists():
            add(ERROR, "base", f"config {rel} not found")
            return None
        add(INFO, "base", f"{rel} is generated and not in this checkout; linting configs/pancake/{m.group(1)}")
        p = tpl
    try:
        cfg = json.loads(p.read_text())
    except ValueError as e:
        add(ERROR, "base", f"{p} is not valid JSON: {e}")
        return None
    if not isinstance(cfg.get("args"), list):
        add(ERROR, "base", f"{p.name}: \"args\" must be a list (abtest.make_config edits it for every arm)")
        return None
    return cfg


def check_base(schema, bc, add):
    """The engine switches of the base config: its exe, flags and env names."""
    exe = Path(str(bc.get("exe") or "")).name
    if not exe:
        add(WARN, "base", "the config has no exe: the flags of its arms are not checked")
        return
    if exe not in schema.targets:
        add(ERROR, "base", f"exe {bc['exe']!r}: {exe} is not an engine target in CMakeLists.txt")
    parser = schema.flags_of(exe)
    for tok in bc["args"]:
        if isinstance(tok, str) and tok.startswith("--") and parser is not None and tok not in parser:
            add(ERROR, "base", f"flag {tok} is not accepted by the {exe} parser (typo?)")
    for k in bc.get("env") or {}:
        if not schema.env_known(k):
            add(ERROR, "base", f"env {k} is not read by engine or serve source (typo?)")


def check_arm(schema, name, arm, base_exe, add):
    """One arm of a spec (or of replay's add_arms): its keys, names, values and switches."""
    where = f"arm {name}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", str(name)):
        add(ERROR, where, "the name becomes a file name: use letters, digits, '.', '_', '+' or '-'")
    if not isinstance(arm, dict):
        add(ERROR, where, "an arm is a JSON object")
        return
    for k in arm:
        if k not in schema.arm_keys:
            add(ERROR, where, f"unknown arm key {k!r} (the harness reads {sorted(schema.arm_keys)})")
    env = arm.get("env") or {}
    if not isinstance(env, dict):
        add(ERROR, where, '"env" must be an object')
        env = {}
    for k, v in env.items():
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", k):
            add(ERROR, where, f"env name {k!r} is not an ENV_NAME")
        elif not schema.env_known(k):
            add(ERROR, where, f"env {k} is not read by engine or serve source (typo?)")
        if not _is_scalar_or_none(v):
            add(ERROR, where, f"env {k}: the value must be a string, a number or null (removes the key)")
    args = arm.get("args") or {}
    if not isinstance(args, dict):
        add(ERROR, where, '"args" must be an object')
        args = {}
    exe = Path(arm["exe"]).name if arm.get("exe") else base_exe
    parser = schema.flags_of(exe)
    if args and parser is None:
        add(WARN, where, f"no flag parser is known for exe {exe}: its flags are not checked")
    for k, v in args.items():
        if not k.startswith("--"):
            add(ERROR, where, f"args key {k!r} is not a --flag")
        elif parser is not None and k not in parser:
            add(ERROR, where, f"flag {k} is not accepted by the {exe} parser (typo?)")
        if not _is_scalar_or_none(v):
            add(ERROR, where, f"flag {k}: the value must be a string, a number, null (removes it) or \"\" (a switch)")
    cfg_keys = arm.get("config") or {}
    if not isinstance(cfg_keys, dict):
        add(ERROR, where, '"config" must be an object')
        cfg_keys = {}
    for k in cfg_keys:
        if not schema.config_key_known(k):
            add(ERROR, where, f"config key {k!r} is not read by serve/ or strata_cli/ (typo?)")
    if "drafting" in arm and not isinstance(arm["drafting"], str):
        add(ERROR, where, '"drafting" must be a string')
    if "exe" in arm:
        check_exe(schema, where, arm["exe"], add)
    req = arm.get("request") or {}
    if not isinstance(req, dict):
        add(ERROR, where, '"request" must be an object')
        req = {}
    for k, v in req.items():
        if k not in schema.request_keys:
            add(ERROR, where, f"request key {k!r} is not one of {sorted(schema.request_keys)}")
        elif k == "thinking" and not isinstance(v, bool):
            add(ERROR, where, "request thinking must be true or false")
        elif k == "reasoning_effort" and not (isinstance(v, str) and v):
            add(ERROR, where, "request reasoning_effort must be a level name")
        elif k == "max_tokens" and not (_is_int(v) and v >= 1):
            add(ERROR, where, "request max_tokens must be an integer >= 1")
        elif k == "temperature" and not (_is_num(v) and v >= 0):
            add(ERROR, where, "request temperature must be a number >= 0")


def check_exe(schema, where, exe, add):
    """An arm's exe: an engine target, built here (a warning when not) and inside this checkout (a warning when not)."""
    if not isinstance(exe, str) or not exe:
        add(ERROR, where, '"exe" must be a path')
        return
    if Path(exe).name not in schema.targets:
        add(ERROR, where, f"exe {exe}: {Path(exe).name} is not an engine target in CMakeLists.txt")
    p = Path(exe) if Path(exe).is_absolute() else schema.root / exe
    if Path(exe).is_absolute() and not p.is_relative_to(schema.root):
        add(WARN, where, f"exe {exe} lies outside this checkout: it runs another build of the engine")
    if not p.exists():
        add(WARN, where, f"exe {exe} is not built here")


def _check_scalars(spec, add):
    """The types of the spec's own keys (the arms are checked separately)."""
    checks = {
        "reps": (lambda v: _is_int(v) and v >= 1, "an integer >= 1"),
        "prefill": (lambda v: isinstance(v, bool), "true or false"),
        "temperature": (lambda v: _is_num(v) and v >= 0, "a number >= 0"),
        "top_p": (lambda v: _is_num(v) and 0 < v <= 1, "a number in (0, 1]"),
        "seed": (_is_int, "an integer"),
        "margin": (lambda v: _is_num(v) and v >= 0, "a number >= 0"),
        "validity_margin": (lambda v: _is_num(v) and v >= 0, "a number >= 0"),
        "max_tokens": (lambda v: _is_int(v) and v >= 1, "an integer >= 1"),
        "thinking": (lambda v: isinstance(v, bool), "true or false"),
        "time_curve": (lambda v: isinstance(v, bool), "true or false"),
        "time_horizon_s": (lambda v: _is_num(v) and v > 0, "a number > 0"),
        "time_budgets": (lambda v: isinstance(v, list) and all(_is_num(b) and b > 0 for b in v),
                         "a list of numbers > 0"),
        "tasks": (lambda v: isinstance(v, str) and v, "a path"),
    }
    for k, (ok, want) in checks.items():
        if k in spec and not ok(spec[k]):
            add(ERROR, "spec", f"{k} must be {want}")


def _check_replay(schema, spec, arms, add):
    """The "replay" block (read by replay_turns.py only): its keys, its arms and its files.  Returns its add_arms."""
    rp = spec.get("replay")
    if rp is None:
        return {}
    if not isinstance(rp, dict):
        add(ERROR, "replay", "must be an object")
        return {}
    for k in rp:
        if k not in schema.replay_keys:
            add(ERROR, "replay", f"unknown replay key {k!r} (replay_turns.py reads {sorted(schema.replay_keys)})")
    add_arms = rp.get("add_arms") or {}
    if not isinstance(add_arms, dict):
        add(ERROR, "replay", '"add_arms" must be an object')
        add_arms = {}
    for n in add_arms:
        if n in arms:
            add(ERROR, "replay", f"add_arms {n!r} has the same name as a spec arm")
    replay_arms = rp.get("arms") or []
    if not isinstance(replay_arms, list):
        add(ERROR, "replay", '"arms" must be a list of arm names')
        replay_arms = []
    for n in replay_arms:
        if not isinstance(n, str) or (n not in arms and n not in add_arms):
            add(ERROR, "replay", f"replay arm {n!r} is neither a spec arm nor an add_arms arm")
    if "reps" in rp and not (_is_int(rp["reps"]) and rp["reps"] >= 1):
        add(ERROR, "replay", "reps must be an integer >= 1")
    if "shared_system" in rp and not (_is_int(rp["shared_system"]) and rp["shared_system"] >= 0):
        add(ERROR, "replay", "shared_system must be an integer >= 0")
    if "session" in rp and not isinstance(rp["session"], str):
        add(ERROR, "replay", "session must be a string (a window, sN or a request id)")
    if "turns" in rp:
        if not isinstance(rp["turns"], str):
            add(ERROR, "replay", "turns must be a path")
        elif not (schema.root / rp["turns"]).exists():
            add(WARN, "replay", f"turn log {rp['turns']} is not here (a run writes it; replay needs it)")
    return add_arms


def lint_spec(path, schema):
    """Problems of one spec file, as (level, where, message) tuples."""
    out = []

    def add(level, where, msg):
        out.append((level, where, msg))

    dups = []

    def pairs(items):
        d = {}
        for k, v in items:
            if k in d:
                dups.append(k)
            d[k] = v
        return d

    try:
        spec = json.loads(Path(path).read_text(), object_pairs_hook=pairs)
    except (OSError, ValueError) as e:
        return [(ERROR, "file", f"not readable JSON: {e}")]
    for k in dups:
        add(ERROR, "file", f"duplicate key {k!r}: JSON keeps only the last one")
    if not isinstance(spec, dict):
        add(ERROR, "file", "the top level must be a JSON object")
        return out
    for k in spec:
        if k not in schema.top_keys:
            add(ERROR, "spec", f"unknown top-level key {k!r}: no harness reads it (typo?)")
    _check_scalars(spec, add)
    base_rel = spec.get("base")
    if not isinstance(base_rel, str) or not base_rel:
        add(ERROR, "spec", '"base" is required: the config path')
    arms = spec.get("arms")
    if not isinstance(arms, dict) or not arms:
        add(ERROR, "spec", '"arms" is required: a non-empty object')
        return out
    cases = spec.get("cases", DEFAULT_CASES)
    if not isinstance(cases, list) or not cases:
        add(ERROR, "spec", '"cases" must be a non-empty list')
        cases = []
    for c in cases:
        if not isinstance(c, str) or c not in schema.cases:
            add(ERROR, "spec", f"unknown case {c!r} (abtest builds {sorted(schema.cases)})")
    cases = [c for c in cases if isinstance(c, str)]
    if len(set(cases)) != len(cases):
        add(WARN, "spec", "a case is listed twice: it is run twice per arm")
    if any(isinstance(a, dict) and "request" in a for a in arms.values()) and "tasks" not in spec:
        if not (schema.root / schema.default_tasks).exists():
            add(ERROR, "spec", f"the default task set {schema.default_tasks} is missing")
    elif isinstance(spec.get("tasks"), str) and not (schema.root / spec["tasks"]).exists():
        add(ERROR, "spec", f"task set {spec['tasks']} is missing")
    add_arms = _check_replay(schema, spec, arms, add)
    bc = load_base(schema, base_rel, add) if isinstance(base_rel, str) and base_rel else None
    base_exe = ""
    if bc is not None:
        check_base(schema, bc, add)
        base_exe = Path(str(bc.get("exe") or "")).name
    every_arm = {**arms, **add_arms}
    for name, arm in arms.items():
        check_arm(schema, name, arm, base_exe, add)
    for name, arm in add_arms.items():
        check_arm(schema, name, arm, base_exe, add)
    if bc is not None:
        if "env" not in bc and any(isinstance(a, dict) and a.get("env") for a in every_arm.values()):
            add(ERROR, "base", 'the base config has no "env" object, and an arm sets env: abtest.make_config fails '
                               "(KeyError 'env')")
        check_effects(schema, bc, every_arm, cases, add)
    return out


def check_effects(schema, bc, every_arm, cases, add):
    """Arms that change nothing (against the base or an earlier arm), and long cases that do not fit --max-context."""
    base_key = (json.dumps(merged(bc, {}), sort_keys=True), "{}")
    seen = {}
    for name, arm in every_arm.items():
        if not isinstance(arm, dict):
            continue
        try:
            cfg = merged(bc, arm)
        except (TypeError, AttributeError):
            continue
        key = (json.dumps(cfg, sort_keys=True), json.dumps(arm.get("request") or {}, sort_keys=True))
        if key == base_key and arm:
            add(WARN, f"arm {name}", "changes nothing against the base config")
        elif key in seen:
            add(WARN, f"arm {name}", f"changes nothing against arm {seen[key]} (the same config)")
        else:
            seen[key] = name
        mc = _flag_value(cfg["args"], "--max-context")
        for c in cases:
            if c in schema.long_target and mc is not None:
                need = schema.long_target[c] * LONG_SLACK + schema.maxtok.get(c, 512)
                if need > mc:
                    add(WARN, f"arm {name}", f"case {c} needs about {need:.0f} tokens of context; --max-context is {mc}")


def lint_harness(schema):
    """Problems of the harness's own references: its prompt sources, default files and engine targets."""
    out = []
    for rel in schema.prompt_files + [schema.default_tasks, schema.default_quality_set]:
        if not (schema.root / rel).exists():
            out.append((ERROR, "harness", f"{rel} is referenced by the harness and missing"))
    if not schema.targets:
        out.append((ERROR, "harness", "no add_executable target found in CMakeLists.txt"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Static lint of bench/specs/*.json against the harness, its configs and the "
                                             "engine's switches. Reads only; no engine, server or model is started.")
    ap.add_argument("specs", nargs="*", type=Path, help="spec files (default: bench/specs/*.json)")
    ap.add_argument("-v", "--verbose", action="store_true", help="also print the notes")
    a = ap.parse_args(argv)
    schema = Schema(ROOT)
    specs = a.specs or sorted((ROOT / "bench/specs").glob("*.json"))
    n_err = n_warn = 0
    harness = lint_harness(schema)
    for level, where, msg in harness:
        print(f"{level:5s} {where}: {msg}")
        n_err += level == ERROR
    for p in specs:
        items = lint_spec(p, schema)
        shown = [i for i in items if i[0] != INFO or a.verbose]
        try:
            rel = p.resolve().relative_to(ROOT)
        except ValueError:
            rel = p
        if not shown:
            print(f"{rel}: ok")
            continue
        print(f"{rel}:")
        for level, where, msg in shown:
            print(f"  {level:5s} {where}: {msg}")
        n_err += sum(1 for i in items if i[0] == ERROR)
        n_warn += sum(1 for i in items if i[0] == WARN)
    print(f"{len(specs)} specs: {n_err} errors, {n_warn} warnings")
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
