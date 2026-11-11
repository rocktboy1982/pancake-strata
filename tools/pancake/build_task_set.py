#!/usr/bin/env python3
"""build_task_set.py - verifiable coding tasks from this repository's own history and tests, for tools/pancake/task_gate.py.

Why: the 18-task default set decides nothing (MOD_PLAN "Test-protocol findings": it passes an identical arm ~3% of the time and rejects a
true 10-point loss ~3%); the gate needs ~100 tasks, and every checker must run on the CPU with no model.

A task is a pure top-level Python function of this repo (stdlib only) that the repo's own tests call with literal arguments.  Its checker is
a characterisation test: the expected results are what the function at the pinned commit returns for those calls, recorded in the sandbox
task_gate uses (a case whose result is not a literal, or differs between two runs, is dropped).  Three families, three broken starting points:
  history    the function as it was before its most recent change(s) in `git log`; the commit subjects are the instruction;
  bugfix     the function with one mutation (flipped comparison, off-by-one, swapped operator, negated condition ...) that the cases detect;
  implement  signature and docstring only, the body removed; the cases show the behaviour.
Every task states the module-level definitions the function needs (expect.setup, run before the answer) and shows 1-3 of the cases; the others
stay hidden in the checker.  A candidate survives only if its checker passes the reference answer and fails the broken one (task_gate.selftest,
the real checker in the real sandbox).

    python tools/pancake/build_task_set.py --count 120 --out bench/tasks/repo-history.json [--seed 1] [--rev HEAD] [--jobs 4]
                                           [--families history,bugfix,implement] [--per-file 8] [--max-history 40]
    python tools/pancake/build_task_set.py --check bench/tasks/repo-history.json     self-test an existing set; with git also that every reference is
                                           the function the pinned commit holds and that the checker passes on that commit's whole module

--count N: emit at most N validated tasks, ordered so that every prefix is a mix of the families (task_gate.py --limit N).  The funnel
(functions seen -> with tests -> context ok -> recorded -> built -> validated -> emitted) is printed and stored in the file.
Run it under a memory cap (systemd-run --user --scope -p MemoryMax=6G nice -n 19 ...): every candidate starts a few short Python processes.
"""
import argparse
import ast
import builtins
import collections
import concurrent.futures
import copy
import difflib
import functools
import json
import os
import random
import subprocess
import symtable
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task_gate as TG  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
STDLIB = sys.stdlib_module_names
BUILTINS = set(dir(builtins))
FAMILIES = ("history", "bugfix", "implement")
SKIP_DIRS = ("bench/results/", "apps/", "metal/", "packaging/", "docs/", "node_modules/", ".claude/")
MAX_CASES, MAX_FN_LINES, MIN_FN_LINES, MAX_SETUP_CHARS, MAX_PROMPT_CHARS, MAX_VALUE_CHARS = 12, 45, 4, 3000, 6500, 300
BAD_NAMES = {"open", "input", "exec", "eval", "__import__", "breakpoint", "subprocess", "shutil", "socket", "urllib", "http", "requests",
             "ctypes", "multiprocessing", "threading", "signal", "sqlite3", "tempfile", "asyncio"}
BAD_ATTRS = {"system", "popen", "remove", "unlink", "rmdir", "rmtree", "rename", "kill", "killpg", "fork", "execv", "execvp", "_exit",
             "makedirs", "mkdir", "chmod", "chown", "symlink", "write_text", "write_bytes", "touch", "exit", "stdin", "sleep", "connect",
             "urlopen", "open"}


# ------------------------------------------------------------------------------------------------ git
def git(root, *args):
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def is_test_path(p):
    b = os.path.basename(p)
    return b.startswith("test_") or b.endswith("_test.py") or p.startswith("tests/") or "/tests/" in p


# ------------------------------------------------------------------------------------------------ the module view
def referenced_globals(src):
    """Names `src` (one statement or a whole file) uses without binding them: what its context must define (builtins included)."""
    out = set()

    def walk(t):
        for s in t.get_symbols():
            n = s.get_name()
            if n.startswith(".") or not s.is_referenced():
                continue
            if t.get_type() == "module":
                if not (s.is_assigned() or s.is_imported() or s.is_namespace()):
                    out.add(n)
            elif s.is_global():
                out.add(n)
        for c in t.get_children():
            walk(c)
    walk(symtable.symtable(src, "<task>", "exec"))
    return out


def impurity(node):
    """Why `node` is not a pure, side-effect-free definition (a short phrase), or None."""
    for n in ast.walk(node):
        if isinstance(n, (ast.Global, ast.Nonlocal, ast.Yield, ast.YieldFrom, ast.Await, ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith)):
            return type(n).__name__.lower()
        if isinstance(n, ast.Name) and n.id in BAD_NAMES:
            return "uses " + n.id
        if isinstance(n, ast.Attribute) and n.attr in BAD_ATTRS:
            return "uses ." + n.attr
    return None


def bound_names(st):
    """Names a top-level statement binds, or None when the statement is not a plain definition we can copy into a context."""
    if isinstance(st, (ast.FunctionDef, ast.ClassDef)):
        return [st.name]
    if isinstance(st, ast.Import):
        return [a.asname or a.name.split(".")[0] for a in st.names]
    if isinstance(st, ast.ImportFrom):
        return [a.asname or a.name for a in st.names]
    if isinstance(st, ast.Assign):
        return [n.id for t in st.targets for n in ast.walk(t) if isinstance(n, ast.Name)]
    if isinstance(st, ast.AnnAssign) and st.value is not None and isinstance(st.target, ast.Name):
        return [st.target.id]
    return None


class Module:
    """One parsed source file: its top-level functions and an index name -> the statement that defines it (None: bound by code we do not copy)."""

    def __init__(self, path, src):
        self.path, self.src, self.tree = path, src, ast.parse(src)
        self.funcs, self.index = {}, {}
        for st in self.tree.body:
            names = bound_names(st)
            if names is None:     # if / try / with / for / expression: whatever it binds is unusable as context
                names = [n.id for x in ast.walk(st) for n in [x] if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)]
                names += [a.asname or a.name.split(".")[0] for x in ast.walk(st) if isinstance(x, (ast.Import, ast.ImportFrom)) for a in x.names]
                st = None
            for n in names:
                self.index[n] = st
            if isinstance(st, ast.FunctionDef):
                self.funcs[st.name] = st


def unsupported(st):
    """Why a context statement cannot be used (non-stdlib import, side effects), or None."""
    if isinstance(st, ast.Import):
        bad = [a.name for a in st.names if a.name.split(".")[0] not in STDLIB]
        return f"imports {bad[0]}" if bad else None
    if isinstance(st, ast.ImportFrom):
        return None if st.level == 0 and (st.module or "").split(".")[0] in STDLIB else f"imports from {'.' * st.level}{st.module}"
    return impurity(st)


def context_for(mod, fn_src, name):
    """The module-level definitions `fn_src` needs (imports, constants, helpers, classes; transitively), as source -> (text, None), or
    (None, reason) when one is unresolved, non-stdlib or impure, or the context is too big."""
    todo, seen, stmts = list(referenced_globals(fn_src) - {name}), set(), {}
    while todo:
        n = todo.pop()
        if n in seen:
            continue
        seen.add(n)
        if n not in mod.index:
            if n in BUILTINS:
                continue
            return None, f"unresolved name {n}"
        st = mod.index[n]
        if st is None:
            return None, f"{n} is bound by module-level code that is not a plain definition"
        if id(st) in stmts:
            continue
        why = unsupported(st)
        if why:
            return None, f"{n}: {why}"
        stmts[id(st)] = st
        todo.extend(referenced_globals(ast.unparse(st)) - {n})
        if len(stmts) > 12:
            return None, "context needs more than 12 definitions"
    text = "\n".join(ast.unparse(s) for s in sorted(stmts.values(), key=lambda s: s.lineno))
    return (text, None) if len(text) <= MAX_SETUP_CHARS else (None, f"context is {len(text)} characters")


def eligible(node):
    """A top-level function worth a task: undecorated, not a generator, a few lines long, not impure."""
    if node.decorator_list or node.name.startswith("__"):
        return False
    n = len(ast.unparse(node).splitlines())
    return MIN_FN_LINES <= n <= MAX_FN_LINES and impurity(node) is None


# ------------------------------------------------------------------------------------------------ the cases (the repo's tests)
def literal_call(node):
    """A Call whose arguments are all literals -> {"args": [source], "kwargs": {name: source}}, else None."""
    if any(isinstance(a, ast.Starred) for a in node.args) or any(k.arg is None for k in node.keywords):
        return None
    try:
        for a in node.args:
            ast.literal_eval(a)
        for k in node.keywords:
            ast.literal_eval(k.value)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None
    c = {"args": [ast.unparse(a) for a in node.args], "kwargs": {k.arg: ast.unparse(k.value) for k in node.keywords}}
    return c if len(json.dumps(c)) <= 2 * MAX_VALUE_CHARS else None


def resolve_module(dotted, test_path, by_stem):
    """The repo file a test's `import dotted` means: the one whose path ends with the dotted name, else the only one next to the test."""
    cands = by_stem.get(dotted.split(".")[-1], [])
    suffix = "/" + dotted.replace(".", "/") + ".py"
    hit = [c for c in cands if ("/" + c).endswith(suffix)]
    if len(hit) == 1:
        return hit[0]
    near = [c for c in cands if os.path.dirname(c) == os.path.dirname(test_path)]
    return near[0] if len(near) == 1 else None


def mine_calls(test_path, test_src, by_stem):
    """{(file, function): [case, ...]} for every literal call of a repo function in one test file."""
    try:
        tree = ast.parse(test_src)
    except SyntaxError:
        return {}
    mods, names = {}, {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                f = resolve_module(a.name, test_path, by_stem) if (a.asname or "." not in a.name) else None
                if f:
                    mods[a.asname or a.name] = f
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            f = resolve_module(n.module, test_path, by_stem)
            for a in n.names:
                if f:
                    names[a.asname or a.name] = (f, a.name)
                sub = resolve_module(n.module + "." + a.name, test_path, by_stem)
                if sub:
                    mods[a.asname or a.name] = sub
    out = collections.defaultdict(list)
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        fn = n.func
        if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id in mods:
            key = (mods[fn.value.id], fn.attr)
        elif isinstance(fn, ast.Name) and fn.id in names:
            key = names[fn.id]
        else:
            continue
        c = literal_call(n)
        if c and c not in out[key]:
            out[key].append(c)
    return out


# ------------------------------------------------------------------------------------------------ recording and the checker
HARNESS = """import ast, builtins, json
@@SETUP@@

@@FUNC@@

_out = []
for _a, _k in @@CASES@@:
    _args, _kw = ast.literal_eval(_a), ast.literal_eval(_k)
    try:
        _r = @@NAME@@(*_args, **_kw)
    except Exception as _e:
        _out.append(["raises", type(_e).__name__] if getattr(builtins, type(_e).__name__, None) is type(_e) else None)
    else:
        _s = repr(_r)
        try:
            _ok = ast.literal_eval(_s) == _r
        except Exception:
            _ok = False
        _out.append(["ok", _s] if _ok and len(_s) <= @@MAXLEN@@ else None)
print("\\n" + json.dumps(_out))
"""

CHECK_HEAD = '''def _same(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return a == b or abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b
'''


def case_source(c):
    return "[" + ", ".join(c["args"]) + "]", "{" + ", ".join(f"{k!r}: {v}" for k, v in c["kwargs"].items()) + "}"


def record(setup, fn_src, name, cases, timeout=10):
    """What `fn_src` (with `setup` defined first) returns for each case: [["ok", repr] | ["raises", ExcName] | None, ...] (None: not a
    literal / not a builtin exception), or None when the run failed or timed out."""
    code = (HARNESS.replace("@@CASES@@", repr([case_source(c) for c in cases])).replace("@@NAME@@", name)
            .replace("@@MAXLEN@@", str(MAX_VALUE_CHARS)).replace("@@SETUP@@", setup).replace("@@FUNC@@", fn_src))
    rc, out = TG.run_python(code, timeout, tail=10_000_000)
    if rc != 0 or not out.strip():
        return None
    try:
        res = json.loads(out.strip().splitlines()[-1])
    except ValueError:
        return None
    return res if isinstance(res, list) and len(res) == len(cases) else None


def stable_cases(setup, fn_src, name, cases):
    """The cases whose result is a literal and the same in two separate processes (hash randomisation, clocks, ...): [(case, record)]."""
    cases = cases[:MAX_CASES * 2]
    a, b = record(setup, fn_src, name, cases), record(setup, fn_src, name, cases)
    if a is None or b is None:
        return []
    return [(c, x) for c, x, y in zip(cases, a, b) if x is not None and x == y][:MAX_CASES]


def build_test(name, kept):
    """The checker: every kept case must give the recorded result.  kept: [(case, record)]."""
    lines = []
    for c, rec in kept:
        a, k = case_source(c)
        lines.append(f"    ({a}, {k}, {rec[0]!r}, {rec[1] if rec[0] == 'ok' else repr(rec[1])}),")
    return (CHECK_HEAD + "\n_CASES = [\n" + "\n".join(lines) + "\n]\nfor _i, (_a, _k, _kind, _want) in enumerate(_CASES):\n"
            "    try:\n        _got = " + name + "(*_a, **_k)\n    except Exception as _e:\n"
            "        assert _kind == 'raises' and type(_e).__name__ == _want, f\"case {_i}: " + name + " raised {type(_e).__name__}: {_e}\"\n"
            "    else:\n        assert _kind == 'ok' and _same(_got, _want), f\"case {_i}: " + name + " returned {_got!r}, expected {_want!r}\"\n")


def failing(kept, rec_broken):
    """Indexes of the kept cases the broken version gets wrong (a wrong result, an exception, a non-literal)."""
    if rec_broken is None:
        return []
    return [i for i, ((_, want), got) in enumerate(zip(kept, rec_broken)) if got != want]


# ------------------------------------------------------------------------------------------------ mutations (bugfix)
SWAP_CMP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
            ast.In: ast.NotIn, ast.NotIn: ast.In, ast.Is: ast.IsNot, ast.IsNot: ast.Is}
SWAP_BIN = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult, ast.FloorDiv: ast.Div, ast.Mod: ast.FloorDiv}
SWAP_NAME = {"min": "max", "max": "min", "any": "all", "all": "any", "startswith": "endswith", "endswith": "startswith", "lstrip": "rstrip",
             "rstrip": "lstrip", "lower": "upper", "upper": "lower", "floor": "ceil", "ceil": "floor", "split": "rsplit", "rsplit": "split",
             "find": "rfind", "rfind": "find"}


class Mutator(ast.NodeTransformer):
    """Applies the `target`-th mutation site (counted in visiting order); target -1 only counts: self.n + 1 sites."""

    def __init__(self, target=-1):
        self.target, self.n, self.label = target, -1, None

    def hit(self, label):
        self.n += 1
        if self.n == self.target:
            self.label = label
            return True
        return False

    def visit_Compare(self, node):
        self.generic_visit(node)
        new = SWAP_CMP.get(type(node.ops[0]))
        if new and self.hit(f"comparison {type(node.ops[0]).__name__} -> {new.__name__}"):
            node.ops[0] = new()
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        new = SWAP_BIN.get(type(node.op))
        if new and self.hit(f"operator {type(node.op).__name__} -> {new.__name__}"):
            node.op = new()
        return node

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        new = {ast.Add: ast.Sub, ast.Sub: ast.Add}.get(type(node.op))
        if new and self.hit(f"augmented assignment {type(node.op).__name__} -> {new.__name__}"):
            node.op = new()
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if self.hit("and <-> or"):
            node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
        return node

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Not) and self.hit("dropped `not`"):
            return node.operand
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        if self.hit("negated condition"):
            t = node.test
            node.test = t.operand if isinstance(t, ast.UnaryOp) and isinstance(t.op, ast.Not) else ast.UnaryOp(op=ast.Not(), operand=t)
        return node

    visit_While = visit_If

    def visit_Constant(self, node):
        if isinstance(node.value, bool):
            if self.hit("flipped boolean"):
                return ast.copy_location(ast.Constant(value=not node.value), node)
        elif isinstance(node.value, int) and self.hit("off-by-one constant"):
            return ast.copy_location(ast.Constant(value=node.value + 1), node)
        return node

    def visit_Break(self, node):
        return ast.copy_location(ast.Continue(), node) if self.hit("break -> continue") else node

    def visit_Continue(self, node):
        return ast.copy_location(ast.Break(), node) if self.hit("continue -> break") else node

    def visit_Call(self, node):
        self.generic_visit(node)
        f = node.func
        old = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
        if old in SWAP_NAME and self.hit(f"call {old} -> {SWAP_NAME[old]}"):
            if isinstance(f, ast.Name):
                f.id = SWAP_NAME[old]
            else:
                f.attr = SWAP_NAME[old]
        return node


def mutation_sites(fn):
    m = Mutator()
    m.visit(copy.deepcopy(fn))
    return m.n + 1


def mutate(fn, site):
    """(label, mutated copy of fn) for mutation site `site`."""
    m = Mutator(site)
    new = m.visit(copy.deepcopy(fn))
    return m.label, ast.fix_missing_locations(new)


# ------------------------------------------------------------------------------------------------ prompts
def fenced(src):
    return "```python\n" + src + "\n```"


def call_text(name, c):
    return f"{name}(" + ", ".join(c["args"] + [f"{k}={v}" for k, v in c["kwargs"].items()]) + ")"


def should(rec):
    return ("return " if rec[0] == "ok" else "raise ") + rec[1]


def does(rec):
    return ("returns " if rec[0] == "ok" else "raises ") + rec[1]


def examples_text(name, kept, picks, rec_broken=None):
    """One line per picked case: the call, what it should do and, for a broken version, what that version does instead."""
    lines = []
    for i in picks:
        c, rec = kept[i]
        line = f"- `{call_text(name, c)}` should {should(rec)}"
        if rec_broken is not None and rec_broken[i] is not None:
            line += f" (the code above {does(rec_broken[i])})"
        lines.append(line)
    return "\n".join(lines)


def context_block(setup):
    return f"These definitions already exist next to the function; do not repeat them:\n\n{fenced(setup)}\n\n" if setup else ""


def prompt_of(family, name, path, setup, shown_src, examples, subjects=()):
    tail = f"Answer with the complete `{name}` function in one python code block, no tests, no explanation needed."
    ctx = context_block(setup)
    if family == "history":
        what = "\n".join(f"- {s}" for s in subjects)
        return (f"`{name}` in `{path}` was changed in this repository's history. The commit(s), newest first:\n{what}\n\n"
                f"Here is the function before the change(s):\n\n{fenced(shown_src)}\n\n{ctx}"
                f"Bring it up to date so it behaves as the project's tests expect. Some of the expected results:\n{examples}\n\n{tail}")
    if family == "bugfix":
        found = f"Some calls that give the wrong result:\n{examples}" if examples else "The docstring says what it should do."
        return f"`{name}` in `{path}` has a bug.\n\n{fenced(shown_src)}\n\n{ctx}{found}\n\nFix the bug without changing the signature. {tail}"
    return (f"Implement `{name}` from `{path}`; the docstring says what it does.\n\n{fenced(shown_src)}\n\n{ctx}Some expected results:\n{examples}\n\n"
            f"{tail}")


def slug(path, name):
    return "-".join(Path(path).with_suffix("").parts[-2:]).replace("_", "-") + "-" + name.replace("_", "-").strip("-")


def make_task(family, tid, name, path, setup, test, reference, wrong, prompt, meta):
    expect = {"test": test, "timeout": 10}
    if setup:
        expect["setup"] = setup
    return {"id": tid, "kind": "code_edit", "prompt": prompt, "expect": expect, "reference": {"content": fenced(reference)},
            "wrong": {"content": fenced(wrong)}, "meta": {"family": family, "path": path, "function": name, **meta}}


# ------------------------------------------------------------------------------------------------ one function -> tasks
@functools.lru_cache(maxsize=512)
def defs_at(root, commit, path):
    """{name: normalised source} of the top-level functions of `path` at `commit` ({} when it does not parse or exist)."""
    try:
        return {st.name: ast.unparse(st) for st in ast.parse(git(root, "show", f"{commit}:{path}") or "").body if isinstance(st, ast.FunctionDef)}
    except (SyntaxError, ValueError, RecursionError):
        return {}


@functools.lru_cache(maxsize=1024)
def file_log(root, rev, path, limit):
    return [l.split("\t", 1) for l in (git(root, "log", "--no-merges", "--format=%H%x09%s", f"-n{limit}", rev, "--", path) or "").splitlines()]


def function_at(src, name):
    try:
        for st in ast.parse(src).body:
            if isinstance(st, ast.FunctionDef) and st.name == name:
                return st
    except SyntaxError:
        pass
    return None


def history_versions(root, rev, path, name, limit):
    """Older distinct versions of `name` in `path`, newest first: [(source, [subjects of the commits since, newest first])]."""
    out, subjects, prev = [], [], None
    for commit, subject in file_log(str(root), rev, path, limit):
        src = defs_at(str(root), commit, path).get(name)
        if src is None:
            break
        if prev is not None and src != prev[0]:
            subjects.append(prev[1])         # the newer commit changed the function from `src` to the version after it
            out.append((src, list(subjects)))
        prev = (src, subject)
    return out


def diff_lines(a, b):
    return sum(1 for x in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0) if x[:1] in "+-" and x[:3] not in ("+++", "---"))


def tasks_for(mod, name, calls, a, rev, seed):
    """All candidate tasks of one function; also returns its funnel counts."""
    fn = mod.funcs[name]
    fn_src = ast.unparse(fn)
    cnt = collections.Counter(functions=1)
    if not eligible(fn) or len(calls) < 2:
        return [], cnt
    cnt["with_tests"] += 1
    setup, why = context_for(mod, fn_src, name)
    if setup is None:
        return [], cnt
    cnt["context_ok"] += 1
    kept = stable_cases(setup, fn_src, name, calls)
    if len(kept) < 2:
        return [], cnt
    cnt["recorded"] += 1
    rng = random.Random(f"{seed}:{mod.path}:{name}")
    test = build_test(name, kept)
    base = slug(mod.path, name)
    out = []

    def add(task):
        if len(task["prompt"]) <= MAX_PROMPT_CHARS:
            out.append(task)

    def fail_picks(bad, n):
        picks = list(bad)
        rng.shuffle(picks)
        return sorted(picks[:n])

    if "history" in a.families:
        for old_src, subjects in history_versions(a.root, rev, mod.path, name, a.max_history)[:3]:
            if old_src == fn_src or len(old_src.splitlines()) > MAX_FN_LINES or diff_lines(old_src, fn_src) > 30:
                continue
            setup_h, _ = context_for(mod, old_src + "\n\n" + fn_src, name)     # the old body may need a helper the new one dropped
            if setup_h is None:
                continue
            bad = failing(kept, record(setup_h, old_src, name, [c for c, _ in kept]))
            if bad:
                rec_old = record(setup_h, old_src, name, [c for c, _ in kept])
                picks = fail_picks(bad, 2)
                add(make_task("history", f"hist-{base}", name, mod.path, setup_h, test, fn_src, old_src,
                              prompt_of("history", name, mod.path, setup_h, old_src, examples_text(name, kept, picks, rec_old), subjects),
                              {"rev": rev, "commits": subjects, "cases": len(kept), "shown": len(picks)}))
                break
    if "bugfix" in a.families:
        sites = list(range(mutation_sites(fn)))
        rng.shuffle(sites)
        labels = set()
        for site in sites[:a.mutants_tried]:
            label, new = mutate(fn, site)
            new_src = ast.unparse(new)
            if new_src == fn_src or label.split(" ")[0] in labels:
                continue
            rec_bad = record(setup, new_src, name, [c for c, _ in kept], timeout=5)
            bad = failing(kept, rec_bad)
            if bad:
                labels.add(label.split(" ")[0])
                hint = "none" if len(labels) == 2 and ast.get_docstring(fn) else "example"      # the second bug of a documented function: no example
                picks = fail_picks(bad, 1) if hint == "example" else []
                add(make_task("bugfix", f"bug-{base}-{len(labels)}", name, mod.path, setup, test, fn_src, new_src,
                              prompt_of("bugfix", name, mod.path, setup, new_src, examples_text(name, kept, picks, rec_bad)),
                              {"rev": rev, "mutation": label, "hint": hint, "cases": len(kept), "shown": len(picks)}))
                if len(labels) >= a.mutants_per_function:
                    break
    if "implement" in a.families and ast.get_docstring(fn) and len(kept) >= 3:
        stub = copy.deepcopy(fn)
        stub.body = [fn.body[0], ast.Raise(exc=ast.Call(func=ast.Name(id="NotImplementedError", ctx=ast.Load()), args=[], keywords=[]), cause=None)]
        stub_src = ast.unparse(ast.fix_missing_locations(stub))
        picks = list(range(min(3, len(kept) - 1)))
        add(make_task("implement", f"impl-{base}", name, mod.path, setup, test, fn_src, stub_src,
                             prompt_of("implement", name, mod.path, setup, stub_src, examples_text(name, kept, picks)),
                             {"rev": rev, "cases": len(kept), "shown": len(picks)}))
    cnt["built"] += len(out)
    return out, cnt


# ------------------------------------------------------------------------------------------------ the whole repo
def collect(root, rev):
    """(modules {path: Module}, calls {(file, function): [case]}) from the files at `rev`."""
    paths = (git(root, "ls-tree", "-r", "--name-only", rev) or "").split()
    py = [p for p in paths if p.endswith(".py") and not p.startswith(SKIP_DIRS)]
    srcs, by_stem = {}, collections.defaultdict(list)
    tests = []
    for p in py:
        text = git(root, "show", f"{rev}:{p}")
        if text is None or len(text) > 400_000:
            continue
        if is_test_path(p):
            tests.append((p, text))
        else:
            srcs[p] = text
            by_stem[Path(p).stem].append(p)
    mods = {}
    for p, text in srcs.items():
        try:
            mods[p] = Module(p, text)
        except (SyntaxError, ValueError, RecursionError):
            continue
    calls = collections.defaultdict(list)
    for p, text in tests:
        for key, cs in mine_calls(p, text, by_stem).items():
            for c in cs:
                if c not in calls[key]:
                    calls[key].append(c)
    return mods, calls


def validate(task):
    """Problems of one task: its fields, then the self-test (reference passes, wrong fails) in the real checker."""
    return TG.validate_tasks([task]) or TG.selftest([task])


def select(tasks, count, per_file, per_function, seed):
    """At most `count` tasks, round-robin over the families (so every prefix is a mix), at most `per_file` per source file and
    `per_function` per function."""
    rng = random.Random(seed)
    queues = {}
    for t in sorted(tasks, key=lambda t: t["id"]):
        queues.setdefault(t["meta"]["family"], []).append(t)
    for q in queues.values():
        rng.shuffle(q)
    files, funcs, out = collections.Counter(), collections.Counter(), []
    while len(out) < count and any(queues.values()):
        for fam in sorted(queues):
            while queues[fam]:
                t = queues[fam].pop()
                m = t["meta"]
                if files[m["path"]] < per_file and funcs[(m["path"], m["function"])] < per_function:
                    files[m["path"]] += 1
                    funcs[(m["path"], m["function"])] += 1
                    out.append(t)
                    break
            if len(out) >= count:
                break
    return out


def build(a):
    rev = (git(a.root, "rev-parse", "--short=12", a.rev) or "").strip()
    if not rev:
        raise SystemExit(f"{a.rev}: not a revision of {a.root}")
    mods, calls = collect(a.root, rev)
    work = sorted((p, n) for (p, n) in calls if p in mods and n in mods[p].funcs)
    cnt = collections.Counter(modules=len(mods), functions_with_literal_calls=len(work))
    cands = []

    def one(item):
        p, n = item
        try:
            return tasks_for(mods[p], n, calls[item], a, rev, a.seed)
        except (RecursionError, ValueError, OSError):
            return [], collections.Counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for tasks, c in ex.map(one, work):
            cands += tasks
            cnt.update(c)
    cnt["candidates"] = len(cands)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        problems = list(ex.map(validate, cands))
    ok = [t for t, p in zip(cands, problems) if not p]
    cnt["validated"] = len(ok)
    for fam in FAMILIES:
        cnt[f"validated_{fam}"] = sum(1 for t in ok if t["meta"]["family"] == fam)
    for t, p in zip(cands, problems):
        if p:
            print(f"dropped {t['id']}: {p[0]}", file=sys.stderr)
    chosen = select(ok, a.count, a.per_file, a.per_function, a.seed)
    cnt["emitted"] = len(chosen)
    return rev, chosen, cnt


def about(rev, cnt, a):
    fam = collections.Counter(t["meta"]["family"] for t in a)
    return (f"{len(a)} code_edit tasks built by tools/pancake/build_task_set.py from the repository at {rev} ("
            + ", ".join(f"{k} {v}" for k, v in sorted(fam.items()))
            + "): pure stdlib-only top-level Python functions that the repo's own tests call with literal arguments; the checker replays those calls "
            "against the recorded results of the function at that commit.  Every task carries a reference answer that must pass and a broken one "
            "that must fail (task_gate.py --dry-run runs both).")


def is_main_guard(st):
    return isinstance(st, ast.If) and isinstance(st.test, ast.Compare) and isinstance(st.test.left, ast.Name) and st.test.left.id == "__name__"


def whole_module(task, src):
    """The checker against the real module at the pinned commit (its `if __name__ == "__main__":` block dropped): "pass", "fail", or
    "skipped" when the module itself does not run in the sandbox (a non-stdlib import, a relative import, module-level work)."""
    tree = ast.parse(src)
    tree.body = [st for st in tree.body if not is_main_guard(st)]
    code = ast.unparse(tree)
    if TG.run_python(code, 20)[0] != 0:
        return "skipped"
    return "pass" if TG.run_python(code + "\n\n" + task["expect"]["test"], 20)[0] == 0 else "fail"


def check(path, root):
    """Self-test an existing set: the schema; every checker on its reference and wrong answer; and, where git has the pinned commit, that the
    reference is the function that commit holds and that the checker also passes on the whole module of that commit (modules that do not run in
    the sandbox are counted apart).  Returns the number of problems."""
    ts, tasks = TG.load_tasks(path)
    probs = TG.validate_tasks(tasks) or TG.selftest(tasks)
    whole = collections.Counter()
    for t in tasks:
        m = t.get("meta") or {}
        src = git(root, "show", f"{m.get('rev')}:{m.get('path')}") if m.get("rev") else None
        if src is None:
            continue
        ref = function_at(t["reference"]["content"].split("```python\n", 1)[-1].rsplit("\n```", 1)[0], m["function"])
        cur = function_at(src, m["function"])
        if ref is None or cur is None or ast.dump(ref) != ast.dump(cur):
            probs.append(f"{t['id']}: the reference is not {m['function']} as {m['rev']} has it")
        r = whole_module(t, src)
        whole[r] += 1
        if r == "fail":
            probs.append(f"{t['id']}: the checker fails on the whole module of {m['rev']}")
    for p in probs:
        print("PROBLEM: " + p)
    print(f"{path}: {len(tasks)} tasks, {len(probs)} problem(s); on the whole original module: "
          + ", ".join(f"{k} {v}" for k, v in sorted(whole.items())))
    return len(probs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=120, help="emit at most this many validated tasks (default 120)")
    ap.add_argument("--out", default="bench/tasks/repo-history.json")
    ap.add_argument("--check", metavar="FILE", help="self-test an existing task set and exit")
    ap.add_argument("--root", default=str(ROOT), help="the git repository to mine (default: this one)")
    ap.add_argument("--rev", default="HEAD")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--families", default=",".join(FAMILIES), type=lambda s: s.split(","))
    ap.add_argument("--per-file", type=int, default=8, help="at most this many tasks from one source file")
    ap.add_argument("--per-function", type=int, default=2, help="at most this many tasks on one function")
    ap.add_argument("--max-history", type=int, default=40, help="commits to look back through for the history family")
    ap.add_argument("--mutants-tried", type=int, default=10, help="mutation sites tried per function")
    ap.add_argument("--mutants-per-function", type=int, default=2)
    a = ap.parse_args()
    if a.check:
        return 1 if check(a.check, a.root) else 0
    if set(a.families) - set(FAMILIES):
        ap.error(f"--families: any of {','.join(FAMILIES)}")
    rev, tasks, cnt = build(a)
    out = Path(a.out) if os.path.isabs(a.out) else ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"name": "repo history task set", "version": 1, "rev": rev, "about": about(rev, cnt, tasks), "funnel": dict(sorted(cnt.items())),
               "tasks": tasks}, open(out, "w"), indent=1)
    print("funnel: " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items())))
    print(f"{len(tasks)} tasks -> {out}" + ("" if len(tasks) >= a.count else f"  (only {len(tasks)} of the {a.count} asked for survived)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
