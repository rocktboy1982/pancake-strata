#!/usr/bin/env python3
"""build_quality_set.py - grows the token-forcing set (bench/quality/set.json, 20 requests) to 48 requests for tools/pancake/quality_gate.py.

Why: with 20 requests (5,322 tokens) an arm identical to stock is INCONCLUSIVE 22% of the time and a real +1.0 point deviation is rejected only
62% (MOD_PLAN "Test-protocol findings", pancake-docs/batch3/promotion.md); with 40+ requests (~10,600 tokens) both reach ~92%.  The old requests
are 650-1,500 tokens long; the real agent requests of the opencode history (qwen3.8-flash-next-iq2_xs, 2,036 requests, 25 sessions; measured as
numbers only, no text) are 11K-100K tokens with a ~11K system+tools prefix, a median step of +525 tokens (bash / read results), and a reply of
~115-340 tokens.  So the 28 new requests are synthetic but have those shapes, assembled from this repository's own files at a pinned commit
(never from the history's text):
  edit-short  6   one function and "rename / add a guard", no tools (copy-heavy, 0.5-2K tokens; like abtest's `edit` case)
  first       4   system + tools + project instructions (~10.5K) + a first user request              (~11K)
  toolres     6   ... + assistant read call + the file as the tool result (thinking on x3, off x3)   (~13-18K)
  edit-ctx    6   ... + "rename X in <file>" with the file as the tool result: the reply is an edit call (copy-heavy)
  deep32k     4   ... + a session of read / git grep / git log exchanges up to ~32K tokens, then the next step
  deep64k     2   the same up to ~64K tokens (the engine's --kv-resident 32768 is crossed)
Prompts are rendered with the served model's own chat template (tokenizer dir's chat_template.jinja) and tokenized with its tokenizer.

The continuations (`cont`) are NOT made here: they are the stock engine's greedy output (tools/force_check.py gen), which needs the GPU.  The
new items carry "cont": null; quality_gate.py --fill generates them with the stock arm (one engine start, ~10 minutes), writes the complete set,
and that file is then committed and frozen (the same tokens for every later run).  The 20 old requests are copied verbatim as the first items.

    python tools/pancake/build_quality_set.py [--out bench/quality/set-v2.json] [--rev REV] [--tokenizer DIR] [--seed 1]
    python tools/pancake/build_quality_set.py --check bench/quality/set-v2.json      structure, id range, old items verbatim, sizes
"""
import argparse
import ast
import collections
import json
import random
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import quality_gate as QG  # noqa: E402

ROOT = QG.ROOT
OLD_SET = "bench/quality/set.json"
VOCAB = 248320
PREFIX_TOKENS = 10_500
CODE_EXT = (".py", ".cpp", ".hpp", ".h", ".cu", ".cuh")
CODE_DIRS = ("src/", "include/", "tools/", "serve/", "strata_cli/", "strata_control/")
DOCS = ("AGENTS.md", "CONTRIBUTING.md", "docs/AI_REFERENCE.md", "docs/DETAILS.md")
SHAPES = (("edit-short", 6), ("first", 4), ("toolres", 6), ("edit-ctx", 6), ("deep32k", 4), ("deep64k", 2))
DEEP_TARGET = {"deep32k": 32_000, "deep64k": 64_000}
STOP = {"return", "static", "const", "inline", "struct", "class", "break", "continue", "while", "else", "elif", "import", "from", "pass", "lambda",
        "global", "nullptr", "size_t", "uint8_t", "int8_t", "int16_t", "int32_t", "int64_t", "uint32_t", "uint64_t", "string", "vector", "include",
        "define", "ifndef", "endif", "namespace", "template", "typename", "default", "switch", "false", "true", "print", "range", "sorted", "float"}

SYSTEM = """You are a coding agent working in the user's repository, from a terminal. You help with software engineering tasks: reading and explaining code, finding things, fixing bugs, refactoring, writing tests and running commands.

How to work:
- Read before you edit. Use grep and glob to find files, read to look at them, and keep a short todo list for tasks with several steps.
- Make small, focused changes that follow the style of the surrounding code. Do not add features, comments or abstractions nobody asked for.
- After a change, run the relevant test or command with bash and report what happened, including failures.
- Use the tools to answer questions about the repository; do not guess file contents.
- Be concise. Answer in a few short sentences or a short list, and reference code as path:line.
- Never run destructive commands (deleting files, force pushes, resetting history) without being asked to."""

TOOLS = [
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command in the project directory and return its output. Use it for tests, builds, git and other commands. "
        "Quote paths that contain spaces. Output is truncated when it is very long.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The command to run"},
                                                        "description": {"type": "string", "description": "A short description of what the command does"},
                                                        "timeout": {"type": "integer", "description": "Timeout in milliseconds"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read", "description": "Read a file from the filesystem. Returns numbered lines. Use offset and limit to read part of a large file.",
        "parameters": {"type": "object", "properties": {"filePath": {"type": "string", "description": "Path of the file"},
                                                        "offset": {"type": "integer", "description": "The line to start from (1-based)"},
                                                        "limit": {"type": "integer", "description": "The number of lines to read"}}, "required": ["filePath"]}}},
    {"type": "function", "function": {"name": "edit", "description": "Replace an exact string in a file. The oldString must match the file exactly (including whitespace) and be unique "
        "unless replaceAll is set. Read the file first.",
        "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}, "oldString": {"type": "string", "description": "The text to replace"},
                                                        "newString": {"type": "string", "description": "The replacement text"},
                                                        "replaceAll": {"type": "boolean", "description": "Replace every occurrence"}},
                       "required": ["filePath", "oldString", "newString"]}}},
    {"type": "function", "function": {"name": "write", "description": "Write a file, replacing it when it exists. Prefer edit for changes to existing files.",
        "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}, "content": {"type": "string"}}, "required": ["filePath", "content"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Search file contents with a regular expression. Returns matching file paths and lines.",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string", "description": "The regular expression"},
                                                        "path": {"type": "string", "description": "The directory to search"},
                                                        "include": {"type": "string", "description": "A file pattern such as *.py"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "glob", "description": "Find files by name pattern, for example **/*.py. Returns matching paths.",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "todowrite", "description": "Create and update the task list of the session. Use it for tasks with three or more steps.",
        "parameters": {"type": "object", "properties": {"todos": {"type": "array", "items": {"type": "object", "properties": {
            "content": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
            "priority": {"type": "string", "enum": ["high", "medium", "low"]}}, "required": ["content", "status", "priority"]}}}, "required": ["todos"]}}},
    {"type": "function", "function": {"name": "task", "description": "Launch a sub-agent for a self-contained piece of research and return its report.",
        "parameters": {"type": "object", "properties": {"description": {"type": "string"}, "prompt": {"type": "string"}}, "required": ["description", "prompt"]}}},
    {"type": "function", "function": {"name": "webfetch", "description": "Fetch a web page and return its text.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
]
THOUGHTS = ["I should look at the file before answering.", "Let me check where this is used first.", "I need the surrounding code to answer.",
            "A quick search will show the call sites.", "I will read the file and then decide."]
FIRST_ASKS = ["Where is `{ident}` defined in this repository and what calls it? Give me the file paths.",
              "I want to understand `{path}`. Find it, read it and explain in a few sentences what it is for.",
              "Is there a test for `{ident}`? If not, tell me which file a test should go in.",
              "List the places where `{ident}` is used and say which one is the hot path."]
READ_ASKS = ["Read `{path}` and explain what `{ident}` does.", "Look at `{path}`: is there anything in it that could fail on an empty input?",
             "Summarize `{path}` in five bullet points."]
EDIT_ASKS = ["In `{path}` rename `{ident}` to `{new}` everywhere in the file.",
             "In `{path}` add a one-line comment above the definition of `{ident}` saying what it is for."]


# ------------------------------------------------------------------------------------------------ the repo at a pinned commit
def git(root, *args):
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def code_files(root, rev, lo=2_000, hi=40_000):
    """Tracked source files at `rev` with a size in [lo, hi] bytes, sorted: (path, size)."""
    out = []
    for line in git(root, "ls-tree", "-r", "-l", rev).splitlines():
        meta, _, path = line.partition("\t")
        size = int(meta.split()[3]) if meta.split()[3].isdigit() else 0
        name = path.rsplit("/", 1)[-1]
        if path.endswith(CODE_EXT) and path.startswith(CODE_DIRS) and lo <= size <= hi and not name.startswith("test_") and "/tests/" not in path:
            out.append((path, size))
    return sorted(out)


def pick_ident(text, rng):
    """A name used 2-8 times in `text` (a rename that touches a few lines), not a keyword; None when there is none."""
    counts = collections.Counter(re.findall(r"\b[a-z][a-z0-9_]{5,}\b", text))
    cands = sorted(w for w, n in counts.items() if 2 <= n <= 8 and w not in STOP and "_" in w)
    return rng.choice(cands) if cands else None


def number_lines(text):
    lines = text.splitlines()
    return "\n".join(f"{i}: {l}" for i, l in enumerate(lines, 1)) + f"\n\n(End of file - total {len(lines)} lines)"


def read_result(path, text):
    return f"<path>{path}</path>\n<type>file</type>\n<content>\n{number_lines(text)}\n</content>"


def functions_of(text, lo=20, hi=60):
    """Source of the top-level Python functions of `text` with lo..hi lines."""
    try:
        return [ast.get_source_segment(text, n) for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and lo <= n.end_lineno - n.lineno + 1 <= hi]
    except SyntaxError:
        return []


def local_name(fn_src, rng):
    """A parameter or local variable of the function that appears at least twice in it (a rename that touches a few lines), or None."""
    tree = ast.parse(fn_src)
    seen = [n.id for n in ast.walk(tree) if isinstance(n, ast.Name)] + [a.arg for a in ast.walk(tree) if isinstance(a, ast.arg)]
    bound = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)} | {a.arg for a in ast.walk(tree) if isinstance(a, ast.arg)}
    counts = collections.Counter(seen)
    cands = sorted(n for n in bound if len(n) >= 3 and counts[n] >= 2 and n not in ("self", "cls"))
    return rng.choice(cands) if cands else None


# ------------------------------------------------------------------------------------------------ rendering
class Renderer:
    """The served model's chat template + tokenizer."""

    def __init__(self, tokenizer_dir):
        import jinja2
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        import strata_tokenizer as ST
        d = Path(tokenizer_dir)
        vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
        tokens = [None] * len(vocab)
        for t, i in vocab.items():
            tokens[i] = t
        cfg = json.loads((d / "tokenizer.json").read_text(encoding="utf-8"))
        self.tok = ST.Tokenizer(tokens, (d / "merges.txt").read_text(encoding="utf-8").split("\n"), json.loads((d / "token_type.json").read_text()),
                                cfg.get("pre") or "qwen35", cfg.get("special_ids") or None)
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, sort_keys=False: json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, sort_keys=sort_keys)

        def fail(m):
            raise ValueError(m)
        env.globals["raise_exception"] = fail
        self.template = env.from_string((d / "chat_template.jinja").read_text(encoding="utf-8"))
        self.jinja_version = jinja2.__version__

    def text(self, messages, tools, thinking):
        return self.template.render(messages=messages, tools=tools, add_generation_prompt=True, enable_thinking=thinking)

    def ids(self, messages, tools, thinking):
        return self.tok.encode(self.text(messages, tools, thinking), parse_special=True)

    def count(self, s):
        return len(self.tok.encode(s, parse_special=True))


def call(name, **args):
    return {"type": "function", "function": {"name": name, "arguments": args}}


def exchange(thought, text, name, args, result):
    """An assistant turn that calls one tool, and the tool's result."""
    return [{"role": "assistant", "content": text, "reasoning_content": thought, "tool_calls": [call(name, **args)]}, {"role": "tool", "content": result}]


def system_prompt(root, rev, r):
    """SYSTEM + the repository's own instructions, cut so that system + tools come to PREFIX_TOKENS."""
    docs = []
    for d in DOCS:
        t = git(root, "show", f"{rev}:{d}")
        if t:
            docs.append(f"Instructions from: {d}\n{t}")
    base = r.count(r.text([{"role": "system", "content": SYSTEM}, {"role": "user", "content": "x"}], TOOLS, True))
    text = "\n\n".join(docs)
    keep = max(0, int((PREFIX_TOKENS - base) * 3.6))               # ~3.6 characters a token for these docs
    msg = SYSTEM + ("\n\n" + text[:keep] if keep else "")
    for _ in range(6):                                               # trim / extend until the prefix is within 2% of the target
        n = r.count(r.text([{"role": "system", "content": msg}, {"role": "user", "content": "x"}], TOOLS, True))
        if abs(n - PREFIX_TOKENS) <= PREFIX_TOKENS * 0.02:
            break
        keep = max(0, int(keep * PREFIX_TOKENS / n))
        msg = SYSTEM + "\n\n" + text[:keep]
    return msg


def item(kind, ids, messages, thinking, files, rev, n):
    return {"id": f"{kind}-{n}", "kind": kind, "prompt": ids, "cont": None,
            "meta": {"shape": kind, "thinking": thinking, "files": files, "rev": rev, "prompt_tokens": len(ids)}}


def build_items(root, rev, r, seed):
    rng = random.Random(seed)
    files = code_files(root, rev)
    rng.shuffle(files)
    pool = iter(files)
    system = system_prompt(root, rev, r)
    texts = {}

    def source(path):
        if path not in texts:
            texts[path] = git(root, "show", f"{rev}:{path}")
        return texts[path]

    def next_file(small=False):
        for path, size in pool:
            if small and size > 16_000:
                continue
            text = source(path)
            ident = pick_ident(text, rng)
            if text and ident:
                return path, text, ident
        raise SystemExit("not enough source files at this commit")

    def convo(user, exchanges=()):
        return [{"role": "system", "content": system}, {"role": "user", "content": user}, *[m for e in exchanges for m in e]]

    out, counters = [], collections.Counter()

    def add(kind, messages, tools, thinking, used):
        counters[kind] += 1
        out.append(item(kind, r.ids(messages, tools, thinking), messages, thinking, used, rev, counters[kind] - 1))

    for kind, n in SHAPES:
        for i in range(n):
            if kind == "edit-short":
                while True:
                    path, text, _ = next_file(small=True)
                    fns = functions_of(text) if path.endswith(".py") else []
                    if fns:
                        break
                fn = rng.choice(fns)
                var = local_name(fn, rng) or "value"
                ask = (f"Rename `{var}` to `{var}_new` in this function and return the whole function, nothing else.\n\n```python\n{fn}\n```" if i % 2 == 0 else
                       f"Add an early return of `None` at the top of this function when its first argument is falsy, and return the whole function, nothing else.\n\n```python\n{fn}\n```")
                add(kind, [{"role": "user", "content": ask}], None, False, [path])
            elif kind == "first":
                path, text, ident = next_file(small=True)
                add(kind, convo(FIRST_ASKS[i % len(FIRST_ASKS)].format(ident=ident, path=path)), TOOLS, True, [path])
            elif kind in ("toolres", "edit-ctx"):
                path, text, ident = next_file(small=True)
                ask = (READ_ASKS[i % len(READ_ASKS)] if kind == "toolres" else EDIT_ASKS[i % len(EDIT_ASKS)]).format(path=path, ident=ident, new=ident + "_new")
                ex = exchange(THOUGHTS[i % len(THOUGHTS)], "I'll read the file first.", "read", {"filePath": path}, read_result(path, text))
                add(kind, convo(ask, [ex]), TOOLS, kind == "toolres" and i % 2 == 0, [path])
            else:
                target, used, exs = DEEP_TARGET[kind], [], []
                path, text, ident = next_file(small=True)
                ask = READ_ASKS[i % len(READ_ASKS)].format(path=path, ident=ident)
                total = r.count(r.text(convo(ask), TOOLS, True))
                while total < target * 0.93:
                    path, text, ident = next_file()
                    roll = rng.random()
                    if roll < 0.6:
                        ex = exchange(rng.choice(THOUGHTS), "Let me look at this file.", "read", {"filePath": path}, read_result(path, text))
                    elif roll < 0.85:
                        hits = [l.split(":", 1)[1] if l.startswith(rev) else l for l in git(root, "grep", "-n", "-I", "-e", ident, rev, "--", "src", "tools", "serve").splitlines()[:40]]
                        ex = exchange(rng.choice(THOUGHTS), "", "grep", {"pattern": ident}, "\n".join(hits) or "No files found")
                    else:
                        log = git(root, "log", "--oneline", "-n", "15", rev, "--", path)
                        ex = exchange(rng.choice(THOUGHTS), "", "bash", {"command": f"git log --oneline -n 15 -- {path}", "description": "Recent commits"}, log.strip())
                    size = sum(r.count(m["content"] or "") + r.count(m.get("reasoning_content") or "") for m in ex)
                    if total + size > target * 1.06 and total >= target * 0.8:
                        break
                    exs.append(ex)
                    used.append(path)
                    total += size
                add(kind, convo(ask, exs), TOOLS, i % 2 == 0, used)
    return out


# ------------------------------------------------------------------------------------------------ the set
def build(root, rev, tokenizer, seed):
    old = json.load(open(ROOT / OLD_SET))
    new = build_items(root, rev, Renderer(tokenizer), seed)
    return {"config": old["config"], "env": old["env"], "rev": rev, "seed": seed,
            "about": ("20 old requests (verbatim) + 28 synthetic agent-shaped requests built by tools/pancake/build_quality_set.py from the repository at "
                      f"{rev}; the new ones have cont null until quality_gate.py --fill generates them with the stock arm (force_check.py gen)."),
            "items": old["items"] + new}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="bench/quality/set-v2.json")
    ap.add_argument("--check", metavar="FILE", help="validate an existing set and exit")
    ap.add_argument("--rev", default="HEAD", help="the commit whose files the requests are assembled from (default HEAD)")
    ap.add_argument("--tokenizer", help="the model's tokenizer directory (default: the one in bench/quality/flashnext-iq2_xs.json)")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.check:
        probs = QG.set_problems(json.load(open(a.check)), json.load(open(ROOT / OLD_SET)))
        print("\n".join("PROBLEM: " + p for p in probs) or "ok")
        print(QG.set_summary(json.load(open(a.check))))
        return 1 if probs else 0
    rev = git(ROOT, "rev-parse", "--short=12", a.rev).strip()
    tokenizer = a.tokenizer or json.load(open(ROOT / "bench/quality/flashnext-iq2_xs.json"))["tokenizer"]
    st = build(ROOT, rev, tokenizer, a.seed)
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    json.dump(st, open(out, "w"), separators=(",", ":"))
    probs = QG.set_problems(st, json.load(open(ROOT / OLD_SET)))
    print("\n".join("PROBLEM: " + p for p in probs))
    print(f"{out}: " + QG.set_summary(st))
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main())
