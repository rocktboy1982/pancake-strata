#!/usr/bin/env python3
"""doc_check.py - does the three-file documentation rule (AGENTS.md "Documentation rules") still hold?  Read-only, no engine, no GPU (LEADS.md Part O O4).

Idea from caminus' ENTROPY.md checks (`refs`: a path a doc names is in the tree; `orphans`; `copies`; a baseline that only shrinks; read as a description
only, AGPL, no code taken).  Several agents append to the same three files here, so the rules rot silently.  Checks:

  md      every tracked *.md is one of the allowed documents (AGENTS.md rule 3: the root files, inherited docs/, raw archives under bench/results and pancake-docs/research,
          the fork_kb output under data/fork-kb) or is listed in the baseline (what was already there when this check arrived)
  ledger  TEST_LEDGER.md: no T-nnn id used twice (rule 7: ids are taken at commit time; two agents taking the same next id is the failure it catches)
  specs   bench/specs/NNN-*.json: no spec number used twice
  parts   LEADS.md: the `## Part X` letters increase (a repeat or a step back means two agents took the same letter; a gap is allowed)
  refs    a path named in backticks in MOD_PLAN.md / LEADS.md / TEST_LEDGER.md / README.md exists in the tree, or is listed in the baseline (box-only paths
          that start with / or ~, placeholders (<>, *, {}, $) and paths under bench/results/ and pancake-docs/research/ archives are not checked: the
          results of a box run are committed later than the row that names them)

The baseline (tools/pancake/doc_check_baseline.json) only shrinks: a check fails on anything not in it, and `--prune` drops entries that are fixed.  `--write-baseline`
writes it from the current tree (done once, 2026-11-17); adding an entry afterwards is a hand edit that the commit message must explain.

    python tools/pancake/doc_check.py                 run every check; exit 1 on a new failure
    python tools/pancake/doc_check.py --only ledger specs parts
    python tools/pancake/doc_check.py --prune | --write-baseline
Tests: python -m pytest -q tools/pancake/test_doc_check.py
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).resolve().parent / "doc_check_baseline.json"

ALLOWED_MD = {"AGENTS.md", "README.md", "MOD_PLAN.md", "TEST_LEDGER.md", "LEADS.md", "CONTRIBUTING.md", "SECURITY.md"}
# Inherited upstream files (written by the upstream author Niko1221, not by this fork's agents; found by the first run, 2026-11-17): a licence notice that
# ships with a data file, the README of a data file, and three research-tool READMEs.  AGENTS.md rule 3 keeps inherited upstream documents as upstream has
# them, so they are allowed by name; a NEW markdown file beside them is still a failure.
INHERITED_MD = {"data/dflash2-tuned/NOTICE.md", "data/experimental-speed-projection/README.md", "tools/dflash2/README.md",
                "tools/dsv41_lab/hip/README.md", "tools/trellis/README.md"}
ALLOWED_MD_PREFIX = ("bench/results/", "pancake-docs/research/", "data/fork-kb/", "docs/", "third_party/", ".github/", "tests/", "apps/", "packaging/", "integrations/", "ref/", "spark/", "sycl/", "metal/", "docker/")
REF_DOCS = ("MOD_PLAN.md", "LEADS.md", "TEST_LEDGER.md", "README.md")
REF_EXT = (".cpp", ".hpp", ".cu", ".cuh", ".h", ".py", ".json", ".md", ".sh", ".def", ".cmake", ".txt", ".hip", ".html", ".ps1", ".bat", ".c")
SKIP_REF_PREFIX = ("bench/results/", "pancake-docs/research/", "bench/quality-hip/", "bench/prefetch/", "build", "http", "/", "~", "../")


def tracked(pattern: str) -> list:
    try:
        out = subprocess.run(["git", "ls-files", pattern], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [l for l in out.splitlines() if l]


def check_md(files: list) -> list:
    """Tracked markdown files that are neither allowed nor baselined (the names; the caller subtracts the baseline)."""
    return sorted(f for f in files if f not in ALLOWED_MD and f not in INHERITED_MD and not f.startswith(ALLOWED_MD_PREFIX))


def ledger_ids(text: str) -> list:
    return [int(m) for m in re.findall(r"^\|\s*T-(\d+)\s*\|", text, flags=re.M)]


def duplicates(ids: list) -> list:
    return sorted(k for k, n in Counter(ids).items() if n > 1)


def spec_numbers(names: list) -> list:
    out = []
    for n in names:
        m = re.match(r"(\d+)-", Path(n).name)
        if m:
            out.append(int(m.group(1)))
    return out


def part_letters(text: str) -> list:
    return re.findall(r"^## Part ([A-Z]):", text, flags=re.M)


def part_problems(letters: list) -> list:
    """Letters must increase: a repeat or a step back is a failure (two agents took the same letter).  A GAP is allowed: another agent's Part may not be
    committed yet when this one is (they land in any order)."""
    bad = []
    for i in range(1, len(letters)):
        if letters[i] <= letters[i - 1]:
            bad.append(f"Part {letters[i]} follows Part {letters[i - 1]}")
    return bad


def cited_paths(text: str) -> set:
    """Repo-relative file paths named in backticks (a token with a '/' and a known extension; no placeholder, no box-only root)."""
    out = set()
    for tok in re.findall(r"`([^`\n]+)`", text):
        tok = tok.strip().rstrip(".,;:)")
        tok = re.sub(r":\d+(-\d+)?$", "", tok)            # file:line
        if " " in tok or "/" not in tok or not tok.endswith(REF_EXT):
            continue
        if any(c in tok for c in "<>*{}$()|"):
            continue
        if re.match(r"[A-Za-z][\w.-]*:[^\d/]", tok):      # `project:path`: a file of ANOTHER project (LEADS.md Part O writes caminus files as `caminus:docs/x.md`)
            continue
        if tok.startswith(SKIP_REF_PREFIX):
            continue
        out.add(tok)
    return out


def missing_refs(root: Path = ROOT, docs=REF_DOCS) -> list:
    miss = set()
    for d in docs:
        p = root / d
        if not p.exists():
            continue
        for ref in cited_paths(p.read_text(encoding="utf-8", errors="replace")):
            if not (root / ref).exists():
                miss.add(ref)
    return sorted(miss)


def load_baseline() -> dict:
    try:
        return json.loads(BASELINE.read_text())
    except (OSError, ValueError):
        return {}


def run(only=None, root: Path = ROOT) -> dict:
    """{check: [findings]} before the baseline is applied."""
    want = lambda k: not only or k in only
    res: dict = {}
    if want("md"):
        res["md"] = check_md(tracked("*.md"))
    if want("ledger"):
        p = root / "TEST_LEDGER.md"
        res["ledger"] = [f"T-{i}" for i in duplicates(ledger_ids(p.read_text(encoding="utf-8", errors="replace")))] if p.exists() else []
    if want("specs"):
        res["specs"] = [f"{n:03d}" for n in duplicates(spec_numbers([str(x) for x in (root / "bench" / "specs").glob("*.json")]))]
    if want("parts"):
        p = root / "LEADS.md"
        res["parts"] = part_problems(part_letters(p.read_text(encoding="utf-8", errors="replace"))) if p.exists() else []
    if want("refs"):
        res["refs"] = missing_refs(root)
    return res


def apply_baseline(res: dict, base: dict) -> tuple:
    """-> (new failures {check: [..]}, fixed baseline entries {check: [..]})."""
    new, fixed = {}, {}
    for k, v in res.items():
        b = set(base.get(k, []))
        n = [x for x in v if x not in b]
        if n:
            new[k] = n
        f = sorted(b - set(v))
        if f:
            fixed[k] = f
    return new, fixed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="+", choices=("md", "ledger", "specs", "parts", "refs"))
    ap.add_argument("--prune", action="store_true", help="drop baseline entries that are fixed")
    ap.add_argument("--write-baseline", action="store_true", help="write the baseline from the current tree")
    a = ap.parse_args(argv)
    res = run(a.only)
    if a.write_baseline:
        BASELINE.write_text(json.dumps({k: v for k, v in res.items() if v}, indent=1, sort_keys=True) + "\n")
        print(f"baseline written: {', '.join(f'{k} {len(v)}' for k, v in res.items() if v) or 'empty'}")
        return 0
    base = load_baseline()
    new, fixed = apply_baseline(res, base)
    if a.prune and fixed:
        for k, v in fixed.items():
            base[k] = [x for x in base.get(k, []) if x not in v]
        BASELINE.write_text(json.dumps({k: v for k, v in base.items() if v}, indent=1, sort_keys=True) + "\n")
    for k in res:
        print(f"{k}: {len(res[k])} found, {len(res[k]) - len(new.get(k, []))} in the baseline, {len(new.get(k, []))} new")
    for k, v in new.items():
        for x in v[:40]:
            print(f"  NEW {k}: {x}")
    for k, v in fixed.items():
        print(f"  baseline entries now fixed ({k}): {len(v)}  (run --prune)")
    return 1 if new else 0


if __name__ == "__main__":
    sys.exit(main())
