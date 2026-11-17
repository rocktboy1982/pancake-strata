"""Tests for doc_check.py on small fixed inputs (LEADS.md Part O O4); the last test runs the real checks against the committed baseline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import doc_check as D  # noqa: E402


def test_md_allowed_and_not():
    files = ["README.md", "MOD_PLAN.md", "docs/QUANTS.md", "bench/results/2026-09-24-x/README.md", "pancake-docs/research/bench/a.md",
             "pancake-docs/NIGHT_RESULTS_X.md", "NEW_REPORT.md", "tools/foo/NOTES.md"]
    assert D.check_md(files) == ["NEW_REPORT.md", "pancake-docs/NIGHT_RESULTS_X.md", "tools/foo/NOTES.md"]
    # the five inherited upstream files are allowed by name; a new file beside them is not
    assert D.check_md(["tools/trellis/README.md", "tools/dflash2/README.md", "tools/trellis/NEW.md"]) == ["tools/trellis/NEW.md"]


def test_baseline_is_empty_and_hook_exists():
    assert D.load_baseline() == {}                                  # everything the first run found is fixed; the baseline only shrinks
    hook = D.ROOT / "tools/pancake/pre-push-doc-check.sh"
    assert hook.exists() and "doc_check.py" in hook.read_text() and "SKIP_DOC_CHECK" in hook.read_text()


def test_ledger_ids_and_duplicates():
    text = "| T-001 | a |\n| T-002 | b |\n| T-002 | c |\nnot a row T-009\n| T-010 |\n"
    ids = D.ledger_ids(text)
    assert ids == [1, 2, 2, 10]
    assert D.duplicates(ids) == [2]


def test_spec_numbers():
    assert D.duplicates(D.spec_numbers(["bench/specs/163-a.json", "bench/specs/163-b.json", "bench/specs/f12-x.json", "bench/specs/164-c.json"])) == [163]


def test_part_letters():
    assert D.part_problems(D.part_letters("## Part A: x\n## Part B: y\n## Part C: z\n")) == []
    assert D.part_problems(D.part_letters("## Part A: x\n## Part C: y\n")) == []   # a gap is allowed
    assert D.part_problems(D.part_letters("## Part A: x\n## Part A: y\n")) != []


def test_cited_paths():
    text = ("see `src/core/verify.cpp:3331` and `tools/pancake/abtest.py`, not `/home/dan/x.py`, `bench/results/2026-11-17-a/b.json`, "
            "`caminus:docs/dev/x.md`, `src/<x>/y.cpp`, `STRATA_FOO`, `a b/c.py`, `docs/*.md`, `include/strata/{a,b}.hpp`, `README.md`")
    assert D.cited_paths(text) == {"src/core/verify.cpp", "tools/pancake/abtest.py"}


def test_missing_refs_in_a_fake_tree(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.cpp").write_text("x")
    (tmp_path / "LEADS.md").write_text("`src/a.cpp` and `src/gone.cpp` and `pancake-docs/OLD.md`\n")
    assert D.missing_refs(tmp_path, docs=("LEADS.md",)) == ["pancake-docs/OLD.md", "src/gone.cpp"]


def test_baseline_only_shrinks():
    new, fixed = D.apply_baseline({"md": ["a.md", "b.md"], "refs": []}, {"md": ["a.md", "gone.md"], "refs": ["r"]})
    assert new == {"md": ["b.md"]}
    assert fixed == {"md": ["gone.md"], "refs": ["r"]}


def test_real_tree_has_no_new_failure():
    new, _ = D.apply_baseline(D.run(["ledger", "specs"]), D.load_baseline())
    assert new == {}, new
