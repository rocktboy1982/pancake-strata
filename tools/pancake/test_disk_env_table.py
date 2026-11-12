"""Phase 17 review (pancake-docs/REVIEW_P17.md, finding R1): every "STRATA_NAME" string literal in src/ and include/ must
have a line in src/core/conversation_disk_env.def.  This is the source-scan half of the C++ ctest
conversation_disk_env_test (tests/core/conversation_disk_env_test.cpp), in pure Python, so it can be run without a
build (the daytime rule).  The C++ test stays the authority; keep the two scans the same if it changes.

On dev/p17/integration (17b11ef85) this FAILS: STRATA_KV_RING_ADAPT, STRATA_PARK_REWIND_MAX, STRATA_ROUTE_PRIOR_AUTO and
STRATA_ROUTE_PRIOR_AUTO_MIN_T are read by src/program/generate.cpp but have no line in the table, so
`ctest -L cpu` fails on conversation_disk_env_test and the disk tier's key ignores them.

Run: python -m pytest tools/pancake/test_disk_env_table.py   (or python tools/pancake/test_disk_env_table.py)"""
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXTS = {".cpp", ".hpp", ".h", ".cu", ".cuh", ".inc", ".inl", ".c", ".cc", ".mm"}
# the C++ scan: a '"STRATA_' opening quote, then [A-Z0-9_]+, then a closing quote ("STRATA_" alone is not a name)
LITERAL = re.compile(rb'"(STRATA_[A-Z0-9_]+)"')
TABLE_LINE = re.compile(r'^STRATA_ENV\("([A-Z0-9_]+)"', re.M)


def names_in_sources(root: Path) -> dict:
    found = {}
    for d in ("src", "include"):
        for p in sorted((root / d).rglob("*")):
            if p.suffix not in EXTS or not p.is_file():
                continue
            for m in LITERAL.finditer(p.read_bytes()):
                found.setdefault(m.group(1).decode(), str(p.relative_to(root)))
    return found


def names_in_table(root: Path) -> set:
    return set(TABLE_LINE.findall((root / "src/core/conversation_disk_env.def").read_text(encoding="utf-8")))


class DiskEnvTable(unittest.TestCase):
    def test_every_source_name_is_classified(self):
        sources, table = names_in_sources(ROOT), names_in_table(ROOT)
        self.assertGreater(len(sources), 500, "the scan found the engine's switches")
        missing = sorted(n for n in sources if n not in table)
        self.assertEqual(missing, [], "ADD A LINE to src/core/conversation_disk_env.def for: " +
                         ", ".join(f"{n} ({sources[n]})" for n in missing))

    def test_no_stale_table_line(self):
        sources, table = names_in_sources(ROOT), names_in_table(ROOT)
        self.assertEqual(sorted(table - set(sources)), [], "REMOVE these lines from conversation_disk_env.def")

    def test_phase17_switches_are_classified(self):
        # the Phase 17 switches by name, so a later rename cannot hide them from the check above
        table = names_in_table(ROOT)
        names = ("STRATA_KV_RING_ADAPT", "STRATA_PARK_REWIND_MAX", "STRATA_ROUTE_PRIOR_AUTO",
                 "STRATA_ROUTE_PRIOR_AUTO_MIN_T", "STRATA_VERIFY_DEFER_AUTO", "STRATA_VERIFY_DEFER_AUTO_MIN_T")
        self.assertEqual([n for n in names if n not in table], [], "Phase 17 switches without a table line")


if __name__ == "__main__":
    sys.exit(0 if unittest.main(exit=False).result.wasSuccessful() else 1)
