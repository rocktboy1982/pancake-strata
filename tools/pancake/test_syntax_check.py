#!/usr/bin/env python3
"""Tests for syntax_check.py (pure parts and a real compile with the system compiler). Run: python3 -I tools/pancake/test_syntax_check.py"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import syntax_check as S  # noqa: E402


class Pure(unittest.TestCase):
    def test_command_loses_output_and_dep_flags(self):
        out = S.syntax_command("/usr/bin/c++ -DX=1 -I/a -MD -MT t.o -MF t.o.d -o t.o -c /src/f.cpp")
        self.assertEqual(out, ["/usr/bin/c++", "-DX=1", "-I/a", "/src/f.cpp", "-fsyntax-only"])

    def test_quoted_define_survives(self):
        out = S.syntax_command('cc -DV=\\"0.1\\" -c f.c -o f.o')
        self.assertIn('-DV="0.1"', out)


class Compile(unittest.TestCase):
    def run_cc(self, src):
        d = tempfile.mkdtemp(prefix="sc-")
        f = os.path.join(d, "f.cpp")
        open(f, "w").write(src)
        return subprocess.run(S.syntax_command(f"c++ -c {f} -o {f}.o"), capture_output=True, text=True).returncode

    def test_good_file_passes(self):
        self.assertEqual(self.run_cc('#include <cstdio>\nint main(){ std::fprintf(stderr, "a\\n"); }\n'), 0)

    def test_line_break_in_string_literal_fails(self):
        self.assertNotEqual(self.run_cc('#include <cstdio>\nint main(){ std::fprintf(stderr, "a\n"); }\n'), 0)

    def test_lost_closing_brace_fails(self):
        self.assertNotEqual(self.run_cc("int f(){ if (1) { return 1; }\n"), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
