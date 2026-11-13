#!/usr/bin/env python3
"""Tests for read_before_edit_hook.py (A41). Run: python3 -I /home/dan/bin/test_read_before_edit.py

The hook is a Claude Code PreToolUse hook. It gets the call as JSON on stdin (tool_name, tool_input, transcript_path)
and reads the session transcript (JSONL, assistant lines carry tool_use blocks in call order). It must:
  - allow an Edit of a file the session has already Read (earlier in the transcript);
  - block (exit 2, reason on stderr) an Edit of an existing file never Read in this session, in HOLD mode;
  - warn only (exit 0, warning on stderr) in WARN mode;
  - allow Edit of a file that does not exist yet (nothing to read);
  - not count an earlier Edit as a Read;
  - match paths written differently (relative to the cwd vs absolute, trailing slashes).
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "read_before_edit_hook.py")


def tool_line(name, path):
    return json.dumps({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": name, "input": {"file_path": path}}]}}) + "\n"


class HookCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="rbe-")
        self.target = os.path.join(self.dir, "target.py")
        with open(self.target, "w", encoding="utf-8") as f:
            f.write("x = 1\n")
        self.transcript = os.path.join(self.dir, "session.jsonl")

    def write_transcript(self, lines):
        with open(self.transcript, "w", encoding="utf-8") as f:
            f.writelines(lines)

    def run_hook(self, file_path, mode="hold", tool="Edit"):
        env = dict(os.environ, RBE_MODE=mode)
        payload = {"tool_name": tool, "tool_input": {"file_path": file_path},
                   "transcript_path": self.transcript, "cwd": self.dir, "session_id": "t"}
        p = subprocess.run([sys.executable, "-I", HOOK], input=json.dumps(payload), capture_output=True,
                           text=True, env=env, timeout=60)
        return p.returncode, p.stdout, p.stderr


class ReadBeforeEdit(HookCase):
    def test_read_then_edit_is_allowed(self):
        self.write_transcript([tool_line("Read", self.target)])
        rc, _, err = self.run_hook(self.target)
        self.assertEqual(rc, 0, err)

    def test_edit_without_read_is_blocked_in_hold_mode(self):
        self.write_transcript([])
        rc, _, err = self.run_hook(self.target, mode="hold")
        self.assertEqual(rc, 2)
        self.assertIn("not read", err.lower())

    def test_edit_without_read_only_warns_in_warn_mode(self):
        self.write_transcript([])
        rc, _, err = self.run_hook(self.target, mode="warn")
        self.assertEqual(rc, 0)
        self.assertIn("not read", err.lower())

    def test_new_file_is_allowed(self):
        self.write_transcript([])
        rc, _, err = self.run_hook(os.path.join(self.dir, "new.py"), mode="hold")
        self.assertEqual(rc, 0, err)

    def test_an_earlier_edit_is_not_a_read(self):
        self.write_transcript([tool_line("Edit", self.target)])
        rc, _, _ = self.run_hook(self.target, mode="hold")
        self.assertEqual(rc, 2)

    def test_relative_and_absolute_paths_match(self):
        self.write_transcript([tool_line("Read", "target.py")])
        rc, _, err = self.run_hook(self.target, mode="hold")
        self.assertEqual(rc, 0, err)

    def test_read_tool_itself_is_not_checked(self):
        self.write_transcript([])
        rc, _, _ = self.run_hook(self.target, mode="hold", tool="Read")
        self.assertEqual(rc, 0)

    def test_write_is_checked_like_edit(self):
        self.write_transcript([])
        rc, _, _ = self.run_hook(self.target, mode="hold", tool="Write")
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
