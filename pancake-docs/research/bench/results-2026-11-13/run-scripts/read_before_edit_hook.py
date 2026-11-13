#!/usr/bin/env python3
"""A41 read-before-edit check: a Claude Code PreToolUse hook for Edit, MultiEdit and Write.

Reads the call as JSON on stdin, scans the session transcript for an earlier Read of the same file, and:
  - allows the call when the file was read earlier in this session, or does not exist yet;
  - HOLD mode (RBE_MODE=hold, default): blocks the call (exit 2, reason on stderr);
  - WARN mode (RBE_MODE=warn): allows the call and prints a warning on stderr.
Read calls and other tools pass through untouched. Tests: /home/dan/bin/test_read_before_edit.py.
"""
import json
import os
import sys

CHECKED = {"Edit", "MultiEdit", "Write"}


def norm(path, cwd):
    if not path:
        return ""
    p = os.path.expanduser(path)
    if not os.path.isabs(p):
        p = os.path.join(cwd or os.getcwd(), p)
    return os.path.normpath(p).rstrip("/")


def files_read_before(transcript_path, cwd, stop_at=None):
    """Paths the session Read before the current call (the transcript is in call order)."""
    read = set()
    if not transcript_path or not os.path.exists(transcript_path):
        return read
    with open(transcript_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            content = (d.get("message") or {}).get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name") == "Read":
                    fp = (block.get("input") or {}).get("file_path")
                    if fp:
                        read.add(norm(fp, cwd))
    return read


def main():
    call = json.load(sys.stdin)
    tool = call.get("tool_name", "")
    if tool not in CHECKED:
        return 0
    cwd = call.get("cwd") or os.getcwd()
    target = norm((call.get("tool_input") or {}).get("file_path", ""), cwd)
    if not target or not os.path.exists(target):
        return 0
    if target in files_read_before(call.get("transcript_path"), cwd):
        return 0
    mode = os.environ.get("RBE_MODE", "hold").lower()
    msg = (f"read-before-edit: {target} was not read in this session. "
           f"Read it first, then edit.")
    if mode == "warn":
        print(msg, file=sys.stderr)
        return 0
    print(msg, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
