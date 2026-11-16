"""STRATA_TRAILING_REMINDER (opt-in; LEADS.md FK-27 "trailing reminder", Ben1078 8b90ab8077): which final user message
counts as a transient reminder.  No engine, no model."""
import os
import unittest
from unittest import mock

from serve.server import _trailing_reminder

REM = "<system-reminder>memory: a, b</system-reminder>"


def conv(last):
    return [{"role": "system", "content": "s"}, {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"}, last]


class TrailingReminder(unittest.TestCase):
    def test_off_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STRATA_TRAILING_REMINDER", None)
            self.assertIsNone(_trailing_reminder(conv({"role": "user", "content": REM})))

    def test_on(self):
        with mock.patch.dict(os.environ, {"STRATA_TRAILING_REMINDER": "1"}):
            self.assertEqual(_trailing_reminder(conv({"role": "user", "content": "  " + REM + "\n"})), REM)
            parts = [{"type": "text", "text": REM}, {"type": "text", "text": "\n" + REM}]
            self.assertEqual(_trailing_reminder(conv({"role": "user", "content": parts})), REM + "\n" + REM)
            # anything besides the reminder: a real question, not moved
            self.assertIsNone(_trailing_reminder(conv({"role": "user", "content": REM + " and fix the bug"})))
            # no reminder, not a user message, an image part, too short a conversation
            self.assertIsNone(_trailing_reminder(conv({"role": "user", "content": "hello"})))
            self.assertIsNone(_trailing_reminder(conv({"role": "assistant", "content": REM})))
            self.assertIsNone(_trailing_reminder(conv({"role": "user", "content": [{"type": "image_url"}]})))
            self.assertIsNone(_trailing_reminder([{"role": "user", "content": REM}]))
            self.assertIsNone(_trailing_reminder(None))


if __name__ == "__main__":
    unittest.main()
