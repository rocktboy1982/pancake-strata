"""Phase 17 review (pancake-docs/REVIEW_P17.md, findings R7 and R8): regression tests for serve/park_divergence.py.

R7: once the tracker has turned itself off (an error in the log line: `broken`), begin() and end() return at once and
never clear `_gen`, but token() still appends every streamed token to it, so the list grows for the rest of the
server's life.  The test is marked expectedFailure until the fix; remove the decorator with it (unittest then reports
an unexpected success as a failure, so the marker cannot be forgotten).

Run: python -m pytest serve/test_park_divergence_review.py"""
import unittest

from serve import park_divergence as pd


def broken_tracker():
    tr = pd.DivergenceTracker(decode=None, emit=lambda line: (_ for _ in ()).throw(RuntimeError("log full")))
    tr.begin(1, [1, 2, 3])
    tr.resumed(0)
    for t in (4, 5):
        tr.token(t)
    tr.end()
    tr.begin(1, [1, 2, 9])           # differs from 1,2,3,4,5: the emit raises, the tracker turns itself off
    tr.resumed(2)
    assert tr.broken
    return tr


class BrokenTracker(unittest.TestCase):
    @unittest.expectedFailure
    def test_a_broken_tracker_keeps_no_tokens(self):
        tr = broken_tracker()
        for _ in range(3):           # three more requests of 1,000 tokens each after it broke
            tr.begin(1, [7])
            for t in range(1000):
                tr.token(1000 + t)
            tr.end()
        held = len(tr._gen or [])
        self.assertEqual(held, 0, f"a turned-off tracker still holds {held} streamed tokens")

    def test_a_broken_tracker_never_raises(self):
        tr = broken_tracker()
        tr.begin(1, [7])
        tr.token(8)
        self.assertIsNone(tr.resumed(1))
        tr.end()
        tr.forget()


if __name__ == "__main__":
    unittest.main()
