"""Checks the ClassLRU replay (tools/replay_classlru.py) on tiny synthetic traces: the properties that define the policy."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay_classlru as rc  # noqa: E402


def cache(nl=2, ne=8, resident=(), **kw):
    return rc.ClassLRU(nl, ne, [l * ne + e for l, e in resident], **kw)


class ClassLRUTests(unittest.TestCase):
    def test_class_is_floor_log2_of_crf(self):
        c = cache()
        for crf, want in [(1.0, 0), (1.9, 0), (2.0, 1), (4.0, 2), (200.0, 7), (1e6, 7)]:
            c.crf[3] = crf
            self.assertEqual(c.class_of(3), want, crf)

    def test_use_decays_with_the_half_life(self):
        c = cache(half_life=128.0)
        c.use(5)                      # crf 1 at clock 0
        c.clock[0] = 128.0            # one half-life later
        self.assertAlmostEqual(c.decayed(5), 0.5, places=6)
        c.use(5)
        self.assertAlmostEqual(c.crf[5], 1.5, places=6)

    def test_every_miss_is_admitted_and_slots_are_conserved(self):
        c = cache(resident=[(0, 0), (0, 1), (0, 2)])
        before = len(c.cls)
        hits, n, swaps = c.step(0, np.array([[5, 6]]), None)
        self.assertEqual((hits, n, swaps), (0, 2, 2))
        self.assertEqual(len(c.cls), before)                # slots conserved
        self.assertIn(5, c.cls)
        self.assertIn(6, c.cls)

    def test_victim_is_never_in_the_plan(self):
        c = cache(resident=[(0, 0), (0, 1), (0, 2)])
        c.step(0, np.array([[0, 1, 4]]), None)              # 0 and 1 hit; 4 must evict 2, the only slot outside the plan
        self.assertIn(0, c.cls)
        self.assertIn(1, c.cls)
        self.assertIn(4, c.cls)
        self.assertNotIn(2, c.cls)

    def test_hit_is_counted_before_admission(self):
        c = cache(resident=[(0, 0), (0, 1)])
        self.assertEqual(c.step(0, np.array([[7]]), None)[0], 0)   # a miss is not a hit even though it is then admitted
        self.assertEqual(c.step(0, np.array([[7]]), None)[0], 1)   # the next access hits

    def test_budget_caps_the_admissions(self):
        c = cache(resident=[(0, 0), (0, 1), (0, 2), (0, 3)])
        _, _, swaps = c.step(0, np.array([[4, 5, 6]]), 1)
        self.assertEqual(swaps, 1)

    def test_frequent_experts_outlive_one_off_experts(self):
        c = cache(resident=[(0, 0), (0, 1)])
        for _ in range(8):                                   # expert 0 becomes a frequent one (class >= 2)
            c.step(0, np.array([[0]]), None)
        c.step(0, np.array([[5]]), None)                     # a one-off arrives: it must take expert 1's slot, not 0's
        self.assertIn(0, c.cls)
        self.assertNotIn(1, c.cls)

    def test_simulate_conserves_the_slot_count(self):
        rng = np.random.default_rng(0)
        windows = [[(l, rng.integers(0, 8, size=(2, 3))) for l in range(2)] for _ in range(30)]
        init = np.array([[0, 0], [0, 1], [1, 2], [1, 3]])
        r = rc.simulate_classlru(windows, init, nl=2, ne=8)
        self.assertEqual(r["slots"], 4)
        self.assertGreaterEqual(r["share"], 0.0)
        self.assertLessEqual(r["share"], 1.0)


if __name__ == "__main__":
    unittest.main()
