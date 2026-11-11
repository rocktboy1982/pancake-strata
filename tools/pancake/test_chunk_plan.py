#!/usr/bin/env python3
"""Tests for chunk_plan.py (MOD_PLAN 17.5): the same cases as tests/program/prefill_plan_test.cpp, plus the cuts.

    nice -n 19 python3 tools/pancake/test_chunk_plan.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chunk_plan as cp  # noqa: E402

C, S = 32768, 4096


def chunks(n, **kw):
    """The batched chunks of the main segment of a single-message prompt of n tokens (header 3)."""
    parts = [p for p in cp.plan(n, **kw) if p["path"] == "batched"]
    assert len(parts) == 1, parts
    return parts[0]["chunks"]


class ChunkPlan(unittest.TestCase):
    def test_today(self):
        self.assertEqual(chunks(32768), [(32765, "streamed")])
        self.assertEqual(chunks(35000), [(32768, "streamed"), (2229, "staged")])
        self.assertEqual(chunks(65537), [(32768, "streamed"), (32766, "streamed")])   # 3 chunks only from 65,540
        self.assertEqual(chunks(65540)[-1], (1, "staged"))
        self.assertEqual(chunks(70000), [(32768, "streamed")] * 2 + [(4461, "streamed")])
        self.assertEqual(chunks(98305), [(32768, "streamed")] * 2 + [(32766, "streamed")])
        self.assertEqual(chunks(100000), [(32768, "streamed")] * 3 + [(1693, "staged")])
        self.assertEqual(chunks(102400), [(32768, "streamed")] * 3 + [(4093, "staged")])
        self.assertEqual(chunks(128000), [(32768, "streamed")] * 3 + [(29693, "streamed")])

    def test_equal(self):
        self.assertEqual(chunks(49152, equal=True), [(24576, "streamed"), (24573, "streamed")])
        self.assertEqual(chunks(70000, equal=True), [(23552, "streamed")] * 2 + [(22893, "streamed")])
        self.assertEqual(chunks(100000, equal=True)[-1], (1693, "staged"))   # EQUAL leaves a staged tail alone

    def test_equal_tail(self):
        t = cp.EQUAL_TAIL_MIN_TAIL
        self.assertEqual(chunks(100000, tail_min=t), [(25088, "streamed")] * 3 + [(24733, "streamed")])
        self.assertEqual(chunks(102400, tail_min=t), [(25600, "streamed")] * 3 + [(25597, "streamed")])
        self.assertEqual(chunks(66000, tail_min=t), [(22016, "streamed")] * 2 + [(21965, "streamed")])
        self.assertEqual(chunks(35000, tail_min=t), [(17664, "streamed"), (17333, "streamed")])
        self.assertEqual(chunks(70000, tail_min=t)[-1], (4461, "streamed"))   # a streamed tail is EQUAL's case
        self.assertEqual(chunks(65540, tail_min=t)[-1], (1, "staged"))        # below the minimum tail
        self.assertEqual(chunks(65540, tail_min=1), [(22016, "streamed")] * 2 + [(21505, "streamed")])

    def test_rule_matches_cpp_cases(self):
        f = cp.equal_tail_chunk
        self.assertEqual(f(99997, C, S), 25088)
        self.assertEqual(f(102397, C, S), 25600)
        self.assertEqual(f(65997, C, S), 22016)
        self.assertEqual(f(34997, C, S), 17664)
        for n in (32765, 32768, 49149, 69997, 127997, 65537, 98305, 2 * C + 255):
            self.assertEqual(f(n, C, S), 0, n)
        self.assertEqual(f(2 * C + 256, C, S), 22016)
        self.assertEqual(f(65537, C, S, 1), 22016)
        self.assertEqual(f(8192 + 1000, 8192, 1024), 0)
        self.assertEqual(f(8192 + 25, 8192, 1024), 0)
        self.assertEqual(f(2 * C + 5000, C, 20000), 23552)
        self.assertEqual(f(2 * C + 5000, C, 24000), 0)
        self.assertEqual(f(99997, 31488, S), 0)
        self.assertEqual(f(3 * 31488 + 2000, 31488, S), 24320)
        self.assertEqual(cp.chunk_lengths(3 * 31488 + 2000, 24320), [24320, 24320, 24320, 23504])

    def test_rule_invariants(self):
        for cap in (8192, 16384, 31488, 32768):
            for smin in (1024, 4096):
                for n in range(1, 140000, 13):
                    c = cp.equal_tail_chunk(n, cap, smin)
                    k = -(-n // cap)
                    last = n - (k - 1) * cap
                    staged_tail = k >= 2 and smin > last >= cp.EQUAL_TAIL_MIN_TAIL
                    if c == 0:
                        even = min(cap, cp.round256(-(-n // k)))
                        self.assertTrue(not staged_tail or even < cp.EQUAL_TAIL_MIN_CHUNK or even < smin
                                        or n - (k - 1) * even < smin, (n, cap, smin))
                        continue
                    self.assertTrue(staged_tail)
                    self.assertTrue(c % 256 == 0 and cp.EQUAL_TAIL_MIN_CHUNK <= c <= cap and c >= smin)
                    k2 = -(-n // c)
                    self.assertEqual(k2, k)
                    self.assertTrue(smin <= n - (k2 - 1) * c <= c)

    def test_cuts(self):
        # agent shape: the system prompt's end is a segment of its own, so the grid moves
        parts = cp.plan(100000, root=12000)
        self.assertEqual([(p["from"], p["to"]) for p in parts], [(0, 12000), (12000, 99997), (99997, 99999)])
        self.assertEqual(parts[1]["chunks"], [(32768, "streamed"), (32768, "streamed"), (22461, "streamed")])
        self.assertEqual(parts[2]["path"], "windows")
        # a root below --prompt-cache-root is not a cut
        self.assertEqual(len(cp.plan(50000, root=1000)), 2)


if __name__ == "__main__":
    unittest.main()
