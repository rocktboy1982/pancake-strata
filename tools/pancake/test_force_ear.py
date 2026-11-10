"""A56: EAR (expected acceptance rate, SLQ) next to the same-top-token rate in tools/force_check.py.  Synthetic TSVs, no engine.

Run: python -m pytest tools/pancake/test_force_ear.py   (or python tools/pancake/test_force_ear.py)"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import force_check as fc  # noqa: E402


def write_arm(path, requests):
    """requests: {rid: [(forced id, {id: probability})]}; the argmax is the likeliest listed id.  Prompt of 3 tokens, positions 2.."""
    lines, items = [], []
    for rid, rows in requests.items():
        first = len(lines)
        ids = [1, 1, 1] + [t for t, _ in rows]
        for k, (tgt, d) in enumerate(rows):
            pos = 2 + k
            am = max(d, key=d.get)
            top = sorted(d, key=d.get, reverse=True)
            cols = [str(pos), str(tgt), "%.9f" % math.log(d.get(tgt, 1e-9)), str(am), "%.9f" % math.log(d[am]), str(int(am == tgt)), "nan", "nan"]
            cols += ["%d:%.6f" % (i, math.log(d[i])) for i in top]
            lines.append("\t".join(cols))
        items.append({"id": rid, "kind": "code", "first_line": first, "end_line": len(lines), "prompt_len": 3, "ids": ids})
    Path(path).write_text("\n".join(lines) + "\n")
    Path(str(path) + ".meta.json").write_text(json.dumps({"items": items}))


class EarPosition(unittest.TestCase):
    def test_identical_is_one(self):
        d = {1: 0.5, 2: 0.3, 3: 0.2}
        self.assertAlmostEqual(fc.ear_position(d, dict(d), 10)[0], 1.0)

    def test_overlap_is_sum_of_minima(self):
        ref = {1: 0.5, 2: 0.3, 3: 0.2}
        cand = {1: 0.25, 2: 0.45, 3: 0.30}
        ear, miss = fc.ear_position(ref, cand, 10)
        self.assertAlmostEqual(ear, 0.25 + 0.30 + 0.20)
        self.assertEqual(miss, 0)

    def test_renormalised_over_the_reference_top_k(self):
        ref = {1: 0.6, 2: 0.2, 3: 0.1, 4: 0.1}
        cand = {1: 0.3, 2: 0.3, 3: 0.2, 4: 0.2}
        ear, _ = fc.ear_position(ref, cand, 2)   # S = {1, 2}: p = (.75, .25), q = (.5, .5)
        self.assertAlmostEqual(ear, 0.5 + 0.25)

    def test_unlisted_id_takes_the_smallest_listed_probability_and_is_counted(self):
        ref = {1: 0.7, 2: 0.3}
        cand = {1: 0.9, 3: 0.1}                  # id 2 is not listed: 0.1 (an upper bound)
        ear, miss = fc.ear_position(ref, cand, 2)
        self.assertEqual(miss, 1)
        self.assertAlmostEqual(ear, min(0.7, 0.9 / 1.0) + min(0.3, 0.1 / 1.0))

    def test_never_above_one(self):
        ref = {1: 0.5, 2: 0.5}
        cand = {7: 0.5, 8: 0.5}
        self.assertLessEqual(fc.ear_position(ref, cand, 2)[0], 1.0 + 1e-12)


class EarCompare(unittest.TestCase):
    def arms(self):
        a = {"r0": [(5, {5: 0.8, 6: 0.2}), (6, {6: 0.6, 5: 0.4}), (7, {7: 0.9, 8: 0.1})],
             "r1": [(5, {5: 0.5, 6: 0.5}), (5, {5: 0.7, 6: 0.3})]}
        # the candidate flips the argmax of one position and softens another
        b = {"r0": [(5, {5: 0.8, 6: 0.2}), (6, {5: 0.55, 6: 0.45}), (7, {7: 0.9, 8: 0.1})],
             "r1": [(5, {5: 0.5, 6: 0.5}), (5, {5: 0.5, 6: 0.5})]}
        return a, b

    def test_identical_arms(self):
        with tempfile.TemporaryDirectory() as d:
            a, _ = self.arms()
            write_arm(Path(d) / "a.tsv", a)
            write_arm(Path(d) / "b.tsv", a)
            r = fc.ear_compare(fc.load_dists(str(Path(d) / "a.tsv")), fc.load_dists(str(Path(d) / "b.tsv")), 10)
            self.assertAlmostEqual(r["ear"], 1.0, places=5)
            self.assertAlmostEqual(r["same_top_token"], 1.0)
            self.assertEqual((r["requests"], r["positions"]), (2, 5))

    def test_different_arms_numbers(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = self.arms()
            write_arm(Path(d) / "a.tsv", a)
            write_arm(Path(d) / "b.tsv", b)
            r = fc.ear_compare(fc.load_dists(str(Path(d) / "a.tsv")), fc.load_dists(str(Path(d) / "b.tsv")), 10, boots=200)
            # positions: 1.0, (min(.4,.55)+min(.6,.45)=.85), 1.0 | 1.0, (min(.7,.5)+min(.3,.5)=.8)  ->  4.65 / 5
            self.assertAlmostEqual(r["ear"], 4.65 / 5, places=4)
            self.assertAlmostEqual(r["same_top_token"], 4 / 5)   # position 2 of r0 flips its argmax
            self.assertLessEqual(r["ear_ci95"][0], r["ear"])
            self.assertGreaterEqual(r["ear_ci95"][1], r["ear"])
            self.assertEqual(r["substituted_ids"], 0)

    def test_cli_ear_and_check_flag(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = self.arms()
            for n, arm in (("s1", a), ("s2", a), ("c", b)):
                write_arm(Path(d) / (n + ".tsv"), arm)
            out = Path(d) / "ear.json"
            ns = type("A", (), {"stock": str(Path(d) / "s1.tsv"), "cand": str(Path(d) / "c.tsv"), "topk": 10, "json": str(out)})
            self.assertEqual(fc.cmd_ear(ns), 0)
            self.assertAlmostEqual(json.loads(out.read_text())["ear"], 4.65 / 5, places=4)
            ck = type("A", (), {"stock": [str(Path(d) / "s1.tsv"), str(Path(d) / "s2.tsv")], "cand": str(Path(d) / "c.tsv"),
                                "delta": None, "json": None, "ear": True, "topk": 10})
            fc.cmd_check(ck)   # prints; must not raise


if __name__ == "__main__":
    unittest.main()
