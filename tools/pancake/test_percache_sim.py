"""Tests for percache_sim.py: the store loader on a synthetic opencode database, the cause labels, the park model's rules
(budget, pinned entries never leave, disk tier) and the chain micro model.  No real history needed."""
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import percache_sim as P  # noqa: E402


def make_db(path, msgs, parts=(), parents=None):
    con = sqlite3.connect(path)
    con.execute("create table session (id text, parent_id text)")
    con.execute("create table message (id text, session_id text, time_created integer, data text)")
    con.execute("create table part (message_id text, time_created integer, id text, data text)")
    for sid in {m[1] for m in msgs}:
        con.execute("insert into session values (?, ?)", (sid, (parents or {}).get(sid)))
    for i, (mid, sid, t, data) in enumerate(msgs):
        con.execute("insert into message values (?,?,?,?)", (mid, sid, t, json.dumps(data)))
    for i, (mid, t, data) in enumerate(parts):
        con.execute("insert into part values (?,?,?,?)", (mid, t, f"p{i}", json.dumps(data)))
    con.commit()
    con.close()


def asst(total, reused, out, t, agent="build", model="qwen3.8-flash-next-iq2_xs"):
    return {"role": "assistant", "modelID": model, "agent": agent,
            "tokens": {"input": total - reused, "output": out, "cache": {"read": reused}},
            "time": {"created": t, "completed": t + 4000}}


class Loader(unittest.TestCase):
    def test_load_and_label(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "o.db")
            make_db(db, [
                ("m1", "A", 1000, asst(20000, 0, 300, 1000)),
                ("u1", "A", 20000, {"role": "user", "time": {"created": 20000}}),
                ("m2", "A", 21000, asst(21000, 20299, 300, 21000)),          # continues after a user message
                ("m3", "B", 40000, asst(9000, 0, 100, 40000, agent="explore")),
                ("m4", "A", 60000, asst(21700, 0, 100, 60000)),              # B ran between: a switch, read cold
                ("m5", "A", 90000, asst(5000, 0, 100, 90000)),               # the prompt shrank: compaction part below
                ("x1", "A", 95000, asst(100, 0, 1, 95000, model="other-model")),
            ], parts=[("m5", 89000, {"type": "compaction"})], parents={"B": "A"})
            steps, ev = P.load_steps(db)
            P.annotate(steps, ev)
        self.assertEqual(len(steps), 5)                                  # the other model's message is left out
        self.assertEqual([s["kind"] for s in steps], ["first", "cont", "first", "cont", "shrink"])
        self.assertEqual(steps[1]["cause"], "after a new user message")
        self.assertEqual(steps[3]["cause"], "switch (another session ran between)")
        self.assertEqual(steps[3]["excess"], 21700 - (21700 - (21000 + 300)))   # all but the 400 new tokens
        self.assertEqual(steps[4]["cause"], "compaction or prune (prompt shrank)")
        self.assertTrue(steps[2]["sub"])
        c = P.classify(steps)
        self.assertEqual(c["sessions"], 2)
        self.assertGreater(c["reread"]["switch (another session ran between)"], 20000)

    def test_missing_tokens_or_completion_are_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "o.db")
            bad = asst(0, 0, 0, 1000)
            nodone = asst(100, 0, 1, 2000)
            nodone["time"] = {"created": 2000}
            make_db(db, [("m1", "A", 1000, bad), ("m2", "A", 2000, nodone)])
            steps, _ = P.load_steps(db)
        self.assertEqual(steps, [])


class Model(unittest.TestCase):
    def test_self_test(self):
        P.self_test()

    def test_pinned_entries_block_parking_only_in_the_pin_arm(self):
        ev = dict(users={}, comp={}, prune={})
        import collections
        ev = {k: collections.defaultdict(list) for k in ev}
        mk = lambda sid, t, tot, out=100: P._mk(sid, sid.lower(), t, tot, 0, out)   # noqa: E731
        steps = [mk(s, i, 20000) for i, s in enumerate("ABCDEF")] + [mk("A", 6, 20200)]
        P.annotate(steps, ev)
        roots = P.family_roots(steps)
        out = {}
        for nm in ("base", "pin"):
            st = [dict(s) for s in steps]
            P.simulate(st, P.ARMS[nm](), roots)
            out[nm] = st[-1]["sim_src"]
        self.assertIn(out["base"], ("cold", "root"))                     # slots 4: A was the oldest and left
        self.assertEqual(out["pin"], "parked")                           # the first four stay, whatever comes later

    def test_disk_tier_catches_what_ram_evicted(self):
        import collections
        ev = {k: collections.defaultdict(list) for k in ("users", "comp", "prune")}
        mk = lambda sid, t, tot, out=100: P._mk(sid, sid.lower(), t, tot, 0, out)   # noqa: E731
        steps = [mk(s, i, 20000) for i, s in enumerate("ABCDEF")] + [mk("A", 6, 20200)]
        P.annotate(steps, ev)
        st = [dict(s) for s in steps]
        P.simulate(st, P.ARMS["flags+disk"](), P.family_roots(steps))
        self.assertEqual(st[-1]["sim_src"], "disk")
        self.assertGreater(st[-1]["sim_restore_s"], 0.9)                 # read back + copy, not free

    def test_snapshot_size_model(self):
        self.assertAlmostEqual(P.snapshot_mib(60000, 4), 111 + 14.9 / 1024 * 60000 + 4 * 113, places=6)
        self.assertEqual(P.n_checkpoints(100000, 11000, 0), 6)           # capped at --prompt-cache

    def test_micro_orders(self):
        lru = P.micro("lru", False, trials=100)
        spread = P.micro("spread", False, trials=100)
        self.assertLess(spread[0], lru[0])                               # random place: thinned checkpoints read less again
        cold_off, cold_on = P.micro("lru", False, trials=100, cold=True), P.micro("lru", True, trials=100, cold=True)
        self.assertLessEqual(cold_on[2], cold_off[2])                    # marks help the last 8K ...
        self.assertGreaterEqual(cold_on[0], cold_off[0])                 # ... and cost the periodic ones


if __name__ == "__main__":
    unittest.main()
