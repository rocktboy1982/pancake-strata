"""Tests for suffix_sim.py (A24) without the opencode store or the tokenizer: the drafters, eviction, a tiny session
store in SQLite and a replay over it.  python -m pytest tools/pancake/test_suffix_sim.py  (or run this file)"""
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import suffix_sim as SS  # noqa: E402


def test_self_test():
    assert SS.self_test() == 0


def test_cap_tokens_matches_the_engine_ring():
    assert SS.cap_tokens(64) == 1 << 20          # 64 MiB at 40 bytes a token, a power of two
    assert SS.cap_tokens(0) == 4096
    assert SS.cap_tokens(256) == 1 << 22


def test_lookup_queries_as_of_a_place():
    L = SS.Lookup()
    L.extend([1, 2, 3, 9, 1, 2, 3, 8, 1, 2, 3])
    # as of the full history the newest earlier "1 2 3" (followed by 8) wins over the older one (followed by 9)
    assert L.propose(len(L.h), 3)[0][:1] == [8]
    # as of the first 7 tokens only the first occurrence exists
    assert L.propose(7, 3)[0] == [9, 1, 2]


def make_store(path):
    con = sqlite3.connect(path)
    con.execute("create table session (id text primary key, parent_id text)")
    con.execute("create table message (id text primary key, session_id text, time_created integer, data text)")
    con.execute("create table part (id text primary key, message_id text, session_id text, time_created integer, data text)")
    con.execute("insert into session values ('main', null), ('sub1', 'main'), ('sub2', 'main')")
    reply = "I will read the file and then edit the function to fix the bug in the parser"
    n = 0
    for t0, sid in ((100, "main"), (200, "sub1"), (300, "sub2")):
        con.execute("insert into message values (?, ?, ?, ?)", (f"u{sid}", sid, t0, json.dumps({"role": "user"})))
        con.execute("insert into part values (?, ?, ?, ?, ?)",
                    (f"p{n}", f"u{sid}", sid, t0, json.dumps({"type": "text", "text": f"task for {sid}"})))
        n += 1
        meta = {"role": "assistant", "providerID": "strata", "modelID": "qwen3.8-flash-next-iq2_xs"}
        con.execute("insert into message values (?, ?, ?, ?)", (f"a{sid}", sid, t0 + 1, json.dumps(meta)))
        for i, p in enumerate([{"type": "step-start"}, {"type": "reasoning", "text": reply},
                               {"type": "tool", "tool": "read", "state": {"input": {"path": "a.py"}, "output": "x = 1"}},
                               {"type": "step-finish"}, {"type": "step-start"}, {"type": "text", "text": "done " + reply}]):
            con.execute("insert into part values (?, ?, ?, ?, ?)", (f"p{n}", f"a{sid}", sid, t0 + 2 + i, json.dumps(p)))
            n += 1
    con.commit()
    con.close()


def test_replay_over_a_tiny_store():
    with tempfile.TemporaryDirectory() as d:
        db = str(Path(d) / "opencode.db")
        make_store(db)
        reqs = SS.load_turns(db, r"^strata/qwen3\.8")
        assert len(reqs) == 6                                   # 3 sessions x 2 steps
        assert [r[1] for r in reqs] == ["main", "main", "sub1", "sub1", "sub2", "sub2"]
        assert [r[2] for r in reqs] == [False, False, True, True, True, True]
        # the second step's prompt holds the tool's output; the output holds the tool call in the template's format
        assert any("<tool_response>\nx = 1" in p for p in reqs[1][3] if isinstance(p, str))
        assert "<function=read>\n<parameter=path>\na.py\n</parameter>" in reqs[0][4][0]
        tok = SS.Tok("/nonexistent")                            # the word-level stand-in
        rt = [(sid, sub, tok.encode(dl), tok.encode(o)) for _, sid, sub, dl, o in reqs]
        tot, _ = SS.simulate(rt, k=8, cap_tokens=1 << 16)
        # the sub-agents write what the main agent wrote: the shared index drafts it, the per-request lookup cannot
        # draft the first step of a new session
        assert tot["sub"]["index"]["accepted"] > 0
        assert tot["all"]["longest"]["accepted"] >= tot["all"]["lookup"]["accepted"]
        assert tot["all"]["longest"]["rounds"] <= tot["all"]["lookup"]["rounds"]
        assert SS.load_turns(db, r"^nothing$") == []


if __name__ == "__main__":
    test_self_test()
    test_cap_tokens_matches_the_engine_ring()
    test_lookup_queries_as_of_a_place()
    test_replay_over_a_tiny_store()
    print("test_suffix_sim: ok")
