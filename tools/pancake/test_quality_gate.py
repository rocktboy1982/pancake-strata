"""Tests for quality_gate.py's set handling (pending continuations, --fill's merge, --dry-run, the refusal) and build_quality_set.py (the pure
parts, and the committed grown set).  No engine, no GPU.
Run: python -m pytest tools/pancake/test_quality_gate.py   (or python tools/pancake/test_quality_gate.py)"""
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_quality_set as Q  # noqa: E402
import quality_gate as G  # noqa: E402

OLD = json.load(open(G.ROOT / "bench/quality/set.json"))
TOKENIZER = json.load(open(G.ROOT / "bench/quality/flashnext-iq2_xs.json"))["tokenizer"]


def tiny_set(n=3, pending_ids=()):
    return {"config": "c", "env": [], "items": [{"id": f"r{i}", "kind": "k", "prompt": [1, 2, 3 + i], "cont": None if f"r{i}" in pending_ids else [4, 5]}
                                                for i in range(n)]}


def test_set_problems():
    assert G.set_problems(tiny_set()) == []
    assert G.set_problems(tiny_set(pending_ids=("r1",))) == []                 # a pending continuation is allowed
    bad = tiny_set()
    bad["items"][1]["id"] = "r0"
    bad["items"][2]["prompt"] = [1, G.VOCAB]
    bad["items"][0]["cont"] = []
    probs = "\n".join(G.set_problems(bad))
    assert "duplicated" in probs and "prompt must be" in probs and "cont must be" in probs
    assert any("at least 40" in p for p in G.set_problems(tiny_set(), min_requests=40))
    assert G.set_problems(OLD) == []


def test_old_items_must_come_first_and_unchanged():
    new = {**OLD, "items": OLD["items"] + [{"id": "x", "kind": "k", "prompt": [1], "cont": None}]}
    assert G.set_problems(new, OLD) == []
    moved = {**new, "items": [new["items"][1], new["items"][0]] + new["items"][2:]}
    assert len(G.set_problems(moved, OLD)) == 2
    edited = json.loads(json.dumps(new))
    edited["items"][3]["prompt"][5] += 1
    assert any("not copied verbatim" in p for p in G.set_problems(edited, OLD))
    assert any("fewer requests" in p for p in G.set_problems({**OLD, "items": OLD["items"][:5]}, OLD))


def test_pending_merge_and_summary():
    st = tiny_set(pending_ids=("r0", "r2"))
    assert [it["id"] for it in G.pending(st)] == ["r0", "r2"]
    full = G.merge_conts(st, {"items": [{"id": "r2", "kind": "k", "prompt": [9], "cont": [7, 8, 9]}, {"id": "r0", "cont": [6]}]})
    assert G.pending(full) == [] and full["items"][0]["cont"] == [6] and full["items"][2]["cont"] == [7, 8, 9]
    assert full["items"][2]["prompt"] == [1, 2, 5]                              # the prompt stays the set's, not the generator's
    assert st["items"][0]["cont"] is None                                       # the input is not modified
    with pytest.raises(ValueError):
        G.merge_conts(st, {"items": [{"id": "r0", "cont": [6]}]})
    assert "3 requests (2 without continuation)" in G.set_summary(st)
    assert G.estimate_seconds(OLD) > 60


def write_spec(d, base_exists=True):
    base = Path(d) / "base.json"
    if base_exists:
        base.write_text(json.dumps({"args": [], "env": {}}))
    spec = Path(d) / "spec.json"
    spec.write_text(json.dumps({"base": str(base), "arms": {"base": {}, "turbo": {"env": {"X": "1"}}}}))
    return spec


def run_main(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["quality_gate.py", *argv])
    monkeypatch.setattr(G.abtest, "stop_servers", lambda: (_ for _ in ()).throw(AssertionError("stop_servers must not run")))
    return G.main()


def test_dry_run_checks_the_spec_and_the_set_and_starts_nothing(monkeypatch, capsys):
    with tempfile.TemporaryDirectory() as d:
        spec, out = write_spec(d), Path(d) / "out"
        s2 = Path(d) / "set.json"
        s2.write_text(json.dumps(tiny_set(pending_ids=("r1",))))
        assert run_main([str(spec), "--out", str(out), "--set", str(s2), "--dry-run"], monkeypatch) == 1      # pending, no --fill
        assert "no continuation" in capsys.readouterr().out
        assert run_main([str(spec), "--out", str(out), "--set", str(s2), "--dry-run", "--fill"], monkeypatch) == 0
        text = capsys.readouterr().out
        assert "3 engine starts" not in text and "4 engine starts" in text and "gen of 1 continuations" in text and "dry run: ok" in text
        assert run_main([str(spec), "--out", str(out), "--dry-run"], monkeypatch) == 0                         # the old set: nothing pending
        assert not out.exists()
        (Path(d) / "x").mkdir()
        gone = write_spec(Path(d) / "x", base_exists=False)
        assert run_main([str(gone), "--out", str(out), "--dry-run"], monkeypatch) == 1


def test_a_set_with_pending_continuations_is_refused_without_fill(monkeypatch, capsys):
    with tempfile.TemporaryDirectory() as d:
        spec, out = write_spec(d), Path(d) / "out"
        s2 = Path(d) / "set.json"
        s2.write_text(json.dumps(tiny_set(pending_ids=("r1",))))
        assert run_main([str(spec), "--out", str(out), "--set", str(s2)], monkeypatch) == 2    # before any server is touched (stop_servers would raise)
        assert "--fill" in capsys.readouterr().out and not out.exists()
        s2.write_text(json.dumps({"items": [{"id": "a", "kind": "k", "prompt": []}]}))
        assert run_main([str(spec), "--out", str(out), "--set", str(s2)], monkeypatch) == 2    # a broken set is refused too
        assert "prompt must be" in capsys.readouterr().out


def fake_gen(argv, **kw):
    """Stands in for capped.sh -> force_check.py gen: writes a continuation for every row of the corpus file it is given."""
    class R:
        returncode, stdout, stderr = 0, "", ""
    out, corpus = argv[argv.index("gen") + 2], argv[argv.index("--corpus") + 1]
    rows = [json.loads(l) for l in open(corpus)]
    json.dump({"items": [{"id": r["id"], "kind": r["kind"], "prompt": r["ids"], "cont": [7, 8, 9]} for r in rows]}, open(out, "w"))
    fake_gen.argv = argv
    return R()


def test_fill_generates_only_the_pending_requests_with_the_stock_config(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        st = tiny_set(pending_ids=("r0", "r2"))
        monkeypatch.setattr(G.subprocess, "run", fake_gen)
        full = G.fill(st, "stock.json", out)
        argv = fake_gen.argv
        assert argv[argv.index("gen") + 1] == "stock.json" and argv[argv.index("--per-kind") + 1] == "2" and argv[argv.index("--offset") + 1] == "0"
        assert [json.loads(l)["id"] for l in open(out / "pending.jsonl")] == ["r0", "r2"]
        assert G.pending(full) == [] and full["items"][0]["cont"] == [7, 8, 9] and full["items"][1]["cont"] == [4, 5]
        assert json.load(open(out / "set.filled.json")) == full
        monkeypatch.setattr(G.subprocess, "run", lambda argv, **kw: type("R", (), {"returncode": 1, "stdout": "", "stderr": "boom"})())
        assert G.fill(st, "stock.json", out) is None


def test_main_with_fill_scores_with_the_filled_set(monkeypatch, capsys):
    with tempfile.TemporaryDirectory() as d:
        spec, out = write_spec(d), Path(d) / "out"
        s2 = Path(d) / "set.json"
        s2.write_text(json.dumps(tiny_set(pending_ids=("r1",))))
        seen = []
        monkeypatch.setattr(sys, "argv", ["quality_gate.py", str(spec), "--out", str(out), "--set", str(s2), "--fill"])
        monkeypatch.setattr(G.abtest, "stop_servers", lambda: None)
        monkeypatch.setattr(G.abtest, "quiet_gate", lambda *a, **k: None)
        monkeypatch.setattr(G.subprocess, "run", fake_gen)
        monkeypatch.setattr(G, "score", lambda cfg, set_path, tsv, topk=2: seen.append(set_path) or False)
        assert G.main() == 2                                              # nothing scored here: "stock not scored twice"
        assert seen and all(p == str(out / "set.filled.json") for p in seen)
        assert G.pending(json.load(open(out / "set.filled.json"))) == []
        assert "commit it as the set" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------------------- build_quality_set.py
def test_pick_ident_and_local_name_and_read_result():
    rng = random.Random(1)
    text = "int foo_bar = 1; int foo_bar2 = foo_bar + 1; return foo_bar; size_t other = 2; static const x = 3;"
    assert Q.pick_ident(text, rng) == "foo_bar"
    assert Q.pick_ident("a_b a_b", rng) is None                                    # too short to be a name worth renaming
    fn = "def f(self, alpha, beta):\n    total = alpha + beta\n    return total * alpha\n"
    assert Q.local_name(fn, random.Random(0)) in ("alpha", "beta", "total")
    assert Q.local_name("def f(x):\n    return x\n", random.Random(0)) is None
    assert Q.read_result("a/b.py", "x = 1\ny = 2\n") == ("<path>a/b.py</path>\n<type>file</type>\n<content>\n1: x = 1\n2: y = 2\n\n(End of file - total 2 lines)\n</content>")
    src = "def a():\n" + "    x = 1\n" * 25 + "\ndef b():\n    return 1\n"
    assert len(Q.functions_of(src)) == 1 and Q.functions_of("def (") == []


def test_code_files_lists_sized_source_files_at_a_commit():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "tools").mkdir()
        (root / "tools" / "big.py").write_text("x = 1\n" * 1000)
        (root / "tools" / "tiny.py").write_text("x = 1\n")
        (root / "tools" / "test_big.py").write_text("x = 1\n" * 1000)
        (root / "notes.md").write_text("x" * 5000)
        subprocess.run(["git", "-C", d, "init", "-q"], check=True)
        subprocess.run(["git", "-C", d, "add", "-A"], check=True)
        subprocess.run(["git", "-C", d, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x"], check=True)
        assert [p for p, _ in Q.code_files(root, "HEAD")] == ["tools/big.py"]


@pytest.mark.skipif(not Path(TOKENIZER).exists(), reason="the model's tokenizer directory is not on this machine")
def test_the_renderer_uses_the_models_chat_template():
    r = Q.Renderer(TOKENIZER)
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
            *Q.exchange("T", "text", "read", {"filePath": "a.py"}, "RESULT")]
    on, off = r.text(msgs, Q.TOOLS, True), r.text(msgs, Q.TOOLS, False)
    assert on.endswith("<|im_start|>assistant\n<think>\n") and off.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "<function=read>\n<parameter=filePath>\na.py\n</parameter>" in on and "<tool_response>\nRESULT\n</tool_response>" in on
    assert "<tools>" in on and '"name": "grep"' in on
    ids = r.ids(msgs, Q.TOOLS, True)
    assert ids[0] == 248045 and max(ids) < G.VOCAB and r.tok.decode(ids) == on      # <|im_start|> first; the text round-trips


def test_the_committed_grown_set():
    path = G.ROOT / "bench/quality/set-v2.json"
    if not path.exists():
        return
    st = json.load(open(path))
    assert G.set_problems(st, OLD, min_requests=40) == []
    assert len(st["items"]) >= 40 and st["items"][:20] == OLD["items"]
    kinds = {it["kind"] for it in st["items"][20:]}
    assert {"edit-short", "first", "toolres", "edit-ctx", "deep32k", "deep64k"} <= kinds
    new = st["items"][20:]
    assert all(it["meta"]["rev"] == st["rev"] and it["meta"]["prompt_tokens"] == len(it["prompt"]) for it in new)
    assert max(len(it["prompt"]) for it in new) >= 60_000 and min(len(it["prompt"]) for it in new) < 1_000
    # filled or not, the gate must know what to do with it
    assert G.pending(st) == new or G.pending(st) == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
