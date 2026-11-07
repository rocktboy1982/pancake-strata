"""learn_profile.py - pancake-strata: warm a `strata --serve` engine on agent-like prompts (source files the bench does
not use) with --expert-profile-save, then QUIT so it writes the profile.

    .venv/bin/python tools/pancake/learn_profile.py --tokenizer DIR --save PROFILE.bin -- build-cuda/strata ARGS...
"""
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "qwen35"))
sys.path.insert(0, str(ROOT / "tools" / "pancake"))
from q36_harness import Eng  # noqa: E402
from bench_engines import chat, load_tok  # noqa: E402

FILES = ["src/core/verify.cpp", "src/core/mtp.cpp", "serve/server.py", "src/program/generate.cpp",
         "src/kernels/cuda/native_mmvq.cu", "setup.py"]
ASKS = ["Explain what this code does and point out one bug risk.", "Write unit tests for the main function here.",
        "Refactor the longest function into smaller ones; show the code.", "Summarize the design in ten bullets.",
        "Add logging to the error paths; show a diff.", "Which parts are hot paths, and how would you speed them up?"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--save", required=True)
    ap.add_argument("--chars", type=int, default=24000)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    tok = load_tok(a.tokenizer)
    e = Eng(cmd[0], cmd[1:] + ["--expert-profile-save", a.save, "--expert-profile-save-every", "0"],
            log=a.save + ".log")
    print(f"loaded in {e.load_s:.1f}s", flush=True)
    for i, (f, ask) in enumerate(zip(FILES, ASKS)):
        text = (ROOT / f).read_text(encoding="utf-8", errors="replace")
        start = (len(text) // 3) % max(1, len(text) - a.chars)
        ids = tok.encode(chat(f"File {f}:\n\n" + text[start:start + a.chars] + "\n\n" + ask), parse_special=True)
        toks, done, ttft, wall = e.solo(ids, 384)
        print(f"{i}: {f}: prompt {len(ids)}, {len(toks)} new in {wall:.1f}s", flush=True)
    e.send("QUIT")
    e.p.wait(timeout=300)
    print("saved" if Path(a.save).exists() else "NOT saved", a.save)


if __name__ == "__main__":
    main()
