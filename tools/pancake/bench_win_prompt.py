"""Prompt-speed (prefill) bench for strata-poly on Windows: the prompt counterpart of bench_win.py.
One fresh server per session; one warm-up request (not counted, largest size), then REPS requests per size, each with a
text slice of its own so no prefix is reused (the DONE line's <reused> is checked to be 0).  Prompt tok/s is
read tokens / prompt_ms from the DONE line (the engine's own clock, load and decode excluded); decode tok/s of the few
generated tokens and the greedy token ids are recorded too (the ids hash is for off/on comparisons).
env:   STRATA_POLY_EXE, STRATA_MODEL, STRATA_TOKENIZER, CUDA_BIN, BENCH_OUT, BENCH_MAX_BUSY (default 60), BENCH_QUIET_WAIT
usage: bench_win_prompt.py SESSIONS REPS MAX_NEW MAX_CONTEXT SIZES(comma list) [extra strata-poly args...]
       an env var of the caller (e.g. STRATA_POLY_PF_F16=1) reaches the server; the label is BENCH_LABEL."""
import ctypes, hashlib, json, math, os, subprocess, sys, time
from ctypes import wintypes
from tokenizers import Tokenizer

SP = os.environ.get("BENCH_OUT", os.getcwd())
EXE, MODEL, TOKENIZER = os.environ["STRATA_POLY_EXE"], os.environ["STRATA_MODEL"], os.environ["STRATA_TOKENIZER"]
CUDA_BIN = os.environ.get("CUDA_BIN", "")
LABEL = os.environ.get("BENCH_LABEL", "run")
SESSIONS, REPS, MAX_NEW, MAXCTX = (int(x) for x in sys.argv[1:5])
SIZES = [int(x) for x in sys.argv[5].split(",")]
EXTRA = sys.argv[6:]
SRC = os.environ.get("BENCH_TEXT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "MOD_PLAN.md"))
tok = Tokenizer.from_file(TOKENIZER)
ALL = tok.encode(open(SRC, encoding="utf-8", errors="ignore").read(), add_special_tokens=False).ids


def slice_ids(n, k):                       # the k-th distinct window of n tokens
    off = (k * 7919) % max(1, len(ALL) - n)
    return ALL[off:off + n]


def sys_times():
    i, k, u = (wintypes.FILETIME() for _ in range(3))
    ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(i), ctypes.byref(k), ctypes.byref(u))
    q = lambda f: (f.dwHighDateTime << 32) | f.dwLowDateTime
    return q(i), q(k) + q(u)


def busy_pct(sec):
    i0, t0 = sys_times(); time.sleep(sec); i1, t1 = sys_times()
    return 100.0 * (1 - (i1 - i0) / max(1, t1 - t0))


def quiet(max_busy=float(os.environ.get("BENCH_MAX_BUSY", "60")), wait=int(os.environ.get("BENCH_QUIET_WAIT", "180"))):
    t0 = time.time(); b = busy_pct(5)
    while b > max_busy and time.time() - t0 < wait:
        print(f"  waiting for quiet: system CPU {b:.0f}%", flush=True); b = busy_pct(5)
    return b


env = dict(os.environ)
if CUDA_BIN: env["PATH"] = CUDA_BIN + os.pathsep + env["PATH"]
rows = []     # dicts
ctr = 1000 * (abs(hash(LABEL)) % 7)
for sess in range(SESSIONS):
    pre = quiet()
    print(f"[{LABEL}] session {sess}: pre-run system CPU {pre:.0f}%", flush=True)
    proc = subprocess.Popen([EXE, "--model", MODEL, "--serve", "--max-context", str(MAXCTX), *EXTRA], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=open(os.path.join(SP, f"{LABEL}_s{sess}.err"), "w"), text=True,
                            env=env, bufsize=1)
    plan = [("warm", max(SIZES))] + [(n, n) for _ in range(REPS) for n in SIZES]
    for tag, n in plan:
        ctr += 1
        ids = slice_ids(n, ctr)
        proc.stdin.write(f"GEN {MAX_NEW} {','.join(map(str, ids))}\n"); proc.stdin.flush()
        gen, done = [], None
        for line in proc.stdout:
            if line.startswith("T "): gen.append(line.split()[1])
            elif line.startswith("DONE"): done = line.split(); break
            elif line.startswith("ERR"): print(line, flush=True); break
        if tag == "warm" or not done: continue
        produced, pn, pms, dms, fin = int(done[1]), int(done[2]), float(done[3]), float(done[4]), done[5]
        reused, read = int(done[8]), int(done[14])
        r = dict(label=LABEL, session=sess, size=n, prompt_tokens=pn, reused=reused, read=read, prompt_ms=pms,
                 prompt_tps=1000.0 * read / pms if pms > 0 else 0.0, decode_tps=1000.0 * produced / dms if dms > 0 else 0.0,
                 produced=produced, ids_hash=hashlib.sha1(",".join(gen).encode()).hexdigest()[:12], pre_cpu=pre)
        rows.append(r); print(json.dumps(r), flush=True)
    proc.stdin.write("QUIT\n"); proc.stdin.flush()
    try: proc.wait(timeout=30)
    except Exception: proc.kill()
for n in SIZES:
    v = [r["prompt_tps"] for r in rows if r["size"] == n]
    if v:
        m = sum(v) / len(v); sd = math.sqrt(sum((x - m) ** 2 for x in v) / max(1, len(v) - 1))
        print(f"[{LABEL}] size {n}: prompt {m:.1f} tok/s sd {sd:.1f} n={len(v)} {['%.1f' % x for x in v]}")
json.dump(rows, open(os.path.join(SP, f"{LABEL}_rows.json"), "w"))
