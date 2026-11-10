"""Windows bench for strata-poly: fresh server per run, warm-up requests not counted, decode tok/s measured from the
first streamed token to the last (prefill and load excluded), mean and 95% interval over runs.
Windows counterpart of abtest.py's measurement (abtest.py itself needs /proc).  One arm only: it is a baseline tool.
needs: pip install tokenizers; the model's tokenizer.json (Qwen/Qwen3.6-35B-A3B on Hugging Face)
env:   STRATA_POLY_EXE, STRATA_MODEL, STRATA_TOKENIZER (tokenizer.json), CUDA_BIN (folder with cudart64_*.dll)
usage: bench_win.py RUNS WARMUP MEASURED MAX_NEW [extra strata-poly args...]"""
import ctypes, json, math, os, subprocess, sys, time
from ctypes import wintypes
from tokenizers import Tokenizer

SP = os.environ.get("BENCH_OUT", os.getcwd())
EXE = os.environ["STRATA_POLY_EXE"]
MODEL = os.environ["STRATA_MODEL"]
TOKENIZER = os.environ["STRATA_TOKENIZER"]
CUDA_BIN = os.environ.get("CUDA_BIN", "")
RUNS, WARM, MEAS, MAX_NEW = (int(x) for x in sys.argv[1:5])
EXTRA = sys.argv[5:]
T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23}
PROMPTS = {
    "short": "Write a detailed explanation of how a hash map works, with an example in Python.",
    "code": "Write a Python function that parses a CSV file and returns the rows as a list of dictionaries.",
}
tok = Tokenizer.from_file(TOKENIZER)
chat = lambda p: f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def sys_times():
    i, k, u = (wintypes.FILETIME() for _ in range(3))
    ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(i), ctypes.byref(k), ctypes.byref(u))
    q = lambda f: (f.dwHighDateTime << 32) | f.dwLowDateTime
    return q(i), q(k) + q(u)          # idle, kernel+user (kernel includes idle)


def busy_pct(sec):
    i0, t0 = sys_times(); time.sleep(sec); i1, t1 = sys_times()
    return 100.0 * (1 - (i1 - i0) / max(1, t1 - t0))


def quiet(max_busy=12.0, wait=180):
    t0 = time.time(); b = busy_pct(5)
    while b > max_busy and time.time() - t0 < wait:
        print(f"  waiting for quiet: system CPU {b:.0f}%", flush=True); b = busy_pct(5)
    return b


env = dict(os.environ)
if CUDA_BIN:
    env["PATH"] = CUDA_BIN + os.pathsep + env["PATH"]
rows = []                                   # (run, case, decode_tps, ttft_s, n_new)
for run in range(RUNS):
    pre = quiet()
    print(f"run {run}: pre-run system CPU {pre:.0f}%", flush=True)
    proc = subprocess.Popen([EXE, "--model", MODEL, "--serve", "--max-context", "2048", *EXTRA], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=open(os.path.join(SP, "bench.err"), "w"), text=True, env=env, bufsize=1)
    for rep in range(WARM + MEAS):
        for case, p in PROMPTS.items():
            ids = tok.encode(chat(p), add_special_tokens=False).ids
            t_send = time.time(); t_first = t_last = None; n = 0
            proc.stdin.write(f"GEN {MAX_NEW} {','.join(map(str, ids))}\n"); proc.stdin.flush()
            for line in proc.stdout:
                if line.startswith("T "):
                    now = time.time()
                    if t_first is None: t_first = now
                    t_last = now; n += 1
                elif line.startswith(("DONE", "ERR")):
                    break
            if rep >= WARM and n > 8:
                rows.append((run, case, (n - 1) / (t_last - t_first), t_first - t_send, n))
    proc.stdin.write("QUIT\n"); proc.stdin.flush()
    try: proc.wait(timeout=30)
    except Exception: proc.kill()


def ci(xs):
    m = sum(xs) / len(xs)
    if len(xs) < 2: return m, float("nan")
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return m, T95.get(len(xs) - 1, 2.0) * sd / math.sqrt(len(xs))


print("\nextra args:", EXTRA or "(none)", f"| runs={RUNS} warmup={WARM} measured={MEAS} max_new={MAX_NEW}")
for case in PROMPTS:
    per_run = []
    for run in range(RUNS):
        v = [r[2] for r in rows if r[0] == run and r[1] == case]
        if v: per_run.append(sum(v) / len(v))
    m, h = ci(per_run)
    ttft = [r[3] for r in rows if r[1] == case]
    print(f"{case:6s} decode {m:6.2f} +/- {h:5.2f} tok/s (95%, n={len(per_run)} runs)  per-run {['%.1f' % x for x in per_run]}  "
          f"time-to-first-token {sum(ttft)/len(ttft):.2f} s")
json.dump(rows, open(os.path.join(SP, "bench_rows.json"), "w"))
