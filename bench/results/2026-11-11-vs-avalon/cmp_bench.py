"""One neutral client for pancake (Strata server) and Avalon (llama-server): same prompts, thinking off, greedy.
Run 1 of every case is discarded (warm-up); medians over the rest.  usage: cmp_bench.py LABEL PORT OUT.json"""
import json, random, statistics, sys, time, urllib.request
sys.path.insert(0, "/home/dan/clone-of-d/src/Strata/tools")
from ab_engine import long_prompt
label, port, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
DEC = ["Write a short story (about 400 words) about a lighthouse keeper.",
       "Write a JavaScript debounce function with JSDoc and a usage example.",
       "List 15 practical tips for reducing cloud costs, one line each.",
       "Explain in about 300 words how a mixture-of-experts language model works.",
       "Write a Python class implementing an LRU cache with type hints and tests."]
def ask(prompt, max_tokens):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
            json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800))
    t = r["timings"]; t["wall_s"] = time.time() - t0
    return t
res = {"label": label, "runs": {}}
def med(xs): return statistics.median(xs) if xs else None
# decode: 3 rounds over 5 prompts; round 1 discarded
rows = []
for rnd in (1, 2, 3):
    for i, p in enumerate(DEC):
        t = ask(p, 512)
        rows.append({"round": rnd, "p": i, "tok": t["predicted_n"], "tps": t["predicted_per_second"],
                     "acc": (t.get("draft_n_accepted"), t.get("draft_n"))})
        print(f"{label} dec r{rnd} p{i}: {t['predicted_n']} tok @ {t['predicted_per_second']:.1f} t/s", flush=True)
res["decode"] = rows
kept = [r["tps"] for r in rows if r["round"] > 1]
print(f"{label} DECODE (rounds 2-3, run 1 discarded): median {med(kept):.1f} t/s  min {min(kept):.1f} max {max(kept):.1f}", flush=True)
# prompt: 4 runs per size, a unique first line so the cache cannot help; run 1 discarded
for name, words in (("4K", 3000), ("16K", 12000), ("32K", 24000)):
    rr = []
    for k in range(1, 5):
        p = f"Request id {random.getrandbits(48)}.\n" + long_prompt(words)
        t = ask(p, 24)
        rr.append({"run": k, "prompt_n": t["prompt_n"], "pps": t["prompt_per_second"], "ttft_s": t["prompt_ms"] / 1000})
        print(f"{label} {name} run{k}: {t['prompt_n']} tok @ {t['prompt_per_second']:.0f} t/s, ttft {t['prompt_ms']/1000:.2f}s", flush=True)
    res["runs"][name] = rr
    ks = [r for r in rr if r["run"] > 1]
    print(f"{label} PROMPT {name} (run 1 discarded): median {med([r['pps'] for r in ks]):.0f} t/s, ttft {med([r['ttft_s'] for r in ks]):.2f}s ({ks[0]['prompt_n']} tok)", flush=True)
json.dump(res, open(out, "w"), indent=1)
print(label, "DONE", flush=True)
