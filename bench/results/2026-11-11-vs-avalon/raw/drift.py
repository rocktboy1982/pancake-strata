"""Greedy outputs for 6 prompts -> json (text + tokens/s).  usage: drift.py PORT OUT.json"""
import json, sys, urllib.request
port, out = int(sys.argv[1]), sys.argv[2]
CODE = "def merge(a, b):\n    i = j = 0\n    out = []\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n            out.append(b[j]); j += 1\n    out.extend(a[i:]); out.extend(b[j:])\n    return out\n"
P = ["Write a short story (about 400 words) about a lighthouse keeper.",
     "Write a JavaScript debounce function with JSDoc and a usage example.",
     "List 15 practical tips for reducing cloud costs, one line each.",
     "Explain in about 300 words how a mixture-of-experts language model works.",
     "Write a Python class implementing an LRU cache with type hints and tests.",
     "Rewrite this function to be iterative-free using heapq.merge, keep the docstring style, and add type hints. Return the full code.\n\n" + CODE]
res = []
for p in P:
    body = {"messages": [{"role": "user", "content": p}], "max_tokens": 400, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
    res.append({"text": r["choices"][0]["message"]["content"], "tps": r["timings"]["predicted_per_second"], "n": r["timings"]["predicted_n"]})
    print(f"{r['timings']['predicted_n']} tok @ {r['timings']['predicted_per_second']:.1f}", flush=True)
json.dump(res, open(out, "w"))
