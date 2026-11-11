"""fake_engine.py - a stand-in for the engine server, so abtest.py and queue_run.sh can be tested end to end in seconds, with no GPU.

What abtest.py really starts is not the engine: it is  `capped.sh <repo>/.venv/bin/python serve/server.py --engine strata --config CFG
--port N`  (the Python server, which then starts the `strata` exe over a stdin/stdout line protocol).  Everything abtest looks at is on
the server's outside, so this file replaces the server (not the exe):
  - stdout: a line containing "ready: http" once the server is up; "EngineDied" (and exit 1) when it died before that;
  - HTTP:   POST /v1/chat/completions (OpenAI shape, the generated text is read for a hash) and POST /v1/messages/count_tokens;
  - the engine log, the file named by the config key "log" (appended to): per request the line
      strata serve: prompt P tokens = 0 reused + P read in X ms (A tok/s), G generated in Y ms (B tok/s), drafts accepted a of b, 1 checkpoints
    which abtest reads for the figures (the same line the engine writes), and per start "[strata] <time> engine started: ...".
  - SIGTERM ends it (abtest.stop_servers).

Behaviour comes from the config JSON: the optional key "fake" (a profile, see DEFAULTS) and the per-arm env keys abtest.make_config writes
into config["env"] (FAKE_*, see ENV_KEYS), so a spec arm is just  {"env": {"FAKE_GAIN_PCT": "3"}}.  Every request's figure is
    base[case] * (1 + gain) * start factor * cold factor * repeat factor * (1 + noise)
with a start factor drawn once per server start (start-to-start variance), an outlier start with probability outlier_prob (a whole start
runs slow, as seen on the real box), the first requests of a start slower (cold_pct: abtest's own warm-up request is request 0), and a
repeated prompt faster (repeat_pct: on the real engine a second identical `edit` request decodes ~35% faster, the lookup draft finds the
earlier reply; "same" is the exact text, so abtest's "[pass k]" and "[run ...]" first lines make a request new again).  Random numbers come from (seed, start index, request index): the same config gives the same figures.  The start
index is the number of "engine started" lines already in the log.  Optional log lines the harness reads: the tier ("tier": "arena" |
"mmap" writes the engine's own start-up line abtest.tier_of looks for) and one `strata switch: NAME acted: ...` line per STRATA_* key
of the config's env on the first request (the engine's once-only switch line; FAKE_SWITCH_SILENT / FAKE_SWITCH_WILLNOT: comma lists of
keys that say nothing / "will not act").

Use it through a fake repo root (make_fake_root below): a copy of abtest.py / queue_run.sh whose ROOT is a temp directory holding
.venv/bin/python (wrapper), tools/pancake/capped.sh (no systemd, no RAM wait), serve/server.py (this file) and a synthetic src/ tree.
The wrapper also compresses time (FAKE_TIME_SCALE, default 0.05: every time.sleep is 20x shorter, so abtest's 5 s quiet sample and 3 s
ready poll cost 0.25 s and 0.15 s); FAKE_TIME_SCALE=1 runs in real time.

    python tools/pancake/probes/fake_engine.py --make-root DIR          build a fake root (prints the commands to try)
    python tools/pancake/probes/fake_engine.py --engine strata --config CFG --port N        what abtest starts

Hazard: abtest.stop_servers() ends EVERY process whose command line has "server.py --engine strata" and waits for any process named
`strata`, on the whole machine.  Never run abtest from a fake root while a real abtest / engine runs: real_run_active() says whether one
does (the tests skip then).  No model is loaded and no GPU is touched.
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CPT = 2.85               # characters per token (abtest.LONG_CPT)
TEMPLATE_TOKENS = 40     # chat template overhead: the 87-character `short` prompt is 71 tokens on the real engine
CASE_PREFIX = {"Write a detailed explanation of how a hash map": "short", "Rewrite the following C++ file": "edit"}
SUMMARIZE = "Summarize what each of these source files does"
RUN_LINE = re.compile(r"\A\[(?:run|pass) [^\]\n]*\]\n")   # abtest's first line of a long* request ([run ...], unique) or of pass k >= 1 ([pass k])
TARGET_TOKENS = {"long1k": 1000, "long3k": 3000, "longdec10k": 10000, "long": 11978, "long32k": 32000, "longdec32k": 32000,
                 "long64k": 64000, "longdec64k": 64000, "longdec128k": 120000}   # abtest.LONG_TARGET ('long' is 34264 chars)
PREFILL_ONLY = ("long1k", "long3k", "long32k", "long64k")   # max_tokens 32: told from longdec* of the same size by it
PROMPT_JUDGED = ("long", "long1k", "long3k", "long32k", "long64k")   # abtest.metric: judged on prompt tok/s, the rest on decode tok/s

DEFAULTS = {
    "seed": 1,
    "startup_s": 4.0,          # seconds before "ready" (the real engine: ~100 s); shortened by FAKE_TIME_SCALE
    "latency_s": 0.0,          # real seconds each request takes
    "sd_start_pct": 1.5,       # start-to-start sd of the speed (%), one draw per server start
    "sd_request_pct": 1.0,     # request-to-request sd inside a start (%)
    "outlier_prob": 0.0,       # probability that a whole start runs slow (the real box: ~0.15, bench/results/2026-11-11-gpu)
    "outlier_range": [0.6, 0.9],   # speed factor of an outlier start, uniform
    "slow_starts": [],         # start indices that run slow by slow_factor (a deterministic outlier, for tests)
    "slow_factor": 0.6,
    "cold_pct": [25.0, 0.0],   # slowdown (%) of the 1st, 2nd ... request of a start
    "repeat_pct": {"short": 2.0, "edit": 35.0},   # gain (%) of every repeat of the same case inside a start
    "gain_pct": 0.0,           # the arm's true effect on the judged figure (all cases); gain_pct_<case> overrides one case
    "nominal_gen": 512,        # tokens generated when max_tokens allows
    "text_salt": "",           # another salt = another text (a lossy arm)
    "tier": "",                # "arena" / "mmap": say so in the log at start, as the engine does ("" = say nothing)
    "switch_silent": [],       # STRATA_* env keys that never log a switch line (a silent no-op: the arm measures nothing)
    "switch_will_not": [],     # STRATA_* env keys that log "will not act"
    "die": False,              # the server dies before ready
    "die_starts": [],          # start indices that die before ready
    "base_decode": {"short": 121.5, "edit": 141.0, "long": 118.0, "long1k": 120.0, "long3k": 120.0, "long32k": 110.0, "long64k": 100.0,
                    "longdec10k": 118.0, "longdec32k": 105.0, "longdec64k": 90.0, "longdec128k": 70.0},
    "base_prompt": {"short": 50.0, "edit": 650.0, "long": 700.0, "long1k": 560.0, "long3k": 640.0, "long32k": 560.0, "long64k": 470.0,
                    "longdec10k": 640.0, "longdec32k": 560.0, "longdec64k": 470.0, "longdec128k": 380.0},
    "accept": {"short": 0.67, "edit": 0.90},   # drafts accepted / drafted, the other cases 0.75
}
ENV_KEYS = {    # config["env"] key -> (profile key, parser): a spec arm changes the profile without a new config file
    "FAKE_SEED": ("seed", int), "FAKE_STARTUP_S": ("startup_s", float), "FAKE_LATENCY_S": ("latency_s", float),
    "FAKE_SD_START_PCT": ("sd_start_pct", float), "FAKE_SD_REQUEST_PCT": ("sd_request_pct", float),
    "FAKE_OUTLIER_PROB": ("outlier_prob", float), "FAKE_GAIN_PCT": ("gain_pct", float), "FAKE_TEXT_SALT": ("text_salt", str),
    "FAKE_COLD_PCT": ("cold_pct", lambda s: [float(x) for x in s.split(",")]), "FAKE_DIE": ("die", lambda s: s not in ("", "0")),
    "FAKE_TIER": ("tier", str), "FAKE_SWITCH_SILENT": ("switch_silent", lambda s: s.split(",")),
    "FAKE_SWITCH_WILLNOT": ("switch_will_not", lambda s: s.split(",")),
}
TIER_LINE = {"arena": "strata serve: expert arena: 33.0 GiB locked for the GPU", "mmap": "strata serve: experts via mmap (fake engine)"}


# ---------------------------------------------------------------------------------------------------------- the model
def profile(cfg):
    """DEFAULTS, then the config's "fake" dict (one level deep for the table keys), then the FAKE_* keys of config["env"]."""
    p = json.loads(json.dumps(DEFAULTS))
    for k, v in (cfg.get("fake") or {}).items():
        if isinstance(v, dict) and isinstance(p.get(k), dict):
            p[k].update(v)
        else:
            p[k] = v
    for k, v in (cfg.get("env") or {}).items():
        if k in ENV_KEYS:
            name, parse = ENV_KEYS[k]
            p[name] = parse(str(v))
        elif k.startswith("FAKE_GAIN_PCT_"):
            p["gain_pct_" + k[len("FAKE_GAIN_PCT_"):].lower()] = float(v)
    return p


def message_text(body):
    """The text of the last user message (a string, or a list of {"type": "text", "text": ...} parts)."""
    content = ([m for m in body.get("messages", []) if m.get("role") == "user"] or [{}])[-1].get("content", "")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or ""


def prompt_tokens(text):
    return max(1, round(len(text) / CPT)) + TEMPLATE_TOKENS


def classify(text, max_tokens=512):
    """abtest case of a prompt: short / edit by their first words, long* by size (the target token count nearest in log scale; a
    32000 / 64000 tie between long32k and longdec32k is told by max_tokens <= 32).  Anything else counts as short."""
    body = RUN_LINE.sub("", text, count=1)
    for prefix, case in CASE_PREFIX.items():
        if body.startswith(prefix):
            return case
    if body.startswith(SUMMARIZE):
        n = max(1, len(body)) / CPT
        best = min(TARGET_TOKENS, key=lambda c: (abs(math.log(n / TARGET_TOKENS[c])), (c in PREFILL_ONLY) != (max_tokens <= 32)))
        return best
    return "short"


def start_state(prof, start_idx):
    """What one server start decided for all its requests: the speed factor and whether it is an outlier start."""
    rng = random.Random(f"{prof['seed']}/{start_idx}")
    factor = max(0.2, 1 + rng.gauss(0, prof["sd_start_pct"] / 100))
    outlier = rng.random() < prof["outlier_prob"]
    if outlier:
        factor *= rng.uniform(*prof["outlier_range"])
    if start_idx in prof["slow_starts"]:
        factor *= prof["slow_factor"]
        outlier = True
    return {"start": start_idx, "factor": factor, "outlier": outlier}


def gain_factor(prof, case):
    return 1 + prof.get("gain_pct_" + case, prof["gain_pct"]) / 100


def request_speeds(prof, state, case, req_idx, pass_idx):
    """(prompt tok/s, decode tok/s) of the req_idx-th request of a start, the pass_idx-th time its case is asked in it."""
    rng = random.Random(f"{prof['seed']}/{state['start']}/{req_idx}")
    f = state["factor"] * max(0.2, 1 + rng.gauss(0, prof["sd_request_pct"] / 100))
    if req_idx < len(prof["cold_pct"]):
        f *= 1 - prof["cold_pct"][req_idx] / 100
    if pass_idx >= 1:
        f *= 1 + prof["repeat_pct"].get(case, 0.0) / 100
    g = gain_factor(prof, case)
    p, d = prof["base_prompt"].get(case, 100.0) * f, prof["base_decode"].get(case, 100.0) * f
    return (p * g, d) if case in PROMPT_JUDGED else (p, d * g)


def fake_text(text, salt, temperature):
    """Same prompt and salt give the same text (an exact arm); a sampled request never does."""
    h = hashlib.sha256((salt + "\0" + text).encode()).hexdigest()
    if temperature and temperature > 0:
        h += os.urandom(4).hex()
    return f"fake reply {h} " * 4


def log_line(prompt, prompt_tps, gen, decode_tps, accepted):
    drafted = max(1, round(gen * 0.92))
    return (f"strata serve: prompt {prompt} tokens = 0 reused + {prompt} read in {prompt / prompt_tps * 1000:.0f} ms ({prompt_tps:.1f} tok/s), "
            f"{gen} generated in {gen / decode_tps * 1000:.0f} ms ({decode_tps:.1f} tok/s), drafts accepted {round(accepted * drafted)} of {drafted}"
            f", 1 checkpoints")


class Engine:
    """One server start: counts requests and passes, answers a chat request."""

    def __init__(self, prof, log_path, start_idx, env=None):
        self.prof, self.log_path, self.env = prof, log_path, env or {}
        self.state = start_state(prof, start_idx)
        self.requests = 0
        self.seen = {}

    def write_log(self, line):
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def say_switches(self):
        """The engine's once-only `strata switch:` line per STRATA_* key it was given (what switch_report.py reads)."""
        for key in sorted(k for k in self.env if k.startswith("STRATA_") and k != "STRATA_RAM_KEEP_GIB"):   # (a setting, not a switch)
            if key in self.prof["switch_silent"]:
                continue
            if key in self.prof["switch_will_not"]:
                self.write_log(f"strata switch: {key} will not act: the fake engine was told it cannot")
            else:
                self.write_log(f"strata switch: {key} acted: the fake engine pretends it did (={self.env[key]})")

    def chat(self, body):
        text = message_text(body)
        max_tokens = int(body.get("max_tokens") or self.prof["nominal_gen"])
        case = classify(text, max_tokens)
        req_idx = self.requests
        self.requests += 1
        pass_idx = self.seen.get(text, 0)    # the same text again in this start: the engine's conversation cache has it
        self.seen[text] = pass_idx + 1
        if req_idx == 0:
            self.say_switches()
        prompt_tps, decode_tps = request_speeds(self.prof, self.state, case, req_idx, pass_idx)
        prompt, gen = prompt_tokens(text), min(max_tokens, self.prof["nominal_gen"])
        self.write_log(log_line(prompt, prompt_tps, gen, decode_tps, self.prof["accept"].get(case, 0.75)))
        if self.prof["latency_s"]:
            time.sleep(self.prof["latency_s"])
        reply = fake_text(text, self.prof["text_salt"], body.get("temperature", 0))
        return {"id": f"fake-{req_idx}", "object": "chat.completion", "model": body.get("model", "x"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": prompt, "completion_tokens": gen, "total_tokens": prompt + gen}}


def handler_for(engine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, obj, code=200):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.send({"data": [{"id": "fake"}]} if self.path.startswith("/v1/models") else {"status": "ok"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if self.path.startswith("/v1/chat/completions"):
                self.send(engine.chat(body))
            elif self.path.startswith("/v1/messages/count_tokens"):
                self.send({"input_tokens": prompt_tokens(message_text(body))})
            else:
                self.send({"error": "not found"}, 404)
    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description="fake stand-in for `serve/server.py --engine strata` (see the module docstring)")
    ap.add_argument("--engine")
    ap.add_argument("--config")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--make-root", metavar="DIR", help="build a fake repo root instead of serving")
    a, _ = ap.parse_known_args(argv)
    if a.make_root:
        root = make_fake_root(Path(a.make_root))
        print(f"fake root {root}\n  cd {root}\n  .venv/bin/python tools/pancake/abtest.py SPEC.json --out OUT.jsonl --port 18080 "
              f"(SPEC 'base' = an absolute config path with an optional \"fake\" profile; arms: env FAKE_GAIN_PCT ...)")
        return 0
    cfg = json.load(open(a.config))
    prof = profile(cfg)
    log_path = cfg.get("log")
    start_idx = 0
    if log_path and Path(log_path).exists():
        start_idx = Path(log_path).read_text(errors="replace").count("engine started:")
    eng = Engine(prof, log_path, start_idx, cfg.get("env"))
    eng.write_log(f"[strata] {time.strftime('%Y-%m-%d %H:%M:%S')} engine started: {cfg.get('exe', 'fake')} --serve {' '.join(cfg.get('args', []))}")
    if prof["tier"] in TIER_LINE:
        eng.write_log(TIER_LINE[prof["tier"]])
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
    server = ThreadingHTTPServer(("127.0.0.1", a.port), handler_for(eng))   # a taken port ends the process: abtest says "died before ready"
    server.daemon_threads = True
    time.sleep(prof["startup_s"])
    if prof["die"] or start_idx in prof["die_starts"]:
        print("EngineDied: the fake engine was told to die before ready", flush=True)
        return 1
    print(f"ready: http://127.0.0.1:{a.port}/v1  (OpenAI: /v1/chat/completions, Anthropic: /v1/messages, context 131072 tokens)", flush=True)
    server.serve_forever()
    return 0


# ----------------------------------------------------------------------------------------------- the fake repo root
SITECUSTOMIZE = """import os, time
_scale = float(os.environ.get("FAKE_TIME_SCALE", "1"))
if _scale != 1:
    _sleep = time.sleep
    time.sleep = lambda seconds: _sleep(seconds * _scale)
"""


def source_text(name, chars):
    """Deterministic C++-looking filler of exactly `chars` characters."""
    out, i = [], 0
    while sum(map(len, out)) < chars:
        out.append(f"int {name}_{i}(int x) {{ return x * {i} + {i % 7}; }}\n")
        i += 1
    return "".join(out)[:chars]


def real_run_active():
    """The first real abtest / engine / queue found on this machine (argv checked, not substrings: a shell that only mentions them
    does not count), else None.  Processes of this test run's own tree are not looked at: call it before starting any."""
    me = os.getpid()
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == me:
            continue
        try:
            argv = Path(f"/proc/{d}/cmdline").read_bytes().split(b"\0")
            comm = Path(f"/proc/{d}/comm").read_text().strip()
        except OSError:
            continue
        toks = [t.decode("utf-8", "replace") for t in argv if t]
        py = comm.startswith("python")
        if comm == "strata" or (py and any(t.endswith(("pancake/abtest.py", "pancake/restore_check.py")) for t in toks)) \
                or (py and "--engine" in toks and any(t.endswith("serve/server.py") for t in toks)) \
                or (comm == "bash" and any(t.endswith("pancake/queue_run.sh") for t in toks)):
            return f"pid {d}: {' '.join(toks)[:140]}"
    return None


def make_fake_root(dest, abtest=None, queue_run=None, scale=0.05):
    """Builds the fake repo root `dest` (see the module docstring) and returns it.  `abtest` / `queue_run`: the abtest.py / queue_run.sh
    to copy (default: the ones next to this probes directory).  Every other tools/pancake/*.py and *.sh is copied too (they may call
    each other)."""
    dest, here = Path(dest), Path(__file__).resolve()
    pancake = here.parent.parent
    for sub in ("tools/pancake", "serve", "shim", ".venv/bin", "src/spec", "src/big", "configs", "logs"):
        (dest / sub).mkdir(parents=True, exist_ok=True)
    for f in [*pancake.glob("*.py"), *pancake.glob("*.sh")]:
        if not f.name.startswith("test_"):
            shutil.copy2(f, dest / "tools/pancake" / f.name)
    for src, name in ((abtest, "abtest.py"), (queue_run, "queue_run.sh")):
        if src:
            shutil.copy2(src, dest / "tools/pancake" / name)
    (dest / "tools/pancake/capped.sh").write_text(   # no systemd, no RAM wait; the one stderr line the real one prints (tier guess)
        '#!/bin/sh\nif [ "$1" = "-n" ]; then shift; echo "would run: $*"; exit 0; fi\n'
        'echo "capped: tier=${FAKE_CAPPED_TIER:-mmap} keep_gib=${STRATA_RAM_KEEP_GIB:-12} big_free_gib=0" >&2\n'
        'export STRATA_TIER_GUESS=${FAKE_CAPPED_TIER:-mmap}\nexec "$@"\n')
    (dest / "tools/pancake/capped.sh").chmod(0o755)
    (dest / "shim/sitecustomize.py").write_text(SITECUSTOMIZE)
    wrapper = dest / ".venv/bin/python"
    wrapper.write_text(f'#!/bin/sh\nFAKE_TIME_SCALE="${{FAKE_TIME_SCALE:-{scale}}}" PYTHONPATH="{dest}/shim${{PYTHONPATH:+:$PYTHONPATH}}" '
                       f'exec "{sys.executable}" "$@"\n')
    wrapper.chmod(0o755)
    link = dest / "serve/server.py"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(here)
    (dest / "src/spec/draft_policy.cpp").write_text(source_text("policy", 8000))
    for n in "abcd":   # 4 x 8566 characters = the 34264 characters of the real `long` prompt
        (dest / f"src/spec/{n}.cpp").write_text(source_text(n, 8566))
    for i in range(24):   # 24 x 20000 characters: enough corpus for every long* case (longdec128k needs 342000)
        (dest / f"src/big/f{i:02d}.cpp").write_text(source_text(f"f{i}", 20000))
    return dest


if __name__ == "__main__":
    sys.exit(main())
