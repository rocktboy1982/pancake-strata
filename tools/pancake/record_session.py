#!/usr/bin/env python3
"""record_session.py - record a real agent session for the offline tools, check what was recorded, convert it.

Every offline verdict so far (FineMoE, HOBBIT, REAP, EcoSpec, A44, A46) rests on one 219-window bench trace with zero router
weights.  This is the one command that gets a real trace.  Three steps, none of which starts the engine except `plan --run`:

  plan     builds the server command and a config for one preset with every recorder on, checks the box and the preset, and
           PRINTS the command.  Writes OUT/config.json and OUT/plan.json, nothing else.  --run starts it (foreground, through
           tools/pancake/capped.sh: STRATA_RAM_KEEP_GIB=12 and a systemd MemoryMax) and runs `check` when the server exits.
               python tools/pancake/record_session.py plan --preset flashnext-iq2_xs-fast-hip --out DIR
  check    after the session: validates every file, writes DIR/manifest.json, says which tools the recording can feed.
               python tools/pancake/record_session.py check DIR [--min-windows 200] [--require-weights]
  convert  writes DIR/replay/: routing.bin (the --dump-routing format time_ceiling.py, finemoe_replay.py, cache_learn.py read),
           session.stwt (cleaned window trace: reap_saliency.py, gate_importance.py, ecospec_sim.py, draft_costs.py, ...),
           turns.jsonl (replay_turns.py --turns), act_norm.npy (when the engine wrote expert output norms), and prints the
           replay commands.
               python tools/pancake/record_session.py convert DIR [--weights auto|valid|none] [--all-windows]

What is recorded (see pancake-docs/batch4/corpus-recorder.md for the source lines):
  session.stwt   STRATA_WINDOW_TRACE: per verify window every layer's ids, placement, router weights; drafts, row misses, ms
  turns.jsonl    the server's turn log (config "turn_log"; numbers only), with STRATA_AGENT_COUNTERS=1 the behaviour counters
  routing.bin    --dump-routing: ids per token and layer, the legacy file (its weights are zeros on purpose in the serve path;
                 it is the safety net that needs no conversion and does not depend on the window trace)
  hidden/        STRATA_HIDDEN_DUMP (opt-in, --hidden-dump): the head's input at the thinking's check points (thinking_probe.py)
  engine.log     the engine's own log (the start-up lines say whether each recorder opened)
The router weights in the window trace are the router's only where the verify window's doorbell published them: the default
path does not.  `check` says per file how many layer records carry real weights; the notes give the engine patch.
"""
import argparse
import copy
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE / "trace"))
import recording as rec  # noqa: E402

ROUTING_ALTERING = ("STRATA_ROUTE_PRIOR", "STRATA_ROUTE_RESIDENT", "STRATA_ROUTE_TAIL_SKIP", "STRATA_VERIFY_ECAP")
FILES = {"trace": "session.stwt", "turns": "turns.jsonl", "routing": "routing.bin", "hidden": "hidden", "log": "engine.log"}
SERVER_ENV = {"STRATA_RAM_KEEP_GIB": "12"}


# ------------------------------------------------------------------------------------------------ plan

def load_preset(preset, repo, data, models):
    """(config dict, path, unresolved placeholders): the preset with run-pancake.sh's substitutions."""
    p = Path(preset) if preset.endswith(".json") else repo / "configs" / "pancake" / f"{preset}.json"
    if not p.exists():
        sys.exit(f"no preset {p} (the names are the files of configs/pancake without .json)")
    text = p.read_text()
    for key, val in (("<REPO>", repo), ("<STRATA_DATA>", data), ("<MODELS>", models)):
        text = text.replace(key, str(val))
    return json.loads(text), p, sorted(set(re.findall(r"<[A-Z_]+>", text)))


def number(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def altering(env):
    """The routing-changing switches that are on in `env` (they bias the recorded routing towards the cache)."""
    on = []
    for k in ROUTING_ALTERING:
        v = number(env.get(k))
        if v is not None and (v >= 0 if k == "STRATA_VERIFY_ECAP" else v > 0):
            on.append(f"{k}={env[k]}")
    return on


def publishes_weights(env):
    """Whether the verify window's doorbell publishes the router weights with this env (verify.cpp record_window: only a
    window captured with the deferral split, STRATA_VERIFY_DEFER_CPU=1 with BELOW > 0, or STRATA_ROUTE_FAST=1)."""
    below = number(env.get("STRATA_VERIFY_DEFER_CPU_BELOW"))
    defer = env.get("STRATA_VERIFY_DEFER_CPU") in ("1", "check") and (below is None or below > 0)
    return defer or env.get("STRATA_ROUTE_FAST") == "1"


def build(cfg, out, a):
    """(derived config, the recorders' engine environment, the extra server environment, the added engine arguments).  The
    engine environment goes in the config's "env": it wins over the process environment when the server builds the engine's."""
    cfg = copy.deepcopy(cfg)
    env = {"STRATA_WINDOW_TRACE": str(out / FILES["trace"]), "STRATA_WINDOW_TRACE_MAX_MB": str(a.max_mb)}
    if a.every != 1:
        env["STRATA_WINDOW_TRACE_EVERY"] = str(a.every)
    if a.hidden_dump:
        env["STRATA_HIDDEN_DUMP"] = f"{out / FILES['hidden']},every={a.hidden_dump}"
    if a.norms:
        env["STRATA_WINDOW_TRACE_NORMS"] = "1"
    if a.publish_weights:
        env["STRATA_VERIFY_DEFER_CPU"] = "1"
    cfg["env"] = {**cfg.get("env", {}), **env}
    cfg["log"], cfg["turn_log"] = str(out / FILES["log"]), str(out / FILES["turns"])
    if a.port:
        cfg["port"] = a.port
    args = [] if a.no_routing_dump else ["--dump-routing", str(out / FILES["routing"])]
    cfg["args"] = list(cfg.get("args", [])) + args
    server_env = dict(SERVER_ENV)
    if not a.no_agent_counters:
        server_env["STRATA_AGENT_COUNTERS"] = "1"
    if "/build-hip/" in str(cfg.get("exe", "")):
        server_env["HIP_VISIBLE_DEVICES"] = os.environ.get("HIP_VISIBLE_DEVICES", "0")   # run-pancake.sh: GPU 1 is the iGPU
    return cfg, env, server_env, args


def preflight(cfg, repo, out, a):
    """[(level, text)]: level ok | WARN | FAIL.  FAIL stops --run."""
    res = []
    add = lambda lvl, t: res.append((lvl, t))   # noqa: E731
    exe = Path(str(cfg.get("exe", "")))
    if exe.name != "strata":
        add("FAIL", f"exe is {exe.name}: only the main engine (strata) writes the window trace; strata-poly does not")
    elif not exe.exists():
        add("WARN", f"{exe} does not exist (build it before the session)")
    else:
        add("ok", f"engine {exe}")
    args = cfg.get("args", [])
    paths = [x for x in args if isinstance(x, str) and x.startswith("/") and not x.startswith(str(out))]
    missing = [x for x in paths if not Path(x).exists()]
    add("WARN" if missing else "ok", f"{len(paths) - len(missing)} of {len(paths)} files named in the arguments exist"
        + (f"; missing: {', '.join(missing[:4])}" if missing else ""))
    py = repo / ".venv/bin/python"
    add("ok" if py.exists() else "WARN", f"server python {py}" + ("" if py.exists() else " does not exist"))
    if "--no-pool" in args:
        add("FAIL", "--no-pool in the preset: the routed ids reach the host through the pool, so nothing can be recorded")
    if "--spec-split" in args:
        add("ok", "--spec-split: windows come as two token groups per layer; check and convert merge them")
    env = cfg["env"]
    bias = altering(env)
    if bias and not a.allow_biased_routing:
        add("FAIL", f"{', '.join(bias)}: the engine changes the routing towards the cache, so the recorded ids are not the "
                    "model's (every replay would be circular); use a preset without it, or --allow-biased-routing")
    elif bias:
        add("WARN", f"{', '.join(bias)}: the recorded routing is the cache-biased router's")
    if publishes_weights(env):
        add("ok", "router weights: published by the deferral split (STRATA_VERIFY_DEFER_CPU=1) for the windows captured "
                  "with it; `check` reports how many layer records carry them")
    else:
        add("WARN", "router weights: this preset's doorbell publishes none (default path), so the trace will carry zeros "
                    "unless the engine has the patch in the notes (verify.cpp); add --publish-weights to turn the deferral on")
    if a.norms:
        add("WARN", "--norms: STRATA_WINDOW_TRACE_NORMS is read only by an engine with the record-6 patch (pancake-docs/batch4/"
                    "corpus-recorder-engine.patch); any other build ignores it and `check` will say no norms were recorded")
    if env.get("STRATA_CPU_PRESUM") == "1" and a.norms:
        add("FAIL", "STRATA_CPU_PRESUM=1 pre-sums the CPU rows: per-entry output norms cannot be taken")
    if cfg.get("replicas") or cfg.get("layer_split"):
        add("WARN", "replicas / layer_split: the window trace is one engine's verify windows")
    anc = out
    while not anc.exists():
        anc = anc.parent
    keep = shutil.disk_usage(anc).free
    need = (a.max_mb + 1024) << 20
    add("ok" if keep >= need else "WARN", f"disk: {keep >> 30} GiB free at {anc}, the trace may take {a.max_mb} MiB "
        f"(+ routing.bin and the logs)")
    try:
        avail = int([x for x in open("/proc/meminfo") if x.startswith("MemAvailable")][0].split()[1]) >> 20
        add("ok" if avail >= 48 else "WARN", f"MemAvailable {avail} GiB (capped.sh waits for 48 GiB, up to 10 minutes)")
    except (OSError, IndexError, ValueError):
        pass
    return res


def cmd_plan(a):
    repo, out = Path(a.repo).resolve(), Path(a.out).resolve()
    cfg, path, left = load_preset(a.preset, repo, Path(a.data), Path(a.models))
    if left:
        sys.exit(f"{path}: unresolved placeholders {left}")
    taken = [f for f in FILES.values() if (out / f).exists()]
    if taken:
        sys.exit(f"{out} already holds {', '.join(taken)}: a recording is never overwritten (the turn log appends); use a new directory")
    derived, env_added, server_env, args_added = build(cfg, out, a)
    checks = preflight(derived, repo, out, a)
    port = a.port or derived.get("port", 8080)
    srv = [str(repo / "tools/pancake/capped.sh"), str(repo / ".venv/bin/python"), str(repo / "serve/server.py"), "--engine",
           "strata", "--config", str(out / "config.json"), "--port", str(port)]
    line = " ".join([f"{k}={shlex.quote(v)}" for k, v in server_env.items()] + [shlex.quote(x) for x in srv])
    print(f"== record_session plan: {path.stem} -> {out}")
    for lvl, t in checks:
        print(f"  [{lvl:>4}] {t}")
    failed = [t for lvl, t in checks if lvl == "FAIL"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(derived, indent=1) + "\n")
    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        head = ""
    plan = {"preset": path.stem, "name": a.name or out.name, "repo": str(repo), "git": head, "port": port, "max_mb": a.max_mb,
            "every": a.every, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "command": line,
            "env_added": env_added, "args_added": args_added, "server_env": server_env,
            "checks": [{"level": lvl, "text": t} for lvl, t in checks]}
    (out / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    print(f"\nfiles: {out}/config.json (the preset with the recorders on), plan.json; the session writes "
          + ", ".join(FILES.values()))
    print("\nstart the server (nothing was started; --run does it):\n  cd " + shlex.quote(str(repo)) + " && " + line)
    print(f"then work as usual against http://127.0.0.1:{port}/v1 (the agent client's base URL) and stop the server "
          "with Ctrl-C when done.  A smoke run first (a few minutes, then `check --min-windows 5 --min-turns 1`) shows "
          "whether this build's trace carries router weights.")
    print(f"\nafter the session:\n  python3 {repo / 'tools/pancake/record_session.py'} check {out}\n"
          f"  python3 {repo / 'tools/pancake/record_session.py'} convert {out}")
    if failed:
        print("\nnot started: " + "; ".join(failed), file=sys.stderr)
        return 2
    if a.run:
        env = {**os.environ, **server_env}
        print("\n== running (foreground) ==", flush=True)
        try:
            subprocess.run(srv, env=env, cwd=repo)
        except KeyboardInterrupt:
            pass
        print("\n== server exited: check ==", flush=True)
        return cmd_check(argparse.Namespace(dir=str(out), min_windows=a.min_windows, min_turns=a.min_turns, require_weights=False,
                                            gap_ms=rec.GAP_MS, json=False))
    return 0


# ------------------------------------------------------------------------------------------------ check

def log_lines(path, patterns):
    """{name: first matching line} for the regexes in `patterns`, streaming the log."""
    found = {}
    with open(path, errors="replace") as f:
        for line in f:
            for name, rx in patterns.items():
                if name not in found and rx.search(line):
                    found[name] = line.strip()[:200]
    return found


def cmd_check(a):
    d = Path(a.dir).resolve()
    plan = json.loads((d / "plan.json").read_text()) if (d / "plan.json").exists() else {}
    man = {"name": plan.get("name", d.name), "dir": str(d), "checked": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "preset": plan.get("preset"), "git": plan.get("git"), "files": {}}
    problems, warnings = [], []
    st = turns = rt = hid = None
    tp = d / FILES["trace"]
    if not tp.exists() or tp.stat().st_size == 0:
        problems.append(f"{FILES['trace']}: missing or empty (the window trace never opened)")
    else:
        try:
            st = rec.trace_stats(tp, a.gap_ms)
        except ValueError as e:
            problems.append(str(e))
    if st:
        man["files"][FILES["trace"]] = {"bytes": st["bytes"], "sha256": rec.sha256(tp)}
        w = st["weights"]
        if st["decode_windows"] < a.min_windows:
            problems.append(f"{st['decode_windows']} decode windows, fewer than --min-windows {a.min_windows}")
        if st["tail_bytes"]:
            warnings.append(f"the trace ends in the middle of a record ({st['tail_bytes']} bytes): the engine was killed; "
                            "the whole records before it are used")
        if st["incomplete_windows"] > 1 + st["windows"] // 100:
            warnings.append(f"{st['incomplete_windows']} of {st['windows']} windows do not hold all {st['header']['n_layers']} layers")
        if st["ids"]["out_of_range"] or st["ids"]["rows_with_duplicate_expert"]:
            warnings.append(f"ids: {st['ids']['out_of_range']} entries out of range, {st['ids']['rows_with_duplicate_expert']} "
                            "rows name an expert twice")
        if w["ok_share"] != 1.0:
            why = ", ".join(f"{w[s]} {s}" for s in ("missing", "zero", "stale", "bad") if w[s])
            msg = (f"router weights: {w['ok']} of {w['records']} layer records carry real weights ({why}); "
                   f"{w['windows_all_ok']} of {st['windows']} windows are all-ok")
            (problems if a.require_weights else warnings).append(msg)
        if w["ok"] and w["rows_sum_to_one_share"] is not None and w["rows_sum_to_one_share"] < 0.99:
            warnings.append(f"only {w['rows_sum_to_one_share']:.1%} of the weight rows sum to 1 (a router without top-k renormalisation?)")
    sizes = {}
    for key in ("turns", "routing", "log"):
        p = d / FILES[key]
        if p.exists():
            sizes[key] = p
            man["files"][FILES[key]] = {"bytes": p.stat().st_size}
    if "turns" in sizes:
        turns = rec.turn_stats(sizes["turns"])
        if turns["turns"] < a.min_turns:
            problems.append(f"{turns['turns']} turns in the turn log, fewer than --min-turns {a.min_turns}")
        if turns["malformed_lines"]:
            warnings.append(f"turn log: {turns['malformed_lines']} lines are not usable records")
        if st and abs(turns["generating"] - st["turns"]["count"]) > max(3, 0.25 * turns["generating"]):
            warnings.append(f"turn starts in the window trace ({st['turns']['count']}) and generating requests in the turn "
                            f"log ({turns['generating']}) differ: the trace's turn cuts are approximate")
    else:
        problems.append(f"{FILES['turns']}: missing (the server's turn log was off or no request finished)")
    if "routing" in sizes:
        rt = rec.routing_stats(sizes["routing"], st["header"]["n_layers"] if st else 48)
        if st and rt["windows"] != st["windows"]:
            warnings.append(f"routing.bin holds {rt['windows']} windows, the window trace {st['windows']} (EVERY > 1, a cap, or a "
                            "window dropped before its commit)")
        elif st and rt["windows"]:
            rt["ids_match_trace"] = rec.compare_ids(tp, sizes["routing"])
            if not rt["ids_match_trace"]:
                warnings.append("routing.bin and the window trace disagree on the first windows' ids")
        if rt.get("tail_bytes"):
            warnings.append(f"routing.bin ends in the middle of a record ({rt['tail_bytes']} bytes)")
    else:
        warnings.append("routing.bin: absent (the plan writes it unless --no-routing-dump)")
    if (d / FILES["hidden"]).is_dir():
        hid = rec.hidden_stats(d / FILES["hidden"])
        if hid["bad_files"]:
            warnings.append(f"hidden dump: {hid['bad_files']} files are not whole SHDM dumps")
    if "log" in sizes:
        rx = {"opened": re.compile(r"STRATA_WINDOW_TRACE: verify windows ->"), "cap": re.compile(r"STRATA_WINDOW_TRACE reached its cap"),
              "cannot": re.compile(r"STRATA_WINDOW_TRACE: cannot write"), "hidden": re.compile(r"STRATA_HIDDEN_DUMP: .*is not a directory")}
        got = log_lines(sizes["log"], rx)
        man["engine_log"] = got
        if "cannot" in got:
            problems.append("engine log: " + got["cannot"])
        if "cap" in got:
            warnings.append(f"the window trace reached its cap ({plan.get('max_mb', '?')} MiB): later windows are not in it")
        if "opened" not in got:
            warnings.append("engine log: no 'STRATA_WINDOW_TRACE: verify windows ->' line (a build without item 64, or the engine never started)")
    if "log" not in sizes:
        warnings.append("engine.log: absent (the engine's start-up lines say whether each recorder opened)")
    man["window_trace"] = rec.public(st) if st else None
    man["turn_log"], man["routing_dump"], man["hidden_dump"] = turns, rt, hid
    man["usable_for"] = {
        "ids (time_ceiling, finemoe_replay, cache_learn, replay_seeding, selftune)": bool(st and st["decode_windows"] >= a.min_windows),
        "router weights (reap_saliency gate part, gate_importance, finemoe maps)": bool(st and st["weights"]["ok_share"] == 1.0),
        "placement, ms, drafts (ecospec_sim, draft_costs, tail_replay, disco_fit)": bool(st and st["drafts"]["windows_with_confidences"]),
        "turn log (replay_turns)": bool(turns and turns["turns"] >= a.min_turns),
        "expert output norms (reap_saliency --act-norm)": bool(st and st["norms"]["records"]),
    }
    man["problems"], man["warnings"] = problems, warnings
    man["verdict"] = "FAIL" if problems else "ok" if all(man["usable_for"].values()) else "usable, not for everything"
    if man["files"]:
        (d / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    if a.json:
        print(json.dumps(man, indent=1))
    else:
        print(f"== check {d}: {man['verdict']}")
        if st:
            print(f"  window trace: {st['windows']} windows ({st['decode_windows']} decode, {st['merged_prompt_windows']} merged "
                  f"prompt), {st['bytes'] / 2**20:.1f} MiB, {st['turns']['count']} turn starts, median {st['ms']['median']} ms, "
                  f"VRAM hit rate {st['placement']['hit_rate']}, drafts accepted {st['drafts']['rate']}")
        if turns:
            print(f"  turn log: {turns['turns']} turns in {turns['sessions']} sessions, {turns['prompt_tokens']:,} prompt tokens "
                  f"({turns['reused_share']} reused), largest prompt {turns['max_prompt']:,}")
        if rt:
            print(f"  routing.bin: {rt['windows']} windows, {rt['records']} records, k {rt.get('k')}")
        for k, v in man["usable_for"].items():
            print(f"  {'yes' if v else 'NO ':>3}  {k}")
        for t in problems:
            print(f"  PROBLEM {t}")
        for t in warnings:
            print(f"  warning {t}")
        if man["files"]:
            print(f"  manifest: {d / 'manifest.json'}")
    return 2 if problems else 0


# ------------------------------------------------------------------------------------------------ convert

TOOLS = (
    ("time_ceiling.py  (A6 time model)", "python3 tools/pancake/time_ceiling.py {bin} data/expert-profile-pancake.bin {slots} --validate"),
    ("finemoe_replay.py (FineMoE)", "python3 tools/pancake/finemoe_replay.py data/expert-profile-pancake.bin {slots} {bin} {cuts} --json {out}/finemoe.json"),
    ("reap_saliency.py (REAP)", "python3 tools/pancake/reap_saliency.py {stwt} --profile data/expert-profile-pancake.bin --slots {slots}{norm}"),
    ("gate_importance.py (HOBBIT proxy)", "python3 tools/pancake/gate_importance.py {stwt}"),
    ("ecospec_sim.py (EcoSpec)", "python3 tools/pancake/ecospec_sim.py {stwt}"),
    ("draft_costs.py / tail_replay.py", "python3 tools/pancake/trace/draft_costs.py {stwt}; python3 tools/pancake/trace/tail_replay.py {stwt}"),
    ("replay_turns.py (the cache arms)", "python3 tools/pancake/replay_turns.py SPEC.json --out DIR --turns {turns} --list-sessions"),
)


def cmd_convert(a):
    d = Path(a.dir).resolve()
    tp = d / FILES["trace"]
    if not tp.exists():
        sys.exit(f"{tp}: missing")
    out = Path(a.out).resolve() if a.out else d / "replay"
    info = rec.export(tp, out, a.weights, decode_only=not a.all_windows)
    if (d / FILES["turns"]).exists():
        info["turns"] = rec.clean_turns(d / FILES["turns"], out / "turns.jsonl")
    (out / "convert.json").write_text(json.dumps(info, indent=1) + "\n")
    print(f"== convert {d} -> {out}")
    if info.get("routing_bin_skipped"):
        print(f"  routing.bin NOT written: {info['routing_bin_skipped']}")
    else:
        print(f"  routing.bin   {info['windows_routing_bin']} {'decode ' if info['decode_only'] else ''}windows")
    print(f"  session.stwt  {info['windows_stwt']} windows, weights: {info['weights']}"
          + (f" ({info['weights_reason']}; --weights valid keeps the records that have them)" if info["weights_reason"] else ""))
    if "turns" in info:
        print(f"  turns.jsonl   {info['turns']['turns']} turns ({info['turns']['dropped_lines']} lines dropped)")
    if "act_norm_coverage" in info:
        print(f"  act_norm.npy  mean expert output norm per (layer, expert), {info['act_norm_coverage']:.1%} measured")
    cuts = f"--cuts {','.join(map(str, info['cuts']))}" if info["cuts"] else ""
    print(f"\nreplay (from the repo root; SLOTS is the expert-cache size of the run, {a.slots} on the 2026-11-08 trace):")
    fmt = dict(bin=out / "routing.bin", stwt=out / "session.stwt", out=out, slots=a.slots, cuts=cuts,
               norm=f" --act-norm {out / 'act_norm.npy'}" if "act_norm_coverage" in info else "", turns=out / "turns.jsonl")
    for name, tmpl in TOOLS:
        print(f"  # {name}\n  {' '.join(tmpl.format(**fmt).split())}")
    return 0


# ------------------------------------------------------------------------------------------------ cli

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="build and print the server command (--run starts it)")
    p.add_argument("--preset", required=True, help="a configs/pancake preset name (flashnext-iq2_xs-fast-hip) or a json path")
    p.add_argument("--out", required=True, help="the recording's directory (new)")
    p.add_argument("--name", help="a label for the manifest (default: the directory name)")
    p.add_argument("--repo", default=str(REPO))
    p.add_argument("--data", default=os.environ.get("STRATA_DATA", "/home/dan/clone-of-d/Strata-data"))
    p.add_argument("--models", default=os.environ.get("MODELS", "/home/dan/clone-of-d/models"))
    p.add_argument("--port", type=int, help="default: the preset's")
    p.add_argument("--max-mb", type=int, default=4096, help="the window trace's cap (STRATA_WINDOW_TRACE_MAX_MB)")
    p.add_argument("--every", type=int, default=1, help="keep every N-th window (STRATA_WINDOW_TRACE_EVERY)")
    p.add_argument("--hidden-dump", type=int, nargs="?", const=64, metavar="EVERY", help="also STRATA_HIDDEN_DUMP (A19), every N thinking tokens")
    p.add_argument("--norms", action="store_true", help="STRATA_WINDOW_TRACE_NORMS=1 (needs the engine patch in the notes)")
    p.add_argument("--publish-weights", action="store_true", help="add STRATA_VERIFY_DEFER_CPU=1 so the doorbell publishes the weights "
                   "(the fast presets have it; it sums the CPU rows in another order)")
    p.add_argument("--no-routing-dump", action="store_true")
    p.add_argument("--no-agent-counters", action="store_true")
    p.add_argument("--allow-biased-routing", action="store_true")
    p.add_argument("--run", action="store_true", help="start the server (foreground, capped) and check when it exits")
    p.add_argument("--min-windows", type=int, default=200)
    p.add_argument("--min-turns", type=int, default=5)
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("check", help="validate a recording, write manifest.json")
    p.add_argument("dir")
    p.add_argument("--min-windows", type=int, default=200, help="decode windows needed (the bench trace had 219)")
    p.add_argument("--min-turns", type=int, default=5)
    p.add_argument("--require-weights", action="store_true", help="router weights that are not all real are a problem, not a warning")
    p.add_argument("--gap-ms", type=float, default=rec.GAP_MS, help="a window this long after its predecessor starts a turn")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("convert", help="write the replay tools' inputs to DIR/replay")
    p.add_argument("dir")
    p.add_argument("--out", help="default DIR/replay")
    p.add_argument("--weights", choices=("auto", "valid", "none"), default="auto",
                   help="auto: keep the weights only if every exported window has real ones; valid: keep the records that have them")
    p.add_argument("--all-windows", action="store_true", help="routing.bin with the merged prompt windows too (default: decode windows)")
    p.add_argument("--slots", type=int, default=10909)
    p.set_defaults(fn=cmd_convert)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
