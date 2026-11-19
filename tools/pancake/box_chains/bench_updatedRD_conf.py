#!/usr/bin/env python3
"""bench_updatedRD_conf.py - the two server configs of bench-updatedRD.sh.

  bench_updatedRD_conf.py PRESET PLAIN_SRC UPDATED_SRC PLAIN_BUILD UPDATED_BUILD OUT_PLAIN OUT_UPDATED

Owner order 2026-11-19 (late; the 20:28 RD run was aborted because the configs were built from the pancake preset and dropped 9 of Strata-rdna3's
arguments and replaced 4 of its values, TEST_LEDGER.md): "use their full flags with their settings and add our remaining flags, don't override
their values".

PLAIN = Strata-rdna3's own setup config rdna3/configs/iq3-1gpu.json EXACTLY (every env key, every argument with its value, its sampling block,
every other key), with only these substitutions, each one printed to the chain log (SUBST lines):
  - the placeholders as rdna3/run.py fills them: @STRATA@ = PLAIN_SRC, @DATA@ = the box's data folder (from PRESET's --pack), @GPUS@ = [0]
    (run.py: list(range(1)) for a 1-GPU setup), @ROCM_LIBS@ = PRESET's lib_dirs (run.py takes them from the build's BUILD.json);
  - exe = PLAIN_BUILD/strata (the chain's build, not @STRATA@/engine/strata);
  - the MODEL: the box runs Qwen3.8-Flash-Next IQ2_XS like every other benchmark of the day: --pack, --native and the tokenizer are PRESET's
    IQ2_XS paths and PRESET's --ple-gguf (shard 2) is added after --native (the IQ3_S GGUF exists on the box, but its pack packs/iq3_s does not;
    the IQ2_XS model is used either way);
  - the "vision" section is dropped: the server would start the separate picture reader @STRATA@/engine/strata-vision (not built here) with
    the IQ3_S GGUF; a text benchmark never sends a picture.  The engine argument --vision stays: in the engine it only sets up the identity
    image-position table (generate.cpp, o.vision) and starts nothing;
  - harness keys the pancake server that abtest starts needs and Strata-rdna3's file does not have: "log" (PRESET's engine log: abtest reads
    the decode / prompt lines from it) and "drafting": "mtp" (their arguments name the MTP drafter: --mtp, --mtp-q4, --mtp-window; without
    the key the server's "auto" may start DFlash2, which the box has in Strata-data/dflash2 and their data folder does not);
  - --mtp-q4 all -> proj ONLY when the substituted model's output.weight is stored in fewer bytes per row than Q4_0 (a model-file
    dependency, not a preference; TEST_LEDGER.md T-987): the engine's --mtp-q4 head copy (src/core/mtp.cpp make_q4_head) requantizes the
    draft head's rows to Q4_0 and writes them back IN PLACE into the buffer sized for the head's own rows ("they are smaller"), which holds
    for their IQ3_S model (output.weight Q6_K, 2100 bytes per row) but not for the box's IQ2_XS model (output.weight IQ4_XS, 1360 bytes per
    row; Q4_0 needs 1440): the copy overruns the buffer and the engine exits 1 with "strata serve: mtp: writing the draft head".  "proj"
    keeps the Q4_0 projections of theirs; the head stays IQ4_XS, already fewer bytes per draft step than the Q4_0 copy would be.  The head
    type is read from the GGUF header (head_type below); an unreadable file keeps their value and says so.
UPDATED = PLAIN + only what PLAIN does not set: exe = UPDATED_BUILD/strata and the shipped rdna3/recommended-flags file of the updated tree,
applied the way rdna3/run.py applies it (rdna3/tuned_flags.py).  Every key and argument value of PLAIN must be unchanged in UPDATED (checked:
the script exits with an error naming the overridden value).  Where the pancake daily preset has another value for something Strata-rdna3 sets
(PF_F16, the hipBLASLt table, the expert profile, --max-context, --conversation-cache-mib, --mtp-draft-vocab, sampling), Strata-rdna3's value
stays in both configs and the pancake value is a spec arm (250 updated+pf16-1; 251 hipblaslt-100401, profile-pancake, draft-vocab-en,
max-context-131072, conversation-cache-2048; sampling: no arm, abtest sends temperature 0 in every request)."""
import json
import posixpath
import struct
import sys

# ggml type id -> (name, elements per block, bytes per block): the row size of a head row in its GGUF type
GGML_BLOCK = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24),
              8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
              13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292), 16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74),
              18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82),
              23: ("IQ4_XS", 256, 136), 29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66)}


def head_type(gguf_path, name="output.weight"):
    """(ggml type id, n_embd) of tensor NAME from a GGUF file's header, or None when the file cannot be read or has no such tensor."""
    scalar = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    try:
        with open(gguf_path, "rb") as f:
            def u(fmt):
                return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]

            def string():
                return f.read(u("Q"))

            def skip(t):
                if t in scalar:
                    f.seek(scalar[t], 1)
                elif t == 8:
                    f.seek(u("Q"), 1)
                elif t == 9:
                    et, n = u("I"), u("Q")
                    if et in scalar:
                        f.seek(scalar[et] * n, 1)
                    else:
                        for _ in range(n):
                            skip(et)
                else:
                    raise ValueError("gguf value type %d" % t)
            if f.read(4) != b"GGUF":
                return None
            u("I")
            n_t, n_kv = u("Q"), u("Q")
            for _ in range(n_kv):
                string()
                skip(u("I"))
            for _ in range(n_t):
                nm = string().decode("utf-8", "replace")
                dims = [u("Q") for _ in range(u("I"))]
                t = u("I")
                u("Q")
                if nm == name:
                    return t, dims[0]
    except (OSError, ValueError, struct.error):
        return None
    return None


def q4_head_fits(ht):
    """True/False: does the engine's in-place Q4_0 copy of the draft head fit the rows of head type HT ((type, n_embd)); None = unknown type."""
    t, n = ht
    if t not in GGML_BLOCK:
        return None
    _, be, bb = GGML_BLOCK[t]
    return n // 32 * 18 <= n // be * bb

# pancake preset items that are not added to the updated config, with the reason (printed as NOT-ADDED lines)
NOT_ADDED = {
    "STRATA_AGENT_COUNTERS": "log only (T-513); not read by updated-strata-rdna3's engine (a no-op arm key)",
    "env_to_user": "a key of the pancake server (the harness, the same server for both arms), not of updated-strata-rdna3",
    "effort_position": "a key of the pancake server (the harness), not of updated-strata-rdna3",
    "reasoning_budget_grace": "a key of the pancake server (the harness), not of updated-strata-rdna3",
    "reasoning_close_retry": "a key of the pancake server (the harness), not of updated-strata-rdna3",
    "turn_log": "a key of the pancake server (the harness), not of updated-strata-rdna3",
    "port": "abtest gives the port on the server's command line",
    "gpus_asked": "a key of the pancake terminal, not an engine setting",
    "cwd": "Strata-rdna3's own cwd (@STRATA@) is kept",
    "exe": "the chain's build",
    "model_name": "Strata-rdna3's own label is kept",
}


def fill(value, subst):
    """rdna3/run.py fill(): every placeholder in every string of the config."""
    if isinstance(value, str):
        for key, new in subst.items():
            value = value.replace(key, new)
        return value
    if isinstance(value, list):
        return [fill(v, subst) for v in value]
    if isinstance(value, dict):
        return {k: fill(v, subst) for k, v in value.items()}
    return value


def argmap(args):
    """{flag: value or None} of an engine argument list (a flag without a value, e.g. --stats, maps to None)."""
    out, i = {}, 0
    while i < len(args):
        a = str(args[i])
        if a.startswith("--"):
            if i + 1 < len(args) and not str(args[i + 1]).startswith("--"):
                out[a] = str(args[i + 1])
                i += 2
                continue
            out[a] = None
        i += 1
    return out


def set_arg(args, flag, value):
    i = args.index(flag)
    args[i + 1] = value


def overrides(plain, upd):
    """Every value of PLAIN that UPDATED changes or removes (empty = UPDATED is PLAIN plus additions only)."""
    bad = []
    for k, v in plain.items():
        if k in ("exe", "env", "args"):
            continue
        if upd.get(k) != v:
            bad.append("key %s: %r -> %r" % (k, v, upd.get(k)))
    for k, v in plain.get("env", {}).items():
        if upd.get("env", {}).get(k) != v:
            bad.append("env %s: %r -> %r" % (k, v, upd.get("env", {}).get(k)))
    pa, ua = argmap(plain.get("args", [])), argmap(upd.get("args", []))
    for k, v in pa.items():
        if k not in ua or ua[k] != v:
            bad.append("arg %s: %r -> %r" % (k, v, ua.get(k, "<removed>")))
    return bad


def build(preset, plain_src, upd_src, bp, bu, tf, hw=None, head_reader=head_type):
    """(plain, updated, recommended-file path, log lines).  HEAD_READER(gguf path) -> (ggml type, n_embd) of output.weight or None."""
    p = json.load(open(preset))
    their = json.load(open(plain_src + "/rdna3/configs/iq3-1gpu.json"))
    pa = argmap(p["args"])
    data = posixpath.dirname(posixpath.dirname(pa["--pack"]))      # <DATA>/packs/iq2_xs
    log = []
    plain = fill(their, {"@STRATA@": plain_src, "@DATA@": data})
    log.append("SUBST placeholders: @STRATA@ = %s, @DATA@ = %s" % (plain_src, data))
    if plain.get("gpu") == "@GPUS@":
        plain["gpu"] = [0]
        log.append("SUBST gpu: @GPUS@ -> [0] (rdna3/run.py: a 1-GPU setup runs on device 0)")
    if plain.get("lib_dirs") == "@ROCM_LIBS@":
        plain["lib_dirs"] = p.get("lib_dirs") or []
        log.append("SUBST lib_dirs: @ROCM_LIBS@ -> %s (rdna3/run.py takes the build's ROCm library folders)" % plain["lib_dirs"])
    log.append("SUBST exe: %s -> %s/strata (the chain's build)" % (plain["exe"], bp))
    plain["exe"] = bp + "/strata"
    args = plain["args"]
    for flag in ("--pack", "--native"):
        old = argmap(args)[flag]
        set_arg(args, flag, pa[flag])
        log.append("SUBST model %s: %s -> %s (Qwen3.8-Flash-Next IQ2_XS, the model of every benchmark of the day)" % (flag, old, pa[flag]))
    if "--ple-gguf" in pa and "--ple-gguf" not in args:
        i = args.index("--native") + 2
        args[i:i] = ["--ple-gguf", pa["--ple-gguf"]]
        log.append("SUBST model --ple-gguf %s added after --native (the IQ2_XS model's second shard, as the preset runs it)" % pa["--ple-gguf"])
    q4 = argmap(args).get("--mtp-q4")
    if q4 in ("all", "head"):
        ht = head_reader(pa["--native"])
        fits = q4_head_fits(ht) if ht else None
        if fits is False:
            nm, be, bb = GGML_BLOCK[ht[0]]
            new = "proj" if q4 == "all" else None
            why = ("model-file dependency (TEST_LEDGER.md T-987): the IQ2_XS GGUF's output.weight is %s, %d bytes per row; the engine's --mtp-q4 "
                   "head copy rewrites the draft head IN PLACE as Q4_0, %d bytes per row, which only fits a head stored in >= 4.5 bits (their "
                   "IQ3_S head is Q6_K) and otherwise exits 1 with \"mtp: writing the draft head\"; the projections stay Q4_0 as theirs, the head "
                   "stays %s (already fewer bytes per draft step)" % (nm, ht[1] // be * bb, ht[1] // 32 * 18, nm))
            if new:
                set_arg(args, "--mtp-q4", new)
            else:
                i = args.index("--mtp-q4")
                del args[i:i + 2]
            log.append("SUBST --mtp-q4 %s -> %s: %s" % (q4, new or "<removed>", why))
        elif fits is None:
            log.append("UNCHECKED --mtp-q4 %s kept: the head type of %s could not be read (%r)" % (q4, pa["--native"], ht))
        else:
            log.append("CHECKED --mtp-q4 %s kept: the model's output.weight (%s) is >= Q4_0 per row" % (q4, GGML_BLOCK[ht[0]][0]))
    log.append("SUBST model tokenizer: %s -> %s" % (plain.get("tokenizer"), p["tokenizer"]))
    plain["tokenizer"] = p["tokenizer"]
    if "vision" in plain:
        plain.pop("vision")
        log.append("DROPPED the \"vision\" section: the server would start the separate picture reader (engine/strata-vision, not built here; "
                   "the IQ3_S GGUF) and a text benchmark sends no picture; the engine argument --vision is KEPT (it only sets up the identity "
                   "image-position table, generate.cpp o.vision)")
    for k, v, why in (("log", p.get("log"), "abtest reads the engine's decode / prompt lines from it"),
                      ("drafting", "mtp", "their arguments name the MTP drafter (--mtp, --mtp-q4, --mtp-window); the server's default \"auto\" "
                                          "could start DFlash2 (Strata-data/dflash2 is on the box)")):
        if k not in plain and v is not None:
            plain[k] = v
            log.append("HARNESS key %s = %r added: %s" % (k, v, why))
    rest = [s for s in ("@STRATA@", "@DATA@", "@GPUS@", "@ROCM_LIBS@") if s in json.dumps(plain)]
    if rest:
        raise SystemExit("CONFIG: placeholders left in the plain config: %s" % rest)

    upd = json.loads(json.dumps(plain))
    upd["exe"] = bu + "/strata"
    rec, path = tf.load_recommended(hw or {"gfx": "gfx1100", "rocm": "10.0", "vram_gib": 20}, upd_src + "/rdna3/recommended-flags")
    if rec:
        tf.apply_to_config(upd, rec, env={}, root=upd_src)
    bad = overrides(plain, upd)
    if bad:
        raise SystemExit("CONFIG: the updated config overrides Strata-rdna3's values (owner rule: never): " + "; ".join(bad))
    added_env = {k: v for k, v in upd["env"].items() if k not in plain["env"]}
    ua = argmap(upd["args"])
    added_args = {k: v for k, v in ua.items() if k not in argmap(plain["args"])}
    log.append("UPDATED = plain + env %s + args %s (recommended file %s); no value of plain changed" % (json.dumps(added_env), json.dumps(added_args), path))
    # the pancake preset against the two configs: what is kept as theirs, what is added, what is not added and why
    for k, v in sorted(p.get("env", {}).items()):
        if k in plain["env"] and plain["env"][k] != v:
            log.append("THEIRS-KEPT env %s = %s (pancake %s: a spec arm, never the updated config)" % (k, plain["env"][k], v))
        elif k not in upd["env"]:
            log.append("NOT-ADDED env %s = %s: %s" % (k, v, NOT_ADDED.get(k, "not in the shipped recommended-flags file")))
    pl = argmap(plain["args"])
    for k, v in pa.items():
        if k in ("--pack", "--native", "--ple-gguf"):
            continue
        if k in pl and pl[k] != v:
            log.append("THEIRS-KEPT arg %s %s (pancake %s: a spec arm, never the updated config)" % (k, pl[k], v))
        elif k not in ua:
            log.append("NOT-ADDED arg %s %s: not in the shipped recommended-flags file" % (k, v))
    for k in sorted(set(p) - {"args", "env", "lib_dirs", "tokenizer", "log", "drafting", "exe", "gpu"}):
        if k in plain and plain[k] != p[k]:
            log.append("THEIRS-KEPT %s = %s (pancake %s)" % (k, json.dumps(plain[k]), json.dumps(p[k])))
        elif k not in plain:
            log.append("NOT-ADDED %s: %s" % (k, NOT_ADDED.get(k, "not a setting of updated-strata-rdna3")))
    return plain, upd, path, log


def main(argv):
    preset, plain_src, upd_src, bp, bu, out_plain, out_upd = argv[:7]
    sys.path.insert(0, upd_src + "/rdna3")
    import tuned_flags as tf
    plain, upd, path, log = build(preset, plain_src, upd_src, bp, bu, tf)
    json.dump(plain, open(out_plain, "w"), indent=1)
    json.dump(upd, open(out_upd, "w"), indent=1)
    for line in log:
        print("CONFIG " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
