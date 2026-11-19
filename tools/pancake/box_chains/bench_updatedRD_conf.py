#!/usr/bin/env python3
"""bench_updatedRD_conf.py - the two server configs of bench-updatedRD.sh.

  bench_updatedRD_conf.py PRESET PLAIN_SRC UPDATED_SRC PLAIN_BUILD UPDATED_BUILD OUT_PLAIN OUT_UPDATED

PRESET is the box's filled-in pancake preset (model paths, port, lib dirs).  Both configs keep its model arguments, drop --grammar and --lookup-max-t (the plain
build has neither: Strata-rdna3's windows are 8 rows and its engine has no --grammar; the updated config gets both back from the recommended-flags file), point --expert-profile at Strata-rdna3's own profile (the plain build must not run the pancake line's learned
one) and take their env from Strata-rdna3's own setup config (rdna3/configs/iq3-1gpu.json: what the plain build runs with by default).  The updated config is the
plain one plus the shipped recommended-flags file of the updated tree, applied the way rdna3/run.py applies it (rdna3/tuned_flags.py).
Since 2026-11-19 night that file is the whole pancake daily preset on top of Strata-rdna3 (every setting Strata-rdna3 lacks, with the pancake value;
Strata-rdna3's own value stays where both have one): the same diff as spec 250's updated arm."""
import json
import sys

DROP_ARGS = ("--grammar", "--lookup-max-t")
DROP_KEYS = ("grammar", "reasoning_budget_grace", "reasoning_close_retry", "env_to_user", "effort_position", "turn_log")


def build(preset, plain_src, upd_src, bp, bu, tf, hw=None):
    p = json.load(open(preset))
    rd = json.load(open(plain_src + "/rdna3/configs/iq3-1gpu.json"))
    args, keep, i = p["args"], [], 0
    while i < len(args):
        if args[i] in DROP_ARGS:
            i += 2
            continue
        keep.append(args[i])
        i += 1
    if "--expert-profile" in keep:   # the plain build runs with Strata-rdna3's own profile, not the pancake line's learned one
        keep[keep.index("--expert-profile") + 1] = plain_src + "/data/expert-profile.bin"
    p["args"] = keep
    p["env"] = {k: str(v).replace("@STRATA@", plain_src) for k, v in rd["env"].items()}
    for k in DROP_KEYS:
        p.pop(k, None)
    plain = json.loads(json.dumps(p))
    plain["exe"] = bp + "/strata"
    upd = json.loads(json.dumps(plain))
    upd["exe"] = bu + "/strata"
    rec, path = tf.load_recommended(hw or {"gfx": "gfx1100", "rocm": "10.0", "vram_gib": 20}, upd_src + "/rdna3/recommended-flags")
    if rec:
        tf.apply_to_config(upd, rec, env={}, root=upd_src)
    return plain, upd, path


def main(argv):
    preset, plain_src, upd_src, bp, bu, out_plain, out_upd = argv[:7]
    sys.path.insert(0, upd_src + "/rdna3")
    import tuned_flags as tf
    plain, upd, path = build(preset, plain_src, upd_src, bp, bu, tf)
    json.dump(plain, open(out_plain, "w"), indent=1)
    json.dump(upd, open(out_upd, "w"), indent=1)
    print("recommended file:", path, "env keys added:", sorted(set(upd["env"]) - set(plain["env"])) + [k for k in upd["env"] if upd["env"][k] != plain["env"].get(k)])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
