"""hipblaslt_extend.py - pancake-strata MOD_PLAN item 80: extend a hipBLASLt tuning table's T-bucket coverage.

The engine picks, per call shape (dtype, N, K, ldy), the table row whose T bucket is closest to the call's T
(src/prefill/hipblaslt_tuning.hpp, TuningTable::closest).  The gfx1100 table for hipBLASLt 1.4.1
(tools/hip/gfx1100-hipblaslt-100401.txt) has buckets 4096 and 8192 only for most shapes, while prompt chunks go up to
32768 (PREFILL_AUTO_MAX) and agent deltas are 1-3K tokens: those all take the 4096 / 8192 solution.  This script plans the
missing buckets, prints (or runs) the tools/hip/tune_hipblaslt command that calibrates them on the card, and merges the
calibrated rows into a new table.  It never touches the GPU by itself: `run` is the only subcommand that starts the tuner.

Table format (one row per line after the header; '#' starts a comment):
    STRATA_HIPBLASLT_TUNING_V1 <arch> <hipBLASLt version>
    <f16|bf16> <N> <K> <ldy> <T bucket> <solution id>

Subcommands:
    plan   list the shapes and the buckets they lack; print the tuner command (--script FILE writes it as a shell script)
    run    the same, then start the tuner (needs the GPU; for the test phase only) and merge its output
    merge  base table + the tuner's --tuning-out file -> a new table (existing rows kept unless --replace)

    python tools/pancake/hipblaslt_extend.py plan --table tools/hip/gfx1100-hipblaslt-100401.txt
    python tools/pancake/hipblaslt_extend.py run --table tools/hip/gfx1100-hipblaslt-100401.txt \\
        --tuner build-hip/tune_hipblaslt --out tools/hip/gfx1100-hipblaslt-100401-ext.txt
    python tools/pancake/hipblaslt_extend.py merge --table tools/hip/gfx1100-hipblaslt-100401.txt \\
        --tuned /tmp/tuned.txt --out tools/hip/gfx1100-hipblaslt-100401-ext.txt

Shapes come from the table itself and, with --fallback-log, from an engine log run with STRATA_HIPBLASLT_VERBOSE=1
(its "fallback shape dtype=... N=... K=... ldy=..." and "Lt fallback; no calibration for dtype=..." lines).  By default a
shape whose largest bucket is below --min-shape-bucket (4096) is skipped: those are decode-sized shapes (the f16 1280 rows
bucketed 8..999), not prompt chunks; --all-shapes keeps them.

A new bucket changes which solution a call of that size runs only if the tuner picks a different id than the bucket it
used before; such a call's bits can change (same FP32-accumulate class).  Point STRATA_HIPBLASLT_TUNING at the new file
and A/B it against the shipped table before adopting it.
"""
import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

HEADER = "STRATA_HIPBLASLT_TUNING_V1"
DEFAULT_TARGETS = (1024, 2048, 3072, 16384, 32768)
FALLBACK_RE = re.compile(r"dtype=(f16|bf16)\s+T=(\d+)\s+N=(\d+)\s+K=(\d+)\s+ldy=(\d+)")


class TableError(ValueError):
    pass


def load_table(path):
    """-> (arch, version, {(dtype, n, k, ldy, t): solution_id}); the same checks as TuningTable::load."""
    arch = version = None
    rows = {}
    for number, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if arch is None:
            if len(fields) != 3 or fields[0] != HEADER:
                raise TableError(f"{path}:{number}: invalid tuning header")
            arch, version = fields[1], int(fields[2])
            continue
        if len(fields) != 6 or fields[0] not in ("f16", "bf16"):
            raise TableError(f"{path}:{number}: invalid tuning row")
        n, k, ldy, t, sol = (int(x) for x in fields[1:])
        if n <= 0 or k <= 0 or ldy < n or t <= 0 or sol < 0:
            raise TableError(f"{path}:{number}: invalid tuning row")
        key = (fields[0], n, k, ldy, t)
        if key in rows:
            raise TableError(f"{path}:{number}: duplicate tuning key")
        rows[key] = sol
    if arch is None:
        raise TableError(f"{path}: tuning file has no header")
    return arch, version, rows


def shapes_of(rows):
    """{(dtype, n, k, ldy): sorted buckets}"""
    out = {}
    for dtype, n, k, ldy, t in rows:
        out.setdefault((dtype, n, k, ldy), []).append(t)
    return {s: sorted(ts) for s, ts in out.items()}


def fallback_shapes(log_path):
    found = set()
    for line in Path(log_path).read_text(errors="replace").splitlines():
        if "fallback" not in line:
            continue
        m = FALLBACK_RE.search(line)
        if m:
            found.add((m.group(1), int(m.group(3)), int(m.group(4)), int(m.group(5))))
    return found


def plan(rows, targets, min_shape_bucket=4096, all_shapes=False, extra_shapes=()):
    """-> (cases [(dtype, t, n, k, ldy)], skipped shapes).  Cases are the target buckets each shape lacks."""
    shapes = shapes_of(rows)
    for s in extra_shapes:
        shapes.setdefault(s, [])
    cases, skipped = [], []
    for shape in sorted(shapes):
        buckets = shapes[shape]
        if buckets and not all_shapes and max(buckets) < min_shape_bucket:
            skipped.append(shape)
            continue
        dtype, n, k, ldy = shape
        for t in targets:
            if t not in buckets:
                cases.append((dtype, t, n, k, ldy))
    return cases, skipped


def tuner_command(tuner, cases, tuned_out, workspace_mib=None):
    cmd = [str(tuner)]
    if workspace_mib is not None:
        cmd += ["--workspace-mib", str(workspace_mib)]
    for dtype, t, n, k, ldy in cases:
        cmd += ["--case", f"{dtype},{t},{n},{k},{ldy}"]
    cmd += ["--tuning-out", str(tuned_out)]
    return cmd


def merge(base, tuned, replace=False):
    """base/tuned: (arch, version, rows) -> (rows, added, replaced, kept_conflicts)."""
    b_arch, b_ver, b_rows = base
    t_arch, t_ver, t_rows = tuned
    if (b_arch, b_ver) != (t_arch, t_ver):
        raise TableError(f"the tuned table is for {t_arch} {t_ver}, the base table for {b_arch} {b_ver}")
    rows = dict(b_rows)
    added = replaced = 0
    conflicts = []
    for key, sol in t_rows.items():
        if key not in rows:
            rows[key] = sol
            added += 1
        elif rows[key] != sol:
            if replace:
                rows[key] = sol
                replaced += 1
            else:
                conflicts.append((key, rows[key], sol))
    return rows, added, replaced, conflicts


def format_table(arch, version, rows, note=None):
    lines = ["# solution IDs are scoped to this hipBLASLt version and device architecture"]
    if note:
        lines.append(f"# {note}")
    lines.append(f"{HEADER} {arch} {version}")
    for (dtype, n, k, ldy, t) in sorted(rows, key=lambda r: (r[0], r[1], r[2], r[3], r[4])):
        lines.append(f"{dtype} {n} {k} {ldy} {t} {rows[(dtype, n, k, ldy, t)]}")
    return "\n".join(lines) + "\n"


def parse_targets(text):
    out = []
    for x in text.split(","):
        v = int(x)
        if v <= 0:
            raise argparse.ArgumentTypeError(f"bad bucket {x}")
        out.append(v)
    return tuple(sorted(set(out)))


def write_merged(args, base, tuned_path):
    tuned = load_table(tuned_path)
    rows, added, replaced, conflicts = merge(base, tuned, args.replace)
    note = f"extended by tools/pancake/hipblaslt_extend.py from {Path(args.table).name} + {Path(tuned_path).name}"
    Path(args.out).write_text(format_table(base[0], base[1], rows, note))
    print(f"merged: {len(base[2])} base rows + {added} new, {replaced} replaced -> {len(rows)} rows in {args.out}")
    for key, old, new in conflicts:
        print(f"kept base row {' '.join(map(str, key))} solution {old} (tuner gave {new}; --replace takes it)")
    load_table(args.out)   # the engine's checks on the result
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("--table", required=True, help="the base table (tools/hip/<arch>-hipblaslt-<version>.txt)")
        p.add_argument("--targets", type=parse_targets, default=DEFAULT_TARGETS, help="T buckets to add (comma list)")
        p.add_argument("--min-shape-bucket", type=int, default=4096)
        p.add_argument("--all-shapes", action="store_true")
        p.add_argument("--fallback-log", action="append", default=[], help="engine log run with STRATA_HIPBLASLT_VERBOSE=1")
        p.add_argument("--tuner", default="build-hip/tune_hipblaslt")
        p.add_argument("--workspace-mib", type=int, default=None, help="tuner workspace limit (its default: 32)")
        p.add_argument("--tuned", default="hipblaslt-tuned.txt", help="the tuner's --tuning-out file")
        if name == "plan":
            p.add_argument("--script", help="write the tuner command to this shell script")
        else:
            p.add_argument("--log", default=None, help="tuner stdout (default: <tuned>.log)")
            p.add_argument("--out", required=True, help="the merged table")
            p.add_argument("--replace", action="store_true")
    m = sub.add_parser("merge")
    m.add_argument("--table", required=True)
    m.add_argument("--tuned", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--replace", action="store_true", help="a tuned row replaces a base row with the same key")
    args = ap.parse_args(argv)

    base = load_table(args.table)
    if args.cmd == "merge":
        return write_merged(args, base, args.tuned)

    extra = set()
    for log in args.fallback_log:
        extra |= fallback_shapes(log)
    cases, skipped = plan(base[2], args.targets, args.min_shape_bucket, args.all_shapes, extra)
    shapes = shapes_of(base[2])
    print(f"table {args.table}: {base[0]} hipBLASLt {base[1]}, {len(base[2])} rows, {len(shapes)} shapes"
          f" (+{len(extra - set(shapes))} from fallback logs)")
    for s in skipped:
        print(f"skip {' '.join(map(str, s))}: largest bucket {max(shapes[s])} < {args.min_shape_bucket} (--all-shapes keeps it)")
    print(f"{len(cases)} cases to calibrate (buckets {','.join(map(str, args.targets))}):")
    for c in cases:
        print("  case " + ",".join(map(str, c)))
    if not cases:
        return 0
    cmd = tuner_command(args.tuner, cases, args.tuned, args.workspace_mib)
    text = " ".join(shlex.quote(x) for x in cmd)
    if args.cmd == "plan":
        print("tuner command (needs the GPU):\n" + text)
        if args.script:
            Path(args.script).write_text("#!/bin/sh\n# generated by tools/pancake/hipblaslt_extend.py plan\nset -e\n"
                                         + text + "\n")
            Path(args.script).chmod(0o755)
            print(f"wrote {args.script}")
        return 0

    log = Path(args.log or (str(args.tuned) + ".log"))
    print(f"running the tuner, stdout -> {log}")
    with open(log, "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    if rc != 0:
        print(f"tuner exited {rc}; see {log}", file=sys.stderr)
        return rc
    lt_best = sum(1 for line in log.read_text(errors="replace").splitlines() if line.startswith("lt_best "))
    unsupported = [line for line in log.read_text(errors="replace").splitlines()
                   if line.startswith("lt_unsupported ") and "no_valid_heuristics" in line]
    print(f"tuner: {lt_best} of {len(cases)} cases calibrated, {len(unsupported)} without a valid solution")
    return write_merged(args, base, args.tuned)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except TableError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
