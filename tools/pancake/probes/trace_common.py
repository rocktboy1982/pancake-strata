"""trace_common.py - shared helpers of the 4.2 probes: read a rocprofv3 kernel trace CSV and cut out the decode phase.

Not a probe itself.  The trace is what `rocprofv3 --kernel-trace --output-format csv` writes (<prefix>_kernel_trace.csv);
tools/pancake/gpu_profile.py produces it as OUTDIR/run_kernel_trace.csv.  Column names are matched case-insensitively and
a few spellings are accepted, so a different rocprofv3 version still reads.

The decode phase is cut the way gpu_profile.py does it: from the first combine / wait_flag kernel to the last, but not
before the last prompt (prefill) kernel.  `--whole` on the probes keeps the whole trace instead.
"""
import csv
import sys

ALIASES = {
    "name": ("Kernel_Name", "KernelName", "Name", "kernel_name"),
    "start": ("Start_Timestamp", "Start", "BeginNs", "start_timestamp"),
    "end": ("End_Timestamp", "End", "EndNs", "end_timestamp"),
    "wgx": ("Workgroup_Size_X", "Wg_Size_X", "workgroup_size_x"),
    "wgy": ("Workgroup_Size_Y", "Wg_Size_Y", "workgroup_size_y"),
    "wgz": ("Workgroup_Size_Z", "Wg_Size_Z", "workgroup_size_z"),
    "gx": ("Grid_Size_X", "Grid_Size_x", "grid_size_x"),
    "gy": ("Grid_Size_Y", "grid_size_y"),
    "gz": ("Grid_Size_Z", "grid_size_z"),
}


def _pick(header, names):
    low = {h.lower(): h for h in header}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def short(n):
    """Kernel name without the namespace noise and the argument list (template arguments are kept)."""
    n = n.replace("strata::kernels::(anonymous namespace)::", "").replace("strata::kernels::", "").replace("void ", "")
    depth = 0
    for i, ch in enumerate(n):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "(" and depth == 0:
            return n[:i]
    return n


def _load_nsys(rd):
    """`nsys stats --report cuda_gpu_kern_trace --format csv` (the RTX 3060): Start (ns), Duration (ns), GrdX.., BlkX.., Name."""
    rows = []
    for r in rd:
        def n(k):
            try:
                return int(float(r.get(k) or 1))
            except ValueError:
                return 1
        s, d = int(float(r["Start (ns)"])), int(float(r["Duration (ns)"]))
        wg = n("BlkX") * n("BlkY") * n("BlkZ")
        rows.append({"name": r["Name"], "start": s, "end": s + d, "dur": d, "wg": wg,
                     "blocks": n("GrdX") * n("GrdY") * n("GrdZ"), "grid": (n("GrdX"), n("GrdY"), n("GrdZ")),
                     "wgdim": (n("BlkX"), n("BlkY"), n("BlkZ"))})
    rows.sort(key=lambda r: r["start"])
    return rows


def load(path):
    """Rows of the trace as dicts: name, start, end (ns), dur (ns), blocks (workgroups), wg (threads per workgroup).
    Reads rocprofv3 kernel traces and nsys cuda_gpu_kern_trace CSVs."""
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        if rd.fieldnames and "Duration (ns)" in rd.fieldnames and "Start (ns)" in rd.fieldnames:
            return _load_nsys(rd)
        cols = {k: _pick(rd.fieldnames or [], v) for k, v in ALIASES.items()}
        for need in ("name", "start", "end"):
            if cols[need] is None:
                sys.exit(f"{path}: no {ALIASES[need][0]} column (header: {rd.fieldnames})")
        rows = []
        for r in rd:
            def num(k, default=1):
                c = cols[k]
                try:
                    return int(r[c]) if c and r.get(c, "") != "" else default
                except ValueError:
                    return default
            wg = num("wgx") * num("wgy") * num("wgz")
            grid = num("gx", 0) * num("gy") * num("gz")   # HSA grid size is in work-items
            s, e = num("start", 0), num("end", 0)
            rows.append({"name": r[cols["name"]], "start": s, "end": e, "dur": e - s,
                         "wg": wg, "blocks": (grid // wg) if (grid and wg) else 0,
                         "grid": (num("gx", 0), num("gy"), num("gz")), "wgdim": (num("wgx"), num("wgy"), num("wgz"))})
    rows.sort(key=lambda r: r["start"])
    return rows


def decode_window(rows, whole=False):
    """The kernels of the decode phase (see the module doc); the whole list with whole=True or when no marker is found."""
    if whole:
        return rows
    marks = [r["start"] for r in rows if "combine" in r["name"] or "wait_flag" in r["name"]]
    if not marks:
        print("note: no combine/wait_flag marker kernel found, using the whole trace", file=sys.stderr)
        return rows
    d0, d1 = min(marks), max(marks)
    pf = [r["end"] for r in rows if "prefill" in r["name"] or "prompt_" in r["name"] or "pfg::" in r["name"]]
    if pf:
        d0 = max(d0, max(pf))
        d1 = max(d1, d0)
    return [r for r in rows if d0 <= r["start"] <= d1]
