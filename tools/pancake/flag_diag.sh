#!/bin/sh
# flag_diag.sh - "can this flag run on this hardware, and does it help here?" (see flag_diag.py for the verdict rules).
# Read only on the system: no sudo, no sysctl writes.  Starts engines only through tools/pancake/abtest.py (the `run` mode).
#   tools/pancake/flag_diag.sh --hardware | --list
#   tools/pancake/flag_diag.sh run --config BASE.json [--flags a,b] [--ctest BUILD_DIR] [--out DIR] [--dry-run]
#   tools/pancake/flag_diag.sh --write-recommended [--out DIR]     # DIR/verdicts.json -> rdna3/recommended-flags/<gfx>-rocm<M.m>.json
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-python3}
exec "$PY" "$HERE/flag_diag.py" "$@"
