#!/bin/sh
# flag_diag.sh - per-flag diagnostic of the target machine: can this flag run here, and does it help?  READ-ONLY on the system
# (sysfs / proc / rocminfo reads, the engine started as your user, files only under OUT_DIR; no sudo, no sysctl writes).
#
#   tools/pancake/flag_diag.sh hw                                   detected hardware (gfx, ROCm, VRAM, PCIe link, CPU, RAM, hipBLASLt, VMM)
#   tools/pancake/flag_diag.sh list                                 candidate flags: kind, value, exact / lossy, metric, ledger id
#   tools/pancake/flag_diag.sh run CONFIG.json OUT_DIR [--only A,B] [--budget-min 15]
#                                                                   verdict per flag: UNSUPPORTED / NO-GAIN / GAIN / UNTESTED
#   tools/pancake/flag_diag.sh --write-recommended OUT_DIR/verdicts.json [FILE] [--merge]
#                                                                   the recommended-flags file (default ~/.strata/tuned-flags.json) that
#                                                                   every normal start reads (terminal/strata_control/tuned_flags.py);
#                                                                   only exact >= 2 % or lossy >= 3 % with a passing quality gate enter it
# Time: one engine start (1-3 minutes) per flag plus two starts of the default: about 3 minutes per flag, 15 minutes at the default
# budget.  The A/B harness is tools/pancake/abtest.py of a port/rdna3-decode checkout (FLAG_DIAG_ABTEST=path when this tree has none).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-python3}
case "${1:-}" in
  hw|list) cmd=$1; shift; exec "$PY" "$HERE/flag_diag.py" "$cmd" "$@" ;;
  run)
    shift; [ $# -ge 2 ] || { echo "usage: flag_diag.sh run CONFIG.json OUT_DIR [--only A,B] [--budget-min N]" >&2; exit 2; }
    cfg=$1; out=$2; shift 2
    exec "$PY" "$HERE/flag_diag.py" run --config "$cfg" --out "$out" "$@" ;;
  --write-recommended)
    shift; [ $# -ge 1 ] || { echo "usage: flag_diag.sh --write-recommended VERDICTS.json [FILE] [--merge]" >&2; exit 2; }
    v=$1; shift; f=${HOME}/.strata/tuned-flags.json
    if [ $# -ge 1 ] && [ "${1#--}" = "$1" ]; then f=$1; shift; fi
    exec "$PY" "$HERE/flag_diag.py" write-recommended --verdicts "$v" --out "$f" "$@" ;;
  *) sed -n 2,16p "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
