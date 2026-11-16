#!/bin/sh
# host_tune.sh - tune a Linux host for the pancake-strata engine, one reversible step at a time.  Two jobs:
#   1. the test harness of the host-level A/B (tools/pancake/host_tune_ab.sh applies one step, measures, reverts), and
#   2. an easy, safe way for other people to tune their platform: --check tells what each step would change, --apply does
#      it for this boot only, --revert puts back the exact previous values.
#
# Credit: the list of settings comes from reading bramvanoploo/strata-performance-setup (MIT,
# https://github.com/bramvanoploo/strata-performance-setup).  No code was copied: this script is written for pancake's
# own harness (state file with the previous values, fake-root self test, report-only steps).  README.md section 9.
#
#   host_tune.sh --list                       the steps, their target and their risk
#   host_tune.sh --check                      read-only report of every step, no sudo (nothing is written)
#   host_tune.sh --status                     what this script applied (state files) and the files it wrote
#   host_tune.sh --apply NAME[,NAME]          apply steps NOW (runtime only: gone after a reboot); saves each previous value
#   host_tune.sh --apply NAME --persist       also write the boot-time file (udev / sysctl.d / tmpfiles.d / limits.d)
#   host_tune.sh --revert NAME[,NAME]|all     restore the saved previous values, delete the files this script wrote
#   host_tune.sh --probe NAME                 exit 0 at target, 1 not at target, 2 not applicable (for scripts)
#   host_tune.sh --mlock-check [--pid N]      would the engine's mlock fail?  (src/platform/memory.cpp: 'mlock failed
#                                             (raise ulimit -l)'), and what a running engine really pinned (VmLck)
#   -n | --dry-run                            print what would be written, write nothing
#   --nvme nvme0n1[,..]  --gpu card1[,..]  --cstate-latency-us N  --expect-gib N    narrow or tune a step
#
# Steps: nvme_ra thp swappiness memlock cpu_perf amd_pstate gpu_dpm aspm cstate | report only: iommu headless.
# Run as your user (the sysfs/proc writes go through `sudo tee`, so a sudoers rule for `tee` and `mkdir` is enough) or as
# plain `sudo host_tune.sh ...`.  The script never edits sudoers, permissions or boot loader files.  It writes only a
# whitelist of kernel files and its own pancake-* files, refuses unknown names, and checks every write by reading it back.
# State (previous values): $HOST_TUNE_STATE, default ~/.local/state/pancake-host-tune/<step>.state.  A step with a state file
# is "applied": a second --apply is refused until --revert, so the saved value is always the original one.
# Self test: HOST_TUNE_ROOT=<dir> redirects every path into a fake tree (tools/pancake/test_host_tune.py).
# Exit: 0 ok, 1 a step failed or was refused, 2 usage.
set -u
TAB=$(printf '\t')
R=${HOST_TUNE_ROOT:-}
R=${R%/}
STEPS="nvme_ra thp swappiness memlock cpu_perf amd_pstate gpu_dpm aspm cstate iommu headless"
DRY=0; PERSIST=0; ACTION=""; ARG=""; NVME="nvme*n*"; GPUS=""; CLAT=${HOST_TUNE_CSTATE_LATENCY_US:-10}
EXPECT_GIB=${HOST_TUNE_EXPECT_GIB:-33}; PID_ARG=""
TMP=${TMPDIR:-/tmp}/host_tune.$$
trap 'rm -f "$TMP" "$TMP".*' EXIT HUP INT TERM

die() { echo "host_tune: $*" >&2; exit 1; }
usage() { sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

if [ -n "${HOST_TUNE_SUDO+x}" ]; then SUDO=$HOST_TUNE_SUDO
elif [ "$(id -u)" = 0 ]; then SUDO=""
else SUDO="sudo"; fi

# ---------------------------------------------------------------- state dir
if [ -n "${HOST_TUNE_STATE:-}" ]; then STATE_DIR=$HOST_TUNE_STATE
elif [ -n "$R" ]; then STATE_DIR=$R/var/lib/pancake-host-tune
else
  H=${HOME:-}
  if [ "$(id -u)" = 0 ] && [ -n "${SUDO_USER:-}" ]; then
    case $SUDO_USER in *[!A-Za-z0-9_.-]*|"") ;; *) H=$(getent passwd "$SUDO_USER" 2>/dev/null | cut -d: -f6) ;; esac
  fi
  [ -n "$H" ] || H=/root
  STATE_DIR=${XDG_STATE_HOME:-$H/.local/state}/pancake-host-tune
fi

# ---------------------------------------------------------------- step table
step_valid() { case " $STEPS " in *" $1 "*) return 0 ;; esac; return 1; }
step_kind() {   # runtime = sysfs/proc writes; file = a boot-time file only; report = never applied
  case $1 in iommu|headless) echo report ;; memlock) echo file ;; *) echo runtime ;; esac; }
step_target() {
  case $1 in
    nvme_ra) echo "read_ahead_kb 4096, scheduler none" ;;
    thp) echo "transparent_hugepage enabled=always, defrag=defer+madvise" ;;
    swappiness) echo "vm.swappiness=10" ;;
    memlock) echo "limits.d: <user> memlock unlimited" ;;
    cpu_perf) echo "scaling_governor=performance, energy_performance_preference=performance" ;;
    amd_pstate) echo "amd_pstate status=active" ;;
    gpu_dpm) echo "power_dpm_force_performance_level=high (discrete AMD cards)" ;;
    aspm) echo "pcie_aspm policy=performance" ;;
    cstate) echo "cpuidle states with latency > ${CLAT} us disabled" ;;
    iommu) echo "kernel parameter iommu=pt (multi-GPU peer traffic)" ;;
    headless) echo "default target multi-user (no desktop)" ;;
  esac; }
step_risk() {
  case $1 in
    nvme_ra) echo "more RAM page cache per read; random expert reads (mmap faults) may over-read; no reboot" ;;
    thp) echo "compaction stalls and memory bloat possible; changes how the arena is backed; no reboot" ;;
    swappiness) echo "kernel reclaims page cache sooner than swapping; on a RAM-tight box (arena 33 GiB) not safer; no reboot" ;;
    memlock) echo "no runtime form; needs a new login (and a restart of the systemd user manager) to count; lets the user pin all RAM" ;;
    cpu_perf) echo "idle power and heat go up (clocks never drop); no reboot" ;;
    amd_pstate) echo "driver mode switched live (some kernels refuse); EPP exists only in active mode; no reboot" ;;
    gpu_dpm) echo "GPU idles at top clocks: idle power, fan noise, heat up; the iGPU is skipped unless --gpu names it; no reboot" ;;
    aspm) echo "PCIe links never sleep: idle power up; no reboot" ;;
    cstate) echo "deep core sleep off: idle power up by several W, wake-up faster; no reboot" ;;
    iommu) echo "REPORT ONLY: edit the boot loader yourself and reboot; only matters with two or more GPUs" ;;
    headless) echo "REPORT ONLY, never applied: it would stop the desktop / remote desktop session" ;;
  esac; }

# ---------------------------------------------------------------- file helpers
safe_sys() {   # whitelist of kernel files this script may write
  sp=${1#"$R"}
  case $sp in *..*) return 1 ;; esac
  case $sp in
    /sys/block/nvme*/queue/read_ahead_kb|/sys/block/nvme*/queue/scheduler) return 0 ;;
    /sys/kernel/mm/transparent_hugepage/enabled|/sys/kernel/mm/transparent_hugepage/defrag) return 0 ;;
    /proc/sys/vm/swappiness) return 0 ;;
    /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor) return 0 ;;
    /sys/devices/system/cpu/cpu[0-9]*/cpufreq/energy_performance_preference) return 0 ;;
    /sys/devices/system/cpu/cpu[0-9]*/cpuidle/state[0-9]*/disable) return 0 ;;
    /sys/devices/system/cpu/amd_pstate/status) return 0 ;;
    /sys/class/drm/card[0-9]*/device/power_dpm_force_performance_level) return 0 ;;
    /sys/module/pcie_aspm/parameters/policy) return 0 ;;
  esac
  return 1; }
safe_file() {   # the only boot-time files this script writes or deletes
  sp=${1#"$R"}
  case $sp in *..*) return 1 ;; esac
  case $sp in
    /etc/udev/rules.d/60-pancake-*|/etc/sysctl.d/91-pancake-*|/etc/tmpfiles.d/pancake-*|/etc/security/limits.d/99-pancake-*) return 0 ;;
  esac
  return 1; }

read_val() {   # first line; the [selected] token of a "a [b] c" list
  [ -r "$1" ] || return 1
  rv=""
  IFS= read -r rv < "$1" || [ -n "$rv" ] || return 1
  case $rv in *\[*\]*) rv=${rv#*\[}; rv=${rv%%\]*} ;; esac
  printf '%s' "$rv"; }

write_val() {   # path value: write (sudo tee when not writable), then read back
  safe_sys "$1" || { echo "  REFUSED: ${1#"$R"} is not on the whitelist" >&2; return 1; }
  if [ "$DRY" = 1 ]; then echo "  [dry-run] write $2 > ${1#"$R"}"; return 0; fi
  if [ -n "$R" ] || [ -w "$1" ]; then printf '%s\n' "$2" > "$1" 2>/dev/null || return 1
  else printf '%s\n' "$2" | $SUDO tee "$1" > /dev/null 2>&1 || return 1; fi
  got=$(read_val "$1") || return 1
  [ "$got" = "$2" ]; }

write_file() {   # path content (boot-time file, created through sudo mkdir + tee)
  safe_file "$1" || { echo "  REFUSED: ${1#"$R"} is not a pancake-* file" >&2; return 1; }
  if [ "$DRY" = 1 ]; then echo "  [dry-run] write ${1#"$R"}:"; printf '%s\n' "$2" | sed 's/^/      /'; return 0; fi
  d=$(dirname "$1")
  if [ -n "$R" ]; then mkdir -p "$d"; printf '%s\n' "$2" > "$1" || return 1
  else
    [ -d "$d" ] || $SUDO mkdir -p "$d" || return 1
    printf '%s\n' "$2" | $SUDO tee "$1" > /dev/null || return 1
  fi
  [ -e "$1" ]; }

remove_file() {
  safe_file "$1" || return 1
  [ -e "$1" ] || return 0
  if [ "$DRY" = 1 ]; then echo "  [dry-run] remove ${1#"$R"}"; return 0; fi
  if [ -n "$R" ]; then rm -f "$1"
  elif $SUDO rm -f "$1" 2>/dev/null; then :
  else   # sudo without rm (tee-only rule): neutralise the file
    printf '# reverted by host_tune.sh; delete this file\n' | $SUDO tee "$1" > /dev/null || return 1
    echo "  note: could not delete ${1#"$R"}; emptied it (delete it with sudo rm)"
  fi; }

mkstate() { [ "$DRY" = 1 ] && return 0; mkdir -p "$STATE_DIR" || die "cannot create $STATE_DIR"; }
statef() { echo "$STATE_DIR/$1.state"; }

# ---------------------------------------------------------------- plans: "path<TAB>target value", whatever the current value is
plan_nvme_ra() {
  for d in "$R"/sys/block/*; do
    [ -d "$d/queue" ] || continue
    n=${d##*/}; m=0
    case $n in nvme*) ;; *) continue ;; esac
    set -f; for g in $NVME; do case $n in $g) m=1 ;; esac; done; set +f
    [ $m = 1 ] || continue
    [ -e "$d/queue/read_ahead_kb" ] && printf '%s	%s
' "$d/queue/read_ahead_kb" 4096
    if [ -r "$d/queue/scheduler" ]; then
      case $(cat "$d/queue/scheduler") in *none*) printf '%s	%s
' "$d/queue/scheduler" none ;; esac
    fi
  done; }
plan_thp() {
  t=$R/sys/kernel/mm/transparent_hugepage
  [ -e "$t/enabled" ] && printf '%s\t%s\n' "$t/enabled" always
  if [ -r "$t/defrag" ]; then
    case $(cat "$t/defrag") in *defer+madvise*) df=defer+madvise ;; *) df=madvise ;; esac
    printf '%s\t%s\n' "$t/defrag" "$df"
  fi; }
plan_swappiness() { [ -e "$R/proc/sys/vm/swappiness" ] && printf '%s\t%s\n' "$R/proc/sys/vm/swappiness" 10; }
plan_cpu_perf() {
  for c in "$R"/sys/devices/system/cpu/cpu[0-9]*; do
    f=$c/cpufreq
    [ -d "$f" ] || continue
    if [ -e "$f/energy_performance_preference" ]; then
      av=""; [ -r "$f/energy_performance_available_preferences" ] && av=$(cat "$f/energy_performance_available_preferences")
      case " $av " in *" performance "*|"  ") printf '%s\t%s\n' "$f/energy_performance_preference" performance ;; esac
    fi
    if [ -e "$f/scaling_governor" ]; then
      av=""; [ -r "$f/scaling_available_governors" ] && av=$(cat "$f/scaling_available_governors")
      case " $av " in *" performance "*|"  ") printf '%s\t%s\n' "$f/scaling_governor" performance ;; esac
    fi
  done; }
plan_amd_pstate() {
  f=$R/sys/devices/system/cpu/amd_pstate/status
  [ -r "$f" ] || return 0
  case $(read_val "$f") in passive|guided|active) printf '%s\t%s\n' "$f" active ;; esac; }
plan_gpu_dpm() {
  for c in "$R"/sys/class/drm/card[0-9]*; do
    n=${c##*/}
    case $n in *-*) continue ;; esac
    f=$c/device/power_dpm_force_performance_level
    [ -e "$f" ] || continue
    if [ -r "$c/device/vendor" ]; then [ "$(cat "$c/device/vendor")" = 0x1002 ] || continue; fi
    if [ -n "$GPUS" ]; then
      case ",$GPUS," in *",$n,"*) ;; *) continue ;; esac
    elif [ -r "$c/device/mem_info_vram_total" ]; then
      v=$(cat "$c/device/mem_info_vram_total")
      [ "$v" -ge 4294967296 ] 2>/dev/null || continue     # an iGPU (carve-out below 4 GiB) is not touched by default
    fi
    printf '%s\t%s\n' "$f" high
  done; }
plan_aspm() {
  f=$R/sys/module/pcie_aspm/parameters/policy
  [ -r "$f" ] || return 0
  case $(cat "$f") in *performance*) printf '%s\t%s\n' "$f" performance ;; esac; }
plan_cstate() {
  for s in "$R"/sys/devices/system/cpu/cpu[0-9]*/cpuidle/state[0-9]*; do
    [ -e "$s/disable" ] && [ -r "$s/latency" ] || continue
    l=$(cat "$s/latency")
    [ "$l" -gt "$CLAT" ] 2>/dev/null && printf '%s\t%s\n' "$s/disable" 1
  done; }
run_plan() { ( "plan_$1" ); }   # a subshell: a plan never clobbers the caller's variables
plan_memlock() { :; }
plan_iommu() { :; }
plan_headless() { :; }

# ---------------------------------------------------------------- check
add_uniq() { eval "cur=\$$1"; case " $cur " in *" $2 "*) ;; *) eval "$1=\"\$$1 $2\"" ;; esac; }

ulimit_l() {   # soft|hard of ulimit -l in KiB or "unlimited"
  if [ "$1" = soft ]; then echo "${HOST_TUNE_ULIMIT_SOFT:-$(ulimit -l 2>/dev/null || echo ?)}"
  else echo "${HOST_TUNE_ULIMIT_HARD:-$(ulimit -Hl 2>/dev/null || echo ?)}"; fi; }

check_step() {   # sets CHK_STATE (at-target|partial|not-at-target|n/a|report) and CHK_TEXT
  s=$1; k=$(step_kind "$s"); CHK_TEXT=""
  case $s in
    iommu)
      cmd=$(cat "$R/proc/cmdline" 2>/dev/null || true); ng=0
      for c in "$R"/sys/class/drm/card[0-9]*; do
        case ${c##*/} in *-*) continue ;; esac
        [ -r "$c/device/vendor" ] && ng=$((ng + 1))
      done
      case " $cmd " in *" iommu=pt "*) io="iommu=pt set" ;; *) io="iommu=pt not set" ;; esac
      CHK_STATE=report; CHK_TEXT="$io; $ng GPU card(s) seen (only relevant with 2 or more)"; return ;;
    headless)
      d=$(systemctl get-default 2>/dev/null || echo "?")
      CHK_STATE=report; CHK_TEXT="default target: $d (report only, never changed here)"; return ;;
    memlock)
      f=$R/etc/security/limits.d/99-pancake-memlock.conf
      sl=$(ulimit_l soft); hl=$(ulimit_l hard)
      if [ "$sl" = unlimited ]; then CHK_STATE=at-target; else CHK_STATE=not-at-target; fi
      CHK_TEXT="this shell: soft $sl KiB, hard $hl KiB; limits.d file: $([ -e "$f" ] && echo present || echo absent)"; return ;;
  esac
  run_plan "$s" > "$TMP.plan"
  n=0; ok=0; curs=""; tgts=""
  while IFS=$TAB read -r p v; do
    n=$((n + 1)); c=$(read_val "$p") || c="?"
    [ "$c" = "$v" ] && ok=$((ok + 1))
    add_uniq curs "$c"; add_uniq tgts "$v"
  done < "$TMP.plan"
  if [ "$n" = 0 ]; then CHK_STATE=n/a; CHK_TEXT="nothing to set here (file missing or option not offered by the kernel)"
  else
    if [ "$ok" = "$n" ]; then CHK_STATE=at-target; elif [ "$ok" = 0 ]; then CHK_STATE=not-at-target; else CHK_STATE=partial; fi
    CHK_TEXT="now:$curs -> target:$tgts ($ok/$n files at target)"
  fi; }

do_check() {
  echo "pancake host check (read-only, nothing is written)  root=${R:-/}"
  echo "  kernel: $(uname -sr 2>/dev/null)   max_map_count: $(cat "$R/proc/sys/vm/max_map_count" 2>/dev/null || echo ?)"
  for s in $STEPS; do
    check_step "$s"
    printf '  %-11s %-14s %s\n' "$s" "$CHK_STATE" "$CHK_TEXT"
    [ -e "$(statef "$s")" ] && echo "              (applied by this script: state file $(statef "$s"))"
  done
  echo "Run --list for the risk of each step, --mlock-check for the mlock verdict."; }

do_list() {
  for s in $STEPS; do
    printf '%-11s [%s]\n   target: %s\n   risk:   %s\n' "$s" "$(step_kind "$s")" "$(step_target "$s")" "$(step_risk "$s")"
  done; }

# ---------------------------------------------------------------- persist
target_user() {
  u=${HOST_TUNE_USER:-${SUDO_USER:-${USER:-$(id -un)}}}
  case $u in ""|root|*[!A-Za-z0-9_.-]*) return 1 ;; esac
  echo "$u"; }

record_file() { mkstate; [ "$DRY" = 1 ] || printf 'file\t%s\t-\n' "$1" >> "$(statef "$2")"; }

persist_step() {
  s=$1
  hdr="# written by tools/pancake/host_tune.sh (step $s); undo: host_tune.sh --revert $s"
  case $s in
    nvme_ra)
      f=$R/etc/udev/rules.d/60-pancake-nvme.rules
      body="$hdr
ACTION==\"add|change\", SUBSYSTEM==\"block\", KERNEL==\"nvme[0-9]*n[0-9]*\", ENV{DEVTYPE}==\"disk\", ATTR{queue/scheduler}=\"none\", ATTR{queue/read_ahead_kb}=\"4096\"" ;;
    swappiness)
      f=$R/etc/sysctl.d/91-pancake-swappiness.conf
      body="$hdr
vm.swappiness = 10" ;;
    memlock)
      u=$(target_user) || { echo "  REFUSED: no safe target user (root or unnamed); set HOST_TUNE_USER" >&2; return 1; }
      f=$R/etc/security/limits.d/99-pancake-memlock.conf
      body="$hdr
$u - memlock unlimited" ;;
    *)
      f=$R/etc/tmpfiles.d/pancake-$s.conf
      run_plan "$s" > "$TMP.persist"
      body="$hdr"
      while IFS=$TAB read -r p v; do body="$body
w ${p#"$R"} - - - - $v"; done < "$TMP.persist" ;;
  esac
  record_file "$f" "$s"
  write_file "$f" "$body" || return 1
  echo "  persistent file: ${f#"$R"}"; }

# ---------------------------------------------------------------- apply / revert
apply_step() {
  s=$1; k=$(step_kind "$s"); rc=0
  echo "== apply $s: $(step_target "$s")"
  echo "   risk: $(step_risk "$s")"
  [ "$k" = report ] && { echo "  REFUSED: $s is report-only and never applied by this script" >&2; return 1; }
  [ -e "$(statef "$s")" ] && { echo "  REFUSED: $s is already applied (state file $(statef "$s")); run --revert $s first" >&2; return 1; }
  if [ "$k" = file ]; then
    [ "$PERSIST" = 1 ] || { echo "  REFUSED: $s has no runtime form; use --apply $s --persist" >&2; return 1; }
    persist_step "$s" || return 1
    echo "  done (takes effect at the next login; systemd user services need the user manager restarted)"
    return 0
  fi
  run_plan "$s" > "$TMP.plan"
  [ -s "$TMP.plan" ] || { echo "  nothing to do: not applicable on this host"; [ "$PERSIST" = 1 ] || return 0; }
  ch=0; un=0
  while IFS=$TAB read -r p v; do
    safe_sys "$p" || { echo "  REFUSED: ${p#"$R"}" >&2; rc=1; continue; }
    c=$(read_val "$p") || { echo "  REFUSED: cannot read ${p#"$R"}, so the previous value cannot be saved" >&2; rc=1; continue; }
    if [ "$c" = "$v" ]; then un=$((un + 1)); continue; fi
    if [ "$DRY" != 1 ]; then mkstate; printf 'val\t%s\t%s\n' "$p" "$c" >> "$(statef "$s")"; fi
    if write_val "$p" "$v"; then ch=$((ch + 1)); [ "$DRY" = 1 ] || echo "  ${p#"$R"}: $c -> $v"
    else echo "  FAILED: ${p#"$R"} ($c -> $v)" >&2; rc=1; fi
  done < "$TMP.plan"
  echo "  runtime: $ch changed, $un already at target"
  if [ "$PERSIST" = 1 ]; then persist_step "$s" || rc=1; fi
  return $rc; }

revert_step() {
  s=$1; rc=0; sf=$(statef "$s")
  step_valid "$s" || { echo "unknown step $s" >&2; return 1; }
  if [ ! -e "$sf" ]; then echo "== revert $s: nothing to revert (no state file)"; return 0; fi
  echo "== revert $s"
  while IFS=$TAB read -r kind p old; do
    case $kind in
      val)
        c=$(read_val "$p") || c="?"
        if [ "$c" = "$old" ]; then continue; fi
        if write_val "$p" "$old"; then [ "$DRY" = 1 ] || echo "  ${p#"$R"}: $c -> $old"
        else echo "  FAILED: ${p#"$R"} (to $old)" >&2; rc=1; fi ;;
      file) remove_file "$p" && echo "  removed ${p#"$R"}" || { echo "  FAILED: ${p#"$R"}" >&2; rc=1; } ;;
    esac
  done < "$sf"
  if [ $rc = 0 ] && [ "$DRY" != 1 ]; then rm -f "$sf"; elif [ $rc != 0 ]; then echo "  state file kept: $sf" >&2; fi
  return $rc; }

do_status() {
  any=0
  for s in $STEPS; do
    sf=$(statef "$s")
    [ -e "$sf" ] || continue
    any=1; echo "$s: applied (state $sf)"
    while IFS=$TAB read -r kind p old; do
      case $kind in
        val) echo "   ${p#"$R"}: was $old, now $(read_val "$p" || echo '?')" ;;
        file) echo "   file ${p#"$R"} ($([ -e "$p" ] && echo present || echo missing))" ;;
      esac
    done < "$sf"
  done
  [ $any = 0 ] && echo "nothing applied by this script (state dir $STATE_DIR)"
  return 0; }

# ---------------------------------------------------------------- mlock check
do_mlock() {
  sl=$(ulimit_l soft); hl=$(ulimit_l hard)
  exp_kib=$((EXPECT_GIB * 1048576))
  echo "mlock check (engine arena expected ~${EXPECT_GIB} GiB; src/platform/memory.cpp:167,178 'mlock failed (raise ulimit -l)')"
  echo "  this shell: ulimit -l soft=$sl hard=$hl (KiB)"
  lf=$R/etc/security/limits.d/99-pancake-memlock.conf
  echo "  limits.d: $([ -e "$lf" ] && echo "$lf present" || echo "no pancake file")"
  if command -v systemctl > /dev/null 2>&1 && [ -z "$R" ]; then
    echo "  pancake.service LimitMEMLOCK: $(systemctl --user show pancake.service -p LimitMEMLOCK --value 2>/dev/null || echo '?')"
  fi
  pid=${PID_ARG:-${HOST_TUNE_ENGINE_PID:-}}
  [ -n "$pid" ] || pid=$(pgrep -f 'strata --serve' 2>/dev/null | head -n 1)
  verdict_lim="unknown"
  case $sl in unlimited) verdict_lim=ok ;; ''|'?'|*[!0-9]*) ;; *) [ "$sl" -ge "$exp_kib" ] && verdict_lim=ok || verdict_lim=low ;; esac
  if [ -n "$pid" ] && [ -r "$R/proc/$pid/status" ]; then
    lk=$(sed -n 's/^VmLck:[[:space:]]*\([0-9]*\) kB.*/\1/p' "$R/proc/$pid/status")
    pl=$(sed -n 's/^Max locked memory[[:space:]]*\([^ ]*\) .*/\1/p' "$R/proc/$pid/limits" 2>/dev/null)
    echo "  engine pid $pid: VmLck=${lk:-?} KiB, its own limit 'Max locked memory' soft=${pl:-?} bytes"
    case $pl in unlimited) verdict_lim=ok ;; ''|*[!0-9]*) ;; *) [ $((pl / 1024)) -ge "$exp_kib" ] && verdict_lim=ok || verdict_lim=low ;; esac
    if [ -n "$lk" ] && [ "$lk" -ge $((exp_kib * 9 / 10)) ]; then
      echo "  VERDICT: OK, the engine pinned ${lk} KiB (>= 90% of the expected $exp_kib KiB)"; return 0
    elif [ "$verdict_lim" = low ]; then
      echo "  VERDICT: mlock FAILED or partial: limit is below the arena and VmLck=${lk:-0} KiB.  Fix: host_tune.sh --apply memlock --persist, log in again, restart the engine"; return 1
    else
      echo "  VERDICT: VmLck=${lk:-0} KiB is below the expected size but the limit is not the cause; read the engine log (grep 'mlock' / 'arena')"; return 1
    fi
  fi
  [ -n "$pid" ] && echo "  engine pid $pid: /proc/$pid/status not readable" || echo "  no running engine found ('strata --serve'); pass --pid N"
  case $verdict_lim in
    ok) echo "  VERDICT (limit only): an engine started from this session may pin the arena" ;;
    low) echo "  VERDICT (limit only): an engine started from this session would FAIL to mlock the ${EXPECT_GIB} GiB arena (soft limit below it); fix: --apply memlock --persist" ;;
    *) echo "  VERDICT: cannot tell the limit" ;;
  esac
  return 0; }

# ---------------------------------------------------------------- command line
setact() { [ -z "$ACTION" ] || { echo "host_tune: one action only" >&2; exit 2; }; ACTION=$1; }
while [ $# -gt 0 ]; do
  case $1 in
    --check) setact check ;; --status) setact status ;; --list) setact list ;; --mlock-check) setact mlock ;;
    --apply|--revert|--probe)
      [ $# -ge 2 ] || { echo "host_tune: $1 needs a NAME" >&2; exit 2; }
      setact "${1#--}"; ARG=$2; shift ;;
    -n|--dry-run) DRY=1 ;;
    --persist) PERSIST=1 ;;
    --nvme) [ $# -ge 2 ] || usage; NVME=$(echo "$2" | tr ',' ' '); shift ;;
    --gpu) [ $# -ge 2 ] || usage; GPUS=$2; shift ;;
    --cstate-latency-us) [ $# -ge 2 ] || usage; CLAT=$2; shift ;;
    --expect-gib) [ $# -ge 2 ] || usage; EXPECT_GIB=$2; shift ;;
    --pid) [ $# -ge 2 ] || usage; PID_ARG=$2; shift ;;
    -h|--help) usage ;;
    *) echo "host_tune: unknown option $1" >&2; usage ;;
  esac
  shift
done
[ -n "$ACTION" ] || usage
case $CLAT in ""|*[!0-9]*) die "--cstate-latency-us needs a number" ;; esac
case $EXPECT_GIB in ""|*[!0-9]*) die "--expect-gib needs a whole number" ;; esac
case $NVME in *[!A-Za-z0-9_\*\?\ ]*) die "--nvme: bad device name" ;; esac
case $GPUS in *[!A-Za-z0-9,]*) die "--gpu: bad card name" ;; esac
case $ARG in *[!a-z_,]*) die "bad step list '$ARG'" ;; esac
if [ -z "$R" ] && [ "$(uname -s 2>/dev/null)" != Linux ] && [ "$ACTION" != list ]; then die "Linux only"; fi

names() { echo "$1" | tr ',' ' '; }
case $ACTION in
  list) do_list ;;
  check) do_check ;;
  status) do_status ;;
  mlock) do_mlock; exit $? ;;
  probe)
    step_valid "$ARG" || die "unknown step $ARG"
    check_step "$ARG"
    case $CHK_STATE in at-target) exit 0 ;; not-at-target|partial) exit 1 ;; *) exit 2 ;; esac ;;
  apply)
    [ "$ARG" != all ] || die "apply needs explicit step names (no 'all')"
    rc=0
    for s in $(names "$ARG"); do
      step_valid "$s" || { echo "host_tune: unknown step $s (see --list)" >&2; rc=1; continue; }
      apply_step "$s" || rc=1
    done
    exit $rc ;;
  revert)
    rc=0
    if [ "$ARG" = all ]; then lst=""; for s in $STEPS; do [ -e "$(statef "$s")" ] && lst="$lst $s"; done
    else lst=$(names "$ARG"); fi
    [ -n "$lst" ] || echo "nothing to revert"
    for s in $lst; do revert_step "$s" || rc=1; done
    exit $rc ;;
esac
