#!/bin/sh
# rocm102-install.v1.sh - side-by-side install of ROCm Core SDK 10.2.0 beside 10.0.0 on the Linux box (owner order 2026-11-21: "you can install ROCm 10.2 in
# parallel with ours (ROCm 10.0); can be before the race").  LEADS.md Part T, leads T1 / T2 (T.4 breaking-change list).  USER SPACE ONLY.
#
# Facts (box survey 2026-11-21, read-only): Ubuntu 26.04.1, kernel 7.0.0-43-generic; 10.0 came from the AMD apt repo
# /etc/apt/sources.list.d/amdrocm-stable.sources (https://stable.repo.amd.com/rocm/core/packages/ubuntu2604/, suite stable, key /etc/apt/keyrings/amdrocm.gpg);
# that repo already carries 10.0, 10.1 and 10.2 (packages amdrocm-*10.2, version 10.2.0-2; the version is part of the package name, so 10.2 installs into
# /opt/rocm/core-10.2 and conflicts with nothing of 10.0).  /opt/rocm is a plain directory (no symlink) that holds core-10.0.  The amdgpu kernel driver is
# separate (amdgpu-core / amdgpu-lib 7.2.70200 from repo.radeon.com): no amdrocm package depends on it, this script never touches it, no reboot.
#
# What it does (nothing is deleted, nothing is removed, no kernel module, no reboot, no /opt/rocm symlink, no ldconfig / PATH / profile.d change):
#   0. default (no ROCM102_INSTALL=1): prints this plan and exits 0, writes nothing.
#   1. refuses while a measured job runs (the busy() patterns of the chain scripts), ERROR + ABORTED + DONE in the log, exit 1.
#   2. SUDO='sudo -n' must work; if not (password needed) it prints the exact commands for the owner (run them with the ! prefix) and exits 3, nothing done.
#   3. pre-state -> repo (added only if the 10.2 package has no candidate) -> apt-get update (only if the repo was just added, only that source) -> disk check
#      -> dry run (-s) checked: only amdrocm*10.2 packages, nothing removed -> install -> verify (10.2 hipcc, rocminfo lists gfx1100, readlink /opt/rocm,
#      the 10.0 hipcc --version, the amdgpu packages and the kernel are unchanged vs the pre-state, else ERROR) -> print the PATH / LD_LIBRARY_PATH lines.
# Log: /home/dan/clone-of-d/src/rocm102-install.log (START / END per step), last line ROCM102_INSTALL_DONE.  Exit: 0 ok or plan, 1 error, 3 owner needed.
# Start:  cd /home/dan/clone-of-d/src && ROCM102_INSTALL=1 sh ./rocm102-install.v1.sh      (queue: step 'ROCM102_INSTALL', after the gates, before the race)
# Environment: ROCM102_INSTALL (switch), ROCM102_MARGIN_GIB (free space kept after install, default 3), ROCM102_S (work dir), ROCM102_ROOT (default /opt/rocm),
# ROCM102_APT_SRC (default /etc/apt/sources.list.d), ROCM102_SUDO (default 'sudo -n'; tests).
S=${ROCM102_S:-/home/dan/clone-of-d/src}; L=$S/rocm102-install.log
ROOT=${ROCM102_ROOT:-/opt/rocm}; R10=$ROOT/core-10.0; R102=$ROOT/core-10.2
APT_SRC=${ROCM102_APT_SRC:-/etc/apt/sources.list.d}; KEY=/etc/apt/keyrings/amdrocm.gpg
SUDO=${ROCM102_SUDO:-sudo -n}
MARGIN_GIB=${ROCM102_MARGIN_GIB:-3}
# metapackage for gfx1100 plus the four -dev packages that 10.0 has installed on the box (hipcc, headers, rocBLAS / hipBLAS dev); nothing else
PKGS="amdrocm10.2-gfx1100 amdrocm-llvm-dev10.2 amdrocm-runtime-dev10.2 amdrocm-blas-dev10.2 amdrocm-hipblas-common-dev10.2"
BUSY_RE='[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[c]onv_trace.py|[a]uto-chain|[a]uto-post|[i]q_avx2_parity|[t]tft_bench.py|[w]in-chain[A-Za-z0-9]*\.sh|[x-z]tail-chain[A-Z]\.sh|[a]fter7|[b]ench-(updated(AA|RD)|diagUS|exact|rdgate|sweep)[.a-z0-9]*\.sh'

owner_cmds() {
  cat <<EOF
# run these on the box, one by one, with the ! prefix (sudo needs the owner's password); the box must be idle (no measured job):
$SUDO apt-get install -y --no-install-recommends $PKGS
# (the repo amdrocm-stable already lists 10.2; if 'apt-cache policy amdrocm10.2-gfx1100' shows no Candidate, first:
#  wget https://stable.repo.amd.com/rocm/gpg/packages.gpg -O - | gpg --dearmor | sudo tee $KEY > /dev/null ; then the .sources file of the plan ; sudo apt update)
EOF
}

if [ "${ROCM102_INSTALL:-0}" != 1 ]; then
  cat <<EOF
rocm102-install.v1.sh PLAN (nothing is done; set ROCM102_INSTALL=1 to run it, idle box only)
  target     ROCm Core SDK 10.2.0 (packages amdrocm-*10.2, 10.2.0-2) into $R102, beside $R10 (stays the default; the race stays on 10.0)
  repo       $APT_SRC/amdrocm-stable.sources (stable.repo.amd.com/rocm/core/packages/ubuntu<release>/), key $KEY; added only when missing
  packages   $PKGS (+ their amdrocm*10.2 dependencies, about 30 to 34 packages, about 6.4 to 8.4 GiB installed, about 1 GiB download)
  never      no kernel / DKMS / amdgpu package, no removal, no reboot, no /opt/rocm symlink, no ldconfig / PATH / profile.d change, no deletion
  checks     busy refusal, passwordless sudo ($SUDO) else exit 3 with the owner commands, free disk >= installed + download + ${MARGIN_GIB} GiB, dry run clean,
             after install: $R102/bin/hipcc --version, rocminfo lists gfx1100, readlink $ROOT and the 10.0 hipcc --version unchanged
  log        $L (START / END per step, last line ROCM102_INSTALL_DONE)
EOF
  owner_cmds
  exit 0
fi

cd "$S" 2>/dev/null || { echo "ERROR no work dir $S"; exit 1; }
say() { echo "$*" >> "$L"; }
finish() { say "ROCM102_INSTALL_DONE $(date +%T)"; }
fail() { say "ERROR $* $(date +%T)"; say "ROCM102_INSTALL_ABORTED"; finish; echo "ERROR $*" >&2; exit 1; }
step() { STEP=$1; say "START $1 $(date +%T)"; }
done_step() { say "END $STEP $(date +%T)"; }
busy() { pgrep -f "$BUSY_RE" >/dev/null 2>&1; }
say "START $(date +%F_%T) ROCM102_INSTALL=1 sudo='$SUDO'"

step busy_check
if busy; then fail "a measured job runs (busy pattern): nothing done, retry on an idle box"; fi
done_step

step sudo_check
if ! $SUDO true >/dev/null 2>&1; then
  say "NEEDS_OWNER sudo is not passwordless: nothing done; commands printed on stdout"
  owner_cmds
  say "ROCM102_INSTALL_NEEDS_OWNER"; finish
  exit 3
fi
done_step

step pre_state
PRE=$S/rocm102-pre-state.txt
{
  echo "== dpkg amdrocm* amdgpu*"; dpkg -l 'amdrocm*' 'amdgpu*' 2>/dev/null | awk 'NR>5{print $2, $3}'
  echo "== ls $ROOT*"; ls -d "$ROOT"* 2>/dev/null; ls "$ROOT" 2>/dev/null
  echo "== readlink $ROOT"; readlink "$ROOT" || echo "(not a symlink)"
  echo "== ldconfig rocm count"; ldconfig -p 2>/dev/null | grep -c rocm
  echo "== hipcc 10.0"; "$R10/bin/hipcc" --version 2>&1 | head -3
  echo "== kernel"; uname -r
  echo "== df"; df -Pk / "$ROOT" 2>/dev/null
} > "$PRE" 2>&1
PRE_LINK=$(readlink "$ROOT" || echo "(not a symlink)")
PRE_HIPCC=$("$R10/bin/hipcc" --version 2>&1 | head -3)
PRE_LDC=$(ldconfig -p 2>/dev/null | grep -c rocm)
PRE_DRV=$(dpkg -l 'amdgpu*' 2>/dev/null | awk 'NR>5{print $2, $3}')
PRE_R10=$(dpkg -l 'amdrocm*10.0*' 2>/dev/null | awk 'NR>5{print $2, $3}')
PRE_KERN=$(uname -r)
say "pre-state saved in $PRE: link='$PRE_LINK' ldconfig_rocm=$PRE_LDC kernel=$PRE_KERN"
[ -x "$R10/bin/hipcc" ] || fail "10.0 hipcc not found at $R10/bin/hipcc (pre-state is not the expected box)"
done_step

step repo
CAND=$(apt-cache policy amdrocm10.2-gfx1100 2>/dev/null | awk '/Candidate:/{print $2}')
if [ -n "$CAND" ] && [ "$CAND" != "(none)" ]; then
  say "repo already provides amdrocm10.2-gfx1100 $CAND: nothing added"; ADDED=0
else
  # AMD docs, ROCm 10.2.0 install (Ubuntu apt): https://rocm.docs.amd.com/en/latest/install/rocm.html  (section "Register the ROCm apt repository")
  . /etc/os-release; REL=ubuntu$(echo "$VERSION_ID" | tr -d .)
  say "repo missing: adding $APT_SRC/amdrocm-stable.sources for $REL"
  $SUDO mkdir --parents --mode=0755 /etc/apt/keyrings || fail "mkdir keyrings"
  if [ ! -s "$KEY" ]; then
    wget -q https://stable.repo.amd.com/rocm/gpg/packages.gpg -O - | gpg --dearmor | $SUDO tee "$KEY" > /dev/null || fail "key download"
  fi
  if [ ! -e "$APT_SRC/amdrocm-stable.sources" ]; then
    $SUDO tee "$APT_SRC/amdrocm-stable.sources" > /dev/null <<EOF || fail "write sources file"
X-Repo-Id: amdrocm-stable
Types: deb
URIs: https://stable.repo.amd.com/rocm/core/packages/$REL/
Suites: stable
Components: main
Architectures: amd64
Signed-By: $KEY
Enabled: yes
EOF
  fi
  ADDED=1
fi
done_step

step apt_update
if [ "$ADDED" = 1 ]; then
  # only this source; List-Cleanup=0 so no other cached list is dropped
  $SUDO apt-get update -o Dir::Etc::sourcelist="sources.list.d/amdrocm-stable.sources" -o Dir::Etc::sourceparts=- -o APT::Get::List-Cleanup=0 >> "$L" 2>&1 || fail "apt-get update"
else say "not needed (repo lists already cached, 10.2 candidate present)"; fi
CAND=$(apt-cache policy amdrocm10.2-gfx1100 2>/dev/null | awk '/Candidate:/{print $2}')
{ [ -n "$CAND" ] && [ "$CAND" != "(none)" ]; } || fail "no candidate for amdrocm10.2-gfx1100 after the repo step"
done_step

step dry_run
DRY=$S/rocm102-dryrun.txt
apt-get -s --no-install-recommends install $PKGS > "$DRY" 2>&1 || fail "dry run failed, see $DRY"
grep -q '^Remv' "$DRY" && fail "dry run would REMOVE packages, see $DRY"
BAD=$(awk '/^(Inst|Conf|Remv|Purg)/{print $2}' "$DRY" | grep -v -E '^amdrocm.*10\.2' )
[ -z "$BAD" ] || fail "dry run touches packages that are not amdrocm*10.2 (kernel / DKMS / amdgpu / other): $(echo $BAD | tr '\n' ' ')"
BADK=$(awk '/^(Inst|Conf|Remv|Purg)/{print $2}' "$DRY" | grep -i -E 'dkms|linux-|amdgpu|firmware|kernel')
[ -z "$BADK" ] || fail "dry run touches kernel / driver packages: $BADK"
NPKG=$(grep -c '^Inst' "$DRY"); [ "$NPKG" -ge 1 ] || fail "dry run installs nothing"
say "dry run clean: $NPKG packages, all amdrocm*10.2, none removed ($DRY)"
done_step

step disk_check
# installed size (KiB, Installed-Size) + download size (bytes, Size) of the packages of the dry run; the archives land on the same file system as /opt
NEED_KIB=$(for p in $(awk '/^Inst/{print $2}' "$DRY"); do apt-cache show "$p=$CAND" 2>/dev/null | awk '/^Installed-Size/{i=$2} /^Size/{s=$2} END{print i, int(s/1024)}'; done | awk '{t+=$1+$2} END{print t+0}')
FREE_KIB=$(df -Pk "$ROOT" | awk 'NR==2{print $4}')
MARGIN_KIB=$((MARGIN_GIB * 1048576))
say "disk: free ${FREE_KIB} KiB on $ROOT, install+download ${NEED_KIB} KiB, margin ${MARGIN_KIB} KiB"
[ "$NEED_KIB" -gt 0 ] || fail "could not compute the install size"
[ "$FREE_KIB" -ge $((NEED_KIB + MARGIN_KIB)) ] || fail "disk too small: free ${FREE_KIB} KiB < need ${NEED_KIB} KiB + margin ${MARGIN_KIB} KiB: free space first (nothing is deleted by this script)"
done_step

step install
$SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends $PKGS >> "$L" 2>&1 || fail "apt-get install"
done_step

step verify
[ -d "$R102" ] || fail "$R102 does not exist after the install"
V102=$(LD_LIBRARY_PATH="$R102/lib" "$R102/bin/hipcc" --version 2>&1 | head -3); say "10.2 hipcc: $V102"
echo "$V102" | grep -q 'HIP version' || fail "10.2 hipcc --version prints no HIP version"
GFX=$(LD_LIBRARY_PATH="$R102/lib" "$R102/bin/rocminfo" 2>/dev/null | grep -c gfx1100)
[ "$GFX" -ge 1 ] || fail "rocminfo of 10.2 does not list gfx1100"
say "10.2 rocminfo lists gfx1100 ($GFX lines)"
POST_LINK=$(readlink "$ROOT" || echo "(not a symlink)")
[ "$POST_LINK" = "$PRE_LINK" ] || fail "readlink $ROOT changed: '$PRE_LINK' -> '$POST_LINK'"
POST_HIPCC=$("$R10/bin/hipcc" --version 2>&1 | head -3)
[ "$POST_HIPCC" = "$PRE_HIPCC" ] || fail "10.0 hipcc --version changed"
[ "$(ldconfig -p 2>/dev/null | grep -c rocm)" = "$PRE_LDC" ] || fail "ldconfig rocm entries changed"
[ "$(dpkg -l 'amdgpu*' 2>/dev/null | awk 'NR>5{print $2, $3}')" = "$PRE_DRV" ] || fail "amdgpu driver packages changed"
[ "$(dpkg -l 'amdrocm*10.0*' 2>/dev/null | awk 'NR>5{print $2, $3}')" = "$PRE_R10" ] || fail "10.0 packages changed"
[ "$(uname -r)" = "$PRE_KERN" ] || fail "kernel changed"
say "unchanged: readlink, 10.0 hipcc, ldconfig, amdgpu packages, 10.0 packages, kernel"
done_step

step print_env
cat <<EOF | tee -a "$L"
To build against 10.2 (per shell or per script; do NOT export globally):
  export PATH=$R102/bin:\$PATH
  export LD_LIBRARY_PATH=$R102/lib:$R102/lib/llvm/lib:\$LD_LIBRARY_PATH
  cmake ... -DCMAKE_PREFIX_PATH=$R102 -DCMAKE_HIP_COMPILER=$R102/lib/llvm/bin/clang++   # same source, separate build dir
  (hipBLASLt of 10.2 is 1.5.0: the table file tools/hip/gfx1100-hipblaslt-100500.txt is the one the loader looks for, LEADS T2)
EOF
done_step
say "OK ROCm 10.2 installed beside 10.0, nothing else changed"
finish
exit 0
