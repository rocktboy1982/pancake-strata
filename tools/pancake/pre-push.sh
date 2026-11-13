#!/usr/bin/env bash
# tools/pancake/pre-push.sh - git pre-push hook (MOD_PLAN item 92): configure every build-matrix row from the
# committed HEAD (~2 min).  Full link check: PANCAKE_PUSH_FULL=1 git push.  Skip once: SKIP_BUILD_MATRIX=1 git push.
# Install: ln -sf ../../tools/pancake/pre-push.sh "$(git rev-parse --git-common-dir)/hooks/pre-push"
[ "${SKIP_BUILD_MATRIX:-0}" = 1 ] && exit 0
# git runs hooks with GIT_DIR (and friends) set: without this, the `git -C build-matrix/src checkout --detach` in build_matrix.sh
# acts on the OUTER repository and detaches the pushing worktree's HEAD (seen 2026-11-11: the branch stayed behind, the next
# commit landed on a detached HEAD)
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_PREFIX GIT_COMMON_DIR GIT_OBJECT_DIRECTORY GIT_ALTERNATE_OBJECT_DIRECTORIES
repo=$(git rev-parse --show-toplevel)
# spec_lint (MOD_PLAN discipline): warn if bench/specs reference a STRATA_* env key or --flag this fork never reads,
# i.e. an arm that would be inert (engine ignores it = base). Advisory on push; SKIP_BUILD_MATRIX=1 still bypasses.
# (A .github/workflows CI gate is deferred: the git credential lacks the `workflow` scope to add workflow files.)
if command -v python3 >/dev/null 2>&1; then _py=python3; else _py=python; fi
if ! "$_py" "$repo/tools/pancake/spec_lint.py" --quiet >/dev/null 2>&1; then
  echo "pre-push: WARNING - a bench/spec arm references an env/flag this fork does not read (inert arm):" >&2
  "$_py" "$repo/tools/pancake/spec_lint.py" >&2 || true
  echo "pre-push: fix the spec (or SKIP_BUILD_MATRIX=1 to push anyway for once)." >&2
fi
if [ "${PANCAKE_PUSH_FULL:-0}" = 1 ]; then
  exec "$repo/tools/pancake/build_matrix.sh" --head
fi
exec "$repo/tools/pancake/build_matrix.sh" --head --configure-only
