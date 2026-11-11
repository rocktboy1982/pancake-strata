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
if [ "${PANCAKE_PUSH_FULL:-0}" = 1 ]; then
  exec "$repo/tools/pancake/build_matrix.sh" --head
fi
exec "$repo/tools/pancake/build_matrix.sh" --head --configure-only
