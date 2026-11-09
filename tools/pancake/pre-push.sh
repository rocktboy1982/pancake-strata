#!/usr/bin/env bash
# tools/pancake/pre-push.sh - git pre-push hook (MOD_PLAN item 92): configure every build-matrix row from the
# committed HEAD (~2 min).  Full link check: PANCAKE_PUSH_FULL=1 git push.  Skip once: SKIP_BUILD_MATRIX=1 git push.
# Install: ln -sf ../../tools/pancake/pre-push.sh "$(git rev-parse --git-common-dir)/hooks/pre-push"
[ "${SKIP_BUILD_MATRIX:-0}" = 1 ] && exit 0
repo=$(git rev-parse --show-toplevel)
if [ "${PANCAKE_PUSH_FULL:-0}" = 1 ]; then
  exec "$repo/tools/pancake/build_matrix.sh" --head
fi
exec "$repo/tools/pancake/build_matrix.sh" --head --configure-only
