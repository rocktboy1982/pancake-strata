#!/usr/bin/env bash
# tools/pancake/pre-push-doc-check.sh - git pre-push hook: the documentation rules of AGENTS.md as a gate (LEADS.md Part O O4).
# Runs tools/pancake/doc_check.py (markdown outside the allowed set, ledger ids and spec numbers used twice, LEADS part letters, paths the docs name that do
# not exist) and BLOCKS the push on any NEW finding (the baseline tools/pancake/doc_check_baseline.json is empty and only shrinks).  Read-only, a second.
# tools/pancake/pre-push.sh calls this script too, so installing that one hook is enough.  Stand-alone install (does not touch any global git config):
#   ln -sf ../../tools/pancake/pre-push-doc-check.sh "$(git rev-parse --git-common-dir)/hooks/pre-push"
# Skip once: SKIP_DOC_CHECK=1 git push   (SKIP_BUILD_MATRIX=1 also skips, as for pre-push.sh).
[ "${SKIP_DOC_CHECK:-0}" = 1 ] && exit 0
[ "${SKIP_BUILD_MATRIX:-0}" = 1 ] && exit 0
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_PREFIX GIT_COMMON_DIR GIT_OBJECT_DIRECTORY GIT_ALTERNATE_OBJECT_DIRECTORIES
repo=$(git rev-parse --show-toplevel)
if python3 -c "" >/dev/null 2>&1; then _py=python3; else _py=python; fi   # (the Windows Store python3 stub exists but does not run)
if ! "$_py" "$repo/tools/pancake/doc_check.py" >&2; then
  echo "pre-push: doc_check found a NEW documentation-rule failure (above). Fix it (AGENTS.md rules 2-4, 7), or SKIP_DOC_CHECK=1 to push anyway for once." >&2
  exit 1
fi
exit 0
