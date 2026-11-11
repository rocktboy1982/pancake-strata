#!/usr/bin/env python3
"""drop_cache.py - ask the kernel to drop the page cache of the big files you own, without root.

Why: the RAM fills with clean page cache of model, build and checkout files.  The expert arena (33 GiB) needs free
2 MiB blocks, and after builds there are few of them (2026-11-11: 10 GiB free in blocks of 2 MiB or more).  This helper
hints the kernel to drop those pages with posix_fadvise(POSIX_FADV_DONTNEED).  On 2026-11-11 it took free RAM from
19 to 32 GiB and the free 2 MiB blocks from 10 to 13 GiB.  It is the non-root fallback for `sudo os_tune.sh compact`.

What it does, and what it never does:
- It only hints.  The kernel keeps the pages that a process has mapped and the dirty ones (for those it starts the
  writeback and drops nothing).  Nothing is deleted, renamed, truncated or written by this script.
- It touches regular files only (symlinks are not followed), owned by the user running it, at least 4 MiB in size.
  Files are opened read-only with O_NOATIME, so their access times stay as they were.
- It walks the directories given on the command line recursively.  Default: ~/.cache, ~/clone-of-d/src and
  ~/backup-pancake-* (the defaults that do not exist are skipped).  Given directories must exist.
- It prints the free memory before and after: MemFree and the free blocks of buddy order >= 9 (2 MiB with 4 KiB pages)
  from /proc/buddyinfo.  Linux only.

Usage:
    python3 tools/pancake/drop_cache.py [--dry-run] [DIR ...]
    --dry-run   count the files that would be hinted and send no hint (no "after" line)

Exit status: 0 done, 2 a given DIR is not a directory.
"""
import argparse
import glob
import os
import stat
import sys

MIN_BYTES = 4 << 20                                  # smaller files are not worth a hint
NOATIME = getattr(os, "O_NOATIME", 0)                # reading our own files must not change their atime
DEFAULT_DIRS = ["~/.cache", "~/clone-of-d/src"]
DEFAULT_GLOBS = ["~/backup-pancake-*"]


def hint(path):
    """Ask the kernel to drop the cached pages of one file (a hint: see the module docstring)."""
    fd = os.open(path, os.O_RDONLY | NOATIME)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def hint_tree(roots, dry_run=False, uid=None):
    """Hint every big regular file under roots that belongs to uid (default: us).  Returns (files, bytes): the files
    hinted, or that would be hinted with dry_run."""
    uid = os.getuid() if uid is None else uid
    files = size = 0
    for root in roots:
        for dirpath, _dirs, names in os.walk(root):  # os.walk does not descend into symlinked directories
            for name in names:
                path = os.path.join(dirpath, name)
                try:
                    st = os.lstat(path)              # lstat: a symlink is neither followed nor hinted
                    if not stat.S_ISREG(st.st_mode) or st.st_uid != uid or st.st_size < MIN_BYTES:
                        continue
                    if not dry_run:
                        hint(path)
                except OSError:                      # vanished or unreadable: nothing to hint
                    continue
                files += 1
                size += st.st_size
    return files, size


def mem_free_gib(path="/proc/meminfo"):
    with open(path) as f:
        for line in f:
            if line.startswith("MemFree:"):
                return int(line.split()[1]) / 2**20  # kB -> GiB
    raise ValueError(f"no MemFree line in {path}")


def big_block_gib(path="/proc/buddyinfo"):
    """Free blocks of buddy order >= 9 (2 MiB with 4 KiB pages), in GiB, summed over all zones."""
    page = os.sysconf("SC_PAGE_SIZE")
    total = 0
    with open(path) as f:
        for line in f:
            if not line.startswith("Node"):
                continue
            counts = line.split()[4:]                # "Node", "0,", "zone", <zone name>, then one count per order
            total += sum(int(c) * (1 << order) * page for order, c in enumerate(counts) if order >= 9)
    return total / 2**30


def snapshot():
    return f"MemFree {mem_free_gib():.1f} GiB, free 2 MiB blocks {big_block_gib():.1f} GiB"


def default_roots():
    roots = [os.path.expanduser(d) for d in DEFAULT_DIRS]
    roots += sorted(p for g in DEFAULT_GLOBS for p in glob.glob(os.path.expanduser(g)))
    return [r for r in roots if os.path.isdir(r)]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Drop the page cache of the big files you own (posix_fadvise, no root).")
    ap.add_argument("dirs", nargs="*", help="directories to walk (default: ~/.cache, ~/clone-of-d/src, ~/backup-pancake-*)")
    ap.add_argument("--dry-run", action="store_true", help="count what would be hinted; send no hint")
    a = ap.parse_args(argv)
    bad = [d for d in a.dirs if not os.path.isdir(d)]
    if bad:
        print(f"drop_cache: not a directory: {', '.join(bad)}", file=sys.stderr)
        return 2
    roots = a.dirs or default_roots()
    print(f"before: {snapshot()}")
    files, size = hint_tree(roots, dry_run=a.dry_run)
    verb = "would hint" if a.dry_run else "hinted"
    print(f"{verb} {files} files, {size / 2**30:.1f} GiB (owned regular files of at least 4 MiB)")
    if not a.dry_run:
        print(f"after:  {snapshot()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
