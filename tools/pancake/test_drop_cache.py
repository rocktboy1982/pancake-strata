"""Unit tests for drop_cache.py: which files are hinted (owner, regular, size, symlinks), that nothing is modified, the
dry run, the /proc parsers (fixtures) and the command line.  Temp dirs only, no root, no GPU.
Run: python -m pytest tools/pancake/test_drop_cache.py   (or python tools/pancake/test_drop_cache.py)"""
import contextlib
import io
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drop_cache as D  # noqa: E402

MIB = 1 << 20


def make_file(path, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def make_tree(root):
    """big.bin (5 MiB), sub/exact.bin (exactly 4 MiB), sub/deep/near.bin (4 MiB - 1), small.bin (1 MiB), link -> big."""
    big = make_file(root / "big.bin", 5 * MIB)
    make_file(root / "sub" / "exact.bin", 4 * MIB)
    make_file(root / "sub" / "deep" / "near.bin", 4 * MIB - 1)
    make_file(root / "small.bin", 1 * MIB)
    (root / "link").symlink_to(big)
    return big


def test_hints_big_owned_regular_files_recursively_and_nothing_else():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_tree(root)
        with mock.patch("os.posix_fadvise") as fadvise:
            files, size = D.hint_tree([str(root)])
        assert (files, size) == (2, 5 * MIB + 4 * MIB)            # big.bin and sub/exact.bin only
        assert fadvise.call_count == 2
        assert all(c.args[1:] == (0, 0, os.POSIX_FADV_DONTNEED) for c in fadvise.call_args_list)   # whole file, DONTNEED


def test_dry_run_counts_the_same_and_sends_no_hint():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_tree(root)
        with mock.patch("os.posix_fadvise") as fadvise:
            files, size = D.hint_tree([str(root)], dry_run=True)
        assert (files, size) == (2, 9 * MIB)
        fadvise.assert_not_called()


def test_other_owners_are_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_tree(root)
        other = os.getuid() + 1                                   # not our uid: a file of someone else
        with mock.patch("os.posix_fadvise") as fadvise:
            assert D.hint_tree([str(root)], uid=other) == (0, 0)
        fadvise.assert_not_called()


def test_nothing_is_modified_or_removed():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_tree(root)
        before = {p: (p.stat().st_mtime_ns, p.stat().st_size, p.is_symlink()) for p in root.rglob("*")}
        contents = (root / "big.bin").read_bytes()
        D.hint_tree([str(root)])                                  # real syscalls, no mock
        after = {p: (p.stat().st_mtime_ns, p.stat().st_size, p.is_symlink()) for p in root.rglob("*")}
        assert before == after
        assert (root / "big.bin").read_bytes() == contents


def test_symlink_to_a_big_file_is_not_hinted_or_followed():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        big = make_file(root / "real" / "big.bin", 5 * MIB)
        (root / "link.bin").symlink_to(big)
        with mock.patch("os.posix_fadvise") as fadvise:
            files, _ = D.hint_tree([str(root)])
        assert files == 1                                         # the real file only, via real/
        assert fadvise.call_count == 1


def test_parsers_read_fixture_files():
    with tempfile.TemporaryDirectory() as tmp:
        meminfo = Path(tmp, "meminfo")
        meminfo.write_text("MemTotal:       60000000 kB\nMemFree:        20971520 kB\nMemAvailable:   30000000 kB\n")
        assert abs(D.mem_free_gib(str(meminfo)) - 20.0) < 1e-9
        buddy = Path(tmp, "buddyinfo")
        # order 8 = 2 blocks (ignored), order 9 = 3 blocks, order 10 = 4 blocks
        buddy.write_text("Node 0, zone      DMA      0 0 0 0 0 0 0 0 0 0 0\n"
                         "Node 0, zone   Normal      0 0 0 0 0 0 0 0 2 3 4\n")
        page = os.sysconf("SC_PAGE_SIZE")
        expect = (3 * 2**9 + 4 * 2**10) * page / 2**30
        assert abs(D.big_block_gib(str(buddy)) - expect) < 1e-12


def test_default_roots_exist_only():
    roots = D.default_roots()
    assert all(os.path.isdir(r) for r in roots)


def test_cli_dry_run_prints_before_and_no_after():
    with tempfile.TemporaryDirectory() as tmp:
        make_tree(Path(tmp))
        out = io.StringIO()
        with contextlib.redirect_stdout(out), mock.patch("os.posix_fadvise") as fadvise:
            assert D.main(["--dry-run", tmp]) == 0
        fadvise.assert_not_called()
        text = out.getvalue()
        assert text.startswith("before: MemFree ") and "free 2 MiB blocks" in text
        assert "would hint 2 files" in text and "after:" not in text


def test_cli_hints_then_prints_after():
    with tempfile.TemporaryDirectory() as tmp:
        make_tree(Path(tmp))
        out = io.StringIO()
        with contextlib.redirect_stdout(out), mock.patch("os.posix_fadvise") as fadvise:
            assert D.main([tmp]) == 0
        assert fadvise.call_count == 2
        text = out.getvalue()
        assert "hinted 2 files" in text and "after:  MemFree " in text


def test_cli_rejects_a_missing_directory():
    with tempfile.TemporaryDirectory() as tmp:
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            assert D.main([os.path.join(tmp, "missing")]) == 2
        assert "not a directory" in err.getvalue()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
