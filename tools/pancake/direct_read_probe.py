#!/usr/bin/env python3
"""direct_read_probe.py - what can this disk deliver to the expert-arena loader? (MOD_PLAN A52 / item 38)

The loader reads ~33 GiB of expert tensors in 8 MB chunks.  The plan estimated "start 28-65 s -> ~10-14 s" with O_DIRECT
and 16 threads from the NVMe's PCIe 3.0 x4 link; this tool measures the part that decides it - sequential read
throughput of the file system the model sits on - on a small window of a model file, buffered against O_DIRECT, and
projects the arena read time (default 33.02 GiB, the IQ2_XS arena).  It never reads more than HARD_CAP_MIB in total.

    python tools/pancake/direct_read_probe.py MODEL.gguf [--offset-mib 8192] [--window-mib 96] [--chunk-mib 8]
                                              [--arms buffered:6,direct:16,direct:6] [--arena-gib 33.02] [--json OUT]

Arm "MODE:THREADS" reads its own window (the windows follow each other, so no arm is warmed by another) with THREADS
threads, each pread()ing CHUNK_MIB pieces into one page-aligned buffer (O_DIRECT needs aligned memory, offset and
length).  Before an arm its window is dropped from the page cache (posix_fadvise DONTNEED) and the share still resident
is reported; a window that is resident reads at RAM speed and says nothing about the disk.  A file system that
refuses O_DIRECT (EINVAL) falls back to buffered reads, as the engine's platform::open_read_try_direct does; the
arm's "mode" column says which one ran.  Read-only: nothing is written, so the stuck-writeback problem of ntfs3 does not
come into play.  Run it niced and with the machine quiet: another process reading the same disk lowers the result.
"""
import argparse
import ctypes
import errno
import fcntl
import json
import mmap
import os
import sys
import threading
import time
import zlib

HARD_CAP_MIB = 512          # total bytes one run may read, whatever the arguments say
ALIGN = 4096                # O_DIRECT: offset, length and memory aligned to this
MIB = 1 << 20


def mincore_resident(fd, offset, length):
    """Share (0..1) of the file range that is in the page cache, or None when mincore is unavailable."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        mm = mmap.mmap(fd, length, flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=offset)
    except (OSError, ValueError):
        return None
    try:
        anchor = ctypes.c_char.from_buffer(mm)       # only to get the address; no page is touched
        pages = (length + ALIGN - 1) // ALIGN
        vec = (ctypes.c_ubyte * pages)()
        libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        ok = libc.mincore(ctypes.addressof(anchor), length, vec) == 0
        del anchor
        return sum(v & 1 for v in vec) / pages if ok else None
    finally:
        mm.close()


def read_window(path, offset, length, chunk, threads, direct):
    """Reads [offset, offset+length) with `threads` threads, `chunk` bytes per pread.  Returns (seconds, crc32, mode).
    mode is "direct" when every read ran under O_DIRECT, else "buffered" (refused at open or by a pread)."""
    assert offset % ALIGN == 0 and length % ALIGN == 0 and chunk % ALIGN == 0
    out = mmap.mmap(-1, length)                     # page-aligned, so every chunk start is aligned too
    view = memoryview(out)
    pieces = list(range(0, length, chunk))
    lock = threading.Lock()
    state = {"next": 0, "error": None, "buffered": not direct}

    def open_fd():
        if direct:
            try:
                return os.open(path, os.O_RDONLY | os.O_DIRECT)
            except OSError as e:
                if e.errno != errno.EINVAL:
                    raise
                state["buffered"] = True            # the file system refuses O_DIRECT altogether
        return os.open(path, os.O_RDONLY)

    def leave_direct(fd):
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags & ~os.O_DIRECT)
        state["buffered"] = True

    def worker():
        try:
            fd = open_fd()
            try:
                while True:
                    with lock:
                        i = state["next"]
                        state["next"] += 1
                    if i >= len(pieces):
                        return
                    at, n = pieces[i], min(chunk, length - pieces[i])
                    got = 0
                    while got < n:
                        try:
                            k = os.preadv(fd, [view[at + got:at + n]], offset + at + got)
                        except OSError as e:
                            if e.errno == errno.EINTR:
                                continue
                            if e.errno == errno.EINVAL and direct and not state["buffered"]:
                                leave_direct(fd)
                                continue
                            raise
                        if k == 0:
                            raise EOFError("short read at byte %d" % (offset + at + got))
                        got += k
            finally:
                os.close(fd)
        except BaseException as e:                  # reported by the caller after the join
            state["error"] = e

    t0 = time.monotonic()
    pool = [threading.Thread(target=worker) for _ in range(threads)]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    seconds = time.monotonic() - t0
    if state["error"] is not None:
        view.release()
        out.close()
        raise state["error"]
    crc = zlib.crc32(view)
    view.release()
    out.close()
    return seconds, crc, "buffered" if state["buffered"] else "direct"


def parse_arms(text):
    arms = []
    for part in text.split(","):
        mode, _, n = part.partition(":")
        if mode not in ("buffered", "direct") or not n.isdigit() or int(n) < 1:
            raise ValueError("arm %r: expected buffered:N or direct:N" % part)
        arms.append((mode, int(n)))
    return arms


def project_seconds(arena_gib, gib_per_s):
    return arena_gib / gib_per_s if gib_per_s > 0 else float("inf")


def device_info(path):
    """Model and PCIe link of the NVMe disk under `path` (sysfs; None when it is not an NVMe disk)."""
    try:
        st = os.stat(path)
        node = os.path.realpath("/sys/dev/block/%d:%d" % (os.major(st.st_dev), os.minor(st.st_dev)))
        disk = node if os.path.exists(os.path.join(node, "device")) else os.path.dirname(node)
        dev = os.path.join(disk, "device")

        def rd(*parts):
            with open(os.path.join(*parts)) as f:
                return f.read().strip()
        return {"disk": os.path.basename(disk), "model": rd(dev, "model"),
                "link": "%s x%s" % (rd(dev, "device", "current_link_speed"), rd(dev, "device", "current_link_width"))}
    except OSError:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("file")
    ap.add_argument("--offset-mib", type=int, default=8192, help="start of the first window (default 8192: inside the expert tensors)")
    ap.add_argument("--window-mib", type=int, default=96)
    ap.add_argument("--chunk-mib", type=int, default=8, help="bytes per pread (the engine reads 16 experts, about 8 MB)")
    ap.add_argument("--arms", default="buffered:6,direct:16,direct:6")
    ap.add_argument("--arena-gib", type=float, default=33.02)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    arms = parse_arms(a.arms)
    total_mib = a.window_mib * len(arms)
    if total_mib > HARD_CAP_MIB:
        sys.exit("refusing to read %d MiB (%d arms x %d MiB): the cap is %d MiB" % (total_mib, len(arms), a.window_mib, HARD_CAP_MIB))
    size_mib = os.path.getsize(a.file) // MIB
    if a.offset_mib + total_mib > size_mib:
        sys.exit("%s holds %d MiB, the windows end at %d MiB" % (a.file, size_mib, a.offset_mib + total_mib))
    print("device:", device_info(a.file) or "unknown")
    rows = []
    fd = os.open(a.file, os.O_RDONLY)
    try:
        for i, (mode, threads) in enumerate(arms):
            off, length = (a.offset_mib + i * a.window_mib) * MIB, a.window_mib * MIB
            os.posix_fadvise(fd, off, length, os.POSIX_FADV_DONTNEED)
            resident = mincore_resident(fd, off, length)
            seconds, crc, ran = read_window(a.file, off, length, a.chunk_mib * MIB, threads, mode == "direct")
            rate = a.window_mib / 1024 / seconds
            rows.append({"arm": "%s:%d" % (mode, threads), "ran": ran, "window_mib": a.window_mib, "seconds": round(seconds, 3),
                         "gib_s": round(rate, 3), "resident_before": None if resident is None else round(resident, 3),
                         "crc32": "%08x" % crc, "arena_s": round(project_seconds(a.arena_gib, rate), 1)})
    finally:
        os.close(fd)
    print("%-12s %-9s %8s %8s %10s %12s" % ("arm", "ran", "seconds", "GiB/s", "resident", "%.2f GiB in" % a.arena_gib))
    for r in rows:
        res = "n/a" if r["resident_before"] is None else "%.0f%%" % (100 * r["resident_before"])
        print("%-12s %-9s %8.3f %8.3f %10s %10.1f s" % (r["arm"], r["ran"], r["seconds"], r["gib_s"], res, r["arena_s"]))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"file": a.file, "device": device_info(a.file), "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
