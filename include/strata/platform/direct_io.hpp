// include/strata/platform/direct_io.hpp - O_DIRECT switched on an open descriptor, with the buffered fallback.
//
// One implementation of the try-direct / leave-direct pattern (first in conversation_file.cpp): a file is opened
// buffered and O_DIRECT is then set with fcntl.  A filesystem that refuses direct I/O (EINVAL: tmpfs, overlayfs, a
// compressed or sparse NTFS file on ntfs3) keeps the buffered descriptor, so the file is never opened twice and the
// caller decides what a buffered read costs it.  A transfer the filesystem refuses later (EINVAL from pread/pwrite on
// a descriptor that took O_DIRECT) is handled with leave_direct: nothing was transferred, the same call is retried
// buffered.  POSIX only; where O_DIRECT does not exist (macOS) every descriptor stays buffered.
#pragma once

#if !defined(_WIN32)
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>

namespace strata::platform {

/// Set O_DIRECT on `fd`; true when the descriptor now reads/writes past the page cache.
inline bool try_direct(int fd) {
#if defined(O_DIRECT)
    if (fd < 0 || O_DIRECT == 0) return false;   // (0: a build that defines it away)
    const int fl = ::fcntl(fd, F_GETFL);
    return fl >= 0 && ((fl & O_DIRECT) != 0 || ::fcntl(fd, F_SETFL, fl | O_DIRECT) == 0);
#else
    (void) fd;
    return false;
#endif
}

/// Clear O_DIRECT on `fd` (a filesystem that took it at open but refuses this transfer); true when it is cleared.
inline bool leave_direct(int fd) {
#if defined(O_DIRECT)
    if (fd < 0) return false;
    if (O_DIRECT == 0) return true;
    const int fl = ::fcntl(fd, F_GETFL);
    return fl >= 0 && ((fl & O_DIRECT) == 0 || ::fcntl(fd, F_SETFL, fl & ~O_DIRECT) == 0);
#else
    (void) fd;
    return true;
#endif
}

/// Open `path` read-only and try O_DIRECT on it (`want_direct`); `direct` says whether it was taken.  -1 (errno set)
/// only when the file itself cannot be opened - a refused O_DIRECT is a buffered descriptor, not a failure.
inline int open_read_try_direct(const char* path, bool& direct, bool want_direct = true) {
    direct = false;
    const int fd = ::open(path, O_RDONLY | O_CLOEXEC);
    if (fd >= 0 && want_direct) direct = try_direct(fd);
    return fd;
}

}  // namespace strata::platform
#endif
