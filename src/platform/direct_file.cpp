// src/platform/direct_file.cpp - see include/strata/platform/direct_file.hpp.
#include "strata/platform/direct_file.hpp"
#include "strata/platform/aux_cpus.hpp"
#include "strata/platform/direct_io.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__linux__) && defined(__has_include)
#if __has_include(<linux/io_uring.h>)
#include <cerrno>
#include <cstdio>
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#define STRATA_HAVE_URING 1
#endif
#endif
#endif

namespace strata::platform {

double now_us() {
    using namespace std::chrono;
    return (double) duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count() / 1000.0;
}

void* DirectFile::alloc_aligned(size_t bytes) {
#if defined(_WIN32)
    return VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* p = nullptr;
    return posix_memalign(&p, alignment(), bytes) == 0 ? p : nullptr;
#endif
}

void DirectFile::free_aligned(void* p) {
    if (p == nullptr) return;
#if defined(_WIN32)
    VirtualFree(p, 0, MEM_RELEASE);
#else
    std::free(p);
#endif
}

namespace {
/// A queued read: `submit` only queues it; the pool's threads issue it (perf-review C-4 / F-2).  One thread
/// issuing every read was the limit of the n-gram table's prompt reads: ~12 us of kernel time per read, ~80K
/// reads/s, while an NVMe drive serves several times that at depth.
struct Pending {
    uint64_t offset;
    void* buffer;
    uint32_t length;
    uint64_t tag;
};

/// The number of issuing threads: STRATA_IO_THREADS, else 4 (Windows: overlapped submits) / 16 (Linux: each
/// thread does one blocking pread, so the thread count is the queue depth).
int io_threads(int dflt) {
    const char* v = std::getenv("STRATA_IO_THREADS");
    const int n = v ? std::atoi(v) : dflt;
    return std::clamp(n, 1, 64);
}
}  // namespace

#if defined(_WIN32)
// ------------------------------------------------------------------------------------------------ Windows
namespace {
/// One in-flight request. The OVERLAPPED must be the first member so a completion packet's OVERLAPPED*
/// converts back to the request.
struct Req {
    OVERLAPPED ov;
    uint64_t tag;
};
}  // namespace

struct DirectFile::Impl {
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE port = nullptr;
    uint64_t size = 0;
    std::mutex mu;                      // everything below
    std::deque<Req*> free_reqs;
    std::vector<Req*> all_reqs;
    std::deque<Completion> immediate;   // requests that completed without a port packet (EOF) or failed
    std::deque<Pending> queue;          // submitted, not yet issued
    std::condition_variable cv;
    bool stop = false;
    std::vector<std::thread> pool;

    Req* take() {
        if (free_reqs.empty()) {
            Req* r = new Req();
            all_reqs.push_back(r);
            return r;
        }
        Req* r = free_reqs.front();
        free_reqs.pop_front();
        return r;
    }

    void issue(const Pending& p) {
        Req* r;
        {
            std::lock_guard<std::mutex> lk(mu);
            r = take();
        }
        std::memset(&r->ov, 0, sizeof r->ov);
        r->ov.Offset = (DWORD) (p.offset & 0xFFFFFFFFull);
        r->ov.OffsetHigh = (DWORD) (p.offset >> 32);
        r->tag = p.tag;
        if (!ReadFile(file, p.buffer, p.length, nullptr, &r->ov)) {
            const DWORD e = GetLastError();
            if (e != ERROR_IO_PENDING) {
                std::lock_guard<std::mutex> lk(mu);
                free_reqs.push_back(r);
                // at or past end of file: a zero-byte completion; anything else: a failed one
                immediate.push_back(Completion{p.tag, 0, e == ERROR_HANDLE_EOF});
                PostQueuedCompletionStatus(port, 0, 0, nullptr);   // wake a waiter to collect it
            }
        }
    }

    void worker() {
        std::unique_lock<std::mutex> lk(mu);
        for (;;) {
            cv.wait(lk, [&] { return stop || !queue.empty(); });
            if (stop && queue.empty()) return;
            const Pending p = queue.front();
            queue.pop_front();
            lk.unlock();
            issue(p);
            lk.lock();
        }
    }

    void stop_pool() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
        }
        cv.notify_all();
        for (std::thread& t : pool) t.join();
        pool.clear();
        stop = false;
    }
};

DirectFile::DirectFile() : impl_(new Impl) {}
DirectFile::~DirectFile() {
    close();
    for (Req* r : impl_->all_reqs) delete r;
    delete impl_;
}

bool DirectFile::open(const std::string& path, std::string& err) {
    close();
    const int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring wpath((size_t) (wlen > 0 ? wlen : 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wlen);
    // NO_BUFFERING: the cache manager never holds these pages, so the table cannot grow into RAM.
    // RANDOM_ACCESS: no read-ahead. OVERLAPPED: many reads in flight, completed through the port below.
    impl_->file = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                              FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (impl_->file == INVALID_HANDLE_VALUE) {
        err = "DirectFile: cannot open " + path + " (error " + std::to_string(GetLastError()) + ")";
        return false;
    }
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(impl_->file, &sz)) {
        err = "DirectFile: cannot size " + path;
        close();
        return false;
    }
    impl_->size = (uint64_t) sz.QuadPart;
    impl_->port = CreateIoCompletionPort(impl_->file, nullptr, 0, 1);
    if (impl_->port == nullptr) {
        err = "DirectFile: CreateIoCompletionPort failed (error " + std::to_string(GetLastError()) + ")";
        close();
        return false;
    }
    // Completions of reads that finish synchronously are still queued to the port, so every issued read
    // produces exactly one packet; `wait` is the only completion path.
    const int n = io_threads(4);
    for (int i = 0; i < n; ++i) impl_->pool.emplace_back([this] { strata::aux_cpus::pin_current_thread(); impl_->worker(); });
    return true;
}

void DirectFile::close() {
    impl_->stop_pool();
    if (impl_->port != nullptr) CloseHandle(impl_->port);
    if (impl_->file != INVALID_HANDLE_VALUE) CloseHandle(impl_->file);
    impl_->port = nullptr;
    impl_->file = INVALID_HANDLE_VALUE;
    impl_->size = 0;
    impl_->immediate.clear();
    impl_->queue.clear();
}

bool DirectFile::is_open() const { return impl_->file != INVALID_HANDLE_VALUE; }
uint64_t DirectFile::size() const { return impl_->size; }

bool DirectFile::submit(uint64_t offset, void* buffer, uint32_t length, uint64_t tag, std::string& err) {
    if (!is_open()) { err = "DirectFile: not open"; return false; }
    if (offset % alignment() || length % alignment() || ((uintptr_t) buffer) % alignment() || length == 0) {
        err = "DirectFile: unaligned request";
        return false;
    }
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        impl_->queue.push_back(Pending{offset, buffer, length, tag});
    }
    impl_->cv.notify_one();
    return true;
}

int DirectFile::wait(Completion* out, int max, int timeout_ms) {
    int n = 0;
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        while (n < max && !impl_->immediate.empty()) {
            out[n++] = impl_->immediate.front();
            impl_->immediate.pop_front();
        }
    }
    if (n == max || !is_open()) return n;
    OVERLAPPED_ENTRY entries[64];
    const ULONG want = (ULONG) (max - n < 64 ? max - n : 64);
    ULONG got = 0;
    const DWORD t = timeout_ms < 0 ? INFINITE : (DWORD) timeout_ms;
    if (!GetQueuedCompletionStatusEx(impl_->port, entries, want, &got, n > 0 ? 0 : t, FALSE)) return n;
    std::lock_guard<std::mutex> lk(impl_->mu);
    for (ULONG i = 0; i < got; ++i) {
        if (entries[i].lpOverlapped == nullptr) {          // a wake() packet (or an immediate completion's nudge)
            while (n < max && !impl_->immediate.empty()) {
                out[n++] = impl_->immediate.front();
                impl_->immediate.pop_front();
            }
            if (n < max) out[n++] = Completion{WAKE_TAG, 0, true};
            continue;
        }
        Req* r = (Req*) entries[i].lpOverlapped;
        const uint32_t status = (uint32_t) r->ov.Internal;   // an NTSTATUS
        Completion c;
        c.tag = r->tag;
        c.bytes = entries[i].dwNumberOfBytesTransferred;
        // STATUS_END_OF_FILE (0xC0000011) is a legal short read at the table's last page.
        c.ok = status == 0 || status == 0xC0000011u;
        out[n++] = c;
        impl_->free_reqs.push_back(r);
    }
    return n;
}

void DirectFile::wake() {
    if (impl_->port != nullptr) PostQueuedCompletionStatus(impl_->port, 0, 0, nullptr);
}

#else
// ------------------------------------------------------------------------------------------------ POSIX
// perf-review F-2: a pool of threads, each doing blocking O_DIRECT preads, so reads run in parallel (the thread
// count is the queue depth).  It replaced one synchronous pread inside `submit`, which read the n-gram table's
// rows one at a time.
#if defined(STRATA_HAVE_URING)
namespace {
// STRATA_IO_URING=N (opt-in, Linux; unset or 0 = the pread pool, as before): one thread issues the reads through an
// io_uring of depth N (8..1024) instead of N blocking pread threads.  The same reads of the same bytes into the same
// buffers, completed in whatever order the drive finishes them (the pool's completions are unordered too).  Raw
// syscalls, no liburing.  A kernel that refuses io_uring_setup keeps the pread pool (said once).  Lead: LEADS.md Part D
// "io_uring in DirectFile" (data-path report: ~1.06x drop-in; SQPOLL not used, it burns a core).
int uring_depth() {
    static const int d = [] {
        const char* e = std::getenv("STRATA_IO_URING");
        const int v = e ? std::atoi(e) : 0;
        return v <= 0 ? 0 : std::clamp(v, 8, 1024);
    }();
    return d;
}

struct Uring {
    int fd = -1;
    unsigned entries = 0;
    void* sq_ptr = nullptr;
    void* cq_ptr = nullptr;
    size_t sq_sz = 0, cq_sz = 0, sqes_sz = 0;
    io_uring_sqe* sqes = nullptr;
    unsigned *sq_head = nullptr, *sq_tail = nullptr, *sq_mask = nullptr, *sq_array = nullptr;
    unsigned *cq_head = nullptr, *cq_tail = nullptr, *cq_mask = nullptr;
    io_uring_cqe* cqes = nullptr;

    bool init(unsigned n) {
        io_uring_params p;
        std::memset(&p, 0, sizeof p);
        fd = (int) syscall(__NR_io_uring_setup, n, &p);
        if (fd < 0) return false;
        entries = p.sq_entries;
        sq_sz = p.sq_off.array + p.sq_entries * sizeof(unsigned);
        cq_sz = p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe);
        const bool single = (p.features & IORING_FEAT_SINGLE_MMAP) != 0;
        if (single) sq_sz = cq_sz = std::max(sq_sz, cq_sz);
        sq_ptr = mmap(nullptr, sq_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQ_RING);
        if (sq_ptr == MAP_FAILED) { sq_ptr = nullptr; destroy(); return false; }
        if (single) cq_ptr = sq_ptr;
        else {
            cq_ptr = mmap(nullptr, cq_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_CQ_RING);
            if (cq_ptr == MAP_FAILED) { cq_ptr = nullptr; destroy(); return false; }
        }
        sqes_sz = p.sq_entries * sizeof(io_uring_sqe);
        void* s = mmap(nullptr, sqes_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQES);
        if (s == MAP_FAILED) { destroy(); return false; }
        sqes = (io_uring_sqe*) s;
        char* sq = (char*) sq_ptr;
        sq_head = (unsigned*) (sq + p.sq_off.head);
        sq_tail = (unsigned*) (sq + p.sq_off.tail);
        sq_mask = (unsigned*) (sq + p.sq_off.ring_mask);
        sq_array = (unsigned*) (sq + p.sq_off.array);
        char* cq = (char*) cq_ptr;
        cq_head = (unsigned*) (cq + p.cq_off.head);
        cq_tail = (unsigned*) (cq + p.cq_off.tail);
        cq_mask = (unsigned*) (cq + p.cq_off.ring_mask);
        cqes = (io_uring_cqe*) (cq + p.cq_off.cqes);
        return true;
    }

    void destroy() {
        if (sqes != nullptr) munmap(sqes, sqes_sz);
        if (cq_ptr != nullptr && cq_ptr != sq_ptr) munmap(cq_ptr, cq_sz);
        if (sq_ptr != nullptr) munmap(sq_ptr, sq_sz);
        if (fd >= 0) ::close(fd);
        sqes = nullptr;
        sq_ptr = cq_ptr = nullptr;
        fd = -1;
    }

    /// Queues one read (the caller keeps at most `entries` in flight, so a slot is always free).
    void push_read(int file, const Pending& r) {
        const unsigned tail = *sq_tail;   // only this thread writes the tail
        const unsigned idx = tail & *sq_mask;
        io_uring_sqe* e = &sqes[idx];
        std::memset(e, 0, sizeof *e);
        e->opcode = IORING_OP_READ;
        e->fd = file;
        e->addr = (unsigned long long) (uintptr_t) r.buffer;
        e->len = r.length;
        e->off = r.offset;
        e->user_data = r.tag;
        sq_array[idx] = idx;
        __atomic_store_n(sq_tail, tail + 1, __ATOMIC_RELEASE);
    }
};
}  // namespace
#endif

struct DirectFile::Impl {
    int fd = -1;
    uint64_t size = 0;
    std::mutex mu;
    std::condition_variable cv_work, cv_done;
    std::deque<Pending> queue;
    std::deque<Completion> done;
    bool stop = false;
    std::vector<std::thread> pool;
#if defined(STRATA_HAVE_URING)
    Uring ring;   // STRATA_IO_URING: ring.fd >= 0 while the uring worker runs

    void uring_worker() {
        unsigned inflight = 0, unsubmitted = 0;   // this thread only
        std::vector<Pending> batch;
        for (;;) {
            batch.clear();
            {
                std::unique_lock<std::mutex> lk(mu);
                if (inflight == 0) cv_work.wait(lk, [&] { return stop || !queue.empty(); });
                if (stop && queue.empty() && inflight == 0) return;
                while (!queue.empty() && inflight + batch.size() < ring.entries) {
                    batch.push_back(queue.front());
                    queue.pop_front();
                }
            }
            for (const Pending& p : batch) ring.push_read(fd, p);
            inflight += (unsigned) batch.size();
            unsubmitted += (unsigned) batch.size();
            const long r = syscall(__NR_io_uring_enter, ring.fd, unsubmitted, inflight > 0 ? 1u : 0u,
                                   IORING_ENTER_GETEVENTS, nullptr, 0);
            if (r >= 0) unsubmitted -= std::min<unsigned>(unsubmitted, (unsigned) r);
            else if (errno != EINTR && errno != EAGAIN && errno != EBUSY) {
                static bool said = false;
                if (!said) { said = true; std::fprintf(stderr, "strata: STRATA_IO_URING: io_uring_enter failed (errno %d); retrying\n", errno); }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            unsigned head = *ring.cq_head;
            const unsigned tail = __atomic_load_n(ring.cq_tail, __ATOMIC_ACQUIRE);
            if (head == tail) continue;
            std::lock_guard<std::mutex> lk(mu);
            while (head != tail) {
                const io_uring_cqe& c = ring.cqes[head & *ring.cq_mask];
                done.push_back(Completion{(uint64_t) c.user_data, c.res < 0 ? 0u : (uint32_t) c.res, c.res >= 0});
                ++head;
                --inflight;
            }
            __atomic_store_n(ring.cq_head, head, __ATOMIC_RELEASE);
            cv_done.notify_all();
        }
    }
#endif

    void worker() {
        std::unique_lock<std::mutex> lk(mu);
        for (;;) {
            cv_work.wait(lk, [&] { return stop || !queue.empty(); });
            if (stop && queue.empty()) return;
            const Pending p = queue.front();
            queue.pop_front();
            lk.unlock();
            const ssize_t got = pread(fd, p.buffer, p.length, (off_t) p.offset);
            lk.lock();
            done.push_back(Completion{p.tag, got < 0 ? 0u : (uint32_t) got, got >= 0});
            cv_done.notify_all();
        }
    }

    void stop_pool() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
        }
        cv_work.notify_all();
        for (std::thread& t : pool) t.join();
        pool.clear();
        stop = false;
    }
};

DirectFile::DirectFile() : impl_(new Impl) {}
DirectFile::~DirectFile() { close(); delete impl_; }

bool DirectFile::open(const std::string& path, std::string& err) {
    close();
#if defined(__APPLE__)   // macOS has no O_DIRECT: F_NOCACHE keeps the reads out of the unified buffer cache
    impl_->fd = ::open(path.c_str(), O_RDONLY);
    if (impl_->fd >= 0 && fcntl(impl_->fd, F_NOCACHE, 1) != 0) { ::close(impl_->fd); impl_->fd = -1; }
#else
    // a filesystem that refuses O_DIRECT (tmpfs, overlayfs, a compressed NTFS file) keeps a buffered descriptor: the
    // reads are aligned either way, they only go through the page cache
    bool direct = false;
    impl_->fd = open_read_try_direct(path.c_str(), direct);
#endif
    if (impl_->fd < 0) { err = "DirectFile: cannot open " + path; return false; }
    struct stat st;
    if (fstat(impl_->fd, &st) != 0) { err = "DirectFile: cannot size " + path; close(); return false; }
    impl_->size = (uint64_t) st.st_size;
#if defined(STRATA_HAVE_URING)
    if (uring_depth() > 0) {   // STRATA_IO_URING (opt-in): one issuing thread over a ring instead of the pread pool
        if (impl_->ring.init((unsigned) uring_depth())) {
            impl_->pool.emplace_back([this] { strata::aux_cpus::pin_current_thread(); impl_->uring_worker(); });
            return true;
        }
        static bool said = false;
        if (!said) { said = true; std::fprintf(stderr, "strata: STRATA_IO_URING: io_uring_setup refused; pread pool kept\n"); }
    }
#endif
    const int n = io_threads(16);
    for (int i = 0; i < n; ++i) impl_->pool.emplace_back([this] { strata::aux_cpus::pin_current_thread(); impl_->worker(); });
    return true;
}

void DirectFile::close() {
    impl_->stop_pool();
#if defined(STRATA_HAVE_URING)
    impl_->ring.destroy();   // (after the worker joined: no read is in flight)
#endif
    if (impl_->fd >= 0) ::close(impl_->fd);
    impl_->fd = -1;
    impl_->size = 0;
    impl_->done.clear();
    impl_->queue.clear();
}

bool DirectFile::is_open() const { return impl_->fd >= 0; }
uint64_t DirectFile::size() const { return impl_->size; }

bool DirectFile::submit(uint64_t offset, void* buffer, uint32_t length, uint64_t tag, std::string& err) {
    if (offset % alignment() || length % alignment() || ((uintptr_t) buffer) % alignment() || length == 0) {
        err = "DirectFile: unaligned request";
        return false;
    }
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        impl_->queue.push_back(Pending{offset, buffer, length, tag});
    }
    impl_->cv_work.notify_one();
    return true;
}

void DirectFile::wake() {
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->done.push_back(Completion{WAKE_TAG, 0, true});
    impl_->cv_done.notify_all();
}

int DirectFile::wait(Completion* out, int max, int timeout_ms) {
    std::unique_lock<std::mutex> lk(impl_->mu);
    if (impl_->done.empty() && timeout_ms != 0) {
        auto ready = [&] { return !impl_->done.empty(); };
        if (timeout_ms < 0) impl_->cv_done.wait(lk, ready);
        else impl_->cv_done.wait_for(lk, std::chrono::milliseconds(timeout_ms), ready);
    }
    int n = 0;
    while (n < max && !impl_->done.empty()) {
        out[n++] = impl_->done.front();
        impl_->done.pop_front();
    }
    return n;
}
#endif

}  // namespace strata::platform
