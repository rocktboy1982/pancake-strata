// LEADS.md "bugs 5.5-5.10", 5.9 (aktalascloud/strata-legacy-gpu d2fb1b9e4c, upstream PR #5530): ConversationCache::put
// publishes an image as one transaction.  Every allocation of a put() is failed in turn (the nth one after the cache
// holds `before` entries); a put that throws must leave the cache as it was (size, bytes) and the caller's image whole,
// and a retry must then park it with a serial that find() answers.  CPU only.
#include "strata/core/conversation_cache.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <vector>

static std::atomic<long> fail_n{0};
void* operator new(std::size_t n) {
    if (fail_n.load() > 0 && fail_n.fetch_sub(1) == 1) {
        fail_n = 0;
        throw std::bad_alloc();
    }
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

using strata::core::ConversationCache;
using strata::core::SavedConversation;

static SavedConversation make_image(int key) {
    SavedConversation s;
    s.live.ids = {key, key + 1};
    s.live.gdn.resize(64, 7);
    s.cvec = true;
    return s;
}

int main() {
    int failures = 0, cases = 0;
    for (int before = 0; before < 40; ++before)   // (a deque's block fills at a few dozen entries: its growth too)
        for (int nth = 1; nth <= 8; ++nth) {
            ConversationCache c(1 << 22, 64);
            for (int i = 0; i < before; ++i) {
                auto s = make_image(100 + i * 10);
                if (!c.put(std::move(s))) return 2;
            }
            auto image = make_image(9999);
            const auto ids = image.live.ids;
            const size_t bytes = image.bytes(), old_bytes = c.bytes();
            bool caught = false;
            fail_n = nth;
            try {
                if (!c.put(std::move(image))) return 3;
            } catch (const std::bad_alloc&) {
                caught = true;
            }
            fail_n = 0;
            ++cases;
            if (caught) {
                ++failures;
                const bool intact = c.size() == (size_t) before && c.bytes() == old_bytes && image.live.ids == ids &&
                                    image.live.gdn.size() == 64;
                if (!intact) {
                    std::printf("FAIL: the cache or the image changed after an allocation failure (before=%d nth=%d)\n",
                                before, nth);
                    return 1;
                }
                if (!c.put(std::move(image))) return 4;
            }
            if (c.size() != (size_t) before + 1 || c.bytes() != old_bytes + bytes) {
                std::printf("FAIL: size/bytes after the put (before=%d nth=%d)\n", before, nth);
                return 5;
            }
            auto q = ids;
            q.push_back(8);
            const auto m = c.best(q, {}, true);
            if (m.tokens != 2 || m.serial == 0 || c.find(m.serial) == nullptr) {
                std::printf("FAIL: the parked image is not found by its serial (before=%d nth=%d)\n", before, nth);
                return 6;
            }
        }
    if (failures == 0) {
        std::printf("FAIL: no allocation failure was injected into a put\n");
        return 7;
    }
    std::printf("conversation cache publication: %d cases, %d injected failures, all consistent: PASS\n", cases, failures);
    return 0;
}
