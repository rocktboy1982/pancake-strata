// include/strata/kernels/klog.hpp - pancake-strata: the one-line error report the kernels ported from basalt use.
#pragma once

#include <cstdio>

namespace strata::kernels {
template <class... A>
inline void klog_error(const char* fmt, A... a) {
    std::fputs("strata: ", stderr);
    std::fprintf(stderr, fmt, a...);
    std::fputc('\n', stderr);
}
inline void klog_error(const char* msg) { std::fprintf(stderr, "strata: %s\n", msg); }
}  // namespace strata::kernels
