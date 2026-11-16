// src/qwen35/q36_prompt_swaps_test.cpp - STRATA_Q36_PROMPT_SWAPS's parse (LEADS.md gputier/StrataGP 5f7c616102 / 819b1dba79;
// q36_prompt_swaps.hpp), no GPU: unset / empty keep this tree's 2048, 0 is no cap, digits only, anything else is refused with a line.
#include "q36_prompt_swaps.hpp"

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
}  // namespace

int main() {
    using namespace strata::q36;
    std::string err;
    check(prompt_swaps_from_env(nullptr, &err) == 2048 && err.empty(), "unset keeps 2048 (byte-identical default)");
    check(prompt_swaps_from_env("", &err) == 2048 && err.empty(), "empty keeps 2048");
    check(prompt_swaps_from_env("4096", &err) == 4096 && err.empty(), "4096 (gputier's measured value) is taken");
    check(prompt_swaps_from_env("1", &err) == 1 && err.empty(), "1 is taken");
    check(prompt_swaps_from_env("0", &err) == INT_MAX && err.empty(), "0 is no cap");
    check(prompt_swaps_from_env("0008", &err) == 8 && err.empty(), "leading zeros read as decimal");
    check(prompt_swaps_from_env("2147483647", &err) == INT_MAX && err.empty(), "INT_MAX is taken");
    check(prompt_swaps_from_env("2147483648", &err) == 2048 && !err.empty(), "above INT_MAX is refused");
    err.clear();
    check(prompt_swaps_from_env("-1", &err) == 2048 && !err.empty(), "a sign is refused");
    err.clear();
    check(prompt_swaps_from_env(" 8", &err) == 2048 && !err.empty(), "a leading space is refused");
    err.clear();
    check(prompt_swaps_from_env("8k", &err) == 2048 && !err.empty(), "trailing text is refused");
    err.clear();
    check(prompt_swaps_from_env("99999999999999999999999", &err) == 2048 && !err.empty(), "out of long long range is refused");
    check(prompt_swaps_from_env("abc", nullptr) == 2048, "a null err is allowed");
    std::printf("q36_prompt_swaps_test: %d checks passed\n", checks);
    return 0;
}
