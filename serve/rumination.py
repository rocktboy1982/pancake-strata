"""serve/rumination.py - pancake item 34: a rumination signal in the thinking.

DeepSeek-R1 Thoughtology (Mila/McGill, arXiv 2504.07128) measures rumination - a model going back to approaches it has
already tried - by the repetition rate of word 5-grams; correct thoughts are on average shorter than wrong ones.  The
loop guard (serve/loop_guard.py) catches exact blocks repeated back to back; this catches the near-repetition below
that: the same phrases coming back with other words between them.

    RuminationMeter   one request's meter.  It reads the thinking text as it streams, splits it into words (runs of
                      letters and digits, lower case) and keeps the word 5-grams of the last WINDOW words in a
                      counter, so a word costs O(1).  Every EVERY words (once MIN_WORDS words were read) it takes the
                      rate: the share of the window's 5-grams that already occurred earlier in the window,
                      (grams - distinct grams) / grams.  0 is text that never repeats a 5-gram; 1 is impossible
                      (the first copy of each gram counts as new).
                      With a threshold (the config's "rumination_close"): the rate at or above it for CHECKS checks
                      in a row make it `high`; the server then closes the thinking (once) the way the soft budget does
                      (reasoning_budget_grace: at the first sentence or line end, the wrap-up and the model's own
                      end-of-thinking).

The paper gives no threshold (its budget figures are only in plots), so the close is opt-in and its threshold must be
measured on our own turn logs first (the per-request max and mean rates are logged without it).
"""
from __future__ import annotations

import re
from collections import Counter, deque

N = 5               # word n-gram length (Thoughtology)
WINDOW = 256        # the rolling window, in words
EVERY = 32          # a check every this many words
MIN_WORDS = 128     # no check before this many words of thinking (a short window's rate is noise)
CHECKS = 3          # checks in a row at or above the threshold before the close fires
LONG_WORD = 200     # a "word" longer than this (a digit string with no break) is counted as one at this length

_WORD = re.compile(r"\w+")


def close_setting(value) -> dict | None:
    """The config's "rumination_close": a threshold (0 < t <= 1), or {"threshold": t, "checks": K, "min_tokens": M}
    (M: thinking tokens before the close may fire, default 0).  None, 0 or false: off.  Raises ValueError otherwise."""
    if value is None or value is False or value == 0:
        return None
    if isinstance(value, bool):
        raise ValueError(f"\"rumination_close\" must be a number in (0, 1] or an object, not {value!r}")
    if isinstance(value, (int, float)):
        value = {"threshold": value}
    if not isinstance(value, dict):
        raise ValueError(f"\"rumination_close\" must be a number in (0, 1] or an object, not {value!r}")
    unknown = set(value) - {"threshold", "checks", "min_tokens"}
    if unknown:
        raise ValueError(f"\"rumination_close\" has unknown keys: {', '.join(sorted(unknown))}")
    t, k, m = value.get("threshold"), value.get("checks", CHECKS), value.get("min_tokens", 0)
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not 0 < t <= 1:
        raise ValueError(f"\"rumination_close\" threshold must be a number in (0, 1], not {t!r}")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"\"rumination_close\" checks must be a whole number >= 1, not {k!r}")
    if isinstance(m, bool) or not isinstance(m, int) or m < 0:
        raise ValueError(f"\"rumination_close\" min_tokens must be a whole number >= 0, not {m!r}")
    return {"threshold": float(t), "checks": k, "min_tokens": m}


class RuminationMeter:
    def __init__(self, threshold: float | None = None, checks: int = CHECKS, n: int = N, window: int = WINDOW,
                 every: int = EVERY, min_words: int = MIN_WORDS):
        self.threshold, self.need = threshold, checks
        self.n, self.every, self.min_words = n, every, max(min_words, n)
        self.grams: deque = deque()                 # the window's grams, oldest first
        self.cap = max(1, window - n + 1)           # (grams in a window of `window` words)
        self.count: Counter = Counter()
        self.last: deque = deque(maxlen=n)          # the last n words
        self.part = ""                              # a word not finished yet (the text may end inside one)
        self.words = 0
        self.checks, self.rate_sum, self.rate_max, self.rate = 0, 0.0, 0.0, None
        self.streak = 0                             # checks in a row at or above the threshold
        self.closed = False                         # the server closed the thinking on it

    @property
    def high(self) -> bool:
        """The rate has been at or above the threshold for `checks` checks in a row (never without a threshold)."""
        return self.threshold is not None and self.streak >= self.need

    def _word(self, w: str) -> bool:
        self.words += 1
        self.last.append(w)
        if len(self.last) == self.n:
            g = tuple(self.last)
            self.grams.append(g)
            self.count[g] += 1
            if len(self.grams) > self.cap:
                old = self.grams.popleft()
                c = self.count[old] - 1
                if c:
                    self.count[old] = c
                else:
                    del self.count[old]
        if self.words >= self.min_words and self.words % self.every == 0:
            self._check()
            return True
        return False

    def _check(self) -> None:
        total = len(self.grams)
        rate = (total - len(self.count)) / total if total else 0.0
        self.rate = rate
        self.checks += 1
        self.rate_sum += rate
        self.rate_max = max(self.rate_max, rate)
        if self.threshold is not None:
            self.streak = self.streak + 1 if rate >= self.threshold else 0

    def feed(self, text: str) -> bool:
        """Reads more thinking text.  True when it took a check (`rate`, `high` are new)."""
        if not text:
            return False
        s = self.part + text
        found = [m.group() for m in _WORD.finditer(s)]
        if found and _WORD.match(s[-1]):
            self.part = found.pop()                 # (it may go on in the next piece)
            if len(self.part) > LONG_WORD:
                found.append(self.part[:LONG_WORD])
                self.part = ""
        else:
            self.part = ""
        checked = False
        for w in found:
            checked = self._word(w.lower()) or checked
        return checked

    def summary(self) -> dict | None:
        """{"max", "mean", "checks", "closed"}, or None when no check was taken (thinking under MIN_WORDS words)."""
        if not self.checks:
            return None
        return {"max": round(self.rate_max, 3), "mean": round(self.rate_sum / self.checks, 3),
                "checks": self.checks, "closed": self.closed}
