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

A second signal (A20, opt-in, config "rumination_entropy": K): the entropy of the engine's top-K distribution at every
thinking token (EntropyMeter; the engine prints the K best logprobs when asked, see emit_lp_line).  A model that
ruminates is usually sure of itself (low entropy) while one that explores is not; "rumination_close" may carry
"entropy_below": H, and the close then needs the rate high AND the rolling mean entropy at or below H.  The engine's
logprobs path copies the logits of every token to the host and sorts the vocabulary, so it costs decode speed: it is not
free, it stays off by default and the per-token cost must be measured before it is left on.

The paper gives no threshold (its budget figures are only in plots), so the close is opt-in and its threshold must be
measured on our own turn logs first (the per-request max and mean rates are logged without it).
"""
from __future__ import annotations

import math
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
    unknown = set(value) - {"threshold", "checks", "min_tokens", "entropy_below"}
    if unknown:
        raise ValueError(f"\"rumination_close\" has unknown keys: {', '.join(sorted(unknown))}")
    t, k, m = value.get("threshold"), value.get("checks", CHECKS), value.get("min_tokens", 0)
    h = value.get("entropy_below")
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not 0 < t <= 1:
        raise ValueError(f"\"rumination_close\" threshold must be a number in (0, 1], not {t!r}")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"\"rumination_close\" checks must be a whole number >= 1, not {k!r}")
    if isinstance(m, bool) or not isinstance(m, int) or m < 0:
        raise ValueError(f"\"rumination_close\" min_tokens must be a whole number >= 0, not {m!r}")
    if h is not None and (isinstance(h, bool) or not isinstance(h, (int, float)) or not h > 0):
        raise ValueError(f"\"rumination_close\" entropy_below must be a number > 0 (nats), not {h!r}")
    out = {"threshold": float(t), "checks": k, "min_tokens": m}
    if h is not None:
        out["entropy_below"] = float(h)     # A20: the second signal (needs "rumination_entropy")
    return out


ENTROPY_WINDOW = 64     # tokens in the rolling mean of the top-K entropy


def entropy_setting(value) -> int | None:
    """The config's "rumination_entropy": K, the number of top logprobs the engine prints per token (2..20).  None, 0 or
    false: off.  Raises ValueError otherwise."""
    if value is None or value is False or value == 0:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 2 <= value <= 20:
        raise ValueError(f"\"rumination_entropy\" must be a whole number 2..20 (top-K), not {value!r}")
    return value


def lp_entropy(line: str) -> float | None:
    """The entropy in nats of the top-K distribution on an engine `LP <logprob> [id:logprob ...]` line, renormalised over
    the K listed ids (the head logprob is the chosen token's own; it is used alone when the list is empty).  None for a
    line that cannot be read (`LP nan`, a cut line)."""
    f = line.split()
    if len(f) < 2 or f[0] != "LP":
        return None
    try:
        vals = [float(x.partition(":")[2]) for x in f[2:]]
        head = float(f[1])
    except ValueError:
        return None
    if not vals:
        vals = [head]
    if not all(math.isfinite(v) for v in vals):
        return None
    mx = max(vals)
    w = [math.exp(v - mx) for v in vals]
    z = sum(w)
    return max(0.0, -sum((x / z) * math.log(x / z) for x in w if x > 0))


class EntropyMeter:
    """One request's top-K entropy of the thinking tokens: the mean over all of them, the lowest rolling mean (a window of
    ENTROPY_WINDOW tokens, once one is full) and the current rolling mean.  `low(h)`: the window is full and its mean is at
    or below h (the second signal of the rumination close)."""

    def __init__(self, window: int = ENTROPY_WINDOW):
        self.window = window
        self.recent: deque = deque(maxlen=window)
        self.total, self.tokens = 0.0, 0
        self.min_window = None

    def add(self, h: float) -> None:
        self.recent.append(h)
        self.total += h
        self.tokens += 1
        if len(self.recent) == self.window:
            m = self.rolling()
            if self.min_window is None or m < self.min_window:
                self.min_window = m

    def feed_line(self, line: str) -> bool:
        h = lp_entropy(line)
        if h is None:
            return False
        self.add(h)
        return True

    def rolling(self) -> float | None:
        return sum(self.recent) / len(self.recent) if self.recent else None

    def low(self, h: float) -> bool:
        return len(self.recent) == self.window and self.rolling() <= h

    def summary(self) -> dict | None:
        """{"mean", "min_window", "tokens"} (nats), or None when no thinking token had a line."""
        if not self.tokens:
            return None
        return {"mean": round(self.total / self.tokens, 3),
                "min_window": round(self.min_window, 3) if self.min_window is not None else None,
                "tokens": self.tokens}


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
