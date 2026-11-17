"""serve/prompt_normalize.py - STRATA_PROMPT_NL_NORMALIZE=1 (opt-in, default off; LEADS.md Part P lead L2).

hipfire (warpfront/hipfire, Apache-2.0; AGENTS.md "Prompt-shape adaptation") collapses every run of 3 or more
newlines to 2 before tokenizing.  On the Qwen vocabulary "\n\n\n" is one rare BPE token and "\n\n" a hot one, and
a drafter conditioned on the rare token accepted far less (their number: +14..27% decode tok/s on PEP-8 style code
prompts, DFlash, 27B dense, RX 7900 XTX; theirs, not ours).  This is our own implementation of the idea, no code
copied: a pure function over the rendered prompt.

It is LOSSY with respect to the prompt: the model reads a file with its blank lines squeezed, so a client that
matches the model's quote against the file byte for byte (an edit tool) can fail, and a prompt-lookup copy of
the file text no longer equals the file.  Default off for that reason; the A/B (bench/specs) decides.  The prompt
cache key follows the normalized text, so a request is consistent with itself across turns.
"""
from __future__ import annotations

import os
import re

_RUN = re.compile(r"\n{3,}")


def normalize_prompt_newlines(text: str) -> str:
    """Every run of 3 or more '\n' becomes exactly 2.  Idempotent; text without such a run is returned as is."""
    if "\n\n\n" not in text:
        return text
    return _RUN.sub("\n\n", text)


def prompt_nl_normalize_enabled() -> bool:
    return os.environ.get("STRATA_PROMPT_NL_NORMALIZE", "").strip() == "1"
