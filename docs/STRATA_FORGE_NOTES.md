# strata-forge: what it is and what it gave us

Notes from reading `github.com/gemovicnemanja/strata-forge` (version 0.3.0, Alpha, Apache-2.0) on 2026-11-09/10. This is
static reading of the code and git history by an agent: nothing in it was built, run or tested, and not every module was
read (`agents/tools` and `rag` only in part). Treat each point as unconfirmed.

**What it is.** An async Python toolkit for LLM work (inference through LiteLLM, evals, fine-tuning, RAG, remote compute).
About 27k lines of source and 40k lines of tests, 19 architecture notes, one author, 257 commits between 2026-05-13 and
2026-10-31. It has no inference engine of its own: for serving it passes `tensor_parallel_size`, `max_model_len` and extra
arguments to vLLM, SGLang or TGI (`compute/serving.py`).

**Link to this project.** None in the code: a search for avalon, basalt, faast and pancake found nothing.

**What we could use, and the result.**
- Throughput timed from the first streamed chunk, so dataset loading and engine warm-up stay out of the number
  (`pipelines/inference_runner.py`, around lines 343-413). the warm-up request in `tools/pancake/abtest.py` already sends a warm-up request
  that is not counted, so nothing was ported for this. `tools/pancake/bench_win.py` times decode from the first token to
  the last for the same reason.
- Nothing in it speeds up decode: the tokens per second in this project come from kernels, the expert cache and
  scheduling, which strata-forge does not touch.

**Points about strata-forge itself** (for anyone who wants to use it): `llm/cache.py:215` calls `pickle.loads` on cached
data; `agents/tools/calculator.py:87` uses `eval` behind an AST allow-list; the `fetch_url` and `fs_read` agent tools were
not audited for SSRF or path traversal; GitHub Actions are pinned to tags, not commit hashes; the Dockerfile uses
`uv:0.5`; there is no dependabot or `pip-audit`; it needs Python 3.14 or newer.
