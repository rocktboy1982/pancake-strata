# Runtime intelligence and test-time-compute-aware LLM runtimes (as of Nov 2026)

Scope: checking the claims in the pasted AI-written overview, and how serving runtimes adapt to reasoning models that spend more compute at inference time. The notes follow five layers: survey, techniques, mechanisms, implementations/code, applicability.

Verification legend: **VERIFIED** = primary source confirms. **PARTIALLY ACCURATE** = the item exists but the overview's wording or numbers are off. **NOT FOUND** = no reliable source located.

## Q1. "Exo Architecture (arXiv:2611.00463)": does it exist, and what does it claim?

### Takeaway
**VERIFIED, with one numeric caveat.** arXiv:2611.00463 is "Exo: An Architecture for Self-Improving Systems at Runtime" (Krentsel, Ratnasamy, Stoica, Casado, Goyal; submitted 30 Oct 2026). It is an agent runtime that folds policy optimization into task execution. A "minimal protected mechanism layer" preserves execution history and governs when policies activate. "Up to 53% lower cost" is the top of one range only (standalone game runs, 12-53%). Other settings report 45-82% and 35-61%. Exo is an agent-harness/runtime paper, not an LLM inference engine. It is unrelated to exo-explore/exo.

### Cited Findings
- Title: "Exo: An Architecture for Self-Improving Systems at Runtime". Authors: Alexander Krentsel, Sylvia Ratnasamy, Ion Stoica, Martin Casado, Ankur Goyal. Category cs.AI. v1 submitted 30 Oct 2026. — [arXiv:2611.00463](https://arxiv.org/abs/2611.00463)
- Problem framing: hand-designing the policies that govern LM agents is hard, and automated alternatives need representative datasets and costly offline runs. Exo merges optimization into task execution, so agents revise their own harness and strategies while they work. — [arXiv:2611.00463](https://arxiv.org/abs/2611.00463)
- Abstract wording: "A minimal protected mechanism layer preserves execution history and governs policy activation." The harness changes through inspectable code edits. The abstract says "preserves" history. It does not explicitly say the layer actively "monitors" it, so the overview's "monitoring execution history" slightly overstates the wording. — [arXiv:2611.00463](https://arxiv.org/abs/2611.00463)
- Reported results:
  - Open-world game: model cost 12-53% lower than Codex and Pi in standalone runs, 45-82% lower in transfer runs, with similar task completion.
  - AppWorld and SREGym: held-out model cost 35-61% lower than the same starting policy without editing, with comparable correctness. Mean task time on AppWorld fell 13%. Median diagnosis time on SREGym fell 35%.
  - Continual Learning Bench coding tasks: total cost, learning included, fell 14%.
  — [arXiv:2611.00463](https://arxiv.org/abs/2611.00463)
- Secondary coverage (talks/podcast; not primary): describes a protected stateful "exo harness" holding the event log, secrets, memory artifacts and environment snapshots; a stateless "executor" for policy/logic; and a sandbox. A guardian process rolls back failed self-modifications to the previous snapshot. A related co-authored title, "Exoharness: An Agent Harness Architecture for Fully-Recursive Self-Improvement", is also mentioned. — [startuphub.ai](https://www.startuphub.ai/news/alex-krentsel-on-exo-and-recursive-self-improving-ai-agents), [themodelwire](https://themodelwire.com/article/exo-enables-agents-to-modify-their-own-code-and-policies-recursively-01M032NMZBPCSFD6M9N467YNZ2), [summify](https://summify.io/discover/exo-harnesses-should-see-their-own-code-and-logs-alex-krents-5lFD-3)
- **Separate project, do not confuse: exo-explore/exo** (EXO Labs) is a distributed inference engine that pools home devices, mainly Apple Silicon Macs, into one cluster. Third-party write-ups say exo 1.0 added RDMA over Thunderbolt 5 on macOS 26.2, claiming up to 99% lower inter-device latency (about 1-2 µs vs 100+ µs over standard networking). Hardware requirements differ between sources (M4 Pro/Max vs also M3 Ultra). RDMA is enabled with `rdma_ctl enable` in recovery mode. One analysis argues it adds capacity, not speed, and says NVIDIA support is a community fork. A university report noted transport/execution issues in tensor mode with RDMA. — [noqta.tn](https://noqta.tn/en/blog/exo-distributed-ai-cluster-apple-silicon-local-llm-2026), [dev.to](https://dev.to/jovan_chan_9500711396d4e6/exo-framework-in-2026-can-you-pool-rtx-3090s-to-beat-a-dgx-spark-the-honest-distributed-inference-5ccm), [UMFT report](https://www.umft.ro/wp-content/uploads/2026/02/Project-Report_Bio-AI-Synergy_FDI-2025.pdf)

### Inferences
- In the paper, "runtime" means agent runtime: the harness, policies and tools around the LLM. It is not a token-serving engine. Its savings come from fewer or cheaper model calls through better policies, not from faster decoding. It belongs in an "agent runtime" layer above the inference engine.
- Ion Stoica is a co-author of both Exo and Certaindex/Dynasor (below). This links the Berkeley/Sky lineage to both ideas, but they remain different layers.

### Gaps
- I could not confirm whether "Exoharness" is a separate arXiv paper or an earlier title of 2611.00463.
- I read only the abstract page, not the full paper (code release, models used, hardware).
- exo 1.0 release date and benchmarks come from non-official sources. I did not read the official GitHub release notes.

## Q2. The term "runtime intelligence"

### Takeaway
**NOT FOUND as an established technical term.** No paper, product or standard uses "runtime intelligence" for LLM serving. The field's terms are "test-time compute", "inference-time scaling", and "reasoning-aware serving". The phrase is probably the overview's own label.

### Cited Findings
- A search for the exact phrase together with LLM inference/test-time compute returned only general test-time-compute explainers, plus "The Art of Scaling Test-Time Compute for Large Language Models" (arXiv 2512.02008, Dec 2025). That paper gives a recipe for choosing a TTC strategy by problem difficulty, model type and budget. — [arXiv 2512.02008](https://arxiv.org/pdf/2512.02008)
- A 2025 survey notes that models spend fixed inference compute regardless of difficulty, overthinking easy problems and underthinking hard ones. This motivates adaptive budgets. — [arXiv 2507.02076](https://arxiv.org/html/2507.02076v1)

### Gaps
- If the overview tied "runtime intelligence" to a specific vendor or paper, that source is unidentified.

## Q3. Reasoning-aware serving: techniques and mechanisms

### Takeaway
Six families of techniques adapt runtimes to long-thinking models:
1. Certainty-based early exit of reasoning (Certaindex/Dynasor), used as a scheduler signal.
2. Hard thinking-token budgets enforced by logit processors (vLLM `thinking_token_budget`, SGLang ThinkingBudgetLogitProcessor, llama.cpp `--reasoning-budget`).
3. Training-time length control (ThinkPrune) and models with native budget support (Qwen3 thinking-mode fusion).
4. Reasoning parsers that split `reasoning` from `content`.
5. Step-level speculative reasoning (SpecReason) on top of token-level speculative decoding (EAGLE-3).
6. Decode-time KV compression for very long generations (R-KV, ThinKV, LazyEviction).

### Cited Findings

**Certaindex / Dynasor (Hao AI Lab, UCSD + Berkeley)**
- Paper: "Efficiently Scaling LLM Reasoning with Certaindex" (Fu, Chen, Zhu, …, Stoica, Hao Zhang). arXiv:2412.20993, Dec 2024, revised May 2025, NeurIPS 2025.
  - Certaindex is an algorithm-agnostic measure of how stable intermediate answers are. It signals when more compute is unlikely to change the final answer.
  - Dynasor is a reasoning-aware serving system that uses Certaindex as a scheduler.
  - Reported: up to 50% compute savings and up to 3.3x throughput, with "no accuracy drop".
  — [arXiv:2412.20993](https://arxiv.org/abs/2412.20993)
- Dynasor-CoT mechanism (blog, 16 Feb 2025):
  - Every 32/64/128 tokens, append a probe ("…Final Answer: \boxed{") to pull out the current answer.
  - Discard answers followed by hesitation words ("wait", "hmm").
  - Exit once answers agree across several consecutive probes.
  — [Hao AI Lab blog](https://haoailab.com/blogs/dynasor-cot/)
- Dynasor-CoT savings:
  - Up to 29% of tokens dataset-wide, with no training or model change.
  - Up to 81% on single MATH500 problems; 34% (AIME) and 53% (MATH500) on the top-10% problems.
  - On full DeepSeek-R1: 12% (AIME) and 24% (AMC).
  - Accuracy comparable on AMC23/AIME24/MATH500.
  - Models: R1-Distill-Qwen 7B/14B/32B and R1.
  — [Hao AI Lab blog](https://haoailab.com/blogs/dynasor-cot/)
- Code and integrations:
  - Packages: `dynasor-chat` (CLI), `dynasor-openai` (OpenAI-compatible proxy in front of vLLM), `dynasor-vllm` (built into vLLM's server). The README warns to enable prefix caching or probing will be slow.
  - Integrated into NVIDIA TensorRT-LLM (Apr 2025) and Snowflake Arctic Inference (May 2025). MIT license.
  - The README does not mention SGLang or Ollama.
  — [GitHub hao-ai-lab/Dynasor](https://github.com/hao-ai-lab/Dynasor)

**Thinking-budget control in serving engines**
- vLLM:
  - `--reasoning-parser` with parsers `deepseek_r1`, `qwen3`, `glm45`, `gemma4`, `granite…`, `minimax_m2…` (about 19 families). Output field is `reasoning` (renamed from `reasoning_content`; old clients may silently read empty).
  - `thinking_token_budget` sampling parameter: counting starts at `reasoning_start_str`, and vLLM forces `reasoning_end_str` when the budget is reached. The end string can include a transition phrase so the cutoff reads naturally. Configured via `--reasoning-config`.
  - `reasoning_effort` maps to `enable_thinking`. `--reasoning-effort-map` exists. Anthropic-style `thinking.type` (disabled/enabled/adaptive) is mapped onto these controls.
  - Structured output: when reasoning is present, the grammar engine skips the reasoning section. Tool calls are parsed only from `content`.
  — [vLLM docs: Reasoning Outputs](https://docs.vllm.ai/en/latest/features/reasoning_outputs.html)
- SGLang:
  - Launch with `--enable-custom-logit-processor` and `--reasoning-parser`. Pass a serialized ThinkingBudgetLogitProcessor and a budget per request. Works on `/generate` and chat completions.
  - Example: `DeepSeekR1ThinkingBudgetLogitProcessor` with `thinking_budget=512`.
  - Other models need a subclass with their think-start/end token IDs.
  - Source is a docs mirror, so check names against the repo.
  — [SGLang docs mirror: thinking_budget](https://lmsysorg.mintlify.app/docs/advanced_features/thinking_budget.md)
- DeepSeek-R1 thought length cannot be controlled reliably by prompting alone, which supports enforcement at the logit level. — [DeepSeek-R1 Thoughtology, arXiv 2504.07128](https://arxiv.org/pdf/2504.07128)
- llama.cpp:
  - PR #13771 added `--reasoning-budget 0` to disable thinking.
  - Later builds added a reasoning-budget sampler with a hard cap that forces the end of thinking.
  - Builds after 10 Apr 2026 (commit d7ff074) register think tags for Gemma 4, which triggered the sampler without a flag and caused hangs. Fix #21870 skips the sampler when no budget is requested.
  - The per-request field was renamed from `reasoning_budget` to `reasoning` in tests.
  - Sources are GitHub issues and community posts.
  — [forge issue #54](https://github.com/antoinezambelli/forge/issues/54), [PR 13771 diff](https://app.semanticdiff.com/gh/ggml-org/llama.cpp/pull/13771/overview), [reasoning-budget.h mirror](https://huggingface.co/datasets/echodict/llama.cpp/blob/main/common/reasoning-budget.h), [ramalama #2137](https://github.com/containers/ramalama/issues/2137)

**Model-side budget support**
- Qwen3 technical report (arXiv 2505.09388, May 2025):
  - `/think` and `/no_think` flags; thinking is on by default.
  - The thinking budget is not explicitly trained. It emerges from "Thinking Mode Fusion": the model can answer from incomplete thinking.
  - A larger budget gives consistent gains on math/code/STEM (Qwen3-235B-A22B, Fig. 2).
  - Caveat: in thinking mode, long-context performance slightly degrades.
  — [Qwen3 Technical Report](https://arxiv.org/pdf/2505.09388)
- In vLLM, DeepSeek-V3.1 has thinking disabled by default and Qwen3 has it enabled. Both toggle via `chat_template_kwargs`. — [vLLM docs](https://docs.vllm.ai/en/latest/features/reasoning_outputs.html)
- ThinkPrune (Hou et al., UCSB/MIT-IBM; arXiv 2504.01296, Apr 2025; TMLR 2026):
  - RL with a token limit; outputs over the limit get zero reward. An iterative variant tightens the limit each round.
  - R1-Distill-Qwen-1.5B: average length 10,355 → 3,574 tokens, with slightly higher accuracy on four math benchmarks.
  - It is a training method, not a runtime knob.
  — [arXiv 2504.01296](https://arxiv.org/abs/2504.01296), [TMLR 2026 listing](https://mlanthology.org/tmlr/2026/hou2026tmlr-thinkprune)

**Speculative decoding for reasoning**
- SpecReason (arXiv 2504.07891; NeurIPS 2025):
  - A small model drafts each reasoning step. The large model scores the step and regenerates it if rejected.
  - This exploits the approximation tolerance of thinking tokens: semantic agreement is enough, token-level equality is not required.
  - v2 reports 1.4-3.0x speedup over vanilla LRM inference and +0.4-9.0% accuracy. Combined with token-level speculative decoding, it adds 8.8-58.0% over speculative decoding alone (v1 numbers differ: 1.5-2.5x, +1.0-9.9%).
  — [arXiv 2504.07891](https://arxiv.org/abs/2504.07891)
- SpecGuard (arXiv 2604.15244, 2026): step-level verification from model-internal signals instead of an external PRM (contrasted with RSD, which uses PRMs). — [arXiv 2604.15244](https://arxiv.org/pdf/2604.15244.pdf)
- Adaptive drafting:
  - AdaEAGLE predicts draft length from hidden states, reporting 1.62x over autoregressive decoding.
  - HC-Spec (ACL 2026) reports about 15-30% more speedup than EAGLE-3.
  - SGLang docs recommend EAGLE-3, and adaptive speculative decoding when acceptance varies across requests.
  — [AdaEAGLE arXiv 2412.18910](https://arxiv.org/pdf/2412.18910), [HC-Spec ACL 2026](https://aclanthology.org/2026.acl-long.353.pdf), [SGLang speculative decoding docs](https://docs.sglang.io/advanced_features/speculative_decoding.html)

**KV management for very long generations**
- R-KV (arXiv 2505.24133; NeurIPS 2025; Microsoft Research co-authors):
  - Compresses during decoding. Scores tokens on attention importance and on redundancy (key-vector similarity), then selects jointly.
  - Motivation: reasoning models produce 8-14x more tokens, with 5-7x higher word repetition.
  - Version conflict:
    - arXiv v1: 105% of full-KV accuracy at 16% of the cache, 90% memory saving, 6.6x throughput.
    - NeurIPS version: 105% at 38% of the cache, 50% memory saving, 2x speedup.
  - Code: github.com/Zefan-Cai/R-KV.
  — [arXiv 2505.24133](https://arxiv.org/pdf/2505.24133), [NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/57e0358aed314784cad14a26e1eba642-Abstract-Conference.html)
- ThinKV (arXiv 2510.01290, Oct 2025):
  - Classifies "thoughts" by attention sparsity, then applies hybrid quantization plus progressive eviction.
  - Near-lossless with under 5% of the KV cache on R1-Distill, GPT-OSS and AceReason. Up to 5.8x throughput is a secondary-summary figure.
  — [arXiv 2510.01290](https://arxiv.org/abs/2510.01290v1)
- LazyEviction (arXiv 2506.15969): keeps tokens whose attention recurs after a delay; 50-70% KV reduction at comparable accuracy. — [arXiv 2506.15969](https://arxiv.org/abs/2506.15969v3)
- "Hold Onto That Thought" (arXiv 2512.12008; NeurIPS 2025) benchmark:
  - H2O and a decoding-enabled SnapKV do best on reasoning models.
  - Aggressive eviction at low budgets can make reasoning traces longer, which cancels out the savings.
  — [arXiv 2512.12008](https://arxiv.org/abs/2512.12008)

**Test-time-compute frameworks (best-of-N, tree search)**
- Hugging Face `search-and-learn`:
  - Best-of-N, beam search and DVTS guided by a PRM. The cookbook uses Llama-3.2-1B-Instruct as generator and the RLHFlow Llama3.1-8B PRM.
  - Example with N=8: Best-of-N took 3.54 s and beam search 10.06 s.
  - HF claimed Llama 3B beats 70B on MATH with PRM-guided search. Critics question fairness, because of the 8B verifier and weak baselines.
  — [HF cookbook](https://huggingface.co/learn/cookbook/search_and_learn), [Reddit thread](https://cal1.lr.ggtyler.dev/r/LocalLLaMA/comments/1hfw14v/outperforming_llama_70b_with_llama_3b_on_hard)
- Foundational result: compute-optimal test-time scaling can beat scaling parameters. — [Snell et al., arXiv 2408.03314](https://arxiv.org/pdf/2408.03314)
- A*-decoding: Llama-3.2-1B matches Llama-3.1-70B on MATH500/AIME24 with token-efficient search. — [arXiv 2505.13672](https://arxiv.org/html/2505.13672v1)
- Certaindex is algorithm-agnostic, so the same stopping signal applies to self-consistency, best-of-N and MCTS-style programs. That is how Dynasor's scheduler covers all of them. — [arXiv:2412.20993](https://arxiv.org/abs/2412.20993)

### Inferences
- Layer map for the report:
  - Survey: TTC scaling (Snell; Art of Scaling TTC).
  - Techniques: early exit, budgets, length-RL, speculative reasoning, KV compression, search.
  - Mechanisms: answer-consistency probes, forced `</think>` via logit processor, step-level accept/reject, redundancy-aware eviction.
  - Implementations: vLLM/SGLang/llama.cpp flags, Dynasor packages, R-KV repo, TRT-LLM/Arctic integrations.
  - Applicability: see Q5.
- A budget cap is blunt: it truncates regardless of progress. Certaindex-style exit is adaptive. They combine well, with the cap as a safety ceiling and the probe for early exit.
- Production engines (vLLM/SGLang/llama.cpp) have standardized on hard budgets and parsers. Adaptive early exit and KV compression for reasoning remain mostly in research code or vendor forks (TRT-LLM, Arctic), not upstream vLLM/SGLang defaults. This is an inference from the docs read, not a verified exhaustive check.

### Gaps
- I found no upstream vLLM/SGLang doc showing Certaindex or Dynasor merged as a native scheduler.
- The DeepSeek-V3.1/R1 model card's own thinking-budget guidance was not read.
- Hardware details for Dynasor/SpecReason were not in the pages read. The papers presumably used datacenter NVIDIA GPUs (unverified).

## Q4. Measured gains summary (cost / latency / accuracy / hardware)

### Takeaway
Typical reported gains:
- 20-50% token or compute reduction from early exit/budgets, at roughly equal accuracy.
- 1.4-3x latency speedup from step-level speculation.
- 2-6x throughput, or a 5-10x smaller KV cache, from reasoning-specific KV compression.
- 12-82% agent model-cost reduction from Exo-style policy self-editing (a different layer).

Several papers changed their headline numbers between versions, so cite the version explicitly.

### Cited Findings
- Certaindex/Dynasor: up to 50% compute saved, up to 3.3x throughput, no accuracy drop. — [arXiv:2412.20993](https://arxiv.org/abs/2412.20993)
- Dynasor-CoT: up to 29% fewer tokens, comparable accuracy. — [blog](https://haoailab.com/blogs/dynasor-cot/)
- ThinKV: under 5% of the KV cache, near-lossless. — [arXiv 2510.01290](https://arxiv.org/abs/2510.01290v1)

The remaining figures are listed under Q1 and Q3: Exo, ThinkPrune, SpecReason v2, R-KV (v1 vs NeurIPS) and HC-Spec.

### Gaps
- No source gave consistent per-hardware numbers. Consumer-GPU measurements (e.g. RTX 3060, RX 7900 XT) for these techniques were not found.

## Q5. Applicability to a single-user local engine

### Takeaway
For one user on a consumer GPU, batching-oriented scheduler gains (Dynasor throughput, best-of-N) matter little. Four things matter:
1. A hard thinking budget plus a natural transition string. Available today in llama.cpp, vLLM and SGLang.
2. Certaindex-style probe early exit, implementable as a client-side proxy. It needs prefix caching to stay cheap.
3. Speculative decoding with adaptive draft length during long reasoning traces.
4. Reasoning-aware KV compression when context or VRAM is tight. This is mostly research code.

### Cited Findings
- The Dynasor proxy works with a single vLLM server at TP=1 (example: R1-Distill-Qwen-7B). Probing is slow without prefix caching. — [Dynasor GitHub](https://github.com/hao-ai-lab/Dynasor)
- In llama.cpp, `--reasoning-budget N` gives a hard cap (0 = disable). There is a known 2026 regression where the budget sampler hung on Gemma 4 when no flag was passed; fixed by #21870. — [forge #54](https://github.com/antoinezambelli/forge/issues/54)
- vLLM `thinking_token_budget` plus a `reasoning_end_str` with a transition phrase lets truncated thinking still produce a coherent answer. — [vLLM docs](https://docs.vllm.ai/en/latest/features/reasoning_outputs.html)
- Qwen3 tolerates truncated thinking by design (thinking-mode fusion), so budget truncation is safer on Qwen3 than on models without this property. — [Qwen3 report](https://arxiv.org/pdf/2505.09388)
- Aggressive KV eviction can lengthen traces and cancel the savings, so use moderate budgets. — [arXiv 2512.12008](https://arxiv.org/abs/2512.12008)
- SpecReason-style step speculation needs a second small model in VRAM. It is complementary to token-level speculative decoding. — [arXiv 2504.07891](https://arxiv.org/abs/2504.07891)
- SGLang recommends adaptive speculative decoding when acceptance varies. Reasoning traces are repetitive, which favours high acceptance. — [SGLang docs](https://docs.sglang.io/advanced_features/speculative_decoding.html)

### Inferences
- Cheapest win for a local engine: per-request thinking budget with a sensible default (e.g. 2-4k tokens for chat, uncapped for hard math/code). Implementation: monitor the `<think>` span, inject `</think>` plus a transition phrase at the cap.
- Next win: a Dynasor-CoT-like probe every 64-128 tokens on a forked sequence that shares the KV prefix. Exit after 2-3 consistent answers when the answer has a checkable form (math/boxed, multiple choice). Little value for open-ended chat.
- Exo-style ideas fit the agent layer (opencode-like harnesses) rather than the inference engine.

### Gaps
- No measured results for early exit or budgets on consumer AMD/NVIDIA GPUs in llama.cpp or ExLlama were found.
- ExLlamaV3/TabbyAPI: a targeted search found no documented thinking budget or think-tag budget control. TabbyAPI is described only as the FastAPI/OpenAI-compatible server for ExLlamaV3 ([TabbyAPI README](https://raw.githubusercontent.com/theroyallab/tabbyAPI/HEAD/README.md), [wiki Usage](https://github.com/theroyallab/tabbyAPI/wiki/03.-Usage)). The source code was not checked. Treat it as unsupported until confirmed. A client/proxy-side budget would work there instead.
- Related operational pitfalls:
  - A vLLM forum report says `thinking_token_budget` was silently ignored when passed via `extra_args` in vLLM 0.18.0, so pass it as a top-level sampling parameter ([vLLM forum](https://discuss.vllm.ai/t/thinking-token-budget-silently-ignored-when-passed-via-extra-args-in-vllm-0-18-0/2533)).
  - The llama.cpp budget sampler runs as a state machine: count down, force the end-of-think sequence, then pass through ([reasoning-budget.h mirror](https://huggingface.co/datasets/echodict/llama.cpp/blob/main/common/reasoning-budget.h)). Community reports say a "budget exceeded" message ending in a newline helps the model stop thinking cleanly (anecdotal).
