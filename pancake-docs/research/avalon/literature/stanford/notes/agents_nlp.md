# Stanford LLM/NLP papers (Nov 2022 - Nov 2026) for running a local coding agent (Avalon + opencode)

Setup assumed: llama.cpp "Avalon" serving Qwen3.8-Flash-Next (~125B MoE reasoning model, effort low/medium/xhigh) on an RX 7900 XT 20 GB + 57 GB RAM, so MoE experts are partly offloaded and prefill is the expensive part. The agent is opencode: prompts grow past 131k, it makes many tool calls, it compacts context, and it runs at temperature 0.1. Small Qwen3.8 2B/4B models run on the iGPU/CPU.

How to read this file:
- Stanford role: **LEAD** means the first or senior author is at Stanford, or the work is a Stanford lab project. **CO** means Stanford co-authors on a work led elsewhere.
- Verification: a source marked "[verified this session]" was checked by web search during this research. A plain arXiv link comes from my background knowledge: the ID and headline claims are well known, but the exact numbers were not re-fetched.
- Scope: GPU kernels, quantization/MoE and speculative decoding/serving are excluded, per the brief.

---

## Q1. Programming and optimizing LM pipelines (DSPy, MIPRO, GEPA, TextGrad): useful for tuning our agent prompts?

### Takeaway
Yes, but only for **bounded sub-tasks with a metric**, not the whole opencode system prompt. These sub-tasks are the compaction/summary prompt, the commit-message and title helpers, tool-selection instructions, and the small-model router.
- GEPA (reflective, trace-reading, sample-efficient) is the best fit for our expensive local model. Use Avalon itself as the reflection LM and run the optimization offline overnight.
- BetterTogether, which combines prompt optimization with fine-tuning, is the path for the 2B/4B helper models.

### Cited Findings
- **DSP: Demonstrate-Search-Predict** (Khattab, Santhanam, Li, Hall, Liang, Potts, Zaharia). **LEAD**, Stanford NLP. arXiv Dec 2022. This was the precursor of DSPy: retrieval and LM calls are composed as a program, and the program bootstraps its own demonstrations. — [arXiv 2212.14024](https://arxiv.org/abs/2212.14024)
- **DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines** (Khattab et al.). **LEAD**, Stanford NLP. ICLR 2024 (arXiv Oct 2023).
  - Idea: pipelines are written as modules with typed "signatures". A compiler/optimizer then bootstraps demonstrations and instructions against a metric.
  - Results: the abstract reports compiled pipelines beating standard few-shot prompting by more than 25% (GPT-3.5) and 65% (Llama2-13b-chat).
  - Code: github.com/stanfordnlp/dspy. — [arXiv 2310.03714](https://arxiv.org/abs/2310.03714)
- **DSPy Assertions** (Singhvi, Shetty, Tan, Potts, Sen, Zaharia, Khattab). **LEAD**. arXiv Dec 2023. `Assert`/`Suggest` constraints trigger retry with feedback inside LM pipelines. This maps naturally to validating tool-call JSON or output formats. — [arXiv 2312.13382](https://arxiv.org/abs/2312.13382)
- **MIPRO: Optimizing Instructions and Demonstrations for Multi-Stage LM Programs** (Opsahl-Ong, Ryan, ..., Khattab). **LEAD**, Stanford. EMNLP 2024. [verified this session]
  - Idea: optimizes the instructions and few-shot demonstrations of every module without module-level labels. It uses data/program-aware instruction proposals, a minibatch surrogate (Bayesian) model and meta-optimization.
  - Results: beats baseline optimizers on 5 of 7 programs, by up to 13% accuracy, with Llama-3-8B.
  - Code: shipped as `dspy.MIPROv2`. — [arXiv 2406.11695](https://arxiv.org/abs/2406.11695)
- **Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together ("BetterTogether")** (Soylu, Potts, Khattab). **LEAD**, Stanford. EMNLP 2024. [verified this session]
  - Idea: alternates prompt optimization with BootstrapFinetune, so the LM "teaches itself".
  - Results: up to 60% better than weight-only and 6% better than prompt-only optimization (v2 abstract; v1 said 5%). Tested on 7-8B models: Mistral-7B, Llama-2-7B, Llama-3-8B.
  - Code: in DSPy. — [arXiv 2407.10930](https://arxiv.org/abs/2407.10930), [ACL Anthology](https://aclanthology.org/2024.emnlp-main.597)
- **GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning** (Agrawal et al.). **CO**: the lead is UC Berkeley, with Stanford co-authors (Potts) and Khattab (now MIT). ICLR 2026 oral. arXiv Jul 2025. [verified this session; numbers from secondary sources]
  - Idea: an LLM reads full execution traces, diagnoses failures in natural language and proposes targeted prompt edits. It keeps a Pareto frontier of candidates and can merge them.
  - Results (secondary sources): roughly 10-20% over GRPO with up to 35x fewer rollouts, and more than 10% over MIPROv2.
  - Code: `dspy.GEPA` and the standalone `gepa` package. It needs a separate reflection LM. — [Emergent Mind summary](https://www.emergentmind.com/papers/2507.19457), [DSPy GEPA tutorial](https://dspy.ai/tutorials/gepa_ai_program/), [ICLR 2026 notes](https://en.papernotes.org/ICLR2026/interpretability/gepa_reflective_prompt_evolution_can_outperform_reinforcement_learning/)
- **TextGrad: Automatic "Differentiation" via Text** (Yuksekgonul, Bianchi, Boen, Liu, Huang, Guestrin, Zou). **LEAD**, Stanford (Zou group). arXiv Jun 2024. Published in Nature, March 2025. [verified this session]
  - Idea: an LLM's textual feedback is backpropagated as a "gradient" through a graph of LLM calls, optimizing text variables (prompts, code, solutions).
  - Code: github.com/zou-group/textgrad, with a PyTorch-like API. — [arXiv 2406.07496](https://arxiv.org/html/2406.07496v1), [Stanford DBDS news](https://dbds.stanford.edu/zou-groups-textgrad-method-for-self-improving-generative-ai-is-published-in-nature)
- **Self-Taught Optimizer (STOP): Recursively Self-Improving Code Generation** (Zelikman, Lorch, Mackey, Kalai). **CO-LEAD**: Stanford first author, Microsoft Research. arXiv Oct 2023. An LM-written "improver" scaffold improves programs, including itself. Mostly of conceptual interest to us. — [arXiv 2310.02304](https://arxiv.org/abs/2310.02304)
- **Meta-Prompting** (Suzgun, Kalai). **CO-LEAD**: Stanford first author, OpenAI. arXiv Jan 2024. One LM acts as a conductor that dispatches sub-queries to "expert" instances of itself, including a Python tool. — [arXiv 2401.12954](https://arxiv.org/abs/2401.12954)
- **Archon: An Architecture Search Framework for Inference-Time Techniques** (Saad-Falcon et al.). **LEAD**, Stanford (Hazy/Scaling Intelligence). arXiv Sep 2024. Searches over combinations of ensembling, ranking, fusion and verification layers for a compute budget. — [arXiv 2409.15254](https://arxiv.org/abs/2409.15254)

### Inferences
- **Applicability to opencode:**
  - Optimizing the whole multi-tool coding-agent prompt end-to-end is impractical locally. Each rollout is a multi-minute agent episode, and there is no cheap metric.
  - Optimizing **sub-prompts** is cheap and valuable:
    - (a) The compaction/summarization prompt. Metric: can a fresh agent continue the task from the summary? Or an LLM-judge rubric on retained file paths, decisions and TODOs.
    - (b) Short helper prompts on the 2B/4B models: session titles, "is this tool output relevant", commit messages.
    - (c) Tool-description wording, scored by tool-call validity rate.
  - GEPA's sample efficiency (tens to hundreds of rollouts) matters because Avalon decode is slow. Set it up as an offline DSPy job: Avalon (effort medium) as the reflection LM, the target model as the task LM.
- **BetterTogether** is directly applicable to Qwen3.8-2B/4B helpers. Bootstrap traces from Avalon, LoRA-tune the small model, then re-optimize its prompt. Expected result: helper tasks are moved off the big model, which frees Avalon slots and avoids KV-cache churn.
- **DSPy Assertions-style validate-and-retry** is a harness-level idea for opencode: on malformed tool-call JSON, retry with the error appended. It is cheap and improves turn reliability at temperature 0.1.
- TextGrad is better suited to per-instance refinement (e.g., iteratively improving one patch) than to durable prompt tuning. It adds LLM calls, which is costly locally. It is lower priority than GEPA.

### Gaps
- Primary GEPA numbers were not re-fetched from the arXiv PDF this session. The 10-20% / 35x / >10% figures come from secondary sources.
- I found no Stanford paper that applies DSPy/GEPA specifically to a multi-turn coding-agent system prompt (SWE-bench-style), so benefits for opencode's main prompt are unproven.
- Multi-module GRPO in DSPy (mmGRPO / "Arbor", 2025) was not verified this session.

---

## Q2. Context management and long-context use: keeping prompts short and cache-friendly

### Takeaway
The Stanford evidence points the same way on two things:
- **Placement matters.** Models use information at the start and end of a long context best ("Lost in the Middle").
- **Naive iterative summarization destroys detail** ("context collapse" in ACE).

The actionable recipe for opencode:
- Keep a stable, append-only prefix: system prompt, tool schemas and an "agent playbook". This maximizes llama.cpp prompt-cache reuse.
- Put the current task and fresh tool results at the end.
- Compact with **itemized, incremental delta updates** instead of whole-history rewrites.
- Offload bulk reading (large files and logs) to small local models (Minions pattern).

### Cited Findings
- **Lost in the Middle: How Language Models Use Long Contexts** (Liu, Lin, Hewitt, Paranjape, Bevilacqua, Petroni, Liang). **LEAD**, Stanford NLP. TACL 2024 (arXiv Jul 2023).
  - Finding: on multi-document QA and key-value retrieval, performance is U-shaped in the position of the relevant information. It is highest at the beginning or end and degrades sharply in the middle, even for long-context models.
  - Code/data: github.com/nelson-liu/lost-in-the-middle. — [arXiv 2307.03172](https://arxiv.org/abs/2307.03172)
- **Agentic Context Engineering (ACE): Evolving Contexts for Self-Improving Language Models** (Qizheng Zhang et al.). **LEAD**: Stanford with SambaNova and UC Berkeley. ICLR 2026. arXiv Oct 2025 (v3 Mar 2026). [verified this session]
  - Idea: the context is an itemized "playbook" maintained by Generator / Reflector / Curator roles.
  - Updates are incremental "delta" entries rather than full rewrites. This avoids two failure modes the paper names: "brevity bias" and "context collapse", where repeated LLM rewriting erodes detail.
  - Results: +10.6% on agent benchmarks (AppWorld) and +8.6% on finance, with lower adaptation latency and rollout cost. No labels are needed because it learns from execution feedback.
  - Code: github.com/ace-agent/ace. — [arXiv 2510.04618](https://arxiv.org/html/2510.04618v1), [InfoQ](https://infoq.com/news/2025/10/agentic-context-eng)
- **Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory** (Suzgun, Yuksekgonul, Bianchi, Jurafsky, Zou). **LEAD**, Stanford. EACL 2026 (arXiv Apr 2025). [verified this session]
  - Idea: a self-curated, persistent memory of short reusable strategies and code snippets (not transcripts) is carried across queries.
  - Results: Claude 3.5 Sonnet's AIME accuracy more than doubled. GPT-4o on Game of 24 went from ~10% to 99% by reusing a Python solution. Claude gained ~9% on GPQA-Diamond.
  - Code: github.com/suzgunmirac/dynamic-cheatsheet. ACE builds on it. — [arXiv 2504.07952](https://arxiv.org/abs/2504.07952v1), [ACL Anthology](https://preview.aclanthology.org/credits/2026.eacl-long.333)
- **Minions: Cost-efficient Collaboration Between On-device and Cloud Language Models** (Narayan, Biderman, Eyuboglu, ..., Ré). **LEAD**, Stanford Hazy Research. ICML 2025 (arXiv Feb 2025). [verified this session]
  - Idea, in two protocols:
    - "Minion": a big model chats with a small local model that reads the long context.
    - "MinionS": the big model decomposes the task into subtasks, which small models run in parallel over chunks.
  - Results: Minion cuts remote cost 30.4x and keeps 87% of quality. MinionS cuts it 5.7x and keeps 97.9%.
  - Code: github.com/HazyResearch/minions. — [arXiv 2502.15964](https://arxiv.org/html/2502.15964v1), [Hazy blog](https://hazyresearch.stanford.edu/blog/2025-02-24-minions)
- **Cartridges: Lightweight and general-purpose long context representations via self-study** (Eyuboglu, Ehrlich, Arora, et al.). **LEAD**, Stanford Hazy and Scaling Intelligence. ICLR 2026 (arXiv Jun 2025). [verified this session]
  - Idea: train a small KV cache offline per corpus, using synthetic "self-study" conversations and context distillation, then load it at inference.
  - Results: matches in-context learning quality with 38.6x less memory and 26.4x higher throughput. Effective context extends from 128k to 484k on MTOB. Cartridges are composable.
  - Code: github.com/HazyResearch/cartridges. — [arXiv 2506.06266](https://arxiv.org/pdf/2506.06266), [Hazy blog](https://hazyresearch.stanford.edu/blog/2025-06-08-cartridges)
- **RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval** (Sarthi, Abdullah, Tuli, Khanna, Goldie, Manning). **LEAD**, Stanford. ICLR 2024 (arXiv Jan 2024). Documents are recursively clustered and summarized into a tree, and retrieval happens at multiple abstraction levels. The paper reports +20% absolute on QuALITY with GPT-4. — [arXiv 2401.18059](https://arxiv.org/abs/2401.18059)
- **Generative Agents: Interactive Simulacra of Human Behavior** (Park, O'Brien, Cai, Morris, Liang, Bernstein). **LEAD**, Stanford HCI/NLP. UIST 2023. The origin of the "memory stream + retrieval by recency/importance/relevance + periodic reflection" agent-memory design. — [arXiv 2304.03442](https://arxiv.org/abs/2304.03442)
- **SUPO**: RL fine-tuning of agents for long-horizon multi-turn tool use, with summarization trained end-to-end to go beyond a fixed context. **CO**: ByteDance Seed lead, with Stanford and CMU. ACL 2026. [found this session; details only partially verified] — [ACL Anthology PDF](https://aclanthology.org/2026.acl-long.966.pdf)

### Inferences
- **Prompt layout for llama.cpp cache reuse.** llama.cpp reuses the KV prefix only up to the first changed token.
  - Keep the system prompt, tool schemas and any ACE/Dynamic-Cheatsheet playbook byte-stable and at the very top.
  - Never inject timestamps or changing state into that prefix.
  - Append new turns at the end.
  - Lost in the Middle suggests restating the current goal and constraints at the tail, e.g., a short "current task / next step" block in the last user message.
  - Expected benefit: much smaller re-prefill per turn. On our offloaded MoE, prefill of 100k+ tokens dominates turn latency.
- **Compaction design (ACE lessons):**
  - Avoid opencode-style "rewrite the whole history into one summary" repeated many times, since it compounds detail loss.
  - Prefer a structured, itemized summary (files touched, decisions, failing tests, TODOs, exact error strings) that is updated by delta.
  - Compact rarely and at fixed points, so the cache stays valid between compactions.
  - The summary prompt itself is an ideal GEPA target (Q1).
- **Minions pattern with local helpers.** The 4B model on the iGPU/CPU can read large tool outputs (long logs, big files, grep results) and return extracts to Avalon. Avalon's context then grows more slowly.
  - Same principle as the paper (big model plans, small model reads chunks), but the "cost" saved here is Avalon prefill time, not dollars.
  - Expect quality loss on fine-grained code reasoning. Use it for logs and search output, not for code to be edited.
- **Persistent playbook (Dynamic Cheatsheet/ACE).** Keep a project-level "lessons learned" file loaded into the stable prefix: build commands, repo quirks, recurring tool errors. It grows by curated deltas between sessions, which gives cross-session learning without fine-tuning.
- **Cartridges** needs per-corpus training (GPU hours) and a runtime that loads a trained KV cache. llama.cpp has no off-the-shelf support. Not practical now, but conceptually relevant for a fixed large repo.

### Gaps
- I found no Stanford-led paper specifically on **coding-agent context compaction** (2025-2026). The compaction papers found (CliffCompaction, AutoCompact, CompactionRL, Parallel Context Compaction) are not confirmed to be Stanford and were excluded.
- No Stanford paper found that quantifies Lost-in-the-Middle effects on 2025-2026 reasoning models at 128k+ tokens. Newer models may be less affected.

---

## Q3. Reasoning and test-time compute (STaR/Quiet-STaR, s1, length control): implications for choosing reasoning effort

### Takeaway
Stanford work shows that test-time compute buys accuracy, but with clear trade-offs:
- **Budget forcing (s1):** extending thinking helps up to a point and then saturates.
- **Repeated sampling (Monkeys):** sampling many times plus a verifier gives large gains, but only when a verifier such as unit tests exists.
- **Adaptive length penalties:** short thinking on easy prompts loses little accuracy.

For a coding agent whose turns are mostly "read output, choose next tool":
- Use **low** effort for routine tool-call turns.
- Use **medium/xhigh** only for planning, debugging hard failures or final patch design.
- Spend extra compute on test-verified retries rather than longer thinking.

### Cited Findings
- **s1: Simple test-time scaling** (Muennighoff, Yang, Shi, Li, Fei-Fei, Hajishirzi, Zettlemoyer, Liang, Candès, Hashimoto). **LEAD**, Stanford (with UW and AI2). EMNLP 2025 (arXiv Jan 2025). [verified this session]
  - Data: SFT of Qwen2.5-32B-Instruct on 1,000 curated reasoning traces (s1K).
  - "Budget forcing": thinking is cut off at a token budget, or extended by appending "Wait".
  - Results: s1-32B beats o1-preview on competition math by up to 27%. Budget forcing raises AIME24 from 50% to 57%.
  - Code: github.com/simplescaling/s1. — [arXiv 2501.19393](https://arxiv.org/abs/2501.19393), [ACL Anthology](https://preview.aclanthology.org/credits/2025.emnlp-main.1025)
- **Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking** (Zelikman, Harik, Shao, Jayasiri, Haber, Goodman). **LEAD**, Stanford. COLM 2024 (arXiv Mar 2024).
  - Idea: generalizes STaR (Zelikman et al. 2022, before this window) to learn token-level rationales on unstructured text.
  - Results: zero-shot gains for Mistral-7B, GSM8K 5.9% to 10.9% and CommonsenseQA 36.3% to 47.2%. — [arXiv 2403.09629](https://arxiv.org/abs/2403.09629)
- **Cognitive Behaviors that Enable Self-Improving Reasoners** (Gandhi, Chakravarthy, Singh, Lile, Goodman). **LEAD**, Stanford. arXiv Mar 2025.
  - Finding: RL self-improvement depends on the base model already showing verification, backtracking, subgoal-setting and backward chaining. Qwen exhibits these and improves, while Llama does not until primed with such behaviors. — [arXiv 2503.01307](https://arxiv.org/abs/2503.01307)
- **Large Language Monkeys: Scaling Inference Compute with Repeated Sampling** (Brown, Juravsky, Ehrlich, Clark, Le, Ré, Mirhoseini). **LEAD**, Stanford. arXiv Jul 2024.
  - Finding: coverage scales roughly log-linearly with the number of samples.
  - Example: on SWE-bench Lite, DeepSeek-Coder-V2 goes from 15.9% (1 sample) to 56% (250 samples).
  - Caveat: gains depend on a verifier. Without one, majority voting and reward models plateau. — [arXiv 2407.21787](https://arxiv.org/abs/2407.21787)
- **CodeMonkeys: Scaling Test-Time Compute for Software Engineering** (Ehrlich, Brown, Juravsky, Clark, Mirhoseini). **LEAD**, Stanford. arXiv Jan 2025. The model iteratively writes edits together with its own test scripts, then selects among candidates using model-generated tests. It reports 57.4% on SWE-bench Verified with Claude 3.5 Sonnet. Code and trajectories are released. — [arXiv 2501.14723](https://arxiv.org/abs/2501.14723)
- **Just Enough Thinking: Efficient Reasoning with Adaptive Length Penalties RL (ALP)** (Stanford authors). **LEAD**, Stanford [affiliation per search snippet]. arXiv Jun 2025. An RL length penalty scaled inversely to the per-prompt solve rate, so the model learns to think briefly on easy prompts and long on hard ones. — [arXiv 2506.05256](https://arxiv.org/html/2506.05256v1)
- **Self-Taught Reasoner lineage / Archon.** See also Archon (Q1) for composing sampling, ranking and verification under a budget. — [arXiv 2409.15254](https://arxiv.org/abs/2409.15254)

### Inferences
- **Reasoning-effort policy for Avalon.** Use the routine split from the Takeaway: low for routine tool turns, medium/xhigh for hard steps. Escalate only when a turn fails, i.e., a tool error, failing tests or a repeated identical action.
  - This is the inference-time analogue of ALP's "think proportionally to difficulty". Most opencode turns are "easy" (read result, emit next call), so low effort should dominate.
  - Expected benefit: large cuts in decode tokens per turn on slow MoE decode, with little loss.
- **s1 budget forcing as a server/harness knob.** A hard cap on thinking tokens per effort level stops runaway thoughts that inflate latency and context. Whether llama.cpp exposes a reasoning budget depends on the version (check `--reasoning-budget`). "Wait" extension is only worth trying for one-shot hard problems, not routine turns.
- **Monkeys/CodeMonkeys.** When a test suite exists, k parallel low-effort attempts with test-based selection can beat one xhigh attempt. On one 20 GB GPU, though, parallel sampling competes for the same hardware. Sequential retry-on-failure is the practical version.
- **Cognitive Behaviors** offers a cheap prompt lever: explicitly asking for "verify the result of the last tool call before proceeding" may help. This is an inference; the paper studies training.

### Gaps
- I found no Stanford paper evaluating reasoning-effort levels (low/medium/high) specifically on agentic coding turns. The policy above is extrapolated.
- ALP's numeric results (token reduction, accuracy) were not verified this session.

---

## Q4. Alignment and fine-tuning methods usable locally for small helper models

### Takeaway
The DPO family (DPO, then KTO) and Alpaca-style distillation are all Stanford-originated, simple and runnable on 2B/4B models with LoRA. KTO is especially convenient because it only needs thumbs-up/down labels, which we can harvest automatically from agent logs (e.g., the tool call succeeded or failed). ReFT is an even lighter alternative to LoRA. For agentic skills, SWE-smith- and AgentFlow-style trajectory training shows small models can gain a lot. Training should happen on the 7900 XT when Avalon is idle.

### Cited Findings
- **Alpaca: A Strong, Replicable Instruction-Following Model** (Taori, Gulrajani, Zhang, Dubois, Li, Guestrin, Liang, Hashimoto). **LEAD**, Stanford CRFM. Blog, Mar 2023. LLaMA-7B was fine-tuned on 52K self-instruct demonstrations generated by text-davinci-003, for under $600. This is the template for "distill a big model into a small one". — [CRFM blog](https://crfm.stanford.edu/2023/03/13/alpaca.html)
- **AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback** (Dubois, Li, Taori, Zhang, Gulrajani, Ba, Guestrin, Liang, Hashimoto). **LEAD**, Stanford. NeurIPS 2023. Uses LLM-simulated preference annotators (about 50x cheaper than crowdworkers) to develop RLHF methods. — [arXiv 2305.14387](https://arxiv.org/abs/2305.14387)
- **Direct Preference Optimization (DPO)** (Rafailov, Sharma, Mitchell, Ermon, Manning, Finn). **LEAD**, Stanford. NeurIPS 2023. RLHF is replaced by a closed-form classification loss on preference pairs, with no reward model and no PPO. Implementations: TRL, axolotl, etc. — [arXiv 2305.18290](https://arxiv.org/abs/2305.18290)
- **From r to Q\*: Your Language Model is Secretly a Q-Function** (Rafailov, Hejna, Park, Finn). **LEAD**, Stanford. COLM 2024. Interprets DPO as token-level credit assignment, which is relevant to multi-step/agent settings. — [arXiv 2404.12358](https://arxiv.org/abs/2404.12358)
- **KTO: Model Alignment as Prospect Theoretic Optimization** (Ethayarajh, Xu, Muennighoff, Jurafsky, Kiela). **LEAD**, Stanford and Contextual AI. ICML 2024. A human-aware loss that needs only binary desirable/undesirable labels, not pairs. It matches or exceeds DPO at 1B-30B scale. — [arXiv 2402.01306](https://arxiv.org/abs/2402.01306)
- **ReFT: Representation Finetuning for Language Models** (Wu, Arora, Wang, Geiger, Jurafsky, Manning, Potts). **LEAD**, Stanford. NeurIPS 2024. Trains low-rank interventions on hidden representations, with the weights frozen. Reported as 15x-65x more parameter-efficient than LoRA at competitive quality. Library: github.com/stanfordnlp/pyreft. — [arXiv 2404.03592](https://arxiv.org/abs/2404.03592)
- **BetterTogether** (see Q1): prompt optimization alternated with fine-tuning gives up to 60% over fine-tuning alone on 7-8B models. — [arXiv 2407.10930](https://arxiv.org/abs/2407.10930)
- **SWE-smith: Scaling Data for Software Engineering Agents** (John Yang et al., with Diyi Yang, Ofir Press, Carlos Jimenez). **LEAD**, Stanford with Princeton [Stanford affiliation from background knowledge; not confirmed in this session's snippets]. NeurIPS 2025 (arXiv Apr 2025). [verified this session]
  - Idea: synthesizes bug-fix task instances by breaking tests in 128 Python repos, giving 50k instances.
  - Results: SWE-agent-LM-32B reaches 40.2% Pass@1 on SWE-bench Verified, open-source SOTA at the time.
  - Code: data, trajectories and models at swesmith.com. — [arXiv 2504.21798](https://arxiv.org/abs/2504.21798)
- **AgentFlow: In-the-Flow Agentic System Optimization for Effective Planning and Tool Use** (Z. Li, ..., Pan Lu, James Zou, Yejin Choi, et al.). **CO-LEAD**: Lu and Zou are at Stanford [affiliation from background knowledge]. NeurIPS 2025 / ICLR 2026 oral (arXiv Oct 2025). [verified this session]
  - Idea: four modules (planner, executor, verifier, generator) share memory. Only the planner is trained, with Flow-GRPO, which broadcasts the trajectory outcome to every turn.
  - Results: a 7B (Qwen2.5-7B) system gains on average +14.9% on search, +14.0% agentic, +14.5% math and +4.1% science, and beats GPT-4o. — [arXiv 2510.05592](https://arxiv.org/html/2510.05592v1), [ICLR 2026](https://iclr.cc/virtual/2026/poster/10009931)
- **OctoTools** (Pan Lu et al.). **LEAD**, Stanford. arXiv Feb 2025. A training-free agentic framework built on standardized "tool cards", a planner and an executor. — [arXiv 2502.11271](https://arxiv.org/abs/2502.11271)

### Inferences
- **Concrete local recipe for the 2B/4B helpers:**
  1. Log Avalon's outputs for helper-type tasks: summaries, titles, extraction of relevant lines from tool output.
  2. Alpaca-style SFT (LoRA or ReFT) of Qwen3.8-4B on those logs.
  3. KTO on automatically labeled outcomes, e.g., whether the summary let the agent continue, or whether a generated command ran with exit 0.

  Expected benefit: the helpers become reliable enough that Avalon can offload work to them (Minions/Q2) and skip small requests that churn its KV cache.
- **Fine-tuning the 125B MoE itself is not feasible** on this hardware. All tuning targets small models.
- The **SWE-smith** dataset and trajectories could be used to SFT a small model as an "executor" sub-agent. AgentFlow suggests training only the planner of a modular system. Both are research-grade efforts with uncertain payoff for us.
- Use **DPO** when pairs exist naturally, e.g., a failed vs. successful retry of the same tool step.

### Gaps
- No Stanford paper found specifically on fine-tuning **2B-4B** models as coding-agent helpers (summarizers/routers). Results above are at 7B-32B.
- AMD/ROCm training support for ReFT/KTO toolchains was not researched (out of scope).

---

## Q5. Agent and coding evaluation from Stanford to measure our setup

### Takeaway
- **Terminal-Bench** (Stanford and Laude Institute) is the most directly relevant: CLI tasks in containers, close to what opencode does, run with the Harbor harness.
- **HELM** provides the methodology: multi-metric, standardized, including efficiency.
- **Cybench** and **MLAgentBench** are secondary agent benchmarks.
- **Intelligence-per-Watt** offers a local-efficiency metric and a profiling harness.
- **AlpacaEval LC** is a cheap judge-based check for helper chat quality.

### Cited Findings
- **Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces** (Merrill et al.). **LEAD**, Stanford and Laude Institute. arXiv Jan 2026. [verified this session]
  - Version 2.0 has 89 human-verified tasks in Docker containers. Frontier agents score under 65% per the paper. Version 2.1 (2026) fixes 28 tasks and adds continuous validation.
  - Harness: Harbor (hub.harborframework.com).
  - Leaderboard numbers vary by version and scaffold, so treat them cautiously. — [arXiv 2601.11868](https://arxiv.org/abs/2601.11868v1), [Snorkel blog](https://www.snorkel.ai/blog/terminal-bench-2-0/), [Harbor hub](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1/6)
- **HELM: Holistic Evaluation of Language Models** (Liang, Bommasani, Lee, et al.). **LEAD**, Stanford CRFM. arXiv Nov 2022, TMLR 2023. Covers many scenarios and 7 metrics (accuracy, calibration, robustness, fairness, bias, toxicity, efficiency). The framework (crfm-helm) supports local/OpenAI-compatible endpoints. — [arXiv 2211.09110](https://arxiv.org/abs/2211.09110)
- **AlpacaEval 2 Length-Controlled** (Dubois, Galambosi, Liang, Hashimoto). **LEAD**, Stanford. arXiv Apr 2024. A regression-based length debiasing of the LLM-judge win rate. It raises Spearman correlation with Chatbot Arena to 0.98. — [arXiv 2404.04475](https://arxiv.org/abs/2404.04475)
- **Cybench: A Framework for Evaluating Cybersecurity Capabilities and Risks of LMs** (A. Zhang et al.). **LEAD**, Stanford. ICLR 2025. 40 professional CTF tasks with subtasks, a terminal-agent setting. — [arXiv 2408.08926](https://arxiv.org/abs/2408.08926)
- **MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation** (Huang, Vora, Liang, Leskovec). **LEAD**, Stanford. ICML 2024. Agents perform ML research tasks via file and shell actions. — [arXiv 2310.03302](https://arxiv.org/abs/2310.03302)
- **Intelligence per Watt: Measuring Intelligence Efficiency of Local AI** (Saad-Falcon et al., Hazy Research). **LEAD**, Stanford. arXiv Nov 2025. [verified this session]
  - Scope: 1M+ queries, 20+ models, 8 accelerators, batch size 1.
  - Findings:
    - Local models answer 88.7% of single-turn chat/reasoning queries.
    - IPW improved 5.3x between 2023 and 2025.
    - Local accelerators are at least 1.4x less efficient than cloud accelerators on the same model.
  - Harness: github.com/HazyResearch/intelligence-per-watt. Single-turn only, not agentic. — [arXiv 2511.07885](https://arxiv.org/pdf/2511.07885), [Hazy blog](https://hazyresearch.stanford.edu/blog/2025-11-11-ipw)
- **SWE-smith** (Q4) also ships SWE-bench-style environments usable for evaluation. — [arXiv 2504.21798](https://arxiv.org/abs/2504.21798)

### Inferences
- **Recommended measurement plan:**
  - (1) Pick a fixed subset of Terminal-Bench 2.x, e.g., 15-25 tasks that fit our latency budget. Run opencode against Avalon at each reasoning effort, and record pass rate, wall time per task, tokens and number of compactions. This directly measures "faster and more reliable agent turns".
  - (2) Add a cheap per-turn reliability metric: the tool-call parse/validity rate from opencode logs.
  - (3) Optionally run HELM or IPW's harness for single-turn efficiency (accuracy per joule / per second) when comparing server configs.
- A HELM-style multi-metric report (accuracy plus latency plus tokens) is the right framing: changes that raise accuracy but double turn time may be a net loss for interactive use.
- Cybench and MLAgentBench are less aligned with everyday coding but useful as stress tests of long-horizon terminal use.

### Gaps
- Not checked whether Terminal-Bench/Harbor supports opencode as an agent adapter out of the box. It does support custom agents via its harness, but this was not verified this session.
- No Stanford benchmark found that specifically measures **context-compaction quality** or multi-turn degradation in coding agents.
- HELM Capabilities / HELM Agents (2025) leaderboards were not researched in detail this session.
