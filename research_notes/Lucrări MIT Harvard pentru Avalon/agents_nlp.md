# MIT and Harvard LLM/NLP papers for a local coding agent (Nov 2022 - Nov 2026)

Scope: reasoning efficiency, context and memory, tool use and agent reliability, test-time training, fine-tuning small helpers, evaluation. GPU kernels, quantization/MoE and speculative decoding/serving are excluded. Papers already in `/home/dan/reports/Lucrări Stanford utile pentru Avalon.md` were skipped. None of those were MIT- or Harvard-led, apart from "Scaling Laws for Precision" (Harvard lead), which is quantization and so out of scope.

Target setup: Avalon/pancake serving Qwen3.8-Flash-Next (~125B MoE, effort low/medium/xhigh) on an RX 7900 XT 20 GB with 57 GB RAM, driven by opencode (prompts of 131k+, many tool calls, compaction, temperature 0.1). Qwen3.8 2B/4B models are available as helpers. The server already has thinking budgets, tool-call recovery, a loop guard, env_to_user prefix stabilisation and a conversation cache.

Lead/co labels: "MIT lead" or "Harvard lead" means the first or last author is at that institution according to the cited source. "Joint" means the attribution is mixed or unconfirmed.

Summary table (details and sources are in the sections below):

| Paper | Lead | Venue/date | Where it applies | Expected benefit |
|---|---|---|---|---|
| Recursive Language Models (2512.24601) | MIT lead (CSAIL) | arXiv Dec 2025, v3 May 2026; NeurIPS 2026 poster per secondary sources | harness (opencode sub-agent / REPL tool), prompts | High for 131k+ repo/log reading; replaces some compaction |
| MEM1 (2506.15841) | Joint MIT/NUS/SMART/Yonsei | ICLR 2026 | compaction design (idea); small-model RL (impractical) | Medium as a design idea: constant-size internal state |
| OS-Pruner (2607.11089) | Joint MIT/Columbia | NeurIPS 2026 poster | server (stopping head) | Medium: 20-60% fewer thinking tokens, but needs a trained head |
| Learning How Hard to Think (2410.04707) | MIT lead | ICLR 2025 | server/harness router for effort level | Medium-high: up to 50% compute saved at equal quality |
| ThinkPrune | Co (MIT CSAIL affiliation) | 2025 | none (needs RL on the big model) | Low |
| Is Self-Repair a Silver Bullet? (2306.09896) | MIT lead (with Microsoft) | ICLR 2024 | prompts/harness policy | Medium: guidance on when retry beats repair |
| CodeSteer (2502.04350) / R1-Code-Interpreter (2505.21668) | MIT/Harvard lead (Yongchao Chen) | 2025; R1-CI at ICLR 2026 | prompts (code-vs-text steering); small helper as steerer | Medium |
| Self-Steering LMs / DisCIPL (2504.07081) | MIT lead (with Yale) | ICLR 2025 workshop | helper pattern: big planner + small 2B/4B followers | Low-medium |
| SEAL (2506.10943) | MIT lead | 2025 | small-model fine-tuning only | Low locally |
| TTT for Abstract Reasoning (2411.07279) | MIT lead | Nov 2024 | small-model only | Low |
| RL's Razor (2509.04259) | MIT lead | ICLR 2026 | helper fine-tuning recipe | Medium for 2B/4B helper training |
| Reasoning with Sampling (2510.14901) | Harvard lead | Oct 2025 | server sampler (MCMC) | Low: multiplies decode cost |
| Echo Chamber (2504.07912) | Harvard lead (Kempner) | COLM 2025 | helper fine-tuning caveat | Low (insight only) |

## Reasoning efficiency: adaptive/early-exit thinking, length control, self-verification

### Takeaway
The most actionable MIT result is input-adaptive compute allocation: a cheap predictor decides per query how much computation to spend, which maps directly onto routing each opencode turn to effort low, medium or xhigh. MIT/Columbia's OS-Pruner gives a principled stop-thinking rule (20-60% fewer tokens), but it needs a small trained head and loses accuracy on hard math. The other MIT/Harvard reasoning papers either need RL on the big model or multiply decode cost.

### Cited Findings
- **Learning How Hard to Think: Input-Adaptive Allocation of LM Computation.** Damani, Shenfeld, Peng, Bobu, Andreas (MIT). arXiv 2410.04707, Oct 2024; ICLR 2025. The method trains a lightweight model to predict the reward distribution for a given input and compute budget, then allocates extra compute where it is predicted to help. It has two variants: adaptive best-of-k, and routing between an expensive and a cheap decoder. Reported results: up to 50% less compute at no quality loss on math and code, and up to 10% less compute on chat. — [ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ff414825df833edb8b1839e3d5d495e9-Abstract-Conference.html); [arXiv](https://arxiv.org/abs/2410.04707v1)
- **OS-Pruner: Pruning Chains-of-Thought of Reasoning Models via Optimal Stopping.** Ehab, El Gadarri, Farias, Jozefiak, Moallemi (MIT and Columbia Business School; the page does not map individual authors to institutions). arXiv 2607.11089, 13 Jul 2026; NeurIPS 2026 poster. At each paragraph boundary, the policy compares stopping now against an estimated continuation value, with utility = accuracy - λ·tokens. On R1-Distill-Qwen-7B / GSM8K it reached 88.0% with 496 thinking tokens, against 88.7% with 1,219 for the baseline: 59.3% fewer tokens for 0.7 points of accuracy. On AIME with DRPO-7B, accuracy fell from 44.4% to 36.9% while tokens fell from 4,359 to 2,006, which the paper calls a sizable task-specific loss. The method is not training-free: it adds a linear head and fine-tunes the last two layers using correctness labels from truncated traces. No code link was found. — [alphaXiv 2607.11089](https://www.alphaxiv.org/abs/2607.11089)
- **ThinkPrune** (MIT CSAIL affiliation listed). It continues R1-style RL under a hard token limit, for example 4,000 tokens for reasoning plus answer. This requires RL training of the reasoning model. — [search summary pointing to arXiv 2503.04697](https://arxiv.org/html/2503.04697v1). The lead affiliation was not verified; the search result only said that MIT CSAIL is listed.
- **Reasoning with Sampling: Your Base Model is Smarter Than You Think.** Karan, Du (Harvard). arXiv 2510.14901, Oct 2025. The method samples from a power distribution p^α over the base model's sequence likelihoods, using an iterative MCMC sampler that resamples token subsequences. It needs no training, data or verifier. On MATH500 it matches GRPO-level accuracy and preserves diversity. — [arXiv HTML](https://arxiv.org/html/2510.14901v1); [project page](https://scour.ing/p/https://aakaran.github.io/reasoning_with_sampling)
- **Self-verification via code (CodeSteer / R1-Code-Interpreter, MIT/Harvard).** The R1-CI paper reports emergent self-checking through code generation (details in the tool-use section). — [arXiv 2505.21668v3](https://arxiv.org/html/2505.21668v3)

### Inferences
- **Learning How Hard to Think → server or harness effort router (high value, cheap).** Train a tiny classifier, or prompt a Qwen3.8 2B helper, on features of the current turn to predict whether xhigh helps: last tool result was an error or failed test, the same action is being repeated, it is the first planning turn, the diff is large. Low effort becomes the default, with escalation to medium or xhigh. The labels can come from our own replayed opencode sessions at the three effort levels. This turns the Stanford report's hand-written escalation heuristic into a learned policy. The ≤50% saving was measured on math and code best-of-k and routing, not on agentic turns, so the transfer is an extrapolation.
- **OS-Pruner → server (medium value, real engineering).** Its "stop at a paragraph boundary when expected gain < λ·cost" rule could replace or augment the fixed thinking budget. A cheap first approximation is to check only at `\n\n` boundaries inside `<think>` and use a training-free proxy, such as answer convergence (the paper's own baseline: 82.6% at 598 tokens, i.e. worse) or repetition. The full method needs a head trained on Qwen3.8-Flash-Next hidden states, which is feasible offline only if we can dump hidden states from Avalon. Fine-tuning its two layers is not feasible at 125B locally, but the linear-head-only variant might be. The AIME result warns against using it on hard debugging turns, so pair it with the effort router (prune only at low and medium).
- **Reasoning with Sampling → not applicable.** MCMC resampling multiplies decode passes, and our bottleneck is decode speed on CPU-offloaded experts.
- **ThinkPrune → not applicable** (needs RL on the 125B model).

### Gaps
- I found no MIT or Harvard paper that evaluates reasoning-effort levels or thinking budgets specifically on multi-turn coding-agent turns.
- The OS-Pruner code release and per-author affiliations were not confirmed.
- The ThinkPrune author list and lead institution were not verified in this session. My recollection is that it is a UCSB lead with MIT-IBM co-authors, which would make it a co-authored paper, but this is unverified.

## Context and memory for agents: recursive LMs, context folding, memory, compaction

### Takeaway
Recursive Language Models (MIT CSAIL) is the most relevant paper in this whole survey. It keeps the long prompt as a variable in a Python REPL and has the model inspect, slice and sub-call itself on pieces, instead of stuffing or compacting context. It beats compaction by a median of 26% and Claude Code by 13% (with GPT-5 as the root model), and it shows that an 8B Qwen3 can be post-trained to do it. MEM1 (MIT-affiliated) supports a constant-size, rewritten internal state over append-only histories.

### Cited Findings
- **Recursive Language Models.** Alex L. Zhang, Tim Kraska, Omar Khattab (all MIT CSAIL). arXiv 2512.24601: v1 31 Dec 2025, v3 11 May 2026. Secondary sources list it as a NeurIPS 2026 poster. — [arXiv abs](https://arxiv.org/abs/2512.24601); [arXiv v1 HTML](https://arxiv.org/html/2512.24601v1); [repo README via search](https://github.com/alexzhang13/rlm)
  - The input prompt lives as a variable in a persistent REPL. The model writes code to inspect and partition it and recursively calls itself or sub-LMs on the parts. It handles inputs up to two orders of magnitude beyond the context window. — [arXiv abs](https://arxiv.org/abs/2512.24601)
  - With GPT-5 as root, the median gains across four long-context tasks are +26% over compaction, +130% over CodeAct with sub-calls, and +13% over Claude Code, at comparable cost. — [arXiv abs](https://arxiv.org/abs/2512.24601)
  - The authors post-trained **RLM-Qwen3-8B**, which is +28.3% on average over base Qwen3-8B and approaches vanilla GPT-5 on three long-context tasks. — [arXiv abs](https://arxiv.org/abs/2512.24601)
  - The paper is motivated by "context rot", the quality decay as inputs grow. It argues that compaction is rarely expressive enough for tasks that need dense access to many parts of the prompt. — [arXiv v1 HTML](https://arxiv.org/html/2512.24601v1)
  - Code: `alexzhang13/rlm`, a plug-and-play inference library supporting several sandboxes, from the MIT OASYS lab. — [GitHub](https://github.com/alexzhang13/rlm); [blog](https://alexzhang13.github.io/blog/2025/rlm/)
- **MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents.** Zijian Zhou, Ao Qu et al. (MIT, NUS, SMART, Yonsei). arXiv 2506.15841; ICLR 2026. RL teaches the agent to keep a compact internal state that is rewritten each turn and serves as both memory and reasoning, instead of appending all past turns. MEM1-7B reaches 3.5x higher performance and 3.7x lower memory than Qwen2.5-14B-Instruct on a 16-objective multi-hop QA task, and it generalizes beyond its training horizon. The code is in the MIT-MI GitHub organization. — [arXiv v2](https://arxiv.org/html/2506.15841v2); [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10008961)
- Related, non-MIT evidence on the same failure mode: General AgentBench (2602.18998) reports that agent test-time scaling is bounded by an "effective context ceiling", beyond which more compute causes instability. Affiliation unknown; listed here only as corroboration. — [arXiv 2602.18998](https://arxiv.org/html/2602.18998v1/)

### Inferences
- **RLM → agent harness (opencode), high value.** For tasks that blow past 131k (whole-repo questions, huge logs, large diffs), add an "RLM tool" or sub-agent. It loads the material into a Python sandbox variable, lets the main model write slicing and grep code, and dispatches sub-queries on chunks to Qwen3.8-Flash-Next at low effort, or to the 4B helper for simple extraction. On our hardware this has two concrete wins:
  1. Prompts stay short, so prefill (which is costly with CPU-offloaded experts) shrinks a lot.
  2. Sub-calls share a stable system prefix, so the conversation cache and prefix cache stay hot.
  The `rlm` library can be pointed at an OpenAI-compatible endpoint (Avalon). This is plausible but not verified here.
- **RLM vs compaction.** The 26% median gain over compaction suggests opencode should prefer "keep the raw history and tool outputs on disk or in a REPL variable, query on demand" over lossy summary rewrites, at least for large tool outputs. This complements the ACE-style delta compaction from the Stanford report: compact the narrative, but externalise the bulky data.
- **RLM-Qwen3-8B → small helper, medium value.** It shows that an 8B model can be post-trained into a competent recursive reader. A Qwen3.8 4B helper could serve as the sub-call worker. Training it ourselves is heavy, so the first step is to try the prompt-only RLM with the 125B model as root.
- **MEM1 → compaction design idea, low-medium.** It validates a fixed-size "state block" (goal, facts, open TODOs, last error) rewritten each turn and kept at the tail of the prompt. The RL training is not practical for the main model. As a prompt pattern it is cheap, but rewriting it every turn is at odds with prefix caching unless it sits after the stable prefix (it should go in the last user message, as env_to_user already does).

### Gaps
- RLM's per-task numbers, cost and latency figures, and its behaviour with open-weight models other than Qwen3-8B were not extracted. The abstract only says "comparable cost".
- No independent replication of RLM was found.
- I found no MIT or Harvard paper specifically on compaction policies for coding agents.
- MEM1's first-author affiliation (NUS/SMART vs MIT) was not resolved, so the lead attribution is "joint".

## Tool use, agent reliability and self-correction

### Takeaway
MIT's self-repair study is the key reliability caveat: self-repair gains are modest once its cost is counted, and they are bottlenecked by the quality of the feedback. Real signals (test output, a stronger reviewer) matter more than "reflect on your code". The MIT/Harvard CodeSteer line shows that deciding when to write and run code versus reason in text is a learnable, steerable decision, and that a small model can do the steering.

### Cited Findings
- **Is Self-Repair a Silver Bullet for Code Generation?** Olausson, Inala, Wang, Gao, Solar-Lezama (MIT lead, with Microsoft Research). arXiv 2306.09896; ICLR 2024. Tested Code Llama, GPT-3.5 and GPT-4 on HumanEval and APPS. Once the cost of repair is accounted for, the gains are often modest, vary across data subsets, and are sometimes absent. Self-repair is bottlenecked by the model's ability to give feedback on its own code. Feedback from a stronger model or from humans yields large gains, even for GPT-4. — [arXiv](https://arxiv.org/abs/2306.09896); [ICLR 2024 proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/9ddc141bdbf9d1db510cefff56c586ad-Abstract-Conference.html)
- **CodeSteer: Symbolic-Augmented Language Models via Code/Text Guidance** (Yongchao Chen et al., MIT/Harvard, 2025). A small fine-tuned steering model tells the main LLM when to use code and when to use text. It raised GPT-4o on 37 SymBench tasks from 53.3 to 86.4, above o1 (82.7) and DeepSeek-R1 (76.8), and the benefit transferred to Claude, Mistral and GPT-3.5. The paper also reports inverse-scaling patterns in models' code-vs-text choices and LLM code degenerating into hard-coded scripts. — [summary via R1-CI paper and search](https://arxiv.org/html/2505.21668v2); [Steering LLMs between Code Execution and Textual Reasoning, 2410.03524](https://fugumt.com/fugumt/paper_check/2410.03524v2_enmode)
- **R1-Code-Interpreter: LLMs Reason with Code via Supervised and Multi-stage RL** (Yongchao Chen, Yueying Liu, Junwei Zhou, Yilun Hao et al., MIT/Harvard). arXiv 2505.21668; ICLR 2026. Multi-turn SFT and RL across 144 reasoning and planning tasks. R1-CI-14B improved from 44.1% to 72.4% on 37 test tasks (v3, Mar 2026), versus 58.6% for GPT-4o text-only and 70.9% for GPT-4o with Code Interpreter. v1 reported 44.0% → 64.1%, so the numbers changed between versions. It shows emergent self-checking through code. Code and models are on GitHub and HF. — [arXiv v3](https://arxiv.org/html/2505.21668v3); [alphaXiv](https://www.alphaxiv.org/abs/2505.21668v3)
- **Self-Steering Language Models (DisCIPL).** Grand, Tenenbaum, Mansinghka, Lew, Andreas (MIT, with Yale). arXiv 2504.07081; ICLR 2025 workshop. A planner LM writes a task-specific inference program (LLaMPPL), and a population of small follower LMs executes it with sequential Monte Carlo. Failed programs feed the traceback back to the planner for up to R retries. With a Llama-3.2-1B follower it matches or beats GPT-4o and o1 on constrained generation. On Puzzles it reaches Pass@1 0.42, against 0.08 for the follower alone and 0.25 for the planner alone, but trails o1's 0.82 (secondary-source figure). — [arXiv](https://arxiv.org/abs/2504.07081); [ICLR page](https://iclr.cc/virtual/2025/36211)
- Non-MIT 2026 context, useful as corroboration only: open-weight agents lose about 70% of clean-context tool-call capability by depth 6 because of their own earlier mistakes ([arXiv 2608.26189](https://arxiv.org/abs/2608.26189)). Agents adopt corrupted tool outputs at high rates (68% for web search), and verification prompts reduce this ([arXiv 2609.05587](https://arxiv.org/abs/2609.05587)). These are not MIT or Harvard papers.

### Inferences
- **Self-repair → prompts/harness policy.** Do not spend xhigh thinking on "re-read and fix your code" loops without new external signal. Escalate effort or retry only when there is fresh evidence: a failing test, a compiler error, or a tool error. This supports the server's loop guard and an escalation policy keyed on real tool feedback. "Stronger feedback helps" suggests that when the main model stalls (loop guard fires), a differently configured reviewer pass with a fresh context (or xhigh effort) helps more than another same-context attempt.
- **CodeSteer → prompts and small helper.** For opencode, the equivalent decision is "run a quick script or test now" versus "reason in text". A system-prompt nudge ("when a claim can be checked by running code or a test, run it") is the cheap version. A Qwen3.8 2B steering helper fine-tuned on our own traces is the expensive version, and is low priority.
- **DisCIPL → helper pattern, low-medium.** Big-planner/small-follower designs fit our hardware: the 125B model is slow to decode, while the 2B/4B models are fast. They are useful for constrained, parallelisable sub-tasks such as generating many candidate grep patterns or formatting. They are not useful for core coding turns.
- The compounding tool-error finding (non-MIT) reinforces the existing tool-call recovery: catching a malformed call early saves more than the single turn.

### Gaps
- I found no MIT- or Harvard-led 2026 paper on coding-agent reliability or tool-call robustness. A generic search returned only non-MIT 2026 work.
- The CodeSteer arXiv ID (2502.04350) is from recollection; I did not directly verify its abstract page. The figures above come from the R1-CI paper and search summaries.
- MIT's grammar-constrained / SMC decoding work (e.g. Loula et al., ICLR 2025), which might bear on tool-call JSON validity, was not checked in this session.

## Test-time training / self-adaptation: SEAL, TTT for reasoning — practical locally?

### Takeaway
Both MIT lines are real and strong on their own benchmarks, but they need gradient updates on the model being adapted. That is impossible for the 125B MoE on 20 GB VRAM. They are only relevant for the 2B/4B helpers, and even there the payoff for a coding agent is unproven. RL's Razor (MIT) is the most useful practical guidance for fine-tuning helpers: prefer on-policy RL or self-generated data to keep KL small and avoid forgetting.

### Cited Findings
- **SEAL: Self-Adapting Language Models.** Zweiger, Pari, Guo, Akyürek, Kim, Agrawal (MIT). arXiv 2506.10943, Jun 2025. The model is RL-trained to write "self-edits", i.e. natural-language specs of fine-tuning data and optionally hyperparameters, which are applied as weight updates. Knowledge incorporation (SQuAD without passage): 33.5% → 47.0%, and the self-generated data beat GPT-4.1-generated data. Simplified ARC with Llama-3.2-1B: 72.5% success, against 20% for untrained self-edits and 0% for ICL. Limitations: catastrophic forgetting, and significant tuning time per edit, which makes real-time use impractical. — [arXiv v2](https://arxiv.org/html/2506.10943v2); [Synced](https://syncedreview.com/2025/06/16/mit-researchers-unveil-seal-a-new-step-towards-self-improving-ai/)
- **The Surprising Effectiveness of Test-Time Training for Abstract Reasoning.** Akyürek et al. (MIT; includes Andreas and Kim). arXiv 2411.07279, Nov 2024. Three components are crucial: initial fine-tuning on similar tasks, auxiliary task format with augmentations, and per-instance training. Up to 6x accuracy over fine-tuned baselines. An 8B model reached 53% on the ARC public validation set, and 61.9% when ensembled with program synthesis (around average human level). — [arXiv](https://arxiv.org/abs/2411.07279v1)
- **RL's Razor: Why Online Reinforcement Learning Forgets Less.** Shenfeld, Pari, Agrawal (MIT Improbable AI Lab). arXiv 2509.04259; ICLR 2026. At matched new-task accuracy, RL forgets less than SFT. The KL between the fine-tuned and base model on the new task predicts forgetting. SFT on an oracle distribution that minimises KL forgets even less, so the advantage comes from implicit KL minimisation. — [arXiv](https://arxiv.org/abs/2509.04259); [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/618c95f4557c15b253fb0e6f548ea0c0-Abstract-Conference.html); [author blog](https://jyopari.github.io/posts/rl_razor)
- **Echo Chamber: RL Post-training Amplifies Behaviors Learned in Pretraining.** Zhao, Meterez, Kakade, Pehlevan, Jelassi, Malach (Harvard / Kempner). arXiv 2504.07912; COLM 2025. RL (PPO, GRPO, Expert Iteration) converges to one dominant output distribution already present in pretraining. RL on easier questions transfers to harder ones. The effect is scale-dependent. — [arXiv](https://arxiv.org/abs/2504.07912)

### Inferences
- **Main model:** SEAL and TTT are not practical. Backpropagating through a ~125B MoE is impossible on this machine. The Stanford report reached the same conclusion for Cartridges.
- **Helpers (2B/4B):** if we fine-tune a helper (compaction summariser, effort router, RLM sub-call worker, code-vs-text steerer), RL's Razor argues for on-policy data: train on samples the helper itself generates, scored by our metric (rejection sampling / expert iteration or GRPO), rather than SFT on 125B-model outputs, so the helper keeps its general abilities. SEAL's finding that self-generated data beat GPT-4.1 data points the same way. Echo Chamber warns that RL will collapse the helper onto its dominant pretraining style, which is acceptable for a narrow helper.
- TTT on the 4B helper per repository (e.g. a LoRA on repo-specific Q&A) is theoretically possible on the 7900 XT, but no MIT or Harvard paper shows it helping coding agents. It is speculative and low priority.

### Gaps
- There is no evidence on TTT or SEAL for code or agentic tasks, nor ROCm-specific cost data.
- The SEAL code repository was not confirmed in this session.

## Benchmarks and evaluation from MIT/Harvard usable for our agent

### Takeaway
I found no MIT- or Harvard-led coding-agent benchmark comparable to Terminal-Bench, which is already covered in the Stanford report. The MIT/Harvard evaluation assets that are usable are the long-context tasks in the RLM paper (to test the RLM tool and the compaction policy) and SymBench from CodeSteer/R1-CI (37 to 144 tasks, for code-vs-text decisions). For agent speed and reliability, Terminal-Bench plus our own replayed opencode sessions remain the main harness.

### Cited Findings
- RLM evaluates on four long-context tasks against compaction, CodeAct and Claude Code baselines, and releases the `rlm` library. — [arXiv](https://arxiv.org/abs/2512.24601); [GitHub](https://github.com/alexzhang13/rlm)
- The CodeSteer / R1-CI task suite draws on SymBench, Big-Bench-Hard and Reasoning-Gym (144 tasks, 37 held out for testing), with code and models released. — [arXiv 2505.21668v3](https://arxiv.org/html/2505.21668v3)
- MEM1 builds long-horizon tests by composing single-objective multi-hop QA tasks into 16-objective tasks, which is a cheap recipe for scaling horizon length. — [arXiv 2506.15841](https://arxiv.org/html/2506.15841v2)
- Learning How Hard to Think's evaluation protocol plots compute against quality at matched quality ("up to 50% less compute"), which is the right metric frame for effort-level and thinking-budget A/B tests. — [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ff414825df833edb8b1839e3d5d495e9-Abstract-Conference.html)

### Inferences
- Evaluation plan for our setup:
  1. Effort router and thinking-budget changes: Terminal-Bench subset plus replayed sessions, reporting pass rate, wall time, thinking tokens and compactions per task. Use LHHT-style compute-matched comparison curves.
  2. RLM tool or compaction changes: RLM's long-context tasks, plus a MEM1-style composed multi-objective task built from our own repos (e.g. "answer 8 questions about different files in one session").
  3. Helper fine-tunes: check general-capability regression to verify forgetting (RL's Razor), not only target-task accuracy.

### Gaps
- No MIT or Harvard benchmark for multi-turn coding agents, tool-call validity, or context compaction quality was found in this session.
- MIT's "Reasoning or Reciting?" (Wu et al., 2307.02477) counterfactual evaluation and Harvard's "Repeat After Me" copying benchmark (2402.01032, which is relevant to hybrid/linear models like Qwen3-Next) are known to me from memory but were not verified here.
- Tool-call budget limits meant that MIT-IBM Watson AI Lab and Kempner publication pages were not crawled systematically, so lesser-known 2026 MIT and Harvard papers may be missing.
