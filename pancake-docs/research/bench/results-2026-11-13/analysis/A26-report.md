# A26: draft-source selector (CPU counterfactual replay)

Saved by the main session from the agent's final text (the agent's own write was refused).

## Files
- /home/dan/agent-results/A26/a26_selector.py: per-request replay, imports suffix_sim read-only (suffix_sim.py not modified).
- /home/dan/agent-results/A26/a26_stats.py: selectors, oracle, bootstrap (B=2000, seed 26).
- /home/dan/agent-results/A26/per_request.json (2,045 requests) and stats.json.

## Consistency
Per-request sums reproduce the replay totals exactly: lookup rounds 574,016 / accepted 697,198; index rounds 777,281 / accepted 493,991; fallback 561,643 / 709,564; longest 560,038 / 711,176.

## Data
2,045 requests, 25 sessions (15 main, 10 sub-agent), 276 sub-agent requests, all in the second half of the time order.

## Results (gain over always-lookup; accepted tokens, rounds negative is better; CI95 by percentile bootstrap over requests)
- Always-index: -29.2% accepted. Lookup wins on 2,045 of 2,045 requests in aggregate.
- Sub-agent rule (index for sub-agents, lookup for main): -3.0% accepted, CI [-49,063, -2,305]; cluster-by-session CI [-61,457, +1,010]. Does not help.
- Previous-request winner per session: -1.0% accepted. Does not help.
- Prompt-length threshold: fitted threshold never switches; holdout gain 0. The first half is all main requests, so it cannot learn sub-agent behaviour.
- Per-request oracle (upper bound, not deployable): +0.78% accepted, CI [+3,913, +7,241]; rounds -0.93%.
- Per-window fallback (suffix_sim's own arm): +1.77% accepted, CI [+10,286, +14,557]; rounds -2.2%.
- Per-window longest match (suffix_sim's own arm): +2.00% accepted, CI [+11,715, +16,464]; rounds -2.4%. Main +1.3%, sub +6.0%.

## Verdict
1. No pre-request selector beats always-lookup. Keep always-lookup for per-request source choice. A26 as scoped does not pay.
2. The per-request oracle caps any selector below 1%.
3. Per-window longest match is the only candidate above noise, on both bootstraps. It uses only information available at each window and has no fitted parameters. It is a per-window rule, not a pre-request selector. Replay is in-sample for one model over 25 sessions. This ties to A24 (the suffix index): same +2.0% figure.
4. Follow-up: more sessions, a real verify cost (suffix_sim assumes one window costs one step), and confirmation through suffix_index_test --simulate with --dump.

## Limitations
Counterfactual replay on recorded greedy output (acceptance = exact match). Prompts rebuilt from stored text, so lengths are approximate. 25 sessions, so the cluster-bootstrap CIs are the honest ones for sub-agent and index results. Rounds model each window as one step.
