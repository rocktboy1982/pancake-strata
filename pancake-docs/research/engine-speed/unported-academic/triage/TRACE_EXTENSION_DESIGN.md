# Trace extension for A22 (entropy stop), A25 (position check), A56 (expected acceptance rate): design
Status: design only (2026-11-12). No code yet. Multi-day engine work; needs a GPU run to verify.

## What the trace has now (include/strata/core/window_trace.hpp)
- drafts record (type 4): per draft row, the drafter's probability of the drafted token (q(d)), and the source.
- rowmiss record (type 5): CPU-missed experts per row (A22 price, A23).
- window record: rows, accepted, position, time.

## What is missing
1. **p(d): the target's probability of each drafted token.** Needed for the Leviathan acceptance min(1, p/q) (A56). The verify path computes argmax only (src/core/verify.cpp:3155-3156, argmax_rows). p(d) needs the softmax normaliser (logsumexp) per verify row over the vocabulary, then exp(logit[d]) minus that normaliser.
2. **Top-k probabilities and entropy of the drafter per draft row** (A22: AdaEDL uses top-1; DISCO uses top-10; entropy needs the full distribution). The MTP kernel returns top-1 only (round-1 research), so this needs the drafter's logits reduced to top-k and entropy on device.

## Cost and risk
- p(d): one vocabulary-wide reduction per verify row. Cheap next to the verify GEMMs, but it touches the verify kernel and must not change the greedy token path when the flag is off.
- Drafter top-k and entropy: changes the MTP draft kernel. The largest item.
- Format: add record type 6 (target p(d) per row) and extend record 4 with top-k and entropy behind a version flag. Readers (tools/pancake/trace/wtrace.py) must skip unknown record types (they already skip type 5 for older readers).

## Order
1. Logging only, behind STRATA_WINDOW_TRACE_TARGET=1: compute logsumexp and p(d) for drafted rows. No change to the argmax path. Test: greedy token identity with the flag on and off (must match), then the CPU test suite.
2. The drafter's top-1 probability is already in record 4, and AdaEDL needs only that. Use it before adding top-k.
3. Drafter top-k and entropy only if step 2 shows entropy rules beat top-1 on replay.

## Acceptance
- Flag off: no output change (token identity; timing within noise).
- Flag on: record 6 written; wtrace.py reads it; on one trace the replayed expected acceptance rate matches the observed acceptance within 1 point.
- GPU time: one window-trace run with the flag on (about 15 min) and one identity run (about 15 min).
