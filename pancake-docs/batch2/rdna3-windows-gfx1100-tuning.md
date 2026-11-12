# rdna3 GitHub issue #4: Windows/gfx1100 tuning + #267 first-request stall — analysis

Source: `strata-windows-gfx1100-tuning-sanitized.zip` attached to `nicejunjie/Strata-rdna3` issue #4
(`api.github.com/.../issues/4`; asset `github.com/user-attachments/files/34438435`). Not our fork and not gfx1100-Linux:
**RX 7900 XTX 24 GB, Windows 11, HIP via ROCm-TheRock 10.2.0, IQ3_S / IQ3_XXS Qwen3.8-Flash-Next**. Its own caveats:
small sequential single-machine tests, XTX 24 GB, driver unrecorded, ~10% is not the ABBA-confirmed kind the fork-KB demands.
Resonance candidates only; our box is RX 7900 XT 20 GB, Linux, IQ2_XS/Q2_0, `draft_vocab_en`.

## 1. What the +10% IQ3_S gain actually was (baseline -> final delta)

Baseline already ran the big upstream "group A" env set (MEGAKERNEL, LFUSE, PLE_BATCH, TSUM, MMVF_ROWS, ATTN_LANECELL,
Q8/Q6_PACKED, GR_RDNA, HIP_WMMA...). The final added only:

| Added | What | Portable to our fast-hip? |
|---|---|---|
| `--mtp-q4 all` | Q4 MTP head (fork-KB had this "queued") | **candidate** — resonance arm; verify path unchanged so greedy stays exact, acceptance changes |
| `--mtp-window 4096` | draft window size | **candidate** |
| `--host-core sibling` | MTP host thread on a sibling core | **candidate** (we use `--pool-affinity measure`) |
| `STRATA_GR_RDNA_Q8=A8=INJ_UP=FUSED=1` | GR RDNA fusion set | **already in our preset** (`flashnext-iq2_xs-fast-hip.json`) |
| `presence_penalty: 0`, `drafting: mtp` | sampling / drafter | presence_penalty 0 is a sampling value; we default-mtp already |
| `--adapt-every 1 --adapt-swaps 32 --pcie-frac 0` | tier policy | **already in our preset** |
| `--mtp-draft-vocab draft_vocab_zh60k.bin` | Chinese 60k draft vocab | **NOT portable** — fork-KB: zh60k hurts English (92.5); we use en |

Headline (author, XTX, IQ3_S): code 86.0/88.8 -> 96.5 tok/s, prose 71.2 -> 78.0 (~+10% geomean); prefill flat
(1450.9 -> 1440.8). **The MTP bundle was tested together, not ablated** and it is IQ3_S-on-XTX, so treat as a
queue of A/B arms for our fast-hip (moe-speedup pain-points already flagged `--mtp-q4` as a queued test), not as an
adopt. Spec prepared: `bench/specs/33-gfx1100-mtp-bundle.json` (mtp-q4 all + mtp-window 4096 + host-core sibling).

XXS: author explicitly REJECTED the CPU/router + GR fusion groups (subsequent fresh starts stalled / fell to ~30 tok/s),
and ran original-kernel controls — a real "measured worse, do not port" note for our fusion-group curiosity.

## 2. #267 first-request handshake/graph stall (the high-value part for our project)

Three fresh XXS starts stalled on the FIRST SHORT REQUEST (issue body + `stall-excerpts.txt`), all with ~1.3-1.5 GiB
VRAM free:

```
no progress for 60 s ... reading the prompt (verify windows), from token 0
  verify window (last window, not the current stage): 5 tokens at position 0, host at layer step 1;
                                                      the GPU rang 1; flags: served 0, plan (A) 0, copies (B) 0
  verify: an earlier window never finished on the GPU (#267)
```

`host at layer step 1; the GPU rang 1; served/plan/copies all 0` is the same class as our window-overhead / host-wait
workstream: the GPU did one doorbell, the host published nothing, the window never completed — a handshake/graph hang,
not a compute stall. This is direct evidence for the house (unresolved) 288/window `wait_flag_ge` seat question and the
window-end sync items (10/12) of MOD_PLAN Phase 18. Two house Windows-HIP notes: big pages refused with error 1314
(missing SeLockMemoryPrivilege) -> 4 KB pages, and `STRATA_ARENA_PIN_GIB=none` -> arena NOT registered (UMA mode) yet
still timed out.

Actions for our project (not ports — investigation hooks):
- The two diagnostics I picked into `work/upstream-sync-1112` — `711a7e93` (stall report asks the prompt compute stream
  whether its work is done) and `03e687ac` (AMD GPU-pause duration) — are exactly what turns this `#267` class from
  "no progress for 60 s" into a located cause on our box. Run them under `STRATA_VERIFY_PROFILE=1` on the first short
  request if a window ever hangs.
- Fold this `#267` excerpt into the Phase-18 window-overhead notes as an independent reproduction of the
  "the GPU rang 1, host published 0" first-request hang, so the host-wait/device-wait design (items 10/12/16) treats
  the doorbell-to-first-plan path as in-scope, not just steady-state waits.
- Keep the caveat: it reproduced WITHOUT the new fusion groups too, so it is a Windows-HIP/TheRock handshake issue, not
  proof those groups cause it — it does argue against unguarded device-wait changes (the "deadlock risk" note).

## 3. Keep / drop

- **Keep as arms:** mtp-q4/mtp-window/host-core (spec 33), presence_penalty 0 (gated sampling arm), and the
  GR_RDNA fusion set (already ours).
- **Do not port:** zh60k vocab (English traffic), the rejected XXS CPU/router fusion groups, the Windows-TheRock
  compat edits (`STRATA_NO_HSA=1`, ArenaShm `pid_t`->`int`) — those are for a Windows-HIP build, which we do not ship.
- **Watch:** `--vram-reserve-mib` 1792 (theirs) / `STRATA_ARENA_PIN_GIB=28` — VRAM policy tuned to a 24 GB card; our 20 GB
  XT and the fork-KB "do not go below ~700 MiB on 20 GB" note govern; needs our own A/B.

## 4. Second pass over the zip (github-issue-draft.md + measurements.json + env diff) — corrections and new checks

The API body I read first was the SHORT form; the in-zip `github-issue-draft.md` has the **ablated** IQ3_S breakdown and
`measurements.json` the per-request numbers. This changes two conclusions:

| Arm (IQ3_S, XTX, author) | code / prose tok/s | delta |
|---|---|---|
| Existing config (baseline) | 86.0 / 71.2 | — |
| Complete MTP bundle | 92.4 / 73.1 | **+~7%** — the whole MTP bundle is worth this much ALONE, not +10% |
| + `presence_penalty=0` | 96.4 / 76.8 | **+~4-5% more** — presence_penalty was the second real lever |
| GR fusion screening (group we already carry) | 101.9 / 82.1 | +the rest |

- **`presence_penalty=0` is a genuine +~4-5% on GREEDY** (temp 0) — their Control defaulted to 1.5. **For us it is a TRACK
  CHECK, not a port:** our normal default is 0; `serve/server.py:7667-7671` floors presence_penalty at `max(1.5,..)` ONLY on
  the non-finite-logits RECOVERY retry (#879), never on a normal run. So the gain exists only if our opencode traffic
  actually sends a nonzero penalty. Action: log what `presence_penalty` our requests carry; if >0, A/B forcing 0 (request
  sampling override), else already optimal. do NOT touch the #879 recovery floor.

- **RED FLAG — `STRATA_RDNA_MMVQ_WAVE=3` (which our fast-hip preset sets, line 93).** The author reports `wave=3` at 1536
  MiB reserve collapsed to **code 30.4 / prose 20.6 tok/s**, while free VRAM was still >1 GiB. Caveated ("this alone does
  not prove a kernel defect"; their box is XTX 24 GB + low reserve), but our preset uses the EXACT value. **MUST re-check
  on our XT** — an independent severe-slowdown report on our active value is not something to fold into a generic do-not-
  repeat. Spec 34 adds `STRATA_RDNA_MMVQ_WAVE=1` vs base(=3) as the first thing to run.

- **Env-set diff (issue#4-final vs our fast-hip) — genuine candidates to arm, not default:** `STRATA_PREFILL_MMQ=1`,
  `STRATA_PF_SWITCH_MIN_T=4096`, `--kv-grow`, `STRATA_IO_THREADS=16`, `STRATA_RESIDENT_HEADROOM_GIB=10`,
  `STRATA_ARENA_PIN_GIB=28`. `STRATA_MTP_DRAFT_ALL=1` is NOT arm-able: it is a rdna3-only switch our fork does not read
  (logic-check: def:0, src:0), so an env A/B would be inert (needs a code port). The happy finding is the mirror side:
  the issue baseline runs the whole "group A" set (`Q8_PACKED, Q6_PACKED, MMVF_ROWS, ATTN_LANECELL, TSUM, LFUSE, PLE_BATCH,
  MEGAKERNEL`) which our preset deliberately omits, and it did NOT beat our omission story — consistent with the fork-KB
  do-not-repeat. So keep those off; do NOT add them.

- `measurements.json` per-request decode for the SAME baseline arm swings 84.4 / 86.0 / 92.3 (code) — another data point
  for the bimodal-run work; the profiler `system_before.txt` (THP / `pp_dpm_sclk` / VRAM) is the instrument to explain it.

- The author's own question ("host handshake vs driver/graph?") is exactly what our just-picked AMD diagnostics answer:
  `03e687ac` (driver GPU-pause duration) separates a driver pause from `711a7e93` (host not marking the compute stream
  done) / the `served 0, plan 0, copies 0` host-side publish. Import that pairing when a window hangs.
