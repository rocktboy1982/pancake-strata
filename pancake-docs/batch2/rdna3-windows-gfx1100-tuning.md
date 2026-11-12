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
