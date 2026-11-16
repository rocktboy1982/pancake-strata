#!/usr/bin/env python3
"""plan_auto.py - an offline planner for the pancake-strata presets (MOD_PLAN.md section 6 item 35). No GPU, no engine.

From a hardware description and a model description it proposes the preset knobs that exist in this engine today, with
one rationale per knob (the rule, how it was decided, the ledger rows behind it). The structure of the planner is taken
from BigMoeOnEdge's `--auto` (Apache-2.0, https://github.com/Helldez/BigMoeOnEdge; LEADS.md Part I); the rules and the
code are our own and use this engine's own flags and STRATA_* names.

Rules taken over from BigMoeOnEdge (their docs hardware-planning.md and measured-planning.md):
  P1  decline instead of guessing: a knob whose deciding fact is unknown keeps its carried value and says so.
  P2  a lever must win by a margin (their min_backend_win = 10%), not merely win; half the plausible levers lost on
      measurement.
  P3  memory is accounted (a ledger with margins), speed is measured: the planner never raises a memory ask because a
      model says it would be faster.
  P4  nothing lossy hides: every lossy knob is marked in the rationale, and --no-lossy leaves them all out (the daily preset
      itself carries them, so the default run keeps them to be comparable with it).
  P5  the cache floor is the model's own token cycle (layers x top-k x expert bytes); under it a tier has no use.
Every knob has a basis: DERIVED (a rule over the hardware/model facts), ANCHORED (a rule scaled from the one measured
box value, so it reproduces that box by construction and is a hypothesis elsewhere) or CARRIED (the measured value from
TEST_LEDGER.md; no rule over facts decides it). The proposal never claims more than its basis.

    python tools/pancake/plan_auto.py                       # our box, Flash-Next IQ2_XS: print rationale and diff
    python tools/pancake/plan_auto.py --out proposal.json   # the proposal preset JSON
    python tools/pancake/plan_auto.py --hw what_if.json     # override hardware fields (see BOX)
    python tools/pancake/plan_auto.py --trial-spec out.json # an abtest spec of the knobs worth a measured trial

Not run on the box: this tool predicts nothing about speed. A proposal is a spec to A/B (abtest.py), never a result.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE_PRESET = ROOT / "configs" / "pancake" / "flashnext-iq2_xs-fast-hip.json"

MIB = 1024 * 1024

# ---------------------------------------------------------------------------------------------- descriptions
# Hardware. Every figure names where it comes from. `cpu_expert_us` / `copy_us` / `gpu_read_us` are the figures the engine's
# own startup probe printed on this box (STRATA_CALIBRATE, TEST_LEDGER.md T-640).
BOX = {
    "name": "RX 7900 XT 20 GB gfx1100 + Ryzen 7 5700G + 57 GB DDR4",
    "gpu_arch": "gfx1100",
    "vram_total_mib": 20464,          # RX 7900 XT, 20 GB (20464 MiB as the driver reports it)
    "vram_other_mib": 0,              # desktop / other processes: the box is headless while serving
    "pcie": "3.0 x16",                # forced by the 5700G (MOD_PLAN.md section 1)
    "ram_gib": 57,
    "service_memory_max_gib": 54,     # pancake.service MemoryMax=54G (MOD_PLAN.md section 3)
    "ram_bw_gbs": 33,                 # MOD_PLAN.md section 1
    "storage": "NVMe",
    "storage_mbs": None,              # not measured: only the model load reads it, and no knob here depends on it
    "cpu_cores": 8,
    "cpu_threads": 16,
    "hybrid_cpu": False,
    "cpu_expert_us": 80.6,            # T-640: CPU pool, one missed expert alone (80.9 with the link busy)
    "copy_us": 114.6,                 # T-640: one expert host -> device (12.6 GB/s)
    "gpu_read_us": 9.6,               # T-640
    "pcie_discrete": True,
    # decode windows of the daily preset, ms per verify window, from the spec 40 profile (MOD_PLAN.md section 1):
    # short 24.8 = 15.9 GPU / 5.2 CPU / 2.2 other; edit 37.4 = 15.8 GPU / 17.4 CPU / 2.5 other + 1.7 sync
    "windows": [
        {"case": "short", "gpu_ms": 15.9, "cpu_ms": 5.2, "weight": 0.5},
        {"case": "edit", "gpu_ms": 15.8, "cpu_ms": 17.4, "weight": 0.5},
    ],
}

# Model. `vram_dense_mib` is an ANCHOR: the box has no raw engine log of the `expert cache auto:` line in the tree, so
# the dense weights + session + drafter VRAM was back-solved from the 10909 expert slots the tier runs with (tier_laws
# T-618 input): 20464 - 10909 x 1.370 - 700 reserve - 256 late - 200 draft head - KV(32768 resident) = ~3966 MiB. Replace it with the real
# log line and the slot count stops being a consistency check and becomes a prediction.
FLASHNEXT_IQ2XS = {
    "name": "Qwen3.8-Flash-Next IQ2_XS (GSQ-RCO pack)",
    "moe_layers": 48,
    "experts_per_layer": 512,
    "top_k": 10,
    "expert_mib": 1.370,              # MOD_PLAN.md section 10: a 32K loan of 9,413 slots is 12.59 GiB
    "qsa_layers": 12,
    "kv_bytes_per_token_layer": 1056,  # MOD_PLAN.md section 10 (KV staging, int8), over the QSA layers
    "vram_dense_mib": 3966,           # anchor, see above
    "mtp_bind_mib": 200,              # the draft head that binds after the cache (auto_cache.hpp measured 178-213 MiB on other models)
    "loan_slots_per_token": 9413 / 32768,   # MOD_PLAN.md section 10: loan at 32K = 9,413 slots
    "max_context_cap": 131072,        # the preset's value; the model's trained maximum is not read here
    "ple_file_gib": 28.8,             # shard 2, read through the page cache (docs/MODELS.md pancake note)
    "has_mtp": True,
    "mtp_depth": 4,                   # official MTP depth (T-118, T-149)
}

# Engine constants this planner mirrors, with the file that holds them.
RESERVE_MIB = 700                     # generate.cpp Options::vram_reserve_mib
LATE_ALLOC_MIB = 256                  # include/strata/core/auto_cache.hpp kLateAllocMib
LEND_KEEP = 128                       # generate.cpp g_lend_keep
LOAN_PCT_MAX = 0.90                   # MOD_PLAN.md section 10: auto chunk = loan <= 90% of the slots
CHUNK_GRID = (8192, 16384, 32768, 49152)   # generate.cpp --prefill auto: STRATA_PREFILL_AUTO_MAX clamps to this grid
PCIE_MIN_SHARE = 0.35                 # include/strata/core/calibrate_ratio.hpp calib_min_share default
PCIE_CEILING = 0.95                   # pcie_share_rule.hpp kPcieCeilingFollow (discrete GPU)
MIN_WIN = 0.10                        # BigMoeOnEdge PlannerPolicy::min_backend_win
MARGIN_KILLED = 0.20                  # BigMoeOnEdge PlannerPolicy::margin_when_killed (a hard cgroup cap is fatal)
ANCHOR_SWAPS = 32                     # the preset's value (T-199), the anchor of the swap rule
ANCHOR_BREAK_EVEN = 114.6 / (80.6 - 9.6)   # calibrate_ratio.hpp break_even_reuse on the T-640 figures


@dataclass
class Knob:
    name: str                 # "--pcie-frac" or "STRATA_PREFILL_AUTO_MAX"
    kind: str                 # "arg" or "env"
    value: object             # str / None (None = engine default, nothing emitted)
    default: object           # the engine's own default, what an absent knob means
    rule: str
    basis: str                # DERIVED / ANCHORED / CARRIED
    exact: bool
    text: str
    evidence: str = ""
    trial: tuple = ()         # neighbour values worth a measured trial (see trial_spec)


@dataclass
class Plan:
    knobs: list = field(default_factory=list)
    ledger: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def add(self, k: Knob):
        self.knobs.append(k)
        return k


def fmt(v):
    return str(v)


# ---------------------------------------------------------------------------------------------- the rules
def expert_set(model):
    return model["moe_layers"] * model["experts_per_layer"]


def token_cycle_mib(model):
    """P5: the experts one token routes across every MoE layer, in MiB (BigMoeOnEdge cache floor)."""
    return model["moe_layers"] * model["top_k"] * model["expert_mib"]


def kv_resident_mib(model, cells):
    return cells * model["kv_bytes_per_token_layer"] * model["qsa_layers"] / MIB


def auto_slots(free_mib, reserve_mib, blob_mib, min_slots):
    """include/strata/core/auto_cache.hpp auto_cache_slots, in MiB: the late allocation is left out when the cache still has min_slots."""
    plain = int((free_mib - reserve_mib) // blob_mib)
    late = int((free_mib - reserve_mib - LATE_ALLOC_MIB) // blob_mib)
    return late if late >= min_slots and late > 0 else max(plain, 0)


def rule_vram_tier(hw, model, kv_cells, plan):
    """R1: the VRAM expert tier = what is left of the card after the dense weights, the resident KV, the draft head, the
    reserve and the late allocation (DERIVED, the engine's own sizing rule). `--expert-cache auto` is the engine doing the same."""
    blob = model["expert_mib"]
    kv = kv_resident_mib(model, kv_cells)
    free = hw["vram_total_mib"] - hw["vram_other_mib"] - model["vram_dense_mib"] - kv
    min_slots = math.ceil(model["loan_slots_per_token"] * 256) + LEND_KEEP
    reserve = RESERVE_MIB + model["mtp_bind_mib"]
    slots = min(auto_slots(free, reserve, blob, min_slots), expert_set(model))
    cycle = token_cycle_mib(model)
    tier_mib = slots * blob
    plan.ledger["vram"] = {
        "total_mib": hw["vram_total_mib"], "dense_and_session_mib": model["vram_dense_mib"], "kv_resident_mib": round(kv, 1),
        "reserve_mib": RESERVE_MIB, "late_alloc_mib": LATE_ALLOC_MIB, "draft_head_mib": model["mtp_bind_mib"],
        "expert_tier_mib": round(tier_mib, 1), "expert_tier_slots": slots, "min_slots_for_loan": min_slots,
        "resident_share_of_experts": round(slots / expert_set(model), 3),
    }
    ok = tier_mib >= cycle
    plan.add(Knob(
        "--expert-cache", "arg", "auto" if ok else "0", "0", "R1 vram-tier + P5 token-cycle floor", "DERIVED", True,
        (f"the card has {hw['vram_total_mib']} MiB; dense+session {model['vram_dense_mib']}, resident KV {kv:.0f}, reserve "
         f"{RESERVE_MIB}, draft head {model['mtp_bind_mib']}, late {LATE_ALLOC_MIB} leave {free - reserve - LATE_ALLOC_MIB:.0f} MiB = {slots} slots "
         f"of {blob} MiB ({slots / expert_set(model):.1%} of the {expert_set(model)} experts). The token cycle is {cycle:.0f} MiB "
         f"({model['moe_layers']} layers x top-{model['top_k']} x {blob}); the tier is {tier_mib / cycle:.1f}x it, so it has use."
         if ok else f"the tier ({tier_mib:.0f} MiB) is under the token cycle ({cycle:.0f} MiB): no expert would survive to its next use"),
        "T-618 (10909 slots, 71.9% static / 81.2% tier hit offline)"))
    return slots


def rule_pcie_frac(hw, plan):
    """R2: --pcie-frac. The engine's static rule (calibrate_ratio.hpp) balances the CPU and the link by their per-expert time.
    On this box it gives 0.40 and lost 17-30% (T-640). So the share is judged on the decode windows, with the margin
    (P2): the share f moves f of the CPU rows to the link; a window is max(GPU, CPU, link); the gain is the weighted window-time
    saving, and it must reach MIN_WIN. DERIVED."""
    cpu, copy = hw["cpu_expert_us"], hw["copy_us"]
    share_raw = cpu / (cpu + copy)
    static = 0.0 if share_raw < PCIE_MIN_SHARE else min(PCIE_CEILING, max(0.10, round(share_raw * 20) / 20))
    rows = []
    best_f, best_gain = 0.0, 0.0
    for f in [x / 100 for x in range(0, 96, 5)]:
        gain = 0.0
        for w in hw["windows"]:
            g, c = w["gpu_ms"], w["cpu_ms"]
            misses = c * 1000.0 / cpu
            link = f * misses * copy / 1000.0
            new = max(g, (1.0 - f) * c, link)
            base = max(g, c)
            gain += w["weight"] * (base - new) / base
        rows.append((f, gain))
        if gain > best_gain:
            best_f, best_gain = f, gain
    take = best_gain >= MIN_WIN
    value = f"{best_f:g}" if take else "0"
    plan.add(Knob(
        "--pcie-frac", "arg", value, "auto", "R2 pcie-share window model + margin", "DERIVED", True,
        (f"per expert the CPU pool takes {cpu} us and the link {copy} us, so the engine's balance rule says {share_raw:.3f} -> "
         f"{static:g} (min share {PCIE_MIN_SHARE}), which measured 17-30% slower than 0. Window model: best share {best_f:g} saves "
         f"{best_gain:.1%} of the weighted window time (short windows are GPU-bound, the share cannot help them); the margin is {MIN_WIN:.0%} "
         f"(P2), so {'it is taken' if take else 'it is declined'}: {value}."),
        "T-197, T-268 (0.15/0.3/0.55 all slower), T-640 (the probe's 0.40 lost 17-30%)", trial=()))
    return value, static


def rule_adapt(hw, plan):
    """R3: how the VRAM tier follows the conversation. --adapt-every 1 (one adaptation per window) is CARRIED (T-199).
    --adapt-swaps scales the preset's 32 by how fast a promotion repays: break-even reuse = copy / (cpu - gpu read)
    (calibrate_ratio.hpp). A faster-repaying promotion (smaller break-even) earns more swaps. ANCHORED to the box; off the box it is a
    hypothesis, which is why it is a trial knob (spec 117 already holds 64 and 96)."""
    cpu, copy, gpu = hw["cpu_expert_us"], hw["copy_us"], hw["gpu_read_us"]
    be = copy / max(cpu - gpu, 1e-6)
    swaps = int(min(96, max(16, round(ANCHOR_SWAPS * ANCHOR_BREAK_EVEN / be))))
    plan.add(Knob("--adapt-every", "arg", "1", "100", "R3a carried", "CARRIED", True,
                  "adapt every window (the tier follows the conversation at the same rate the windows run)", "T-199"))
    plan.add(Knob(
        "--adapt-swaps", "arg", str(swaps), "0", "R3b break-even scaled swaps", "ANCHORED", True,
        (f"a promotion repays after {be:.2f} reuses here (copy {copy} us / (CPU {cpu} - GPU read {gpu} us)); the anchor is 32 swaps at "
         f"{ANCHOR_BREAK_EVEN:.2f}, so {swaps}. No rule over facts decides this number; the trial does."),
        "T-199 (e1, 32), T-248 (16/64/96 screen), spec 117 queued", trial=(max(16, swaps // 2), min(96, swaps * 2))))


def rule_pool(hw, plan):
    """R4: CPU expert pool. Bandwidth-bound DDR4: one worker per physical core except the core the host thread spins on, no SMT
    workers (MOD_PLAN.md section 10: SMT adds no bandwidth; T-202 workers 12 lost). On a hybrid CPU the P-cores only."""
    cores = hw["cpu_cores"]
    workers = max(1, cores - 1)
    plan.add(Knob(
        "--pool-workers", "arg", None, "0", "R4a cores-1", "DERIVED", True,
        (f"{cores} physical cores ({hw['cpu_threads']} threads): {workers} workers + the host thread joining the drain = {cores} participating; "
         "no SMT siblings (bandwidth-bound pool). Equals the engine's default 0, so nothing is emitted."),
        "T-195 (4/5/6 workers lost 1-11%), T-202 (12 lost)", trial=(workers - 1, workers + 1)))
    if hw["hybrid_cpu"]:
        plan.add(Knob("--pool-affinity", "arg", "measure", "all", "R4b hybrid", "DERIVED", True,
                      "hybrid CPU: P-cores and their siblings vs all cores is timed at start", "engine usage text"))
    else:
        plan.add(Knob("--pool-affinity", "arg", None, "all", "R4b non-hybrid", "DERIVED", True,
                      "uniform cores: `measure` times all vs p-cores and is documented for hybrid CPUs only; nothing to choose", "generate.cpp usage"))
    return workers


def rule_prefill(hw, model, slots, plan):
    """R5: the prompt path borrows cache slots as its chunk buffers (a loan). The largest chunk is the one whose loan stays within
    90% of the tier (MOD_PLAN.md section 10); STRATA_PREFILL_AUTO_MAX is that size on the engine's grid. DERIVED.
    The CPU share of staged chunks is CARRIED (T-108: the share arms lost; the 0.6 fraction itself has no sweep row)."""
    per_tok = model["loan_slots_per_token"]
    slots = max(slots, 1)
    max_tokens = int(LOAN_PCT_MAX * slots / per_tok)
    cap = max([g for g in CHUNK_GRID if g <= max_tokens] or [8192])
    if cap == 49152 and max_tokens < 49152:
        cap = 32768
    # the engine gives exactly 32768 for any ask below 40960 (generate.cpp --prefill auto), so the grid step above needs the room
    if max_tokens < 40960:
        cap = min(cap, 32768)
    loan = per_tok * cap
    plan.ledger["prompt_loan"] = {"chunk_tokens": cap, "loan_slots": round(loan), "loan_share_of_tier": round(loan / slots, 3), "max_tokens_at_90pct": max_tokens}
    plan.add(Knob("--prefill", "arg", "auto", "0", "R5a chunk auto", "DERIVED", True,
                  "the engine picks the chunk per prompt under the loan rule", "T-096 (hipBLASLt table, 1,850-1,920 tok/s)"))
    plan.add(Knob(
        "STRATA_PREFILL_AUTO_MAX", "env", str(cap), "8192", "R5b loan within 90% of the tier", "DERIVED", False,
        (f"a chunk of T tokens lends {per_tok:.4f} slots/token, so 90% of {slots} slots admits {max_tokens} tokens; the engine's grid "
         f"(8192/16384/32768, 49152 only from 40960) gives {cap}; loan {loan:.0f} slots = {loan / slots:.0%} of the tier. Listed lossy in the "
         "preset (a bigger chunk changes the rounding of the prompt GEMMs)."),
        "T-106 (24576: short -0.4%, dropped), T-614 (49152 inside noise)"))
    hint = hw["ram_bw_gbs"] * 0.89 / (hw["ram_bw_gbs"] * 0.89 + model["expert_mib"] * MIB / hw["copy_us"] / 1000.0)
    plan.add(Knob("STRATA_PREFILL_CPU_SHARE", "env", "0.6", "0", "R5c carried", "CARRIED", False,
                  f"carried. A bandwidth-ratio formula would give {hint:.2f}; it is not used, an unmeasured number may not justify a move (P1).",
                  "T-108 (share0/auto/max 1024/2048 all lost), MOD_PLAN.md section 3", trial=("0.4", "0.8")))
    plan.add(Knob("STRATA_PREFILL_CPU_SHARE_MAX", "env", "4096", "0", "R5d carried", "CARRIED", True,
                  "chunks of 4096 tokens or more stream every expert; below it the staged path shares rows with the CPU", "T-108"))


def rule_kv(hw, model, plan):
    """R6: KV. int8 is CARRIED (q8 costs < 0.00012 KL, MOD_PLAN.md section 10). The resident cells are CARRIED at 32768 (131072 and
    8192 both lost, T-276/T-277) and their VRAM cost is charged to the tier by R1. The context ceiling is DERIVED from the RAM
    ledger: what spills to host RAM beyond the resident cells must fit."""
    cells = 32768
    plan.add(Knob("--kv", "arg", "int8", "fp16", "R6a carried", "CARRIED", False,
                  "int8 KV: < 0.00012 KL vs fp16 (the preset lists it lossy)", "MOD_PLAN.md section 10, T-276"))
    plan.add(Knob("--kv-resident", "arg", str(cells), "0", "R6b carried", "CARRIED", True,
                  f"{cells} resident cells cost {kv_resident_mib(model, cells):.0f} MiB of the card (charged to the tier in R1)",
                  "T-276 (131072 lost), T-277 (8192 not better)"))
    return cells


def rule_ram(hw, model, plan):
    """R7: the RAM ledger (P3). Held for the run: every expert in the resident arena, the conversation cache, the parked conversations,
    host KV beyond the resident cells, engine overhead. A hard cgroup cap is fatal, so the margin is MARGIN_KILLED of the cap (BigMoeOnEdge
    margin_when_killed). The max context is the largest value whose host KV still fits. The PLE table (28.8 GB, page cache) floats and is not held."""
    cap = hw.get("service_memory_max_gib") or hw["ram_gib"]
    budget = cap * (1 - MARGIN_KILLED)
    arena = expert_set(model) * model["expert_mib"] / 1024.0
    conv = 2048 / 1024.0
    parked = 4352 / 1024.0              # MOD_PLAN.md section 10: parked conversations 1,280-6,656 MiB auto, usually 4,352
    misc = 2.0
    kv_tok = model["kv_bytes_per_token_layer"] * model["qsa_layers"]
    cells = 32768
    ctx = 32768
    for c in (65536, 131072, 262144):
        if c > model["max_context_cap"]:
            break
        host_kv = (c - cells) * kv_tok / 1024 ** 3
        if arena + conv + parked + misc + host_kv <= budget:
            ctx = c
    host_kv = (ctx - cells) * kv_tok / 1024 ** 3
    held = arena + conv + parked + misc + host_kv
    plan.ledger["ram"] = {
        "cap_gib": cap, "margin": MARGIN_KILLED, "budget_gib": round(budget, 2), "expert_arena_gib": round(arena, 2),
        "conversation_cache_gib": conv, "parked_conversations_gib": round(parked, 2), "engine_misc_gib": misc,
        "host_kv_gib": round(host_kv, 2), "held_gib": round(held, 2), "fits": held <= budget,
        "ple_page_cache_gib_not_held": model["ple_file_gib"], "ram_total_gib": hw["ram_gib"],
    }
    plan.add(Knob("--max-context", "arg", str(ctx), "0", "R7a RAM ledger", "DERIVED", True,
                  (f"largest of 32768/65536/131072/262144 (capped at the preset's {model['max_context_cap']}) whose host KV fits: held {held:.1f} GiB of a "
                   f"{budget:.1f} GiB budget ({cap} GiB cap less {MARGIN_KILLED:.0%}); arena {arena:.1f}, conversation cache {conv:.1f}, parked {parked:.1f}, "
                   f"host KV {host_kv:.2f}, misc {misc:.1f}"),
                  "pancake.service MemoryMax=54G, T-627 (agent prompts p90 54,313 tokens)"))
    plan.add(Knob("--conversation-cache-mib", "arg", "2048", "0", "R7b carried", "CARRIED", True,
                  "2 GiB conversation cache; it fits the ledger above, its size is a TTFT trade measured, not derived", "T-178"))
    plan.add(Knob("--conversation-cache-min-tokens", "arg", "8192", "0", "R7c carried", "CARRIED", True,
                  "conversations shorter than this are not worth a cache slot", "T-178, T-627"))
    return ctx


def rule_spec(model, plan):
    """R8: drafting. CARRIED: no rule over the hardware decides these; they are properties of this model and the agent workload."""
    if model["has_mtp"]:
        plan.add(Knob("--spec", "arg", str(model["mtp_depth"]), "0", "R8a carried", "CARRIED", True,
                      "the model's official MTP depth (depth 2 lost 5%, 5/6 flat)", "T-118, T-149"))
    plan.add(Knob("--spec-min-p", "arg", "0.5", "0", "R8b carried", "CARRIED", True,
                  "live per-request key (strata_tune spec_min_p): the one decode knob a trial can switch without a restart", "T-389 (0.7 dropped), T-153/T-154 (0 measured on prose)", trial=("0.3", "0.7")))
    plan.add(Knob("--lookup-max-t", "arg", "12", "0", "R8c carried", "CARRIED", True,
                  "lookup window rows; edit +8.8% +-6.8, quality PASS", "T-126, T-127"))


def plan_auto(hw, model, allow_lossy=True):
    plan = Plan()
    kv_cells = rule_kv(hw, model, plan)
    slots = rule_vram_tier(hw, model, kv_cells, plan)
    rule_pcie_frac(hw, plan)
    rule_adapt(hw, plan)
    rule_pool(hw, plan)
    rule_prefill(hw, model, slots, plan)
    rule_ram(hw, model, plan)
    rule_spec(model, plan)
    if not allow_lossy:
        plan.knobs = [k for k in plan.knobs if k.exact]
        plan.notes.append("--no-lossy: lossy knobs left out; the proposal is not the daily preset")
    return plan


# ---------------------------------------------------------------------------------------------- the proposal preset
def split_args(args):
    pairs = []
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--") and i + 1 < len(args) and not args[i + 1].startswith("--"):
            pairs.append([a, args[i + 1]])
            i += 2
        else:
            pairs.append([a, None])
            i += 1
    return pairs


def join_args(pairs):
    out = []
    for a, v in pairs:
        out.append(a)
        if v is not None:
            out.append(v)
    return out


def apply_plan(base, plan):
    """The base preset with every knob the planner owns set to its proposal (an absent value = the engine default = removed)."""
    p = copy.deepcopy(base)
    pairs = split_args(p["args"])
    env = p.setdefault("env", {})
    for k in plan.knobs:
        if k.kind == "arg":
            pairs = [x for x in pairs if x[0] != k.name]
            if k.value is not None:
                pairs.append([k.name, k.value])
        else:
            if k.value is None:
                env.pop(k.name, None)
            else:
                env[k.name] = k.value
    # keep the base order for knobs that were already there
    order = {a: i for i, (a, _) in enumerate(split_args(base["args"]))}
    pairs.sort(key=lambda x: order.get(x[0], 10_000))
    p["args"] = join_args(pairs)
    return p


def effective(preset, names):
    """The value each owned knob has in a preset, the engine default where absent."""
    pairs = {a: v for a, v in split_args(preset["args"])}
    env = preset.get("env", {})
    out = {}
    for n, kind, default in names:
        out[n] = (pairs.get(n) if kind == "arg" else env.get(n))
        if out[n] is None:
            out[n] = default
    return out


def diff_vs_preset(base, plan):
    names = [(k.name, k.kind, k.default) for k in plan.knobs]
    have = effective(base, names)
    rows = []
    for k in plan.knobs:
        want = k.value if k.value is not None else k.default
        same = fmt(have[k.name]) == fmt(want)
        rows.append({"knob": k.name, "preset": have[k.name], "proposal": want, "same": same, "basis": k.basis, "rule": k.rule})
    return rows


# ---------------------------------------------------------------------------------------------- the trial spec
def trial_spec(plan, base_path="configs/pancake/flashnext-iq2_xs-fast-hip.json"):
    """An abtest.py spec (--tier screen, then confirm) with one arm per neighbour value of every trial knob. Interleaving, warm-up and the
    promote/drop thresholds are abtest.py's own (MOD_PLAN.md section 2); no other runner."""
    arms = {"base": {}}
    for k in plan.knobs:
        for v in k.trial:
            tag = f"{k.name.lstrip('-').replace('STRATA_', '').lower().replace('_', '-')}-{v}"
            if k.kind == "arg":
                if k.name == "--spec-min-p":
                    continue            # live per-request key: measured by tools/calibrate.py without a restart (MOD_PLAN.md section 6 item 36)
                arms[tag] = {"args": {k.name: str(v)}}
            else:
                arms[tag] = {"env": {k.name: str(v)}}
    return {
        "base": base_path,
        "lossy": any(not k.exact for k in plan.knobs for v in k.trial if k.name in ("STRATA_PREFILL_CPU_SHARE",)),
        "prefill": True,
        "drift_check": True,      # abtest --tier screen re-runs the base at the end: BigMoeOnEdge lost a day to a baseline that drifted -40% with heat
        "cases": ["short", "edit", "long1k", "long3k"],
        "reps": 4,
        "arms": arms,
        "note": ("Written by tools/pancake/plan_auto.py --trial-spec (MOD_PLAN.md section 6 items 35 and 36): the neighbours of the three knobs "
                 "whose best value depends on this box's CPU/link/core costs (--adapt-swaps, --pool-workers, STRATA_PREFILL_CPU_SHARE). "
                 "short/edit judge the decode arms, long1k/long3k the prompt-share arms. Screen first; a PROMOTE needs confirm."),
    }


# ---------------------------------------------------------------------------------------------- output
def render(plan, rows):
    out = []
    out.append("Rationale (one line per knob: basis, rule, value):")
    for k in plan.knobs:
        v = k.value if k.value is not None else f"(engine default {k.default})"
        out.append(f"  {k.name} = {v}   [{k.basis}, {'exact' if k.exact else 'lossy'}, {k.rule}]")
        out.append(f"      {k.text}")
        if k.evidence:
            out.append(f"      evidence: {k.evidence}")
    out.append("")
    out.append("Ledger:")
    out.append(json.dumps(plan.ledger, indent=1))
    out.append("")
    out.append("Proposal vs the preset (effective values):")
    for r in rows:
        out.append(f"  {'=' if r['same'] else '!'} {r['knob']}: preset {r['preset']} / proposal {r['proposal']}   [{r['basis']}]")
    return "\n".join(out)


def load_override(path, base):
    if not path:
        return base
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out = copy.deepcopy(base)
    out.update(d)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hw", help="JSON file overriding fields of the box description (what-if)")
    ap.add_argument("--model", help="JSON file overriding fields of the model description")
    ap.add_argument("--base", default=str(BASE_PRESET), help="the preset used as the template and the sanity-check target")
    ap.add_argument("--out", help="write the proposal preset JSON here")
    ap.add_argument("--trial-spec", help="write the abtest trial spec here")
    ap.add_argument("--no-lossy", action="store_true", help="leave the lossy knobs out (P4)")
    ap.add_argument("--json", action="store_true", help="print the rationale and the diff as JSON")
    a = ap.parse_args(argv)
    hw = load_override(a.hw, BOX)
    model = load_override(a.model, FLASHNEXT_IQ2XS)
    base = json.loads(Path(a.base).read_text(encoding="utf-8"))
    plan = plan_auto(hw, model, allow_lossy=not a.no_lossy)
    rows = diff_vs_preset(base, plan)
    proposal = apply_plan(base, plan)
    if a.out:
        Path(a.out).write_text(json.dumps(proposal, indent=1) + "\n", encoding="utf-8")
    if a.trial_spec:
        Path(a.trial_spec).write_text(json.dumps(trial_spec(plan), indent=1) + "\n", encoding="utf-8")
    if a.json:
        print(json.dumps({"rationale": [k.__dict__ for k in plan.knobs], "ledger": plan.ledger, "diff": rows}, indent=1, default=str))
    else:
        print(render(plan, rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
