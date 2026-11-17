"""Tests for kernel_resources.py: parser, digest and the regression rule on fixed text (no compiler, no GPU) (LEADS.md Part O O1 / O2)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kernel_resources as K  # noqa: E402

TAG = "[-Rpass-analysis=kernel-resource-usage]"
REMARKS = f"""\
a.cu:74:1: remark: Function Name: _ZN6strata7kernels12_GLOBAL__N_116combine_k10_vec4EPKfi {TAG}
   74 | int n) {{
a.cu:74:1: remark:     SGPRs: 18 {TAG}
a.cu:74:1: remark:     VGPRs: 46 {TAG}
a.cu:74:1: remark:     ScratchSize [bytes/lane]: 0 {TAG}
a.cu:74:1: remark:     Dynamic Stack: False {TAG}
a.cu:74:1: remark:     Occupancy [waves/SIMD]: 16 {TAG}
a.cu:74:1: remark:     SGPRs Spill: 0 {TAG}
a.cu:74:1: remark:     VGPRs Spill: 3 {TAG}
a.cu:74:1: remark:     LDS Size [bytes/block]: 4 {TAG}
a.cu:107:1: remark: Function Name: _Z7kernel2v {TAG}
a.cu:107:1: remark:     VGPRs: 8 {TAG}
"""


def test_parse_remarks():
    r = K.parse_remarks(REMARKS)
    a = r["_ZN6strata7kernels12_GLOBAL__N_116combine_k10_vec4EPKfi"]
    assert a == dict(sgprs=18, vgprs=46, scratch=0, occupancy=16, spills_s=0, spills_v=3, lds=4)
    assert r["_Z7kernel2v"] == {"vgprs": 8}


def test_short_name():
    assert K.short_name("_ZN6strata7kernels12_GLOBAL__N_116combine_k10_vec4EPKfi").endswith("combine_k10_vec4")
    assert K.short_name("_Z7kernel2v") == "kernel2"


ASM = """\
\t.text
_Z2k1v:                                 ; @_Z2k1v
; %bb.0:
\ts_load_b64 s[0:1], s[0:1], 0x0   ; comment
\ts_cbranch_scc1 .LBB0_2
.LBB0_1:
\tv_add_f32 v0, v0, v1
.LBB0_2:
\ts_endpgm
.Lfunc_end0:
\t.size _Z2k1v, .Lfunc_end0-_Z2k1v
_Z2k2v:
\ts_endpgm
.Lfunc_end1:
"""


def test_digest_counts_instructions_and_ignores_comments():
    d = K.asm_digests(ASM)
    assert set(d) == {"_Z2k1v", "_Z2k2v"}
    assert d["_Z2k1v"]["instr"] == 4 and d["_Z2k2v"]["instr"] == 1


def test_digest_survives_label_renumbering_in_another_function():
    # a function added before k1 renumbers .LBB<n>_<bb>; k1's digest must not move
    shifted = ASM.replace(".LBB0_", ".LBB5_").replace(".Lfunc_end0", ".Lfunc_end7")
    assert K.asm_digests(shifted)["_Z2k1v"] == K.asm_digests(ASM)["_Z2k1v"]


def test_digest_moves_when_an_instruction_changes():
    changed = ASM.replace("v_add_f32 v0, v0, v1", "v_mul_f32 v0, v0, v1")
    assert K.asm_digests(changed)["_Z2k1v"]["sha1"] != K.asm_digests(ASM)["_Z2k1v"]["sha1"]
    assert K.asm_digests(changed)["_Z2k2v"] == K.asm_digests(ASM)["_Z2k2v"]


def row(**kw):
    base = dict(short="k", vgprs=40, sgprs=20, scratch=0, spills_v=0, spills_s=0, lds=0, occupancy=12, sha1="aa", instr=10)
    base.update(kw)
    return base


def test_check_regressions():
    base = {"k": row()}
    assert K.check({"k": row()}, base) == ([], [])
    bad, _ = K.check({"k": row(spills_v=2, scratch=8, lds=64, occupancy=10)}, base)
    assert len(bad) == 4
    bad, notes = K.check({"k": row(vgprs=44)}, base)
    assert bad == [] and len(notes) == 1                        # a VGPR change alone is a note ...
    assert len(K.check({"k": row(vgprs=44)}, base, strict=True)[0]) == 1   # ... unless --strict
    assert K.check({"k": row(sha1="bb")}, base)[0] == []        # a digest change is a regression only with --digest
    assert len(K.check({"k": row(sha1="bb")}, base, use_digest=True)[0]) == 1


def test_improvements_and_membership_are_notes_only():
    base = {"k": row(spills_v=4), "gone": row()}
    bad, notes = K.check({"k": row(spills_v=0), "new": row()}, base)
    assert bad == [] and any("gone" in n for n in notes) and any("new" in n for n in notes)
