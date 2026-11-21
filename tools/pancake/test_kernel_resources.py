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


# ---- U1 ISA census (LEADS.md Part U): fixture of a gfx1100 loop, offline ------------------------------------------------------------------------
CENSUS_ASM = """\
\t.text
_Z4hotkv:                               ; @_Z4hotkv
; %bb.0:
\ts_load_b64 s[0:1], s[0:1], 0x0
\ts_mov_b32 s2, 0
.LBB0_1:                                ; =>This Loop Header
\tglobal_load_b128 v[4:7], v0, s[0:1]
\tds_load_b32 v8, v1
\ts_waitcnt vmcnt(0) lgkmcnt(0)
\tv_dot4_i32_iu8 v9, v4, v8, v9
\tv_dual_fmac_f32 v10, v4, v5 :: v_dual_mul_f32 v11, v6, v7
\tv_fma_f64 v[12:13], v[4:5], v[6:7], v[12:13]
\tv_fma_f16 v14, v4, v5, v14
\tv_add_nc_u32_e32 v1, 4, v1
\tv_wmma_f32_16x16x16_f16 v[16:23], v[24:31], v[32:39], v[16:23]
\ts_delay_alu instid0(VALU_DEP_1)
\ts_add_i32 s2, s2, 1
\ts_cmp_lt_i32 s2, 8
\ts_cbranch_scc1 .LBB0_1
; %bb.2:
\tglobal_store_b32 v0, v9, s[0:1]
\ts_endpgm
.Lfunc_end0:
\t.size _Z4hotkv, .Lfunc_end0-_Z4hotkv
"""


def test_classify():
    for mn, cls in [("v_dual_fmac_f32", "VOPD"), ("v_dot4_i32_iu8", "DOT"), ("v_dot2_f32_f16", "DOT"), ("v_wmma_f32_16x16x16_f16", "WMMA"),
                    ("v_fma_f64", "VALU_F64"), ("v_cvt_f32_f64_e32", "VALU_F64"), ("v_fma_f32", "VALU_F32"), ("v_pk_fma_f16", "VALU_F16"),
                    ("v_add_nc_u32_e32", "VALU_I32"), ("ds_load_b128", "LDS"), ("global_load_b128", "VMEM_LD"), ("buffer_store_b32", "VMEM_ST"),
                    ("s_load_b64", "SMEM"), ("s_waitcnt", "WAITCNT"), ("s_delay_alu", "DELAY"), ("s_cbranch_scc1", "BRANCH"), ("s_endpgm", "BRANCH"),
                    ("s_add_i32", "SALU"), ("garbage", "OTHER")]:
        assert K.classify(mn) == cls, mn


def test_census_counts_the_inner_loop():
    c = K.census(CENSUS_ASM, cus=84, clock_ghz=2.0, macs=16)
    r = c["_Z4hotkv"]
    assert r["instr"] == 17
    assert len(r["loops"]) == 1
    lp = r["loops"][0]
    assert lp["label"] == "LBB0_1" and lp["instr"] == 13
    want = dict(VMEM_LD=1, LDS=1, WAITCNT=1, DOT=1, VOPD=1, VALU_F64=1, VALU_F16=1, VALU_I32=1, WMMA=1, DELAY=1, SALU=2, BRANCH=1)
    assert {k: v for k, v in lp["counts"].items() if v} == want
    assert sum(lp["counts"].values()) == lp["instr"]
    assert abs(lp["ceiling"]["iter_per_s"] - 84 * 64 * 2e9 / 13) < 1
    assert abs(lp["ceiling"]["mac_per_s"] - lp["ceiling"]["iter_per_s"] * 16) < 1
    assert "0.81 instr/MAC" in K.census_table(c, 16) and "VOPD=1" in K.census_table(c, 16)


def test_census_nested_keeps_only_the_inner_loop_and_objdump_text():
    asm = """\
0000000000001000 <k>:
\ts_mov_b32 s0, 0 // 000000001000: BE800080
<BB0_1>:
\ts_mov_b32 s1, 0 // 000000001004: BE810080
<BB0_2>:
\tv_add_f32 v0, v0, v1 // 000000001008: 06000300
\ts_cbranch_scc1 BB0_2 // 00000000100c: BFA2FFFE
\ts_cbranch_scc1 BB0_1 // 000000001010: BFA2FFFD
\ts_endpgm // 000000001014: BFB00000
"""
    r = K.census(asm)["k"]
    assert [l["label"] for l in r["loops"]] == ["BB0_2"] and r["loops"][0]["instr"] == 2


def test_census_asm_cli_and_default_path_untouched(tmp_path, capsys, monkeypatch):
    f = tmp_path / "a.s"
    f.write_text(CENSUS_ASM)
    assert K.main(["--census-asm", str(f), "--census-macs", "16"]) == 0
    assert "VOPD=1" in capsys.readouterr().out
    # default path: without --census / --repeat the output is the table alone
    monkeypatch.setattr(K, "analyse", lambda *a, **k: ({"k": dict(short="k", vgprs=1, sgprs=2, scratch=0, spills_v=0, spills_s=0, lds=0, occupancy=8, sha1="aa", instr=3)}, None))
    monkeypatch.setattr(K, "default_clang", lambda: ("clang", None))
    (tmp_path / "k.cu").write_text("")
    assert K.main([str(tmp_path / "k.cu"), "--tmp", str(tmp_path)]) == 0
    base = capsys.readouterr().out
    assert base.rstrip("\n") == K.table({"k": dict(short="k", vgprs=1, sgprs=2, scratch=0, spills_v=0, spills_s=0, lds=0, occupancy=8, sha1="aa", instr=3)})
    assert "census" not in base.lower() and "ceiling" not in base
    (tmp_path / "k.s").write_text(CENSUS_ASM)
    assert K.main([str(tmp_path / "k.cu"), "--tmp", str(tmp_path), "--census"]) == 0
    out = capsys.readouterr().out
    assert out.startswith(base) and "ceiling" in out and "VOPD=1" in out


def test_compile_command_unchanged_by_prefix_param():
    p = Path("a.cu")
    assert K.compile_command(p, "clang", None, Path("o.s"))[0] == "clang"


# ---- U6 repeat loop with a fake compiler ---------------------------------------------------------------------------------------------------------
def test_repeat_clean_and_differing(tmp_path):
    calls = []

    def stable(src, clang, rocm, d, defs, inc, cf, prefix):
        calls.append(tuple(prefix))
        return {"k1": {"sha1": "a"}, "k2": {"sha1": "b"}}, None

    r = K.repeat_check(Path("x.cu"), 4, "clang", None, tmp_path, prefix=("setarch", "x86_64", "-R"), run=stable)
    assert r == {"runs": 4, "differ": [], "errors": []} and len(calls) == 4 and calls[0] == ("setarch", "x86_64", "-R")
    n = [0]

    def flaky(src, clang, rocm, d, defs, inc, cf, prefix):
        n[0] += 1
        return {"k1": {"sha1": "a"}, "k2": {"sha1": "b" if n[0] != 3 else "c"}}, None

    r = K.repeat_check(Path("x.cu"), 4, "clang", None, tmp_path, run=flaky)
    assert r["differ"] == ["k2"]


def test_repeat_records_compile_errors(tmp_path):
    r = K.repeat_check(Path("x.cu"), 2, "clang", None, tmp_path, run=lambda *a: ({}, "clang exit 1"))
    assert r["errors"] == ["clang exit 1", "clang exit 1"] and r["differ"] == []
