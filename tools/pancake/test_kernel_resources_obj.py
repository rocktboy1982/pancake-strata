"""Tests for kernel_resources_obj.py (T9) and isa_audit.py (T10) on fixture text (no compiler, no GPU)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import isa_audit as A  # noqa: E402
import kernel_resources_obj as O  # noqa: E402

NOTES = """\
Displaying notes found in: .note
  Owner                Data size \tDescription
  AMDGPU               0x000012ab\tNT_AMDGPU_METADATA (AMDGPU Metadata)
    amdhsa.kernels:
      - .agpr_count:     0
        .args:
          - .address_space:  global
            .name:           x
            .offset:         0
            .size:           8
          - .address_space:  global
            .name:           y
            .offset:         8
            .size:           8
        .group_segment_fixed_size: 4096
        .kernarg_segment_size: 16
        .max_flat_workgroup_size: 256
        .name:           _ZN6strata7kernels8rdna_kq_EPKf
        .private_segment_fixed_size: 0
        .sgpr_count:     18
        .sgpr_spill_count: 0
        .symbol:         _ZN6strata7kernels8rdna_kq_EPKf.kd
        .vgpr_count:     46
        .vgpr_spill_count: 0
        .wavefront_size: 32
      - .agpr_count:     0
        .args:
          - .address_space:  global
            .name:           z
        .group_segment_fixed_size: 0
        .max_flat_workgroup_size: 128
        .name:           spiller
        .private_segment_fixed_size: 64
        .sgpr_count:     30
        .sgpr_spill_count: 2
        .vgpr_count:     200
        .vgpr_spill_count: 12
        .wavefront_size: 32
    amdhsa.target:   amdgcn-amd-amdhsa--gfx1100
"""
LOG = """\
a.cu:1:1: remark: Function Name: spiller [-Rpass-analysis=kernel-resource-usage]
a.cu:1:1: remark:     Occupancy [waves/SIMD]: 7 [-Rpass-analysis=kernel-resource-usage]
"""


def test_parse_notes_skips_args():
    n = O.parse_notes(NOTES)
    assert set(n) == {"_ZN6strata7kernels8rdna_kq_EPKf", "spiller"}
    k = n["_ZN6strata7kernels8rdna_kq_EPKf"]
    assert (k["vgprs"], k["sgprs"], k["lds"], k["scratch"], k["wave"]) == (46, 18, 4096, 0, 32)
    s = n["spiller"]
    assert (s["vgprs"], s["scratch"], s["spills_v"], s["spills_s"]) == (200, 64, 12, 2)


def test_merge_flags_and_cli(tmp_path, capsys):
    notes = tmp_path / "n.txt"
    log = tmp_path / "b.log"
    notes.write_text(NOTES)
    log.write_text("")
    rows = O.merge(O.parse_notes(NOTES), {})
    by = {r["name"]: r for r in rows}
    assert by["spiller"]["occ_src"] == "model" and by["spiller"]["occupancy"] == 1536 // 216
    assert "SPILL" in O.flag(by["spiller"]) and "VGPR>128" in O.flag(by["spiller"])
    assert O.flag(by["_ZN6strata7kernels8rdna_kq_EPKf"]) == []
    log.write_text(LOG)
    csvp = tmp_path / "o.csv"
    rc = O.main(["--notes", str(notes), "--log", str(log), "--csv", str(csvp), "--fail-on-spill"])
    assert rc == 1 and csvp.exists()
    assert "7" in capsys.readouterr().out


def test_nothing_parsed(tmp_path):
    f = tmp_path / "e.txt"
    f.write_text("nothing\n")
    assert O.main(["--notes", str(f)]) == 2


DIS = """\
0000000000001000 <hot_kernel>:
; %bb.0:
\ts_load_b64 s[0:1], s[0:1], 0x0                             // 000000001000: F4040000 F8000000
\tglobal_load_b32 v1, v0, s[0:1]                              // 000000001008: DC520000
\ts_waitcnt vmcnt(0)                                          // 000000001010: BF8903F7
\tv_dual_mul_f32 v2, v1, v1 :: v_dual_add_f32 v3, v1, v1      // 000000001014: C8C80101
\tv_fmac_f32_e32 v4, v1, v2
\tv_mul_f32_e32 v5, v1, v3
\tv_dot4_i32_iu8 v6, v1, v2, v3
\tv_dot2_f32_f16 v7, v1, v2, v3
\tv_wmma_f32_16x16x16_f16 v[8:15], v[0:7], v[0:7], v[8:15]
\ts_delay_alu instid0(VALU_DEP_1)
\tscratch_store_b32 off, v1, off offset:4
\tds_load_b128 v[0:3], v9 offset:128
\tds_load_b128 v[4:7], v9 offset:256
\tds_load_b128 v[8:11], v9 offset:384
\tds_load_b128 v[12:15], v9 offset:512
\ts_barrier
\ts_endpgm
0000000000002000 <clean_kernel>:
\tv_add_f32_e32 v0, v0, v1
\ts_endpgm
"""


def test_isa_audit_counts():
    f = A.parse(DIS)
    assert list(f) == ["hot_kernel", "clean_kernel"]
    r = A.audit("hot_kernel", f["hot_kernel"])
    assert r["vopd"] == 1 and r["vopd_cand"] == 2 and r["vopd_share"] == 0.5
    assert r["dot"] == 2 and r["sudot"] == 1 and r["wmma"] == 1
    assert r["scratch"] == 1 and r["lds"] == 4 and r["bank_cand"] == 1
    assert r["tight_wait"] == 1 and r["delay_alu"] == 1 and r["barrier"] == 1
    fl = A.findings(r)
    assert any(x.startswith("SPILL") for x in fl) and any(x.startswith("BANK?") for x in fl)
    c = A.audit("clean_kernel", f["clean_kernel"])
    assert "NO-VOPD" in A.findings(c) and not c["scratch"]


def test_isa_audit_cli(tmp_path, capsys):
    p = tmp_path / "d.txt"
    p.write_text(DIS)
    csvp = tmp_path / "o.csv"
    assert A.main([str(p), "--match", "hot", "--csv", str(csvp)]) == 0
    assert "hot_kernel" in capsys.readouterr().out and "clean_kernel" not in csvp.read_text()
    assert A.main([str(p), "--match", "zzz"]) == 2
