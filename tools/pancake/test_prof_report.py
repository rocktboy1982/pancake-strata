"""Tests for prof_report.py (T11) and timeline_gaps.py (T12) on fixture rocprofv3 files (CSV and JSON); no GPU, no ROCm."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prof_report as P  # noqa: E402
import timeline_gaps as T  # noqa: E402

KT = """\
"Kind","Agent_Id","Queue_Id","Kernel_Id","Kernel_Name","Correlation_Id","Start_Timestamp","End_Timestamp","LDS_Block_Size"
"KERNEL_DISPATCH","Agent 1","1","10","rdna_kq(float*)","1","1000000","1600000","0"
"KERNEL_DISPATCH","Agent 1","1","11","wait_flag_ge(int*)","2","1610000","2210000","0"
"KERNEL_DISPATCH","Agent 1","1","10","rdna_kq(float*)","3","2215000","2415000","0"
"KERNEL_DISPATCH","Agent 1","1","12","combine(float*)","4","4415000","4515000","0"
"""
CC = """\
"Correlation_Id","Dispatch_Id","Kernel_Name","Counter_Name","Counter_Value","Start_Timestamp","End_Timestamp"
"1","1","rdna_kq(float*)","SQ_WAVES","100","1000000","1600000"
"1","1","rdna_kq(float*)","GRBM_GUI_ACTIVE","5000","1000000","1600000"
"3","3","rdna_kq(float*)","SQ_WAVES","300","2215000","2415000"
"3","3","rdna_kq(float*)","GRBM_GUI_ACTIVE","0","2215000","2415000"
"4","4","combine(float*)","SQ_WAVES","0","4415000","4515000"
"""
API = """\
"Kind","Function","Process_Id","Thread_Id","Correlation_Id","Start_Timestamp","End_Timestamp"
"HIP_RUNTIME_API","hipLaunchKernel","1","7","1","900000","1000000"
"HIP_RUNTIME_API","hipEventRecord","1","7","2","1600000","1605000"
"HIP_RUNTIME_API","hipEventSynchronize","1","7","5","2500000","4400000"
"HIP_RUNTIME_API","hipStreamWaitEvent","1","7","6","1610000","1611000"
"""


def mk(tmp_path):
    (tmp_path / "kernel_trace.csv").write_text(KT)
    (tmp_path / "counter_collection.csv").write_text(CC)
    (tmp_path / "hip_api_trace.csv").write_text(API)
    return tmp_path


def test_rank_by_time(tmp_path):
    d = P.load([mk(tmp_path)])
    rows = P.rank(d["kernels"])
    assert [r["name"] for r in rows][:2] == ["rdna_kq(float*)", "wait_flag_ge(int*)"]
    assert rows[0]["calls"] == 2 and abs(rows[0]["ns"] - 800000) < 1
    assert abs(sum(r["share"] for r in rows) - 1) < 1e-9
    assert len(d["kernels"]) == 4          # counter-file rows do not double the trace rows


def test_counters_zero_and_mean(tmp_path, capsys):
    d = P.load([mk(tmp_path)])
    t = P.counter_table(d["counters"])
    assert t["rdna_kq(float*)"]["SQ_WAVES"]["mean"] == 200
    assert t["combine(float*)"]["SQ_WAVES"]["sum"] == 0
    assert P.main([str(tmp_path), "--counters", "SQ_WAVES"]) == 0
    out = capsys.readouterr().out
    assert "zero" in out and "200.0" in out


def test_counter_only_dir_dedups_dispatch(tmp_path):
    (tmp_path / "counter_collection.csv").write_text(CC)
    d = P.load([tmp_path])
    assert len(d["kernels"]) == 3          # one per dispatch, not per counter row


def test_nothing(tmp_path):
    (tmp_path / "x.csv").write_text("a,b\n1,2\n")
    assert P.main([str(tmp_path)]) == 2


def test_json_layout(tmp_path):
    j = {"rocprofiler-sdk-tool": [{"kernel_symbols": [{"kernel_id": 10, "formatted_kernel_name": "kk"}],
         "strings": {"buffer_records": [{"kind": 5, "operations": ["hipA", "hipEventSynchronize"]}]},
         "buffer_records": {"kernel_dispatch": [{"start_timestamp": 10, "end_timestamp": 30, "dispatch_info": {"kernel_id": 10}, "correlation_id": {"internal": 1}}],
                            "hip_api": [{"kind": 5, "operation": 1, "start_timestamp": 30, "end_timestamp": 90, "thread_id": 3}]}}]}
    f = tmp_path / "a_results.json"
    f.write_text(json.dumps(j))
    d = P.load([f])
    assert d["kernels"][0]["name"] == "kk" and d["api"][0]["name"] == "hipEventSynchronize"


def test_timeline_gaps(tmp_path, capsys):
    mk(tmp_path)
    d = P.load([tmp_path])
    r = T.analyse(d)
    assert r["kernels"] == 4
    assert abs(r["wall_ms"] - 3.515) < 1e-6
    b = {x["bucket"]: x for x in r["buckets"]}
    assert b["<10us"]["count"] == 1 and b["10-50us"]["count"] == 1 and b[">1ms"]["count"] == 1
    # wait_flag is 600us of 1500us kernel time
    assert abs(r["wait_kernel_share_of_kernel_time"] - 600 / 1500) < 1e-6
    # hipEventSynchronize 2.5 ms..4.4 ms covers the >1 ms gap 2.415..4.415 from 2.5 to 4.415 (clipped by 4.4)
    assert r["host_wait_ms"] > 1.8 and r["gap_covered_by_wait_ms"] > 1.8
    assert r["event_records"] == 1 and r["wait_events"] == 1
    assert r["top_gaps"][0]["prev"] == "rdna_kq(float*)" and r["top_gaps"][0]["next"] == "combine(float*)"
    assert T.main([str(tmp_path)]) == 0 and "host wait" in capsys.readouterr().out


def test_window_regex(tmp_path):
    mk(tmp_path)
    d = P.load([tmp_path])
    r = T.analyse(d, w_from="wait_flag", w_to="combine")
    assert r["kernels"] == 3
    assert T.analyse(d, w_from="nothing_matches") == {}
    assert T.main([str(tmp_path), "--window-from", "zzz"]) == 2
