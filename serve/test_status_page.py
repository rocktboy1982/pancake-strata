"""The project status page (serve/status_page.py): GET /status for a browser, GET /pancake, GET /status.json.  Ported from
pancake-strata (214c8cd31, 19ffe37ab) and extended with this project's state: branch / head, engine build, the
rdna3/run.py setup, the recommended flags of rdna3/tuned_flags.py (rdna3/recommended-flags/<gfx>.json) and their state in
the running engine, the user's settings, the hardware, the flag-diagnostics verdicts and the live counters.  Mock engine,
no GPU, no network.

    python -m pytest serve/test_status_page.py -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import status_page  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DASH = "\u2014"
# rdna3/tuned_flags.py's names (pcie, not pcie_link)
HW = {"gfx": "gfx1100", "rocm": "7.2", "vram_gib": 24, "pcie": "16.0 GT/s PCIe x16", "cpu": "Test CPU", "ram_gib": 63,
      "vmm": True, "hipblaslt": False}
RECOMMENDED = {"schema": 1, "key": {"gfx": "gfx1100", "rocm": "", "vram_gib_min": 16},
               "env": {"STRATA_A": "1", "STRATA_B": "1", "STRATA_C": "1", "STRATA_D": "1"},
               "args": {"--lookup-max-t": "12", "--grammar": "on"},
               "verdicts": {"STRATA_A": {"env": {"STRATA_A": "1"}, "verdict": "GAIN", "exact": True, "gain_pct": 4.0,
                                         "note": "T-1"},
                            "--lookup-max-t 12": {"args": {"--lookup-max-t": "12"}, "verdict": "GAIN", "exact": False,
                                                  "gate_pass": True, "gain_pct": 8.8, "note": "T-126/T-127 " + "x" * 300}},
               "preset": {"STRATA_B": "daily preset note"}}


def make_svc(api_key=""):
    tok = ByteTokenizer()
    svc = Service(MockEngine(tok, "ok", max_context=8192), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
    svc.api_key = api_key
    svc.status_hardware = {}            # no hardware read in a test: "no AMD card read here"
    return svc


def get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, dict(r.headers), r.read()


def section(view, sid):
    return next(s for s in view if s["id"] == sid)


def kv(view, sid):
    return {r[0]: r[1] for r in section(view, sid)["rows"]}


class Routes(unittest.TestCase):
    def setUp(self):
        self.svc = make_svc()
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def chat(self):
        body = {"model": "m", "max_tokens": 20, "messages": [{"role": "user", "content": "hi"}]}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())

    def test_status_json_shape(self):
        status, headers, raw = get(self.base + "/status.json")
        self.assertEqual(status, 200)
        self.assertTrue(headers["Content-Type"].startswith("application/json"))
        d = json.loads(raw)
        for key in ("project", "hardware", "tuned_flags", "settings", "verdicts", "live", "engine_facts", "hardware_now",
                    "view", "time"):
            self.assertIn(key, d)
        self.assertEqual(d["project"]["name"], "updated-strata-rdna3")
        self.assertEqual([s["id"] for s in d["view"]],
                         ["project", "hardware", "tuned-state", "tuned-flags", "settings", "verdicts", "speed", "caches",
                          "facts", "hardware-now"])
        for s in d["view"]:                                   # every cell is text
            for row in s["rows"]:
                self.assertTrue(all(isinstance(c, str) for c in row), (s["id"], row))

    def test_plain_status_stays_the_api_json(self):
        status, headers, raw = get(self.base + "/status")
        self.assertTrue(headers["Content-Type"].startswith("application/json"))
        self.assertIn("busy", json.loads(raw))

    def test_browser_gets_the_page_drawn_on_the_server(self):
        for path, hdr in (("/status", {"Accept": "text/html,application/xhtml+xml"}), ("/status?html=1", {}), ("/pancake", {})):
            status, headers, raw = get(self.base + path, hdr)
            self.assertEqual(status, 200, path)
            self.assertTrue(headers["Content-Type"].startswith("text/html"), path)
            self.assertIn("Content-Security-Policy", headers)
            page = raw.decode("utf-8")
            self.assertNotIn(status_page.MARK, page)          # the sections are filled in: readable with JavaScript off
            self.assertIn('id="s-project"', page)
            self.assertIn('id="s-tuned-flags"', page)
            self.assertIn("<noscript>", page)

    def test_missing_values_are_an_em_dash(self):
        d = json.loads(get(self.base + "/status.json")[2])
        project = kv(d["view"], "project")
        self.assertEqual(project["Engine binary"], DASH)        # the mock engine has no binary
        self.assertEqual(project["Engine build"], DASH)
        self.assertEqual(kv(d["view"], "hardware")["Graphics architecture"], DASH)
        self.assertEqual(kv(d["view"], "speed")["Decode now"], DASH)
        self.assertEqual(kv(d["view"], "caches")["Expert cache hit, mean"], DASH)
        page = get(self.base + "/pancake")[2].decode("utf-8")
        self.assertIn(f"<td>{DASH}</td>", page)                  # the character itself, not "&mdash;" text
        self.assertNotIn("&amp;mdash;", page)
        self.assertNotIn("&mdash;", page)

    def test_counters_after_a_turn(self):
        self.chat()
        d = json.loads(get(self.base + "/status.json")[2])
        self.assertEqual(d["live"]["requests"], 1)
        self.assertGreater(d["live"]["prompt_tokens"], 0)
        self.assertEqual(d["live"]["state"], "idle")
        self.assertEqual(kv(d["view"], "speed")["Requests finished"], "1")

    def test_api_key(self):
        self.svc.api_key = "sekrit"
        with self.assertRaises(urllib.error.HTTPError) as e:
            get(self.base + "/status.json")
        self.assertEqual(e.exception.code, 401)
        page = get(self.base + "/pancake")[2].decode("utf-8")
        self.assertIn('id="locked"', page)                      # the page, without the values
        self.assertNotIn('id="s-project"', page)
        page = get(self.base + "/pancake", {"Authorization": "Bearer sekrit"})[2].decode("utf-8")
        self.assertIn('id="s-project"', page)
        self.assertEqual(get(self.base + "/status.json", {"Authorization": "Bearer sekrit"})[0], 200)

    def test_get_only(self):
        req = urllib.request.Request(self.base + "/status.json", data=b"{}", method="POST",
                                     headers={"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=10)
        self.assertGreaterEqual(e.exception.code, 400)


class FakeEngine:
    """What the report reads of a started engine: its spawn tuple (binary, args, cwd, log, env)."""

    def __init__(self, exe, args, env):
        self.spawn = (exe, list(args), None, None, env, False, None)
        self.info = {"version": "0.1.48", "expert_slots": 512, "windows": 3}


class TunedFlags(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.flags_dir = t / "recommended-flags"
        self.flags_dir.mkdir()
        (self.flags_dir / "gfx1100.json").write_text(json.dumps(RECOMMENDED), encoding="utf-8")
        configs = t / "configs"
        configs.mkdir()
        (configs / "iq4-2gpu.json").write_text(json.dumps({"env": {"STRATA_D": "1"}, "args": ["--grammar", "off"]}),
                                               encoding="utf-8")
        state = t / "rdna3" / "state"
        state.mkdir(parents=True)
        # the run config rdna3/run.py wrote: the setup plus the recommended flags
        (state / "iq4-2gpu-8080.json").write_text(json.dumps({"env": {"STRATA_A": "1", "STRATA_D": "1"},
                                                              "args": ["--lookup-max-t", "12", "--grammar", "off"]}),
                                                  encoding="utf-8")
        engine_dir = t / "engine"
        engine_dir.mkdir()
        (engine_dir / "BUILD.json").write_text(json.dumps({"version": "0.1.48", "source": "local-hip", "backend": "hip"}),
                                               encoding="utf-8")
        self.env = {"STRATA_C": "7"}
        self.svc = make_svc()
        self.svc.config_path = str(state / "iq4-2gpu-8080.json")
        # A applied, B not applied, C the user's, D the setup's default too, --lookup-max-t applied, --grammar different
        self.svc.engine = FakeEngine(str(engine_dir / "strata"), ["--lookup-max-t", "12", "--grammar", "off"],
                                     {"STRATA_A": "1", "STRATA_C": "7", "STRATA_D": "1"})
        self.configs = mock.patch.object(status_page, "CONFIGS", configs)
        self.configs.start()

    def tearDown(self):
        self.configs.stop()
        self.tmp.cleanup()

    def report(self, env=None, hw=HW):
        return status_page.tuned_flags_report(self.svc, status_page.normalized_hardware(hw),
                                              env=self.env if env is None else env, directory=self.flags_dir)

    def test_each_flag_state(self):
        rep = self.report()
        self.assertTrue(rep["enabled"])
        self.assertTrue(rep["file"].endswith("gfx1100.json"))
        state = {f["name"]: (f["status"], f["in_effect"]) for f in rep["flags"]}
        self.assertEqual(state, {
            "STRATA_A": ("applied", "1"),
            "STRATA_B": ("not applied (run.py --no-tuned-flags, or a start without run.py)", None),
            "STRATA_C": ("yours (environment)", "7"),
            "STRATA_D": ("in effect (the setup's own default too)", "1"),
            "--lookup-max-t": ("applied", "12"),
            "--grammar": ("different value in the run config", "off")})
        self.assertEqual(rep["state"], "3 of 6 in effect")
        a = next(f for f in rep["flags"] if f["name"] == "STRATA_A")
        self.assertEqual((a["verdict"], a["gain_pct"], a["note"]), ("GAIN", 4.0, "T-1"))
        b = next(f for f in rep["flags"] if f["name"] == "STRATA_B")
        self.assertEqual((b["verdict"], b["note"]), (None, "daily preset note"))
        lookup = next(f for f in rep["flags"] if f["name"] == "--lookup-max-t")
        self.assertLessEqual(len(lookup["note"]), 160)            # a long note is cut, with an ellipsis
        self.assertTrue(lookup["note"].endswith("\u2026"))

    def test_before_the_engine_started_the_run_config_counts(self):
        self.svc.engine = make_svc().engine                       # the mock engine: no spawn
        state = {f["name"]: f["status"] for f in self.report()["flags"]}
        self.assertEqual(state["STRATA_A"], "applied")
        self.assertEqual(state["--grammar"], "different value in the run config")

    def test_off_switch(self):
        rep = self.report(env={"STRATA_TUNED_FLAGS": "0"})
        self.assertFalse(rep["enabled"])
        self.assertIn("off", rep["state"])
        self.assertEqual(rep["flags"], [])

    def test_unknown_machine(self):
        self.assertIn("no recommended-flags file for gfx90a", self.report(hw=dict(HW, gfx="gfx90a"))["state"])
        self.assertIn("no supported graphics card", self.report(hw=dict(HW, gfx=""))["state"])

    def test_page_sections(self):
        self.svc.status_hardware = HW
        with mock.patch.dict(status_page.START_ENV, self.env, clear=True), \
                mock.patch.object(status_page, "rdna3_flags", self.flags_module()):
            d = status_page.build(self.svc)
        rows = section(d["view"], "tuned-flags")["rows"]
        self.assertEqual(rows[0], ["STRATA_A", "env", "1", "applied", "1", DASH, "GAIN", "4.0 %", "T-1"])
        self.assertEqual(rows[1][3:9], ["not applied (run.py --no-tuned-flags, or a start without run.py)", DASH, DASH,
                                        DASH, DASH, "daily preset note"])
        state = kv(d["view"], "tuned-state")
        self.assertEqual(state["State"], "3 of 6 in effect")
        hw = kv(d["view"], "hardware")
        self.assertEqual((hw["Graphics architecture"], hw["ROCm"], hw["VRAM (largest card)"], hw["PCIe link"],
                          hw["VMM (hipMemCreate)"], hw["hipBLASLt"]),
                         ("gfx1100", "7.2", "24 GiB", "16.0 GT/s PCIe x16", "yes", "no"))
        self.assertEqual(section(d["view"], "settings")["rows"], [["STRATA_C", "7", "environment"]])
        project = kv(d["view"], "project")
        self.assertEqual(project["rdna3/run.py setup"], "iq4-2gpu")
        self.assertEqual((project["Engine version"], project["Engine build"]), ("0.1.48", "local-hip"))
        facts = {r[0]: r[1] for r in section(d["view"], "facts")["rows"]}
        self.assertEqual(facts, {"expert_slots": "512", "windows": "3"})
        self.assertIn("<td>STRATA_A</td>", status_page.render_sections(d["view"]))

    def flags_module(self):
        """rdna3/tuned_flags.py with its folder pointed at the test's recommended file."""
        real = status_page.rdna3_flags()
        real.DIR = self.flags_dir
        return lambda: real

    def test_shipped_file_reads(self):
        """The real rdna3/recommended-flags/gfx1100.json loads and every flag gets a state (none raises)."""
        rep = status_page.tuned_flags_report(self.svc, status_page.normalized_hardware(dict(HW, rocm="")), env={})
        self.assertTrue(rep["file"].endswith("gfx1100.json"), rep["state"])
        self.assertTrue(rep["flags"])
        self.assertTrue(all(f["status"] for f in rep["flags"]))


class Verdicts(unittest.TestCase):
    def test_verdicts_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "verdicts.json"
            f.write_text(json.dumps({
                "PF16": {"env": {"STRATA_PF_F16": "1"}, "verdict": "GAIN", "exact": False, "gate_pass": True, "gain_pct": 10.27},
                "VMM": {"env": {"STRATA_VMM": "1"}, "verdict": "UNSUPPORTED", "exact": True, "note": "hardware lacks: vmm"}}),
                encoding="utf-8")
            with mock.patch.dict(status_page.START_ENV, {"STRATA_FLAG_DIAG_VERDICTS": str(f)}, clear=True):
                d = status_page.build(make_svc())
        s = section(d["view"], "verdicts")
        self.assertEqual(s["rows"], [["PF16", "GAIN", "10.3 %", "no", DASH],
                                     ["VMM", "UNSUPPORTED", DASH, "yes", "hardware lacks: vmm"]])
        self.assertIn(f"({DASH},", s["note"])                     # completeness not recorded: the em dash

    def test_no_verdicts_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(status_page.START_ENV, {"STRATA_FLAG_DIAG_VERDICTS": str(Path(tmp) / "none.json")}, clear=True):
                d = status_page.build(make_svc())
        s = section(d["view"], "verdicts")
        self.assertEqual(s["rows"], [])
        self.assertIn("no verdicts file", s["note"])
        self.assertIn('class="empty">No verdicts<', status_page.render_sections([s]))


class Helpers(unittest.TestCase):
    def test_fmt(self):
        self.assertEqual(status_page.fmt(None), DASH)
        self.assertEqual(status_page.fmt(""), DASH)
        self.assertEqual(status_page.fmt(float("nan")), DASH)
        self.assertEqual(status_page.fmt(None, unit="GiB"), DASH)
        self.assertEqual(status_page.fmt(12345), "12,345")
        self.assertEqual(status_page.fmt(3.14159, 2, "tok/s"), "3.14 tok/s")
        self.assertEqual(status_page.pct(None), DASH)
        self.assertEqual(status_page.duration(None), DASH)
        self.assertEqual(status_page.duration(3725), "62 min 5 s")

    def test_git_head_branch_packed_and_worktree(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            main = t / "main" / ".git"
            (main / "refs" / "heads").mkdir(parents=True)
            (main / "HEAD").write_text("ref: refs/heads/updated-strata-rdna3\n")
            (main / "refs" / "heads" / "updated-strata-rdna3").write_text("0123456789abcdef0123\n")
            self.assertEqual(status_page.git_head(t / "main"), {"branch": "updated-strata-rdna3", "head": "0123456789ab"})
            (main / "refs" / "heads" / "updated-strata-rdna3").unlink()
            (main / "packed-refs").write_text("# pack-refs\nfedcba9876543210ffff refs/heads/updated-strata-rdna3\n")
            self.assertEqual(status_page.git_head(t / "main" / "sub")["head"], "fedcba987654")
            wt = main / "worktrees" / "wt"
            wt.mkdir(parents=True)
            (wt / "HEAD").write_text("ref: refs/heads/updated-strata-rdna3\n")
            (wt / "commondir").write_text("../..\n")
            (t / "wt").mkdir()
            (t / "wt" / ".git").write_text(f"gitdir: {wt}\n")
            self.assertEqual(status_page.git_head(t / "wt"), {"branch": "updated-strata-rdna3", "head": "fedcba987654"})

    def test_page_files_need_no_network(self):
        for name in ("status.html", "status.js"):
            text = (ROOT / "serve" / "web" / name).read_text(encoding="utf-8")
            self.assertNotIn("http://", text, name)
            self.assertNotIn("https://", text, name)
        self.assertEqual((ROOT / "serve" / "web" / "status.html").read_text(encoding="utf-8").count(status_page.MARK), 1)
        js = (ROOT / "serve" / "web" / "status.js").read_text(encoding="utf-8")
        self.assertNotIn("innerHTML", js)                        # cells are set as text
        self.assertNotIn("method:", js)                          # GET only


if __name__ == "__main__":
    unittest.main()
