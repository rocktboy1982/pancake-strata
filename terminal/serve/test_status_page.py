"""The project status page (serve/status_page.py): GET /status for a browser, GET /pancake, GET /status.json.  Ported from
pancake-strata (214c8cd31, 19ffe37ab) and extended with this project's state: branch / head, engine build, the tuned flags
of terminal/strata_control/tuned_flags.py and their state at start, the user's settings, the hardware, the flag-diagnostics
verdicts and the live counters.  Mock engine, no GPU, no network.

    python -m pytest terminal/serve/test_status_page.py -v
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
DASH = "—"
HW = {"gfx": "gfx1100", "rocm": "10.0.1", "vram_gib": 24.0, "pcie_link": "16.0 GT/s PCIe x16", "cpu": "Test CPU",
      "ram_gib": 62.7}
PROFILE = {"schema": "strata.tuned-flags/1", "profiles": [
    {"match": {"gfx": "gfx1100", "rocm": ["10."], "vram_gib": [16, 24]}, "source": "test profile",
     "flags": [
         {"kind": "env", "name": "STRATA_A", "value": "1", "verdict": "GAIN", "exact": True, "gain_pct": 4.0, "ledger": "T-1"},
         {"kind": "env", "name": "STRATA_B", "value": "1", "verdict": "GAIN", "exact": True, "gain_pct": 3.0, "ledger": "T-2"},
         {"kind": "env", "name": "STRATA_C", "value": "1", "verdict": "GAIN", "exact": True, "gain_pct": 3.0, "ledger": "T-3"},
         {"kind": "env", "name": "STRATA_D", "value": "1", "verdict": "GAIN", "exact": True, "gain_pct": 3.0, "ledger": "T-4"},
         {"kind": "env", "name": "STRATA_E", "value": "1", "verdict": "GAIN", "exact": True, "gain_pct": 3.0, "ledger": "T-5"},
         {"kind": "arg", "name": "--pcie-frac", "value": "0", "verdict": "GAIN", "exact": True, "gain_pct": 6.0, "ledger": "T-6"},
         {"kind": "arg", "name": "--lookup-max-t", "value": "12", "verdict": "GAIN", "exact": False, "gain_pct": 8.8,
          "quality_gate": "PASS", "ledger": "T-7"},
         # not accepted (no gain): never shown as a tuned flag
         {"kind": "env", "name": "STRATA_NOPE", "value": "1", "verdict": "NO-GAIN", "exact": True, "gain_pct": 0.5}]}]}


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
        self.assertEqual(d["project"]["name"], "updated-strata")
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
        self.info = {"version": "0.2.16", "expert_slots": 512, "windows": 3}


class TunedFlags(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        (t / "tuned.json").write_text(json.dumps(PROFILE), encoding="utf-8")
        (t / "run.json").write_text(json.dumps({"env": {"STRATA_B": "0"}, "args": ["--lookup-max-t", "6"]}), encoding="utf-8")
        engine_dir = t / "engine"
        engine_dir.mkdir()
        (engine_dir / "BUILD.json").write_text(json.dumps({"version": "0.2.16", "build_id": "abc123-rocm10"}), encoding="utf-8")
        self.env = {"STRATA_TUNED_FLAGS_FILE": str(t / "tuned.json"), "STRATA_C": "7"}
        self.svc = make_svc()
        self.svc.config_path = str(t / "run.json")
        # what the start made of it: A applied, B / C the user's, D changed by a saved tuning, E not set, --pcie-frac applied
        self.svc.engine = FakeEngine(str(engine_dir / "strata"), ["--pcie-frac", "0", "--lookup-max-t", "6"],
                                     {"STRATA_A": "1", "STRATA_B": "0", "STRATA_C": "7", "STRATA_D": "2"})

    def tearDown(self):
        self.tmp.cleanup()

    def report(self, env=None, argv=()):
        return status_page.tuned_flags_report(self.svc, HW, env=self.env if env is None else env, argv=list(argv))

    def test_each_flag_state(self):
        rep = self.report()
        self.assertTrue(rep["enabled"])
        self.assertTrue(rep["file"].endswith("tuned.json"))
        self.assertEqual(rep["source"], "test profile")
        state = {f["name"]: (f["status"], f["in_effect"]) for f in rep["flags"]}
        self.assertEqual(state, {"STRATA_A": ("applied", "1"), "STRATA_B": ("yours (run config)", "0"),
                                 "STRATA_C": ("yours (environment)", "7"), "STRATA_D": ("changed by a saved tuning", "2"),
                                 "STRATA_E": ("not applied", None), "--pcie-frac": ("applied", "0"),
                                 "--lookup-max-t": ("yours (run config)", "6")})
        self.assertNotIn("STRATA_NOPE", state)                  # a NO-GAIN record never enters a profile
        self.assertEqual(rep["state"], "2 of 7 applied at start")

    def test_off_switch(self):
        rep = self.report(env={**self.env, "STRATA_TUNED_FLAGS": "0"})
        self.assertFalse(rep["enabled"])
        self.assertIn("off", rep["state"])
        self.assertEqual(rep["flags"], [])
        self.assertFalse(self.report(argv=["--no-tuned-flags"])["enabled"])

    def test_unknown_machine(self):
        rep = status_page.tuned_flags_report(self.svc, dict(HW, gfx="gfx90a"), env=self.env, argv=[])
        self.assertIn("no measured profile for gfx90a", rep["state"])
        rep = status_page.tuned_flags_report(self.svc, None, env=self.env, argv=[])
        self.assertIn("no supported graphics card", rep["state"])

    def test_page_sections(self):
        self.svc.status_hardware = HW
        with mock.patch.dict(status_page.START_ENV, self.env, clear=True):
            d = status_page.build(self.svc)
        rows = section(d["view"], "tuned-flags")["rows"]
        self.assertEqual(rows[0], ["STRATA_A", "env", "1", "applied", "1", "yes", "4.0 %", "T-1"])
        self.assertEqual(rows[4][3:5], ["not applied", DASH])   # nothing in effect: the em dash
        state = kv(d["view"], "tuned-state")
        self.assertEqual(state["State"], "2 of 7 applied at start")
        self.assertTrue(state["Profile file"].endswith("tuned.json"))
        hw = kv(d["view"], "hardware")
        self.assertEqual((hw["Graphics architecture"], hw["ROCm"], hw["VRAM (largest card)"], hw["PCIe link"]),
                         ("gfx1100", "10.0.1", "24.0 GiB", "16.0 GT/s PCIe x16"))
        settings = section(d["view"], "settings")["rows"]
        self.assertIn(["STRATA_B", "0", "run config env"], settings)
        self.assertIn(["--lookup-max-t", "6", "run config args"], settings)
        self.assertIn(["STRATA_C", "7", "environment"], settings)
        project = kv(d["view"], "project")
        self.assertEqual(project["Engine version"], "0.2.16")
        self.assertEqual(project["Engine build"], "abc123-rocm10")
        facts = {r[0]: r[1] for r in section(d["view"], "facts")["rows"]}
        self.assertEqual(facts, {"expert_slots": "512", "windows": "3"})
        html = status_page.render_sections(d["view"])
        self.assertIn("<td>STRATA_A</td>", html)


class Verdicts(unittest.TestCase):
    def test_verdicts_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "verdicts.json"
            f.write_text(json.dumps({"schema": "strata.flag-diag/1", "complete": False, "minutes": 12.5, "verdicts": [
                {"name": "STRATA_PF_F16", "kind": "env", "verdict": "GAIN", "gain_pct": 10.3, "noise_pct": 1.2},
                {"name": "--pcie-frac", "kind": "arg", "verdict": "UNSUPPORTED", "reason": "PCIe link not readable"}]}),
                encoding="utf-8")
            svc = make_svc()
            with mock.patch.dict(status_page.START_ENV, {"STRATA_FLAG_DIAG_VERDICTS": str(f)}, clear=True):
                d = status_page.build(svc)
        s = section(d["view"], "verdicts")
        self.assertEqual(s["rows"], [["STRATA_PF_F16", "GAIN", "10.3 %", "1.2 %", DASH],
                                     ["--pcie-frac", "UNSUPPORTED", DASH, DASH, "PCIe link not readable"]])
        self.assertIn("incomplete", s["note"])

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
            (main / "HEAD").write_text("ref: refs/heads/updated-strata\n")
            (main / "refs" / "heads" / "updated-strata").write_text("0123456789abcdef0123\n")
            self.assertEqual(status_page.git_head(t / "main"), {"branch": "updated-strata", "head": "0123456789ab"})
            (main / "refs" / "heads" / "updated-strata").unlink()
            (main / "packed-refs").write_text("# pack-refs\nfedcba9876543210ffff refs/heads/updated-strata\n")
            self.assertEqual(status_page.git_head(t / "main" / "sub")["head"], "fedcba987654")
            wt = main / "worktrees" / "wt"
            wt.mkdir(parents=True)
            (wt / "HEAD").write_text("ref: refs/heads/updated-strata\n")
            (wt / "commondir").write_text("../..\n")
            (t / "wt").mkdir()
            (t / "wt" / ".git").write_text(f"gitdir: {wt}\n")
            self.assertEqual(status_page.git_head(t / "wt"), {"branch": "updated-strata", "head": "fedcba987654"})

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
