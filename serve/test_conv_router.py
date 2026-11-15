"""STRATA_CONV_ROUTE / STRATA_TUNE (opt-in; Strata PR #5516, LEADS.md "POST /v1/tune + --conv-route"): the
traffic-shape router, the TUNE line both ways, and the Service decision with a fake engine."""
import io
import queue
import threading
import unittest
from unittest import mock

from serve import conv_router
from serve.server import EngineDied, Service, StrataEngine


def _single(text):
    return [{"role": "user", "content": text}]


class RouterTest(unittest.TestCase):
    def test_run_of_single_shots_switches_off_and_chat_switches_back(self):
        r = conv_router.Router(cache_on=6, batch_after=3)
        d = [r.decide(_single(f"prompt {i}")) for i in range(3)]
        self.assertEqual([x.changed for x in d], [False, False, True])
        self.assertEqual(d[2].wanted_cache, 0)
        self.assertEqual(r.mode, conv_router.BATCH)
        self.assertFalse(r.decide(_single("prompt 9")).changed)           # already batch
        back = r.decide([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                         {"role": "user", "content": "c"}])
        self.assertTrue(back.changed)
        self.assertEqual(back.wanted_cache, 6)
        self.assertEqual(r.report()["switches_off"], 1)
        self.assertEqual(r.report()["switches_on"], 1)

    def test_repeat_prefix_and_tools_keep_chat(self):
        r = conv_router.Router(batch_after=2)
        r.decide(_single("same"))
        self.assertEqual(r.decide(_single("same")).reason, "repeat prefix")
        self.assertEqual(r.decide(_single("other"), tools=[{"name": "x"}]).reason, "tools")
        self.assertEqual(r.mode, conv_router.CHAT)

    def test_revert(self):
        r = conv_router.Router(batch_after=1)
        self.assertTrue(r.decide(_single("x")).changed)
        r.revert()
        self.assertEqual(r.mode, conv_router.CHAT)


class ProtocolTest(unittest.TestCase):
    def test_tune_line(self):
        self.assertEqual(conv_router.tune_line({"prompt_cache": 0, "spec-min-p": 0.4}),
                         "TUNE prompt_cache=0 spec_min_p=0.4")
        with self.assertRaises(ValueError):
            conv_router.tune_line({})
        with self.assertRaises(ValueError):
            conv_router.tune_line({"a b": 1})

    def test_parse_tuned(self):
        out = conv_router.parse_tuned("TUNED prompt_cache=0 prefill_chunk=512(capped at the startup chunk) "
                                      "refused: spec=4(unknown key, or a value it cannot take)")
        self.assertEqual(out["applied"], {"prompt_cache": 0, "prefill_chunk": 512})
        self.assertEqual(out["refused"], ["spec=4"])
        self.assertEqual(conv_router.parse_tuned("TUNED none"), {"applied": {}, "refused": []})

    def test_engine_round_trip(self):
        e = StrataEngine("missing-executable", [], lazy=True)
        e.proc = mock.Mock()
        e.proc.poll.return_value = None
        e.proc.stdin = io.StringIO()
        e.ended = False
        e.lines = queue.Queue()
        e.lines.put("strata serve: noise\n")
        e.lines.put("TUNED spec_min_p=0.400000\n")
        self.assertEqual(e.tune({"spec_min_p": 0.4}), {"applied": {"spec_min_p": 0.4}, "refused": []})
        self.assertEqual(e.proc.stdin.getvalue(), "TUNE spec_min_p=0.4\n")
        e.lines.put(None)
        with self.assertRaises(EngineDied):
            e.tune({"spec_min_p": 0.4})


class _FakeSvc:
    """Just the attributes Service.tune / Service.conv_route_decide read."""
    def __init__(self, applied=True):
        self.fifo = threading.Lock()
        self.calls = []

        class _E:
            batch = 0

            def tune(_, settings):
                self.calls.append(dict(settings))
                return {"applied": dict(settings) if applied else {}, "refused": []}
        self.engine = _E()

    def loaded(self):
        return True

    tune = Service.tune
    conv_route_decide = Service.conv_route_decide


class ServiceTest(unittest.TestCase):
    def _cfg(self, mode, batch_after=2):
        return mock.patch.object(conv_router, "_CFG", {"mode": mode, "batch_after": batch_after, "cache_on": 6,
                                                       "tune": False})

    def test_off_never_touches_the_engine(self):
        svc = _FakeSvc()
        with self._cfg("off"):
            for i in range(5):
                svc.conv_route_decide({"messages": _single(f"p{i}")})
        self.assertEqual(svc.calls, [])
        self.assertNotIn("conv_router", svc.__dict__)

    def test_default_env_is_off(self):
        with mock.patch.object(conv_router, "_CFG", None), mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(conv_router.config()["mode"], "off")
            self.assertFalse(conv_router.config()["tune"])

    def test_on_switches_off_then_back(self):
        svc = _FakeSvc()
        with self._cfg("on"):
            svc.conv_route_decide({"messages": _single("a")})
            svc.conv_route_decide({"messages": _single("b")})
            svc.conv_route_decide({"messages": _single("a"), "tools": [{"name": "t"}]})
        self.assertEqual(svc.calls, [{"prompt_cache": 0}, {"prompt_cache": 6}])
        self.assertEqual(svc.conv_route_stats["switches"], 2)

    def test_shadow_only_logs(self):
        svc = _FakeSvc()
        with self._cfg("shadow"):
            for i in range(4):
                svc.conv_route_decide({"messages": _single(f"p{i}")})
        self.assertEqual(svc.calls, [])

    def test_refused_reverts(self):
        svc = _FakeSvc(applied=False)
        with self._cfg("on", batch_after=1):
            svc.conv_route_decide({"messages": _single("a")})
        self.assertEqual(svc.conv_router.mode, conv_router.CHAT)
        self.assertEqual(svc.conv_route_stats["refused"], 1)


if __name__ == "__main__":
    unittest.main()
