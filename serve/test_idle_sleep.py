"""LEADS.md B74, sleep in place (opt-in --idle-sleep; basalt 64167ba73): the service's sleep and wake around a fake
engine (no driver, no GPU).  Off by default: a service without idle_sleep_s never sleeps."""
import time
import unittest
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service

ROOT = Path(__file__).resolve().parents[1]


class SleepyEngine(MockEngine):
    """A mock engine with StrataEngine's sleep interface."""

    def __init__(self, *a, can=True, refuse=None, **kw):
        super().__init__(*a, **kw)
        self.running, self.asleep, self.restore_s = True, False, 0.0
        self.can, self.refuse, self.calls = can, refuse, []

    def alive(self):
        return self.running

    def unload(self):
        self.running = False

    def restart(self):
        self.running = True

    def can_sleep(self):
        return self.can

    def sleep(self, ram_check=None):
        self.calls.append("sleep")
        why = self.refuse or (ram_check() if ram_check is not None else None)
        if why is not None:
            raise ValueError(why)
        self.asleep = True
        return {}

    def wake(self):
        self.calls.append("wake")
        self.asleep = False
        return {}


def service(engine):
    tok = ByteTokenizer()
    svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
    svc._sleep_fits = lambda: None                      # (no /proc/meminfo on every test box)
    return svc


class IdleSleep(unittest.TestCase):
    def make(self, **kw):
        return SleepyEngine(ByteTokenizer(), "</think>\n\nok", max_context=4096, **kw)

    def test_sleep_and_wake_on_the_next_load(self):
        eng = self.make()
        svc = service(eng)
        self.assertEqual(svc.sleep(), "asleep")
        self.assertTrue(eng.asleep)
        self.assertEqual(svc.sleep(), "asleep")             # already: nothing sent again
        self.assertEqual(eng.calls, ["sleep"])
        self.assertEqual(svc.metrics()["live"]["state"], "asleep")
        svc.load()                                         # the next request wakes it
        self.assertFalse(eng.asleep)
        self.assertEqual(eng.calls, ["sleep", "wake"])

    def test_refusals(self):
        self.assertEqual(service(self.make(can=False)).sleep(), "unsupported")
        self.assertEqual(service(self.make(refuse="SLEEP needs an engine started with --vram-elastic")).sleep(),
                         "unsupported")
        self.assertEqual(service(self.make(refuse="RAM: 100 MiB free")).sleep(), "no ram")
        eng = self.make()
        svc = service(eng)
        with svc.fifo:                                     # a request runs
            self.assertEqual(svc.sleep(), "busy")
        svc.last_request_at = time.time()
        self.assertEqual(svc.sleep(idle_for=60), "busy")   # one ran just now
        eng.running = False
        self.assertEqual(svc.sleep(), "not loaded")
        self.assertEqual(eng.calls, [])

    def test_unload_wakes_first(self):
        eng = self.make()
        svc = service(eng)
        self.assertEqual(svc.sleep(), "asleep")
        self.assertEqual(svc.unload(), "unloaded")
        self.assertEqual(eng.calls, ["sleep", "wake"])
        self.assertFalse(eng.running)

    def test_off_by_default(self):
        eng = self.make()
        svc = service(eng)
        self.assertEqual(svc.idle_sleep_s, 0)
        svc.start_idle_sleep()                             # no thread, nothing sent
        time.sleep(0.05)
        self.assertEqual(eng.calls, [])


if __name__ == "__main__":
    unittest.main()
