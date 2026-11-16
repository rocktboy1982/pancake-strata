"""LEADS.md "bugs 5.5-5.10", 5.10 (aktalascloud/strata-legacy-gpu 3da17a80fe): the disconnect watcher and a finished
JSON response.  The real Handler._watch_client and Handler._json are taken out of server.py and run against a fake
socket: a client that closes right after the last body byte is not a disconnect, a client that closes before the
answer is, and a failed write still raises to the caller.  No model, engine or network."""
import ast
import json
import socket
import threading
import types
import unittest
from pathlib import Path


def _methods(source):
    wanted = {}
    for n in ast.walk(ast.parse(source)):
        if isinstance(n, ast.FunctionDef) and n.name in {"_watch_client", "_json"}:
            wanted.setdefault(n.name, n)
    assert set(wanted) == {"_watch_client", "_json"}
    return list(wanted.values())


def probe(source):
    rows = []
    for case in ("complete_then_close", "early_EOF", "write_failure"):
        closed, exited, cancel = threading.Event(), threading.Event(), threading.Event()
        svc = types.SimpleNamespace(status_lock=threading.RLock(), active_cancels=set(), disconnects=0)

        def disconnected(event, svc=svc):
            svc.disconnects += 1
            event.set()

        svc.note_client_disconnect = disconnected
        real_thread = threading.Thread

        def thread(*args, exited=exited, **kw):
            target = kw["target"]

            def observed():
                try:
                    target()
                finally:
                    exited.set()

            kw["target"] = observed
            return real_thread(*args, **kw)

        env = dict(threading=types.SimpleNamespace(Event=threading.Event, Lock=threading.Lock, Thread=thread),
                   socket=socket, json=json, svc=svc,
                   select=types.SimpleNamespace(select=lambda *a, closed=closed: ([1] if closed.is_set() else [], [], [])))
        cls = ast.ClassDef(name="Handler", bases=[], keywords=[], body=_methods(source), decorator_list=[],
                           type_params=[])
        module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
        exec(compile(module, "server-handler", "exec"), env)
        h = env["Handler"]()
        h.path, h.record, h.capture = "/v1/chat/completions", None, None
        h.watch_done = h.watch_lock = None
        h.connection = types.SimpleNamespace(recv=lambda *a: b"")
        h._note = lambda **kw: setattr(h, "outcome", kw.get("outcome"))
        h.send_response = h.send_header = lambda *a: None
        h.end_headers = h._cors = lambda: None

        def write(data, case=case, closed=closed, exited=exited):
            if case == "write_failure":
                raise BrokenPipeError("write failure")
            closed.set()                                  # the client closes right after the last byte
            assert exited.wait(3), "the watcher did not end after the body"

        h.wfile = types.SimpleNamespace(write=write)
        h._watch_client(cancel)
        write_failed = False
        try:
            if case == "early_EOF":
                closed.set()
                assert exited.wait(3)
            else:
                try:
                    h._json(200, {"choices": []})
                except BrokenPipeError:
                    write_failed = True
            rows.append(dict(case=case, cancelled=cancel.is_set(), disconnects=svc.disconnects,
                             write_error_propagated=write_failed))
        finally:
            h.watch_done.set()
            assert exited.wait(3)
    return rows


class JsonWatcherRetirement(unittest.TestCase):
    def test_completed_body_early_eof_and_write_failure(self):
        rows = probe(Path(__file__).with_name("server.py").read_text(encoding="utf-8-sig"))
        self.assertEqual(rows, [
            dict(case="complete_then_close", cancelled=False, disconnects=0, write_error_propagated=False),
            dict(case="early_EOF", cancelled=True, disconnects=1, write_error_propagated=False),
            dict(case="write_failure", cancelled=False, disconnects=0, write_error_propagated=True),
        ])


if __name__ == "__main__":
    unittest.main()
