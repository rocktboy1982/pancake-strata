"""STRATA_SERVE_ASYNC_STREAM=1 (opt-in; unset or any other value = the stream is read and written on one thread, as before).

"Server never blocks the engine" (LEADS.md Part D, llm-runtimes report): with one thread, a slow client socket stalls
the chunk generator, the generator stops reading the engine's pipe, and the engine blocks on its own write.  Here a
reader thread drains the generator into a queue at the engine's pace and the request thread writes from the queue.
The same items in the same order; exceptions raised by the generator (EngineDied, an ERR line) are re-raised on the
request thread at the same point of the stream.  close() asks the reader to stop and close the generator after the
item it is waiting for (the engine itself is stopped by the caller's cancel event, as before).
"""
from __future__ import annotations

import os
import queue
import threading

_ON: bool | None = None


def on() -> bool:
    global _ON
    if _ON is None:
        _ON = os.environ.get("STRATA_SERVE_ASYNC_STREAM", "").strip() == "1"
    return _ON


class AsyncChunks:
    _ITEM, _ERROR, _END = 0, 1, 2

    def __init__(self, gen):
        self._gen = gen
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._done = False
        self._t = threading.Thread(target=self._run, name="strata-stream-reader", daemon=True)
        self._t.start()

    def _run(self):
        try:
            for c in self._gen:
                self._q.put((self._ITEM, c))
                if self._stop.is_set():
                    break
        except BaseException as e:  # noqa: BLE001 - handed to the request thread unchanged
            self._q.put((self._ERROR, e))
            return
        if self._stop.is_set():
            close = getattr(self._gen, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass
        self._q.put((self._END, None))

    def __iter__(self):
        return self

    def __next__(self):
        if self._done:
            raise StopIteration
        kind, v = self._q.get()
        if kind == self._ITEM:
            return v
        self._done = True
        if kind == self._ERROR:
            raise v
        raise StopIteration

    def close(self):
        self._stop.set()


def wrap(gen):
    """`gen` itself when the switch is off, else an AsyncChunks over it."""
    return AsyncChunks(gen) if on() else gen
