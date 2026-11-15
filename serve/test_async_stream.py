"""STRATA_SERVE_ASYNC_STREAM (opt-in; LEADS.md Part D "server never blocks engine")."""
import unittest
from unittest import mock

from serve import async_stream


class AsyncStreamTest(unittest.TestCase):
    def test_off_returns_same_object(self):
        g = iter([1, 2])
        with mock.patch.object(async_stream, "_ON", False):
            self.assertIs(async_stream.wrap(g), g)

    def test_same_items_in_order(self):
        items = [None, {"a": 1}, None, {"b": 2}]
        self.assertEqual(list(async_stream.AsyncChunks(iter(items))), items)

    def test_exception_reraised_after_items(self):
        def gen():
            yield 1
            raise ValueError("ERR line")
        it = async_stream.AsyncChunks(gen())
        self.assertEqual(next(it), 1)
        with self.assertRaises(ValueError):
            next(it)
        with self.assertRaises(StopIteration):
            next(it)

    def test_close_stops_and_closes_generator(self):
        closed = []

        def gen():
            try:
                for i in range(1000000):
                    yield i
            finally:
                closed.append(True)
        it = async_stream.AsyncChunks(gen())
        self.assertEqual(next(it), 0)
        it.close()
        it._t.join(5)
        self.assertFalse(it._t.is_alive())
        self.assertEqual(closed, [True])


if __name__ == "__main__":
    unittest.main()
