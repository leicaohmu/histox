"""Tests for slide prefetching."""

import os
import tempfile
import threading
import unittest
from unittest import mock

from histox.util import prefetch as pf


class TestPrefetch(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.paths = []
        for i in range(4):
            p = os.path.join(self.tmp.name, f's{i}.bin')
            with open(p, 'wb') as f:
                f.write(os.urandom(1024 * (i + 1)))
            self.paths.append(p)

    def tearDown(self):
        self.tmp.cleanup()

    def test_prefetch_file_reads_file(self):
        self.assertTrue(pf.prefetch_file(self.paths[0]))

    def test_missing_or_directory_is_skipped(self):
        self.assertFalse(pf.prefetch_file(os.path.join(self.tmp.name, 'nope')))
        self.assertFalse(pf.prefetch_file(self.tmp.name))

    def test_env_var_disables(self):
        with mock.patch.dict(os.environ, {'HX_PREFETCH': '0'}):
            self.assertFalse(pf.prefetch_enabled())
            self.assertFalse(pf.prefetch_file(self.paths[0]))

    def test_large_file_is_skipped(self):
        with mock.patch.object(pf, '_available_memory', return_value=1024):
            self.assertFalse(pf.should_prefetch(self.paths[3]))   # 4 KiB > 0.5 * 1 KiB
        with mock.patch.object(pf, '_available_memory', return_value=None):
            self.assertTrue(pf.should_prefetch(self.paths[3]))

    def test_prefetcher_order_and_lookahead(self):
        read = []
        lock = threading.Lock()
        def fake(path, max_fraction, stop=None):
            with lock:
                read.append(path)
            return True
        with mock.patch.object(pf, 'prefetch_file', side_effect=fake):
            with pf.SlidePrefetcher(self.paths) as p:
                p.wait(0)
                # Consumer is on slide 0: at most slides 0 and 1 may be read.
                p._ready[1].wait(5)
                self.assertEqual(read, self.paths[:2])
                for i in range(1, len(self.paths)):
                    p.wait(i)
        self.assertEqual(read, self.paths)

    def test_disabled_prefetcher_does_not_read(self):
        with mock.patch.object(pf, 'prefetch_file') as m:
            with pf.SlidePrefetcher(self.paths, enabled=False) as p:
                for i in range(len(self.paths)):
                    p.wait(i)
            m.assert_not_called()

    def test_early_exit_does_not_hang(self):
        with pf.SlidePrefetcher(self.paths) as p:
            p.wait(0)
        self.assertIsNone(p._thread)

    def test_failure_in_thread_does_not_hang(self):
        with mock.patch.object(pf, 'prefetch_file', side_effect=RuntimeError('boom')):
            with pf.SlidePrefetcher(self.paths) as p:
                for i in range(len(self.paths)):
                    p.wait(i)


if __name__ == '__main__':
    unittest.main()
