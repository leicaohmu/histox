"""Tests for the default tile-reading worker pool."""

import multiprocessing as mp
import sys
import unittest
from unittest import mock

import histox as hx
from histox.slide import wsi as wsi_mod


class TestDefaultGeneratorWorkers(unittest.TestCase):

    def _call(self, backend, platform='linux', daemon=False, cores=24):
        proc = mock.Mock(daemon=daemon)
        with mock.patch.object(hx, 'slide_backend', return_value=backend), \
             mock.patch.object(hx.util, 'num_cpu', return_value=cores), \
             mock.patch.object(wsi_mod.sys, 'platform', platform), \
             mock.patch.object(wsi_mod.mp, 'current_process', return_value=proc):
            return wsi_mod._default_generator_workers()

    def test_cucim_linux_uses_processes(self):
        self.assertEqual(self._call('cucim'), (None, 12))

    def test_libvips_uses_processes(self):
        self.assertEqual(self._call('libvips', platform='darwin'), (None, 12))

    def test_cucim_non_linux_uses_threads(self):
        self.assertEqual(self._call('cucim', platform='darwin'), (24, None))

    def test_daemonic_worker_uses_threads(self):
        self.assertEqual(self._call('cucim', daemon=True), (8, None))
        self.assertEqual(self._call('libvips', daemon=True), (8, None))

    def test_process_count_is_capped(self):
        self.assertEqual(self._call('cucim', cores=256), (None, 32))
        self.assertEqual(self._call('cucim', cores=1), (None, 1))


if __name__ == '__main__':
    unittest.main()
