"""Tests for writing TFRecords during direct feature extraction."""

import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

import histox as hx
from histox.model.extractors._slide._utils import _tee_to_tfrecord, _tfrecord_img_format


def _jpeg(v):
    import cv2
    img = np.full((8, 8, 3), v, dtype=np.uint8)
    return cv2.imencode('.jpg', img)[1].tobytes()


class TestTFRecordTee(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.slide = SimpleNamespace(name='slide_a')
        self.tiles = [{'image': _jpeg(i * 20), 'loc': (i * 10, i * 20), 'grid': (i, 0)}
                      for i in range(5)]

    def tearDown(self):
        self.tmp.cleanup()

    def _gen(self):
        return lambda: iter(self.tiles)

    def test_records_match_extract_tiles_layout(self):
        out = os.path.join(self.tmp.name, 'tfr')
        yielded = list(_tee_to_tfrecord(self._gen(), self.slide, out)())
        self.assertEqual(len(yielded), 5)
        path = os.path.join(out, 'slide_a.tfrecords')
        self.assertTrue(os.path.exists(path))
        from histox.util import tfrecord2idx
        self.assertIsNotNone(tfrecord2idx.find_index(path))
        self.assertFalse(os.path.exists(path + '.partial'))
        tfr = hx.TFRecord(path)
        self.assertEqual(len(tfr), 5)
        for i in range(5):
            slide, image = tfr.get_record_by_xy(i * 10, i * 20)
            if isinstance(slide, bytes):
                slide = slide.decode('utf-8')
            self.assertEqual(slide, 'slide_a')
            self.assertEqual(bytes(image), self.tiles[i]['image'])

    def test_interrupted_run_leaves_no_tfrecord(self):
        out = os.path.join(self.tmp.name, 'tfr')
        it = _tee_to_tfrecord(self._gen(), self.slide, out)()
        next(it); next(it)
        it.close()
        self.assertEqual(os.listdir(out), [])

    def test_empty_slide_leaves_no_tfrecord(self):
        out = os.path.join(self.tmp.name, 'tfr')
        list(_tee_to_tfrecord(lambda: iter([]), self.slide, out)())
        self.assertEqual(os.listdir(out), [])

    def test_img_format(self):
        self.assertEqual(_tfrecord_img_format('numpy'), 'jpg')
        self.assertEqual(_tfrecord_img_format('jpg'), 'jpg')
        self.assertEqual(_tfrecord_img_format('JPEG'), 'jpg')
        with self.assertRaises(ValueError):
            _tfrecord_img_format('png')


if __name__ == '__main__':
    unittest.main()
