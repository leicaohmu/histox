import tempfile
import unittest
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')

import histox as hx


class TestMosaicFromArrays(unittest.TestCase):
    """Mosaic maps built directly from images and coordinates."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.images = [rng.integers(0, 255, (16, 16, 3), dtype=np.uint8) for _ in range(40)]
        self.coords = rng.random((40, 2))

    def test_points_are_a_dataframe_with_xy(self):
        mosaic = hx.Mosaic(self.images, self.coords)
        self.assertEqual(len(mosaic.points), 40)
        np.testing.assert_allclose(mosaic.points.x.values, self.coords[:, 0])
        np.testing.assert_allclose(mosaic.points.y.values, self.coords[:, 1])

    def test_generate_grid_selects_points(self):
        mosaic = hx.Mosaic(self.images, self.coords)
        mosaic.generate_grid(num_tiles_x=5)
        self.assertGreater(int(mosaic.points.selected.sum()), 0)

    def test_save(self):
        mosaic = hx.Mosaic(self.images, [tuple(c) for c in self.coords])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'mosaic.png'
            mosaic.save(str(out))
            self.assertTrue(out.exists() and out.stat().st_size > 0)

    def test_tuples_without_tfrecords(self):
        # Exercise point preparation only; full construction reads real TFRecords.
        tuples = [(f'/data/slide_{i % 3}.tfrecords', i) for i in range(40)]
        mosaic = hx.Mosaic.__new__(hx.Mosaic)
        mosaic.tfrecords = None
        mosaic._prepare_from_tuples(tuples, self.coords)
        self.assertEqual(set(mosaic.points.slide), {'slide_0', 'slide_1', 'slide_2'})
        self.assertEqual(list(mosaic.points.tfr_index), list(range(40)))
        self.assertEqual(len(mosaic.tfrecords), 3)
        self.assertEqual(mosaic._get_tfrecords_from_slide('slide_1'), '/data/slide_1.tfrecords')

if __name__ == '__main__':
    unittest.main()
