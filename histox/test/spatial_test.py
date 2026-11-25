"""Tests for histox.SpatialMap."""

import importlib.util
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

import histox as hx
from histox.spatial import SpatialMap


class SpatialMapTest(unittest.TestCase):

    def test_validation(self):
        with self.assertRaises(ValueError):
            SpatialMap(np.zeros(5))
        with self.assertRaises(ValueError):
            SpatialMap(np.zeros((2, 2)), kind='heat')
        with self.assertRaises(ValueError):
            SpatialMap(np.zeros((2, 2)), stride=0)
        with self.assertRaises(ValueError):
            SpatialMap(np.zeros((2, 2, 3)), channels=['a', 'b'])
        m = SpatialMap(np.zeros((2, 3)), origin=(10, 20), stride=4)
        self.assertEqual((m.origin, m.stride), ((10.0, 20.0), (4.0, 4.0)))
        self.assertIs(hx.SpatialMap, SpatialMap)

    def test_extent_and_lookup(self):
        vals = np.arange(6, dtype=float).reshape(2, 3)   # 2 rows, 3 cols
        m = SpatialMap(vals, origin=(100, 50), stride=(10, 20))
        self.assertEqual(m.size_level0, (30, 40))
        self.assertEqual(m.extent(), (100, 130, 90, 50))
        self.assertEqual(m.extent(downsample=10), (10, 13, 9, 5))
        # (x, y) -> [row, col]
        self.assertEqual(m.value_at((125, 55)), 2.0)     # row 0, col 2
        self.assertEqual(m.value_at((100, 89.9)), 3.0)   # row 1, col 0
        np.testing.assert_array_equal(m.cell_of([[125, 55], [99, 55]]), [[0, 2], [0, -1]])
        out = m.value_at([[125, 55], [99, 55], [131, 60]])
        self.assertEqual(out[0], 2.0)
        self.assertTrue(np.isnan(out[1]) and np.isnan(out[2]))
        self.assertEqual(m.value_at((0, 0), default=-1), -1)

    def test_channels_and_masked(self):
        vals = np.ma.masked_array(np.ones((2, 2, 3)), mask=False)
        vals[1, 1] = np.ma.masked
        m = SpatialMap(vals, stride=1, channels=['a', 'b', 'c'])
        np.testing.assert_array_equal(m.value_at((0.5, 0.5)), [1, 1, 1])
        self.assertTrue(np.isnan(m.value_at((1.5, 1.5))).all())

    def test_from_points_roundtrip_odd_stride(self):
        # Tile centers on a 443-px grid (112 um at ~0.2527 mpp), as in bags.
        rng = np.random.default_rng(0)
        stride, half = 443, 221
        gx, gy = rng.integers(0, 50, 300), rng.integers(0, 40, 300)
        keep = np.unique(np.stack([gx, gy], 1), axis=0)
        coords = keep * stride + half + np.array([17, 9])     # non-zero grid origin
        vals = rng.random(len(coords))
        m = SpatialMap.from_points(coords, vals, stride=stride, slide='s', mpp=0.2527)
        np.testing.assert_array_equal(m.value_at(coords), vals)
        self.assertEqual(int(np.isfinite(m.values).sum()), len(coords))
        # multi-channel values
        m2 = SpatialMap.from_points(coords, np.stack([vals, -vals], 1), stride=stride)
        np.testing.assert_array_equal(m2.value_at(coords)[:, 1], -vals)

    def test_from_bag(self):
        bag = hx.Bag(np.zeros((3, 2), np.float32), np.array([[221, 221], [664, 221], [221, 664]]),
                     slide='s1')
        m = SpatialMap.from_bag(bag, [0.1, 0.2, 0.3], stride=443)
        self.assertEqual(m.slide, 's1')
        np.testing.assert_allclose(m.value_at(bag.coords), [0.1, 0.2, 0.3])
        self.assertTrue(np.isnan(m.value_at((664, 664))))

    def test_save_load(self):
        vals = np.ma.masked_array(np.arange(12.).reshape(2, 2, 3), mask=False)
        vals[0, 1] = np.ma.masked
        m = SpatialMap(vals, (5, 6), (2, 3), slide='s', mpp=0.25, kind='scores',
                       channels=['a', 'b', 'c'], metadata={'model': 'x'})
        with tempfile.TemporaryDirectory() as tmp:
            path = m.save(os.path.join(tmp, 'map'))
            self.assertTrue(path.endswith('.npz'))
            back = SpatialMap.load(path)
        np.testing.assert_array_equal(np.ma.getdata(back.values), np.ma.getdata(vals))
        np.testing.assert_array_equal(np.ma.getmaskarray(back.values), np.ma.getmaskarray(vals))
        self.assertEqual((back.origin, back.stride, back.slide, back.mpp, back.kind,
                          back.channels, back.metadata),
                         ((5.0, 6.0), (2.0, 3.0), 's', 0.25, 'scores', ['a', 'b', 'c'],
                          {'model': 'x'}))

    def test_from_qc_mask(self):
        from histox.slide.utils import QCMask
        mask = np.zeros((50, 100), dtype=bool)
        mask[10:20, 30:40] = True
        wsi = SimpleNamespace(dimensions=(4000, 2000), name='s', mpp=0.5)
        m = QCMask(mask, filter_threshold=0.6).to_spatial_map(wsi)
        self.assertEqual((m.kind, m.stride, m.slide), ('mask', (40.0, 40.0), 's'))
        self.assertEqual(m.value_at((35 * 40, 15 * 40)), 1.0)
        self.assertEqual(m.value_at((5 * 40, 5 * 40)), 0.0)
        self.assertEqual(m.metadata['filter_threshold'], 0.6)

    def test_from_heatmap_matches_tile_grid(self):
        # A fake slide with a 3 x 2 tile grid, tile size 443, stride 443, origin (17, 9).
        full, stride = 443, 443
        coord = np.array([[17 + gx * stride, 9 + gy * stride, gx, gy]
                          for gy in range(2) for gx in range(3)])
        wsi = SimpleNamespace(coord=coord, full_stride=stride, full_extract_px=full,
                              origin=(17, 9), name='s', mpp=0.25, tile_px=224, tile_um=112)
        preds = np.ma.masked_all((2, 3, 2))           # [grid_y, grid_x, classes]
        for x, y, gx, gy in coord:
            preds[gy, gx] = (gx, gy)
        heatmap = SimpleNamespace(slide=wsi, predictions=preds, uncertainty=None)
        m = SpatialMap.from_heatmap(heatmap)
        centers = coord[:, :2] + 221
        np.testing.assert_array_equal(m.value_at(centers), coord[:, 2:4])
        self.assertEqual(m.origin, (17 + 221 - 221.5, 9 + 221 - 221.5))
        with self.assertRaises(ValueError):
            SpatialMap.from_heatmap(heatmap, uncertainty=True)


@unittest.skipUnless(importlib.util.find_spec('cellpose'), 'cellpose not installed')
class SegmentationSpatialMapTest(unittest.TestCase):

    def test_from_segmentation(self):
        from histox.cellseg import Segmentation
        masks = np.zeros((100, 200), dtype=np.int32)
        masks[10:21, 40:51] = 1
        seg = Segmentation(masks, wsi_dim=(400, 200), wsi_offset=(1000, 500))
        m = seg.to_spatial_map()
        self.assertEqual((m.kind, m.stride, m.origin), ('labels', (2.0, 2.0), (1000.0, 500.0)))
        # The level-0 centroid of the cell lies in the cell's label.
        self.assertEqual(m.value_at(seg.centroids(wsi_dim=True)[0]), 1)
        with self.assertRaises(ValueError):
            Segmentation(masks).to_spatial_map()


if __name__ == '__main__':
    unittest.main()
