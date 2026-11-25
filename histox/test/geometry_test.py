"""Tests for histox.geometry and the code that now uses it."""

import importlib.util
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from histox import geometry as g


class GeometryTest(unittest.TestCase):

    def test_half_tile_rounds_down(self):
        self.assertEqual(g.half_tile(512), 256)
        self.assertEqual(g.half_tile(443), 221)

    def test_matches_historical_tile_record_formula(self):
        # Tile records were written as int(x + full_extract_px / 2).
        rng = np.random.default_rng(0)
        for size in (224, 443, 512, 1001):
            xs = rng.integers(0, 200_000, size=200)
            ys = rng.integers(0, 200_000, size=200)
            old = np.array([[int(x + size / 2), int(y + size / 2)]
                            for x, y in zip(xs, ys)])
            new = g.tile_center(np.stack([xs, ys], axis=1), size)
            np.testing.assert_array_equal(old, new)

    def test_center_topleft_roundtrip_keeps_dtype(self):
        corners = np.array([[0, 0], [443, 886]], dtype=np.int64)
        centers = g.tile_center(corners, 443)
        self.assertEqual(centers.dtype, np.int64)
        np.testing.assert_array_equal(centers, [[221, 221], [664, 1107]])
        np.testing.assert_array_equal(g.tile_topleft(centers, 443), corners)
        x, y = g.tile_center((10, 20), 100)
        self.assertEqual((x, y), (60, 70))

    def test_level0_roundtrip(self):
        pts = np.array([[1.0, 2.0], [10.0, 5.0]])
        lv0 = g.to_level0(pts, 4.0, offset=(100, 50))
        np.testing.assert_allclose(lv0, [[104, 58], [140, 70]])
        np.testing.assert_allclose(g.from_level0(lv0, 4.0, offset=(100, 50)), pts)
        np.testing.assert_allclose(g.to_level0((1, 2), 2), [2, 4])

    def test_rc_xy(self):
        rc = np.array([[3, 7], [1, 2]])
        np.testing.assert_array_equal(g.rc_to_xy(rc), [[7, 3], [2, 1]])
        np.testing.assert_array_equal(g.xy_to_rc(g.rc_to_xy(rc)), rc)
        np.testing.assert_array_equal(g.rc_to_xy((5, 9)), [9, 5])


class HeatmapExtentTest(unittest.TestCase):

    def _extent(self, wsi):
        from histox.heatmap import calculate_heatmap_extent
        thumb = SimpleNamespace(size=(wsi.dimensions[0] // 32,
                                      wsi.dimensions[1] // 32))
        grid = np.zeros((20, 30))   # 20 rows, 30 columns
        return calculate_heatmap_extent(wsi, thumb, grid)

    def test_uses_the_slide_grid_geometry(self):
        # Tile size given as a magnification: WSI stores a rounded tile_um, so
        # tile_um / mpp (510) no longer equals the real tile size (512).
        wsi = SimpleNamespace(tile_um=129, mpp=0.2525, stride_div=1,
                              full_extract_px=512, full_stride=512,
                              dimensions=(32000, 32000))
        left, right, bottom, top = self._extent(wsi)
        self.assertAlmostEqual(left, 0.0)
        self.assertAlmostEqual(top, 0.0)
        self.assertAlmostEqual(right, 512 * 30 / 32)
        self.assertAlmostEqual(bottom, 512 * 20 / 32)

    def test_half_stride(self):
        wsi = SimpleNamespace(tile_um=112, mpp=0.25, stride_div=2,
                              full_extract_px=448, full_stride=224,
                              dimensions=(32000, 32000))
        left, right, bottom, top = self._extent(wsi)
        self.assertAlmostEqual(left, (224 - 112) / 32)
        self.assertAlmostEqual(right, 224 * 30 / 32)


@unittest.skipUnless(importlib.util.find_spec('cellpose'), 'cellpose not installed')
class SegmentationCoordinatesTest(unittest.TestCase):

    def _segmentation(self, **kwargs):
        from histox.cellseg import Segmentation
        masks = np.zeros((100, 200), dtype=np.int32)
        masks[10:21, 40:51] = 1      # one cell, centroid at (row 15, col 45)
        return Segmentation(masks, **kwargs)

    def test_centroids_level0_with_offset(self):
        seg = self._segmentation(wsi_dim=(400, 200), wsi_offset=(1000, 500))
        np.testing.assert_array_equal(seg.centroids(wsi_dim=True),
                                      [[45 * 2 + 1000, 15 * 2 + 500]])

    def test_centroids_level0_without_offset(self):
        # wsi_offset is optional; it used to raise a TypeError here.
        seg = self._segmentation(wsi_dim=(400, 200))
        np.testing.assert_array_equal(seg.centroids(wsi_dim=True), [[90, 30]])

    def test_extract_centroids_reads_at_the_cell(self):
        seg = self._segmentation(wsi_dim=(400, 200), wsi_offset=(1000, 500))
        reader = mock.Mock(dimensions=(5000, 5000))
        with mock.patch('histox.slide.wsi_reader', return_value=reader):
            list(seg.extract_centroids('slide.svs', tile_px=64)())
        (loc, *_), _ = reader.read_from_pyramid.call_args
        # x from the column, y from the row; offset added to the matching axis;
        # scale from wsi_dim (2), not from the full slide height (50).
        self.assertEqual(tuple(np.round(loc, 6)), (90 + 1000 - 32, 30 + 500 - 32))

    def test_apply_rois_respects_offset(self):
        from shapely.geometry import box
        seg = self._segmentation(wsi_dim=(400, 200), wsi_offset=(1000, 500))
        # ROI around the cell in level-0 pixels: x 1080-1100, y 520-540.
        seg.apply_rois(1, [box(1070, 510, 1110, 550)])
        self.assertEqual(int(seg.masks.max()), 1)
        # An ROI elsewhere removes it.
        seg = self._segmentation(wsi_dim=(400, 200), wsi_offset=(1000, 500))
        seg.apply_rois(1, [box(0, 0, 60, 60)])
        self.assertEqual(int(seg.masks.max()), 0)


if __name__ == '__main__':
    unittest.main()
