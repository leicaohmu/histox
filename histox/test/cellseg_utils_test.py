import unittest

import numpy as np

from histox.cellseg import seg_utils


class TestSparseCentroids(unittest.TestCase):

    def test_split_indices_cover_every_label_once(self):
        for shape in range(1, 120):
            for splits in (1, 2, 3, 7, 64):
                chunks = seg_utils.sparse_split_indices(shape, splits)
                rows = [r for i, j in chunks for r in range(i, j)]
                self.assertEqual(rows, list(range(1, shape)), (shape, splits))
                self.assertLessEqual(len(chunks), max(1, splits))

    def test_centroids_include_every_cell(self):
        masks = np.zeros((64, 64), dtype=np.int32)
        masks[2:6, 2:6] = 1
        masks[20:24, 30:34] = 2
        masks[50:54, 10:14] = 3          # highest label
        sparse = seg_utils.sparse_mask(masks)
        centroids = seg_utils.get_sparse_centroid(masks, sparse)
        self.assertEqual(len(centroids), 3)
        np.testing.assert_array_equal(centroids[2], (51, 11))

    def test_single_cell(self):
        masks = np.zeros((32, 32), dtype=np.int32)
        masks[10:14, 10:14] = 1
        centroids = seg_utils.get_sparse_centroid(masks, seg_utils.sparse_mask(masks))
        self.assertEqual(len(centroids), 1)


    def test_no_cells(self):
        masks = np.zeros((32, 32), dtype=np.int32)
        centroids = seg_utils.get_sparse_centroid(masks, seg_utils.sparse_mask(masks))
        self.assertEqual(centroids.shape, (0, 2))


if __name__ == '__main__':
    unittest.main()
