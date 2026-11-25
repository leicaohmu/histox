"""Tests for histox.Bag and the code that reads and writes feature bags."""

import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import histox as hx
from histox.bag import (FLOAT16_MAX, Bag, list_bags, load_coords,
                        load_features)


def _bag(n=6, f=4, slide='slide_a', seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    feats = (rng.standard_normal((n, f)) * scale).astype(np.float32)
    coords = np.stack([np.arange(n) * 443 + 221, np.arange(n) * 10 + 221], axis=1)
    return Bag(feats, coords, slide=slide,
               config={'tile_px': 224, 'tile_um': 112, 'num_features': f,
                       'extractor': {'class': 'histox.model.extractors.Foo'}})


class BagTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_validation(self):
        with self.assertRaises(ValueError):
            Bag(np.zeros(5, dtype=np.float32), slide='s')
        with self.assertRaises(ValueError):
            Bag(np.zeros((5, 2), dtype=np.float32), np.zeros((4, 2)), slide='s')
        with self.assertRaises(ValueError):
            Bag(np.zeros((5, 2), dtype=np.float32), slide='')
        with self.assertRaises(TypeError):
            Bag([[1.0, 2.0]], slide='s')

    def test_properties(self):
        bag = _bag()
        self.assertEqual(len(bag), 6)
        self.assertEqual(bag.num_features, 4)
        self.assertEqual((bag.tile_px, bag.tile_um), (224, 112))
        self.assertEqual(bag.extractor, 'histox.model.extractors.Foo')
        self.assertIs(hx.Bag, Bag)

    def test_save_load_float16_default(self):
        bag = _bag()
        path = bag.save(self.dir)
        self.assertEqual(path, os.path.join(self.dir, 'slide_a.pt'))
        self.assertEqual(torch.load(path).dtype, torch.float16)     # on disk
        loaded = Bag.load(self.dir, 'slide_a')
        self.assertEqual(loaded.features.dtype, torch.float32)      # in memory
        torch.testing.assert_close(loaded.features, bag.features, rtol=1e-3, atol=1e-3)
        np.testing.assert_array_equal(loaded.coords, bag.coords)
        self.assertEqual(loaded.config, bag.config)
        self.assertEqual(list_bags(self.dir), ['slide_a'])

    def test_save_float32_on_request(self):
        bag = _bag()
        path = bag.save(self.dir, dtype=torch.float32)
        self.assertEqual(torch.load(path).dtype, torch.float32)
        torch.testing.assert_close(Bag.load(self.dir, 'slide_a').features, bag.features,
                                   rtol=0, atol=0)

    def test_out_of_range_falls_back_to_float32(self):
        bag = _bag(scale=FLOAT16_MAX * 2)
        with self.assertLogs('histox', level='WARNING') as logs:
            path = bag.save(self.dir)
        self.assertEqual(torch.load(path).dtype, torch.float32)
        self.assertTrue(any('float16 range' in m for m in logs.output))
        self.assertTrue(torch.isfinite(Bag.load(self.dir, 'slide_a').features).all())

    def test_reads_legacy_layout(self):
        # Bags written before Bag existed: float32 .pt, np.savez index, config.
        feats = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        torch.save(feats, os.path.join(self.dir, 'old.pt'))
        np.savez(os.path.join(self.dir, 'old.index'), arr_0=np.array([[1, 2], [3, 4], [5, 6]]))
        with open(os.path.join(self.dir, 'bags_config.json'), 'w') as f:
            json.dump({'tile_px': 299}, f)
        bag = Bag.load(self.dir, 'old')
        torch.testing.assert_close(bag.features, feats)
        np.testing.assert_array_equal(bag.coords, [[1, 2], [3, 4], [5, 6]])
        self.assertEqual(bag.tile_px, 299)
        bag2 = Bag.from_path(os.path.join(self.dir, 'old.pt'))
        torch.testing.assert_close(bag2.features, feats)

    def test_npy_index_when_zip_disabled(self):
        bag = _bag()
        with mock.patch('histox.util.zip_allowed', return_value=False):
            bag.save(self.dir)
        self.assertTrue(os.path.exists(os.path.join(self.dir, 'slide_a.index.npy')))
        np.testing.assert_array_equal(Bag.load(self.dir, 'slide_a').coords, bag.coords)

    def test_missing_index_and_bag(self):
        torch.save(torch.zeros(2, 3), os.path.join(self.dir, 'noidx.pt'))
        self.assertIsNone(Bag.load(self.dir, 'noidx').coords)
        self.assertIsNone(load_coords(os.path.join(self.dir, 'noidx.pt')))
        with self.assertRaises(FileNotFoundError):
            Bag.load(self.dir, 'absent')

    def test_config_mismatch_keeps_existing(self):
        _bag(slide='a').save(self.dir)
        other = _bag(slide='b')
        other.config['tile_px'] = 512
        with self.assertLogs('histox', level='WARNING'):
            other.save(self.dir)
        with open(os.path.join(self.dir, 'bags_config.json')) as f:
            self.assertEqual(json.load(f)['tile_px'], 224)

    def test_subset(self):
        bag = _bag()
        mask = np.array([True, False, True, False, False, True])
        sub = bag.subset(mask)
        self.assertEqual(len(sub), 3)
        np.testing.assert_array_equal(sub.coords, bag.coords[[0, 2, 5]])
        torch.testing.assert_close(sub.features, bag.features[[0, 2, 5]])
        np.testing.assert_array_equal(bag.subset([1, 3]).coords, bag.coords[[1, 3]])
        with self.assertRaises(ValueError):
            bag.subset(np.array([True, False]))

    def test_helpers(self):
        bag = _bag()
        path = bag.save(self.dir)
        feats = load_features(path)
        self.assertEqual(feats.dtype, torch.float32)
        np.testing.assert_array_equal(load_coords(path), bag.coords)


class BagReadersTest(unittest.TestCase):
    """Readers of bag directories accept float16 bags and return float32."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_dataset_features_roundtrip(self):
        df = hx.DatasetFeatures(None, None)
        df.slides = ['s1', 's2']
        df.activations = {'s1': np.ones((3, 5), np.float32),
                          's2': np.full((2, 5), 0.5, np.float32)}
        df.locations = {'s1': np.array([[1, 1], [2, 2], [3, 3]]),
                        's2': np.array([[7, 7], [8, 8]])}
        with mock.patch.object(hx.DatasetFeatures, 'dump_config',
                               return_value={'tile_px': 224, 'tile_um': 112}):
            df.to_torch(self.dir, verbose=False)
        self.assertEqual(torch.load(os.path.join(self.dir, 's1.pt')).dtype, torch.float16)
        back = hx.DatasetFeatures.from_bags(self.dir)
        self.assertEqual(sorted(back.slides), ['s1', 's2'])
        self.assertEqual(back.activations['s2'].dtype, np.float32)
        np.testing.assert_allclose(back.activations['s2'], 0.5)
        np.testing.assert_array_equal(back.locations['s1'], df.locations['s1'])

    def test_mil_readers_return_float32(self):
        from histox.mil.data import BagDataset
        from histox.mil.utils import _load_bag
        path = _bag().save(self.dir)
        feats, n = BagDataset([path])[0]
        self.assertEqual((feats.dtype, n), (torch.float32, 6))
        self.assertEqual(_load_bag(path).dtype, torch.float32)

    def test_filter_bags_by_roi(self):
        import pandas as pd
        bag = _bag()
        bag.save(self.dir)
        dest = os.path.join(self.dir, 'filtered')
        x, y = bag.coords[[1, 4]].T
        tile_df = pd.DataFrame({
            'slide': ['slide_a', 'slide_a', 'slide_a', 'other'],
            'loc_x': [x[0], x[1], bag.coords[0, 0], 0],
            'loc_y': [y[0], y[1], bag.coords[0, 1], 0],
            'roi_name': ['tumor', 'tumor', None, 'tumor'],   # tile 0 is outside
        })
        hx.Dataset.filter_bags_by_roi(None, self.dir, dest, tile_df=tile_df)
        out = Bag.load(dest, 'slide_a')
        np.testing.assert_array_equal(out.coords, bag.coords[[1, 4]])
        torch.testing.assert_close(out.features, Bag.load(self.dir, 'slide_a').features[[1, 4]])
        self.assertEqual(out.config, bag.config)     # config copied


if __name__ == '__main__':
    unittest.main()
