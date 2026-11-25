"""Tests for histox.TileSpec and where it is accepted."""

import dataclasses
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

import histox as hx
from histox.slide.tile_spec import TileSpec, resolve_tile_args


class TileSpecTest(unittest.TestCase):

    def test_defaults_and_normalization(self):
        spec = TileSpec(224, 112)
        self.assertEqual((spec.tile_px, spec.tile_um, spec.stride_div, spec.origin),
                         (224, 112, 1, (0, 0)))
        self.assertEqual(TileSpec(np.int64(224), 112.0).tile_um, 112)
        self.assertIsInstance(TileSpec(np.int64(224), 112).tile_px, int)
        self.assertEqual(TileSpec(256, '20X').tile_um, '20x')
        self.assertEqual(TileSpec(256, '20x', origin=[5, 6]).origin, (5, 6))
        self.assertEqual(TileSpec(256, '20x', origin='random').origin, 'random')
        self.assertTrue(TileSpec(256, '20x').is_magnification)
        self.assertFalse(spec.is_magnification)

    def test_invalid(self):
        for kwargs in (dict(tile_px=0, tile_um=112), dict(tile_px=224.5, tile_um=112),
                       dict(tile_px=True, tile_um=112), dict(tile_px=224, tile_um=-1),
                       dict(tile_px=224, tile_um='20'), dict(tile_px=224, tile_um='abc'),
                       dict(tile_px=224, tile_um=112, stride_div=0),
                       dict(tile_px=224, tile_um=112, origin='center'),
                       dict(tile_px=224, tile_um=112, origin=(1, 2, 3)),
                       dict(tile_px=224, tile_um=112, origin=5)):
            with self.assertRaises(ValueError, msg=kwargs):
                TileSpec(**kwargs)

    def test_label(self):
        self.assertEqual(TileSpec(224, 112).label, '224px_112um')
        self.assertEqual(TileSpec(256, '20X').label, '256px_20x')
        self.assertEqual(TileSpec(224, 112, stride_div=2).label, '224px_112um')
        self.assertEqual(TileSpec.from_label('224px_112um'), TileSpec(224, 112))
        self.assertEqual(TileSpec.from_label('256px_20x'), TileSpec(256, '20x'))
        self.assertEqual(TileSpec.from_label('512px_2.5x').tile_um, '2.5x')
        with self.assertRaises(ValueError):
            TileSpec.from_label('224_112')

    def test_value_semantics(self):
        a, b = TileSpec(224, 112), TileSpec(224, 112)
        self.assertEqual(a, b)
        self.assertEqual(len({a, b}), 1)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            a.tile_px = 256
        c = a.replace(stride_div=2)
        self.assertEqual((c.stride_div, a.stride_div), (2, 1))

    def test_dict_roundtrip(self):
        spec = TileSpec(256, '10x', stride_div=2, origin=(3, 4))
        self.assertEqual(TileSpec.from_dict(spec.to_dict()), spec)
        # Extraction configs carry other keys and only tile_px / tile_um.
        cfg = {'tile_px': 224, 'tile_um': 112, 'extractor': {}, 'num_features': 2048}
        self.assertEqual(TileSpec.from_dict(cfg), TileSpec(224, 112))

    def test_resolve_tile_args(self):
        spec = TileSpec(224, 112, stride_div=2, origin='random')
        self.assertEqual(resolve_tile_args(spec, None), (224, 112, 2, 'random'))
        self.assertEqual(resolve_tile_args(224, 112, 1, (0, 0)), (224, 112, 1, (0, 0)))
        with self.assertRaisesRegex(ValueError, 'stride_div'):
            resolve_tile_args(spec, None, stride_div=2)
        with self.assertRaisesRegex(ValueError, 'tile_um'):
            resolve_tile_args(spec, 112)

    def test_top_level_exports(self):
        self.assertIs(hx.TileSpec, TileSpec)
        self.assertIs(hx.slide.TileSpec, TileSpec)


class TileSpecIntegrationTest(unittest.TestCase):

    def test_dataset_accepts_spec(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = hx.Dataset(tile_px=TileSpec(299, '10X'), tfrecords=tmp, slides=tmp)
            self.assertEqual((ds.tile_px, ds.tile_um), (299, '10x'))
            self.assertEqual(ds.tile_spec, TileSpec(299, '10x'))
            ds2 = hx.Dataset(tile_px=299, tile_um=302, tfrecords=tmp, slides=tmp)
            self.assertEqual(ds2.tile_spec, TileSpec(299, 302))
            with self.assertRaisesRegex(ValueError, 'extract_tiles'):
                hx.Dataset(tile_px=TileSpec(299, 302, stride_div=2), tfrecords=tmp, slides=tmp)

    def test_project_extract_tiles_unpacks_spec(self):
        project = hx.Project.__new__(hx.Project)
        fake_ds = mock.Mock()
        with mock.patch.object(hx.Project, 'dataset', return_value=fake_ds) as ds_fn:
            project.extract_tiles(TileSpec(224, 112, stride_div=2, origin='random'), qc='otsu')
        args, kwargs = ds_fn.call_args
        self.assertEqual(args[:2], (224, 112))
        fake_ds.extract_tiles.assert_called_once_with(
            qc='otsu', stride_div=2, randomize_origin=True)
        with self.assertRaises(ValueError):
            project.extract_tiles(TileSpec(224, 112), stride_div=2)

    def test_bag_tile_spec(self):
        bag = hx.Bag(np.zeros((2, 3), np.float32), slide='s',
                     config={'tile_px': 224, 'tile_um': 112})
        self.assertEqual(bag.tile_spec, TileSpec(224, 112))
        self.assertIsNone(hx.Bag(np.zeros((2, 3), np.float32), slide='s').tile_spec)


if __name__ == '__main__':
    unittest.main()
