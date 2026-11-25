"""Tests for histox.transforms."""

import json
import unittest
from types import SimpleNamespace

import numpy as np
import torch

import histox as hx
from histox import transforms as T
from histox.registry import get_registry


class TransformsTest(unittest.TestCase):

    def test_lazy_attribute(self):
        self.assertIs(hx.transforms, T)

    def test_registries(self):
        self.assertIs(get_registry('qc'), T.QC_METHODS)
        self.assertIs(T.QC_METHODS['OTSU'], T.Otsu)
        self.assertIs(T.QC_METHODS['blur'], T.GaussianV2)
        self.assertIn('summary', T.QC_METHODS.metadata('otsu'))
        names = T.NORMALIZERS.names()
        for m in ('macenko', 'reinhard', 'reinhard_fast', 'vahadane'):
            self.assertIn(m, names)
        self.assertTrue(T.NORMALIZERS.metadata('reinhard')['class'].startswith('histox.norm'))

    def test_describe_plain_objects(self):
        d = T.describe(T.Gaussian(sigma=5, threshold=0.1))
        self.assertEqual(d['type'], 'histox.slide.qc.gaussian.Gaussian')
        self.assertEqual((d['params']['sigma'], d['params']['threshold']), (5, 0.1))
        d = T.describe(T.RandomJPEGCompression(p=0.3, q_min=60, q_max=90))
        self.assertEqual(d['params'], {'p': 0.3, 'q_max': 90, 'q_min': 60})
        json.dumps(T.describe(T.Otsu()))          # always serializable

    def test_tile_normalizer_matches_normalizer(self):
        norm = T.normalizer('reinhard_fast', backend='opencv')
        tn = T.TileNormalizer(norm)
        rng = np.random.default_rng(0)
        tile = torch.from_numpy(rng.integers(0, 256, (3, 32, 32), dtype=np.uint8))
        out = tn(tile)
        self.assertEqual((out.dtype, tuple(out.shape)), (torch.uint8, (3, 32, 32)))
        torch.testing.assert_close(out, norm.torch_to_torch(tile))
        batch = torch.stack([tile, tile])
        self.assertEqual(tuple(tn(batch).shape), (2, 3, 32, 32))
        cfg = T.describe(tn)
        self.assertEqual((cfg['method'], cfg['augment']), ('reinhard_fast', False))
        self.assertIn('target_means', cfg['fit'])
        json.dumps(cfg)

    def test_tile_normalizer_by_name(self):
        tn = T.TileNormalizer('reinhard_fast')
        self.assertEqual(tn.normalizer.method, 'reinhard_fast')

    def test_compose(self):
        calls = []
        def a(x): calls.append('a'); return x + 1
        def b(x): calls.append('b'); return x * 2
        pipe = T.Compose([a, b])
        self.assertEqual(pipe(1), 4)
        self.assertEqual(calls, ['a', 'b'])
        tn = T.TileNormalizer(T.normalizer('reinhard_fast', backend='opencv'))
        cfg = T.describe(T.Compose([tn, T.RandomColorDistortion(0.5)]))
        self.assertEqual([s.get('method', s['type']) for s in cfg['transforms']],
                         ['reinhard_fast', 'histox.io.torch.augment.RandomColorDistortion'])
        json.dumps(cfg)

    def test_describe_torchvision_compose(self):
        from torchvision import transforms as tvt
        cfg = T.describe(tvt.Compose([T.RandomCardinalRotation()]))
        self.assertEqual(cfg['type'], 'torchvision.transforms.transforms.Compose')
        self.assertEqual(len(cfg['transforms']), 1)

    def test_qc_spatial_map(self):
        mask = np.zeros((20, 40), dtype=bool)
        mask[:, :10] = True

        class FakeQC:
            threshold = 0.5
            def __init__(self):
                self.level = 2
            def __call__(self, wsi):
                return mask

        wsi = SimpleNamespace(dimensions=(4000, 2000), name='s', mpp=0.25)
        m = T.qc_spatial_map(FakeQC(), wsi)
        self.assertEqual((m.kind, m.stride, m.slide), ('mask', (100.0, 100.0), 's'))
        self.assertEqual(m.value_at((50, 50)), 1.0)
        self.assertEqual(m.metadata['qc']['params'], {'level': 2})


if __name__ == '__main__':
    unittest.main()
