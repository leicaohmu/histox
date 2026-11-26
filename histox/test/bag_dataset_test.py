import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import histox as hx
from histox.data import BagDataset, collate_bags


def _bag(slide, n, f=4, offset=0):
    # Feature column 0 equals the tile x coordinate, so tests can check that
    # features and coordinates stay aligned after sampling and padding.
    coords = np.stack([np.arange(n) * 10 + offset, np.arange(n) * 10 + 1], 1)
    feats = np.zeros((n, f), np.float32)
    feats[:, 0] = coords[:, 0]
    feats[:, 1] = 1.0
    return hx.Bag(feats, coords, slide=slide, config={'tile_px': 224})


class BagDatasetTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.bags = os.path.join(cls.tmp, 'bags')
        cls.bags2 = os.path.join(cls.tmp, 'bags2')
        sizes = {'s1': 5, 's2': 3, 's3': 7, 's4': 2}
        for i, (slide, n) in enumerate(sizes.items()):
            _bag(slide, n, offset=1000 * i).save(cls.bags if slide != 's4' else cls.bags2,
                                                 dtype=torch.float32)
        cls.table = pd.DataFrame({
            'patient': ['p1', 'p1', 'p2', 'p3', 'p4'],
            'slide': ['s1', 's2', 's3', 's4', 's5'],     # s5 has no bag
            'subtype': ['b', 'b', 'a', 'a', 'b'],
            'age': [61, 61, 50, '', 70],
        })
        cls.cohort = hx.Cohort(cls.table)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_slide_items(self):
        ds = BagDataset(self.cohort, [self.bags, self.bags2], target='subtype')
        self.assertEqual(len(ds), 4)
        self.assertEqual(ds.slides, ['s1', 's2', 's3', 's4'])
        self.assertEqual(ds.classes, ['a', 'b'])
        self.assertEqual(ds.target_type, 'categorical')
        self.assertEqual(ds.num_features, 4)
        item = ds[0]
        self.assertEqual(set(item), {'features', 'coords', 'slide', 'patient', 'target'})
        self.assertEqual(item['features'].dtype, torch.float32)
        self.assertEqual(tuple(item['features'].shape), (5, 4))
        self.assertEqual(item['coords'].dtype, torch.int64)
        self.assertEqual(item['slide'], 's1')
        self.assertEqual(item['patient'], 'p1')
        self.assertEqual(item['target'].item(), 1)     # 'b'
        self.assertEqual(ds[2]['target'].item(), 0)    # 'a'

    def test_missing_bags(self):
        ds = BagDataset(self.cohort, self.bags)        # s4 only in bags2, s5 nowhere
        self.assertEqual(ds.slides, ['s1', 's2', 's3'])
        with self.assertRaises(FileNotFoundError):
            BagDataset(self.cohort, self.bags, missing='error')

    def test_patient_level(self):
        ds = BagDataset(self.cohort, [self.bags, self.bags2], target='subtype', level='patient')
        self.assertEqual(ds.patients, ['p1', 'p2', 'p3'])
        item = ds[0]
        self.assertEqual(item['slide'], ['s1', 's2'])
        self.assertEqual(item['features'].shape[0], 8)
        torch.testing.assert_close(item['features'][:, 0], item['coords'][:, 0].float())

    def test_numeric_target_drops_empty(self):
        ds = BagDataset(self.cohort, [self.bags, self.bags2], target='age')
        self.assertEqual(ds.target_type, 'numeric')
        self.assertEqual(ds.slides, ['s1', 's2', 's3'])            # s4 has no age
        self.assertEqual(ds[2]['target'].dtype, torch.float32)
        self.assertEqual(ds[2]['target'].item(), 50.0)
        forced = BagDataset(self.cohort, self.bags, target='age', target_type='categorical')
        self.assertEqual(forced.classes, ['50', '61'] if isinstance(forced.classes[0], str)
                         else [50, 61])

    def test_explicit_classes(self):
        ds = BagDataset(self.cohort, [self.bags, self.bags2], target='subtype', classes=['b', 'a'])
        self.assertEqual(ds[0]['target'].item(), 0)
        with self.assertRaises(ValueError):
            BagDataset(self.cohort, self.bags, target='subtype', classes=['a'])

    def test_max_tiles_keeps_alignment(self):
        ds = BagDataset(self.cohort, self.bags, max_tiles=2)
        torch.manual_seed(0)
        item = ds[2]                                    # s3, 7 tiles
        self.assertEqual(item['features'].shape[0], 2)
        torch.testing.assert_close(item['features'][:, 0], item['coords'][:, 0].float())
        self.assertEqual(ds[1]['features'].shape[0], 2)
        self.assertEqual(BagDataset(self.cohort, self.bags, max_tiles=10)[2]['features'].shape[0], 7)

    def test_collate_and_dataloader(self):
        ds = BagDataset(self.cohort, [self.bags, self.bags2], target='subtype')
        loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_bags)
        batch = next(iter(loader))
        self.assertEqual(tuple(batch['features'].shape), (4, 7, 4))
        self.assertEqual(batch['lens'].tolist(), [5, 3, 7, 2])
        self.assertEqual(batch['mask'].sum(1).tolist(), [5, 3, 7, 2])
        self.assertTrue(torch.all(batch['features'][~batch['mask']] == 0))
        self.assertTrue(torch.all(batch['coords'][~batch['mask']] == -1))
        real = batch['mask']
        torch.testing.assert_close(batch['features'][real][:, 0], batch['coords'][real][:, 0].float())
        self.assertEqual(batch['target'].tolist(), [1, 1, 0, 0])
        self.assertEqual(batch['slide'], ['s1', 's2', 's3', 's4'])

    def test_transform_and_dataset_input(self):
        def add_len(item):
            item['n'] = item['features'].shape[0]
            return item
        ds = BagDataset(self.cohort, self.bags, transform=add_len)
        self.assertEqual(ds[0]['n'], 5)
        batch = collate_bags([ds[0], ds[1]])
        self.assertEqual(batch['n'], [5, 3])

        class _HasCohort:
            cohort = self.cohort
        self.assertEqual(len(BagDataset(_HasCohort(), self.bags)), 3)

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            BagDataset(self.cohort, self.bags, level='tile')
        with self.assertRaises(ValueError):
            BagDataset(self.cohort, self.bags, max_tiles=0)
        with self.assertRaises(ValueError):
            BagDataset(self.cohort, os.path.join(self.tmp, 'empty'))


if __name__ == '__main__':
    unittest.main()
