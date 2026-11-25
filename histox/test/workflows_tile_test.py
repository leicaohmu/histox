"""Tests for the tile_classifier workflow method (Project.train / evaluate mocked)."""

import os
import tempfile
import unittest

import pandas as pd

import histox as hx
from histox.workflows import Experiment, TileClassifierMethod, list_methods


class FakeDataset:
    def __init__(self, slides=None):
        self.slides_ = slides

    def filter(self, filters):
        return FakeDataset(filters['slide'])


class FakeProject:
    """Writes the files Project.train / Project.evaluate write."""

    def __init__(self, root):
        self.models_dir = os.path.join(root, 'models')
        self.eval_dir = os.path.join(root, 'eval')
        self.train_calls, self.eval_calls = [], []

    def train(self, outcome, hp, **kwargs):
        self.train_calls.append((outcome, hp, kwargs))
        d = os.path.join(self.models_dir, '00000-label-HP0')
        os.makedirs(d)
        for name in ('label-HP0_epoch1.zip', 'label-HP0_epoch3.zip', 'params.json',
                     'slide_predictions_val_epoch3.parquet.gzip'):
            open(os.path.join(d, name), 'w').close()
        return {'label-HP0': {'epochs': {'epoch3': {'val_metrics': {'accuracy': 0.5}}}}}

    def evaluate(self, model, outcome, dataset=None, **kwargs):
        self.eval_calls.append((model, outcome, dataset, kwargs))
        d = os.path.join(self.eval_dir, '00001-eval')
        os.makedirs(d)
        pd.DataFrame({'slide': ['s1', 's2', 's3', 's4'],
                      f'{outcome}-y_pred0': [0.9, 0.6, 0.3, 0.45],
                      f'{outcome}-y_pred1': [0.1, 0.4, 0.7, 0.55],
                      f'{outcome}-y_true': [0.0, 0.0, 1.0, 1.0]}).to_parquet(
            os.path.join(d, 'slide_predictions_eval.parquet.gzip'))
        return {'eval': {'slide_auc': {outcome: [1.0, 1.0]}}}


def _spec():
    cohort = hx.Cohort(pd.DataFrame({'patient': ['p1', 'p2'], 'slide': ['s1', 's2'],
                                     'subtype': ['a', 'b'], 'age': ['50', '60']}))
    return cohort.data_spec({'label': 'subtype', 'target': 'age'})


class TileClassifierTest(unittest.TestCase):

    def test_registered_and_params(self):
        self.assertIn('tile_classifier', list_methods())
        m = TileClassifierMethod(project='/x', tile_px=224, tile_um=112, model='resnet18', epochs=[1])
        self.assertEqual(m.model_params(hx.ClassificationTask()).loss, 'CrossEntropy')
        self.assertEqual(m.model_params(hx.RegressionTask()).loss, 'MSE')
        self.assertEqual(m.model_params().tile_px, 224)
        hp = hx.ModelParams(tile_px=299, tile_um=302, model='resnet18')
        self.assertIs(TileClassifierMethod(params=hp).model_params(), hp)
        with self.assertRaises(ValueError):
            TileClassifierMethod().project()

    def test_build_model(self):
        import torch
        m = TileClassifierMethod(tile_px=64, tile_um=32, model='resnet18', n_out=3)
        self.assertIsInstance(m.build_model(_spec()), torch.nn.Module)
        with self.assertRaises(ValueError):
            TileClassifierMethod(tile_px=64, tile_um=32).build_model(_spec())

    def test_fit_and_evaluate(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = FakeProject(tmp)
            exp = Experiment(task='classification', method='tile_classifier', data_spec=_spec(),
                             cohort=FakeDataset(),
                             method_config={'project': project, 'tile_px': 224, 'tile_um': 112,
                                            'model': 'resnet18', 'epochs': [3]})
            split = hx.splits.Split(('p1',), ('p2',), ('s1',), ('s2',))
            fit = exp.fit(split=split, max_tiles=40)
            outcome, hp, kw = project.train_calls[0]
            self.assertEqual(outcome, 'subtype')
            self.assertEqual((kw['dataset'].slides_, kw['val_dataset'].slides_), (['s1'], ['s2']))
            self.assertEqual((kw['max_tiles'], hp.loss), (40, 'CrossEntropy'))
            self.assertTrue(fit.metadata['model_path'].endswith('label-HP0_epoch3.zip'))
            self.assertEqual(fit.metadata['results']['label-HP0']['epochs']['epoch3']['val_metrics'],
                             {'accuracy': 0.5})

            res = exp.evaluate(split=split)
            model, outcome, ds, _ = project.eval_calls[0]
            self.assertTrue(model.endswith('epoch3.zip'))
            self.assertEqual(ds.slides_, ['s2'])
            self.assertEqual((res.metrics['auroc'], res.metrics['accuracy']), (1.0, 1.0))
            self.assertEqual(res.metadata['trainer_metrics']['eval']['slide_auc']['subtype'], [1.0, 1.0])
            self.assertEqual(len(res.predictions), 4)

    def test_no_validation_means_none_strategy(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = FakeProject(tmp)
            exp = Experiment(task='classification', method='tile_classifier', data_spec=_spec(),
                             cohort=FakeDataset(['s1', 's2']),
                             method_config={'project': project, 'tile_px': 224, 'tile_um': 112})
            exp.fit()
            self.assertEqual(project.train_calls[0][2]['val_strategy'], 'none')


if __name__ == '__main__':
    unittest.main()
