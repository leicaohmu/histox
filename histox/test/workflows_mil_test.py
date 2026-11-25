"""Tests for the classification / regression tasks and the MIL workflow method."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

import histox as hx
from histox.workflows import (ClassificationTask, Experiment, IncompatibleWorkflowError,
                              MILMethod, RegressionTask, build_method, build_task,
                              list_methods, list_tasks)


def _spec(fields=None, modalities=('histology',)):
    cohort = hx.Cohort(pd.DataFrame({'patient': ['p1', 'p2'], 'slide': ['s1', 's2'],
                                     'subtype': ['a', 'b'], 'age': ['50', '60'],
                                     'os': ['10', '20'], 'event': ['1', '0']}))
    return cohort.data_spec(fields or {'label': 'subtype'}, modalities=modalities)


class TaskTest(unittest.TestCase):

    def test_registered(self):
        self.assertIn('classification', list_tasks())
        self.assertIn('regression', list_tasks())
        self.assertIn('mil', list_methods())
        self.assertIsInstance(build_task('classification'), ClassificationTask)
        self.assertIs(hx.ClassificationTask, ClassificationTask)

    def test_classification_score(self):
        task = ClassificationTask(metrics=('auroc', 'accuracy', 'balanced_accuracy'))
        y = np.array([0, 0, 1, 1])
        p = np.array([[0.9, 0.1], [0.6, 0.4], [0.3, 0.7], [0.45, 0.55]])
        s = task.score(y, p)
        self.assertEqual((s['n'], s['auroc'], s['accuracy']), (4, 1.0, 1.0))
        p3 = np.eye(3)[[0, 1, 2, 2]] * 0.8 + 0.2 / 3
        s3 = task.score([0, 1, 2, 2], p3)
        self.assertAlmostEqual(s3['auroc'], 1.0)
        self.assertIsNone(task.score([1, 1], np.array([[0.2, 0.8], [0.4, 0.6]]))['auroc'])
        with self.assertRaises(ValueError):
            ClassificationTask(metrics=('f1',))
        with self.assertRaises(ValueError):
            task.score([0, 1], np.array([0.1, 0.9]))

    def test_regression_score(self):
        task = RegressionTask(metrics=('r2', 'mae', 'mse', 'pearson'))
        s = task.score([1, 2, 3], [1, 2, 4])
        self.assertAlmostEqual(s['mae'], 1 / 3)
        self.assertAlmostEqual(s['mse'], 1 / 3)
        self.assertAlmostEqual(s['r2'], 0.5)
        self.assertEqual(task.label_field, 'target')

    def test_validation_against_data_spec(self):
        Experiment(task='classification', method='mil', data_spec=_spec(),
                   method_config={'bags': '/tmp/bags'})
        with self.assertRaises(hx.workflows.InvalidDataSpecError):
            Experiment(task='classification', method='mil', data_spec=_spec({'y': 'subtype'}))
        Experiment(task=RegressionTask(target_field='age'), method='mil',
                   data_spec=_spec({'age': 'age'}))
        with self.assertRaises(IncompatibleWorkflowError):
            Experiment(task='survival', method='mil',
                       data_spec=_spec({'duration': 'os', 'event': 'event'}))
        with self.assertRaises(hx.workflows.InvalidDataSpecError):
            Experiment(task='classification', method='mil', data_spec=_spec(modalities=('rna',)))


class MILMethodTest(unittest.TestCase):

    def test_trainer_config_loss_follows_task(self):
        m = MILMethod(model='attention_mil', epochs=3)
        self.assertEqual(m.trainer_config(ClassificationTask()).model_config.loss, 'cross_entropy')
        self.assertEqual(m.trainer_config(RegressionTask()).model_config.loss, 'mse')
        self.assertEqual(MILMethod(loss='mae').trainer_config(RegressionTask()).model_config.loss, 'mae')
        self.assertEqual(m.trainer_config().epochs, 3)

    def test_build_model(self):
        import torch
        exp = Experiment(task='classification', method=build_method('mil', n_feats=8, n_out=2),
                         data_spec=_spec())
        self.assertIsInstance(exp.build_model(), torch.nn.Module)
        with self.assertRaises(ValueError):
            Experiment(task='classification', method='mil', data_spec=_spec()).build_model()

    def test_fit_and_evaluate_call_histox_mil(self):
        class FakeDataset:
            def __init__(self, slides=None):
                self.slides_ = slides
            def filter(self, filters):
                return FakeDataset(filters['slide'])

        cohort = FakeDataset()
        exp = Experiment(task='classification', method='mil', data_spec=_spec(), cohort=cohort,
                         method_config={'model': 'attention_mil', 'bags': '/b', 'epochs': 2},
                         output='/tmp/exp')
        split = hx.splits.Split(('p1',), ('p2',), ('s1',), ('s2',))
        learner = SimpleNamespace(path='/tmp/exp/mil/00000-attention_mil', model='M')
        with mock.patch('histox.mil.train_mil', return_value=learner) as train:
            fit = exp.fit(split=split)
        config, train_ds, val_ds = train.call_args[0]
        kw = train.call_args[1]
        self.assertEqual((train_ds.slides_, val_ds.slides_), (['s1'], ['s2']))
        self.assertEqual((kw['outcomes'], kw['bags'], kw['outdir']), ('subtype', '/b', '/tmp/exp/mil'))
        self.assertEqual(config.epochs, 2)
        self.assertEqual(fit.artifacts['model'].path.as_posix(), learner.path)
        df = pd.DataFrame({'slide': ['s1', 's2', 's3', 's4'], 'y_true': [0, 0, 1, 1],
                           'y_pred0': [0.9, 0.6, 0.3, 0.45], 'y_pred1': [0.1, 0.4, 0.7, 0.55]})
        with mock.patch('histox.mil.eval_mil', return_value=df) as ev:
            res = exp.evaluate(split=split)
        self.assertEqual(ev.call_args[0][0], learner.path)
        self.assertEqual(ev.call_args[0][1].slides_, ['s2'])
        self.assertEqual((res.metrics['auroc'], res.metrics['accuracy']), (1.0, 1.0))
        self.assertIs(res.predictions, df)

    def test_evaluate_without_model(self):
        exp = Experiment(task='classification', method='mil', data_spec=_spec(), cohort=object(),
                         method_config={'bags': '/b'})
        with self.assertRaises(ValueError):
            exp.evaluate()


if __name__ == '__main__':
    unittest.main()
