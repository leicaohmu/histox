"""Tests for histox.Cohort and histox.splits."""

import os
import tempfile
import unittest
from collections import Counter

import pandas as pd

import histox as hx
from histox.cohort import Cohort
from histox import splits


def _table(n_patients=20, slides_per_patient=2):
    rows = []
    for p in range(n_patients):
        label = 'a' if p % 3 else 'b'
        for s in range(slides_per_patient):
            rows.append({'patient': f'P{p:02d}', 'slide': f'P{p:02d}_S{s}',
                         'label': label, 'site': f'site{p % 4}'})
    return pd.DataFrame(rows)


class CohortTest(unittest.TestCase):

    def test_basics(self):
        c = Cohort(_table())
        self.assertEqual(len(c), 40)
        self.assertEqual(len(c.patients), 20)
        self.assertEqual(c.fields, ['label', 'site'])
        self.assertEqual(c.patient_of('P03_S1'), 'P03')
        self.assertEqual(c.slides_of('P03'), ['P03_S0', 'P03_S1'])
        self.assertIn('P03_S1', c)
        self.assertIs(hx.Cohort, Cohort)

    def test_no_patient_column(self):
        c = Cohort(pd.DataFrame({'slide': ['a', 'b'], 'y': ['1', '0']}))
        self.assertEqual(c.patients, ['a', 'b'])

    def test_validation(self):
        with self.assertRaises(ValueError):
            Cohort(pd.DataFrame({'patient': ['p'], 'x': [1]}))
        with self.assertRaises(ValueError):
            Cohort(pd.DataFrame({'slide': ['a', 'a']}))
        c = Cohort(pd.DataFrame({'slide': ['a', '', None]}))
        self.assertEqual(c.slides, ['a'])

    def test_labels(self):
        t = _table(4, 1)
        t.loc[0, 'label'] = ''
        c = Cohort(t)
        self.assertEqual(c.labels('label'), {'P01_S0': 'a', 'P02_S0': 'a', 'P03_S0': 'b'})
        self.assertEqual(len(c.labels('label', drop_empty=False)), 4)
        self.assertEqual(c.patient_labels('label')['P03'], 'b')
        with self.assertRaises(KeyError):
            c.labels('nope')
        t2 = _table(2, 2)
        t2.loc[0, 'label'] = 'z'                     # P00 slides disagree
        with self.assertRaises(ValueError):
            Cohort(t2).patient_labels('label')

    def test_selection(self):
        c = Cohort(_table())
        self.assertEqual(len(c.filter(label='b')), 14)
        self.assertEqual(len(c.filter(site=['site0', 'site1'])), 20)
        self.assertEqual(c.subset(patients=['P01']).slides, ['P01_S0', 'P01_S1'])
        self.assertEqual(c.subset(slides=['P01_S0']).slides, ['P01_S0'])
        t = _table(3, 1); t.loc[1, 'label'] = ''
        self.assertEqual(len(Cohort(t).drop_empty('label')), 2)

    def test_slide_files_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            sub = os.path.join(tmp, 'nested'); os.makedirs(sub)
            for name in ('P00_S0.svs', 'P00_S1.tiff'):
                open(os.path.join(sub, name), 'w').close()
            csv = os.path.join(tmp, 'ann.csv')
            _table(2, 2).to_csv(csv, index=False)
            c = Cohort.from_csv(csv, slides=tmp)
            self.assertEqual(c.slide_path('P00_S0'), os.path.join(sub, 'P00_S0.svs'))
            self.assertEqual(set(c.slide_paths()), {'P00_S0', 'P00_S1'})
            self.assertEqual(c.missing_slides(), ['P01_S0', 'P01_S1'])
            self.assertEqual(c.filter(label='b').slide_paths(), {
                'P00_S0': os.path.join(sub, 'P00_S0.svs'),
                'P00_S1': os.path.join(sub, 'P00_S1.tiff')})

    def test_data_spec(self):
        c = Cohort(_table())
        spec = c.data_spec({'label': 'label'})
        self.assertEqual(spec.fields['patient_id'], 'patient')
        self.assertEqual(spec.fields['label'], 'label')
        self.assertEqual(spec.metadata['patients'], 20)
        with self.assertRaises(KeyError):
            c.data_spec({'duration': 'os_days'})


class SplitsTest(unittest.TestCase):

    def setUp(self):
        self.c = Cohort(_table(30, 2))

    def test_kfold_patient_level_and_complete(self):
        folds = splits.patient_kfold(self.c, 5, stratify='label', seed=3)
        self.assertEqual(len(folds), 5)
        val_patients = [p for f in folds for p in f.val_patients]
        self.assertEqual(sorted(val_patients), self.c.patients)      # each once
        for f in folds:
            self.assertFalse(set(f.train_patients) & set(f.val_patients))
            self.assertEqual(len(f.train_patients) + len(f.val_patients), 30)
            # slides follow their patient
            self.assertTrue(all(s[:3] in f.val_patients for s in f.val_slides))
            self.assertEqual(len(f.val_slides), 2 * len(f.val_patients))
            self.assertEqual(len(f.val_patients), 6)
            labels = Counter(self.c.patient_labels('label')[p] for p in f.val_patients)
            self.assertEqual(labels, Counter({'a': 4, 'b': 2}))      # 20 a / 10 b overall

    def test_deterministic(self):
        a = splits.patient_kfold(self.c, 3, seed=7)
        b = splits.patient_kfold(self.c, 3, seed=7)
        c = splits.patient_kfold(self.c, 3, seed=8)
        self.assertEqual([f.val_patients for f in a], [f.val_patients for f in b])
        self.assertNotEqual([f.val_patients for f in a], [f.val_patients for f in c])

    def test_holdout(self):
        s = splits.patient_holdout(self.c, 0.2, stratify='label', seed=1)
        self.assertEqual(len(s.val_patients), 6)
        labels = Counter(self.c.patient_labels('label')[p] for p in s.val_patients)
        self.assertEqual(labels, Counter({'a': 4, 'b': 2}))
        self.assertFalse(set(s.train_slides) & set(s.val_slides))
        with self.assertRaises(ValueError):
            splits.patient_holdout(self.c, 0.0)

    def test_empty_strata_left_out(self):
        t = _table(6, 1); t.loc[0, 'label'] = ''
        folds = splits.patient_kfold(Cohort(t), 2, stratify='label')
        everyone = {p for f in folds for p in f.val_patients}
        self.assertNotIn('P00', everyone)
        self.assertEqual(len(everyone), 5)

    def test_errors(self):
        with self.assertRaises(ValueError):
            splits.patient_kfold(self.c, 1)
        with self.assertRaises(ValueError):
            splits.patient_kfold(Cohort(_table(2, 1)), 3)
        with self.assertRaises(ValueError):
            splits.Split(('a',), ('a',), (), ())

    def test_to_dict(self):
        d = splits.patient_holdout(self.c, 0.5).to_dict()
        self.assertEqual(set(d), {'train_patients', 'val_patients', 'train_slides', 'val_slides'})


if __name__ == '__main__':
    unittest.main()
