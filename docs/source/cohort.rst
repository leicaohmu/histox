.. currentmodule:: histox

histox.Cohort and histox.splits
===============================

A :class:`Cohort` holds the patients of a study, their slides, and their
labels, and nothing else: no tile sizes, TFRecords, or feature bags. The same
cohort can then be tiled, split, and modeled in different ways.

.. code-block:: python

    import histox as hx

    cohort = hx.Cohort.from_csv('annotations.csv', slides='/data/slides')
    cohort                                   # <Cohort 120 patients, 241 slides, fields=[...]>
    cohort.labels('subtype')                 # {slide: label}
    cohort.patient_labels('subtype')         # {patient: label}; error if a patient's slides disagree
    cohort.missing_slides()                  # annotated slides without a file

    luad = cohort.filter(subtype='LUAD')
    dataset.cohort                           # the cohort of an existing Dataset

The annotation table needs a ``slide`` column. A ``patient`` column groups
slides by patient; without one, each slide is its own patient.

Patient-level splits
********************

:mod:`histox.splits` assigns whole patients, never single slides, so no
patient contributes slides to both training and validation. Splits are
deterministic: the same cohort, arguments, and ``seed`` give the same result
on any machine.

.. code-block:: python

    folds = hx.splits.patient_kfold(cohort, k=5, stratify='subtype', seed=1)
    for fold in folds:
        train = cohort.subset(slides=fold.train_slides)
        val = cohort.subset(slides=fold.val_slides)

    holdout = hx.splits.patient_holdout(cohort, val_fraction=0.2, stratify='subtype')
    by_site = hx.splits.site_preserved_kfold(cohort, k=3, site='site', stratify='subtype')

With ``stratify``, each fold gets the same mix of label values as far as the
numbers allow, and patients with an empty value are left out.
:meth:`Dataset.split` is unchanged and keeps using its own split files.

API reference
*************

.. autoclass:: Cohort
   :members:

.. automodule:: histox.splits
   :members: Split, patient_kfold, patient_holdout, site_preserved_kfold
   :no-index:
