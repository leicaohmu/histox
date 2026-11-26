.. currentmodule:: histox.data

.. _data_datasets:

PyTorch datasets
================

:mod:`histox.data` provides PyTorch datasets for pathology data. They are
ordinary :class:`torch.utils.data.Dataset` subclasses: use them with a standard
:class:`~torch.utils.data.DataLoader` and your own training loop, or subclass
them to change how items are built.

.. note::

   These classes are **beta** (see the :doc:`release_notes`). They implement
   RFC 0002, proposal P2. Tile and slide datasets will follow
   :class:`BagDataset`.

Item format
-----------

Every dataset returns one ``dict`` per item, with standard keys:

.. list-table::
   :header-rows: 1
   :widths: 18 32 50

   * - Key
     - Type
     - Meaning
   * - ``features``
     - float32 tensor ``(N, F)``
     - One feature vector per tile.
   * - ``coords``
     - int64 tensor ``(N, 2)``
     - Tile centers, ``(x, y)`` in level-0 pixels (see :doc:`coordinates`).
       ``None`` if the bag has no tile index.
   * - ``slide``
     - ``str``
     - Slide name. A list of slide names for patient-level items.
   * - ``patient``
     - ``str``
     - Patient identifier.
   * - ``target``
     - tensor
     - Class index (``int64``) for categorical targets, value (``float32``)
       for numeric ones. Absent when no target is requested.

A ``transform`` callable receives this ``dict`` and returns it, so it can
change values or add keys.

Feature bags
------------

.. code-block:: python

    import torch
    from torch.utils.data import DataLoader
    import histox as hx

    cohort = hx.Cohort.from_csv("annotations.csv")
    fold = hx.splits.patient_holdout(cohort, 0.2, stratify="subtype")

    train = hx.data.BagDataset(
        cohort.subset(slides=fold.train_slides),
        "bags/",
        target="subtype",
        max_tiles=4096,          # random subset per access, for large slides
    )
    loader = DataLoader(train, batch_size=8, shuffle=True,
                        collate_fn=hx.data.collate_bags, num_workers=4)

    for batch in loader:
        batch["features"]        # (B, N_max, F), zero-padded
        batch["mask"]            # (B, N_max), True for real tiles
        batch["lens"]            # (B,), number of real tiles
        batch["target"]          # (B,), class indices; see train.classes

Pass ``level="patient"`` for one item per patient, with the bags of the
patient's slides joined. Cohort slides without a bag are skipped with a
warning, or raise an error with ``missing="error"``.

API
---

.. autoclass:: BagDataset
   :members: slides, patients, num_features

.. autofunction:: collate_bags
