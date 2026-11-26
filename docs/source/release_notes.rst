.. _release_notes:

Release notes
=============

HistoX follows the stability policy of
`RFC 0001 <https://github.com/leicaohmu/histox/blob/develop/rfcs/0001-core-abstractions.md>`_,
section 6. Features are **stable**, **beta**, or **prototype**. A deprecated
feature emits :class:`histox.HistoXDeprecationWarning`, names its replacement,
and stays for at least two minor releases. Earlier releases are listed on
`GitHub <https://github.com/leicaohmu/histox/tags>`_.

0.4.0 (in development)
----------------------

This release implements RFC 0002: documented base classes that users subclass
to extend HistoX, and training with ordinary PyTorch loops.

New features
************

- :class:`histox.data.BagDataset` and :func:`histox.data.collate_bags`: a
  PyTorch dataset of feature bags built from a :class:`histox.Cohort`, one item
  per slide or per patient, returning ``dict`` items. ``collate_bags`` pads the
  bags in a batch and returns a ``mask`` and ``lens``. See
  :doc:`data_datasets`. The existing ``histox.mil.data.BagDataset`` used by
  the MIL trainer is unchanged.

0.3.0
-----

This release adds the core abstractions proposed in RFC 0001. Each is a small,
documented type that replaces several ad hoc conventions in the code. They are
**beta**: the API may still change, with a note here.

New core abstractions
*********************

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - API
     - What it is
   * - :doc:`histox.registry <registry>`
     - One registry type for every pluggable component. Workflows,
       feature extractors, and MIL models now use it.
   * - :doc:`histox.geometry <coordinates>`
     - One documented coordinate convention: ``(x, y)`` in level-0 pixels,
       tiles referenced by their center.
   * - :doc:`histox.Bag <bag>`
     - A feature bag with its coordinates and provenance, loaded and saved
       as one object.
   * - :doc:`histox.TileSpec <tile_spec>`
     - Tile size, resolution, stride, and origin as one validated value,
       accepted by ``WSI``, ``Dataset``, and ``Project.extract_tiles``.
   * - :doc:`ModelCard <model_card>`
     - License, citation, weights, and input contract of a feature
       extractor, available as ``extractor.card``.
   * - ``histox.mil.SlideAggregator``
     - One calling convention for MIL models, used by MIL inference and
       ``MILFeatures``.
   * - :doc:`histox.SpatialMap <spatial>`
     - One type for slide-aligned arrays: QC masks, heatmaps, segmentations,
       and attention maps.
   * - :doc:`histox.transforms <transforms>`
     - QC methods, stain normalizers, and augmentations as composable
       transforms, with ``qc`` and ``normalizer`` registries.
   * - :doc:`histox.Cohort and histox.splits <cohort>`
     - Patients, slides, and labels, with patient-level k-fold, holdout, and
       site-preserved splits.
   * - :doc:`Tasks and methods <workflows_api>`
     - ``ClassificationTask``, ``RegressionTask``, and the ``mil`` and
       ``tile_classifier`` methods for :class:`histox.workflows.Experiment`.

Other new features
******************

- PS3 multimodal survival model, the PLIP feature extractor, and the
  :mod:`histox.workflows` package (``Task``, ``Method``, ``Experiment``).
- ``extractor(wsi, tfrecord_dir=...)`` writes the tiles it reads to TFRecords
  while computing features, so one pass over the slides serves later passes.
- ``histox.util.prefetch``: slides are read sequentially into the page cache
  before their tiles are read. Cold tile extraction on spinning disks was
  3.2 times faster in our benchmark. Set ``HX_PREFETCH=0`` to disable it.

Changes in behavior
*******************

- **Python 3.11 or newer is required.** HistoX follows the Python versions
  supported by current PyTorch releases and is tested on Python 3.11, 3.12,
  and 3.13. Projects on older Python versions can stay on ``0.2.x``.
- **Feature bags are stored as float16.** All HistoX readers return float32,
  and existing float32 bags still load. Bags with values outside the float16
  range are stored as float32 automatically. To keep float32 on disk, pass
  ``dtype=torch.float32`` to ``DatasetFeatures.to_torch`` or ``Bag.save``.
- **NumPy 2 and pandas 3 are supported.** Code paths that used APIs removed
  in these releases were fixed.
- **Two dependency groups are now optional.** HistoX Studio's graphical
  dependencies moved to ``histox[studio]``, and SMAC3 and ConfigSpace for
  Bayesian hyperparameter search moved to ``histox[hpo]``. Both were
  previously installed by default. Using either feature without its group
  raises an ``ImportError`` that names the group to install.
- The legacy ``histox[tf]`` group now installs TensorFlow 2.12 to 2.15, which
  supports Python 3.11 and still uses Keras 2.
- On Linux with the cuCIM backend, reading tiles directly from slides uses
  a process pool by default instead of threads. In our benchmark,
  end-to-end feature extraction was 2.5 times faster.

Deprecations
************

- **The TensorFlow backend** is deprecated and will be removed no earlier
  than 0.5. Selecting it, or using a TensorFlow code path, emits
  :class:`histox.HistoXDeprecationWarning`. Use the PyTorch backend
  (``HX_BACKEND=torch``). New APIs in this release are PyTorch-only.

Bug fixes
*********

- ``Segmentation.extract_centroids`` mixed the ``(x, y)`` offset with
  ``(row, col)`` centroids and used the wrong scale, so tiles were read at
  shifted positions.
- Heatmap extents and the segmentation tile dataset recomputed the tile size
  from rounded microns, so overlays drifted for tiles given as a
  magnification.
- ``Segmentation.apply_rois`` ignored the mask offset, and empty cell masks
  crashed ``seg_utils.sparse_mask``.
- ``hx.Mosaic(images, coords)`` always failed with array input.
- ``WSI.extract_cells(apply_masks=True)`` with libvips scaled pixels by the
  cell label instead of masking them.
- Cell segmentation dropped the highest-labelled cell of every slide.
- ``Dataset.filter_bags_by_roi`` wrote filtered bags without a tile index.
- GoogLeNet failed to train without pretrained weights.
- Training a MIL classifier with class weighting failed on pandas 3.
- JPEG decoding in stain normalizers, ``Mosaic``, and direct feature
  extraction failed on NumPy 2.
- ``hx.mil`` and other subpackages were available after ``import histox``
  only if some other code had already imported them.
