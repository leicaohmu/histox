.. _tutorial-feature-bags:

Generate feature bags from a WSI project
========================================

In this tutorial, you will turn tiles from an existing HistoX project into
one feature bag per slide. The result is the input used by many
multiple-instance learning (MIL) workflows.

This is a task tutorial: the steps form one continuous workflow and produce
files you can inspect. For model-specific checkpoint arguments, use the
:doc:`encoder Method Guides <../methods/encoders/index>`.

.. note::

   This draft has been checked against the current source signatures, but the
   complete workflow and displayed outputs have not yet been reproduced in a
   clean environment. Treat it as a preview until runtime validation is added.

What you will build
-------------------

.. grid:: 1 1 3 3
   :gutter: 2

   .. grid-item-card:: Input

      A HistoX project containing at least one whole-slide image and its
      annotation row.

   .. grid-item-card:: Processing

      Tissue detection, tile extraction, and ImageNet ResNet-50 feature
      encoding.

   .. grid-item-card:: Output

      A ``.pt`` feature tensor and ``.index.npz`` coordinate array for each
      slide, plus ``bags_config.json``.

Prerequisites
-------------

You need:

* an existing HistoX project with at least one readable WSI;
* enough disk space for TFRecords and feature bags;
* the PyTorch dependency group; and
* network access on the first run if the pretrained ResNet-50 weights are not
  already cached by TorchVision.

Install an editable development checkout with:

.. code-block:: bash

   cd /path/to/histox
   python -m pip install -e ".[torch]"

.. warning::

   Tile extraction and feature generation create derived data. Start with one
   or two slides, inspect the output size, and only then scale to a cohort.
   The example ``tile_um="20x"`` is a working scale, not a universal research
   choice.

1. Load the project
-------------------

Replace the path below with the directory containing the HistoX project
configuration.

.. code-block:: python

   from pathlib import Path
   import histox as hx

   project_root = Path("/path/to/my-histox-project")
   project = hx.load_project(str(project_root))

   dataset = project.dataset(tile_px=224, tile_um="20x")
   print(f"Slides selected: {len(dataset.slides())}")

Expected result: ``dataset.slides()`` contains the slide identifiers you intend
to process. If it is empty, fix the project sources or annotations before
continuing.

2. Prepare tiles
----------------

Feature-bag generation reads TFRecords. Check whether the selected tile size
and scale already have TFRecords:

.. code-block:: python

   tfrecords = dataset.tfrecords()
   print(f"TFRecords found: {len(tfrecords)}")

If the count is zero, extract tiles explicitly:

.. code-block:: python

   dataset.extract_tiles(qc="otsu")

Then rebuild the dataset and verify the result:

.. code-block:: python

   dataset = project.dataset(tile_px=224, tile_um="20x")
   tfrecords = dataset.tfrecords()
   if not tfrecords:
       raise RuntimeError("No TFRecords were created; inspect the extraction log.")

``qc="otsu"`` removes much of the empty background. Review the extracted tiles
before treating them as analysis-ready; tissue detection is not a substitute
for cohort-specific quality control.

3. Build and inspect an encoder
-------------------------------

This tutorial uses the registered ImageNet ResNet-50 extractor because it has
a stable public constructor and does not require you to supply a separate
checkpoint path.

.. code-block:: python

   import torch

   encoder = hx.build_feature_extractor(
       "resnet50_imagenet",
       tile_px=224,
       backend="torch",
   )
   print(f"Features per tile: {encoder.num_features}")

Confirm that ``encoder.num_features`` is ``2048``. HistoX applies the
extractor's resize and ImageNet normalization pipeline while reading the tile
batches used below.

4. Generate one bag per slide
-----------------------------

Write bags to an explicit directory so the output is easy to review:

.. code-block:: python

   output_dir = project_root / "pt_files" / "resnet50_imagenet_20x"

   bag_dir = project.generate_feature_bags(
       encoder,
       dataset=dataset,
       outdir=str(output_dir),
       batch_size=32,
       min_tiles=1,
   )
   print(f"Feature bags: {bag_dir}")

Reduce ``batch_size`` if GPU memory is insufficient. HistoX skips existing
``.pt`` files unless ``force_regenerate=True`` is supplied.

5. Validate the output
----------------------

Do not stop at a success message. Load one bag, load its coordinates, and
confirm that they contain the same number of tiles:

.. code-block:: python

   import json
   import numpy as np

   bag_dir = Path(bag_dir)
   bag_paths = sorted(bag_dir.glob("*.pt"))
   if not bag_paths:
       raise RuntimeError("No .pt bags were created; inspect the HistoX log.")

   bag_path = bag_paths[0]
   index_path = bag_path.with_suffix(".index.npz")

   bag = torch.load(bag_path, map_location="cpu")
   coordinates = np.load(index_path)["arr_0"]
   config = json.loads((bag_dir / "bags_config.json").read_text())

   print(f"Bag: {bag_path.name}")
   print(f"Feature shape: {tuple(bag.shape)}")
   print(f"Coordinate shape: {coordinates.shape}")
   print(f"Extractor: {config}")

   assert bag.ndim == 2
   assert bag.shape[1] == 2048
   assert bag.shape[0] == coordinates.shape[0]

For a slide with *N* retained tiles, the feature tensor has shape
``(N, 2048)``. The coordinate array preserves the spatial location associated
with each row. ``bags_config.json`` records the extractor configuration needed
to interpret and reproduce the bags.

6. Use the bags downstream
---------------------------

The output directory can now be used as the ``bags`` input to a compatible MIL
workflow. Before training, record:

* the HistoX version and Git revision;
* ``tile_px`` and ``tile_um``;
* tissue and quality-control settings;
* the encoder name and weight provenance; and
* the generated ``bags_config.json`` alongside any external checkpoint name.

To replace ResNet-50 with DINOv2, Virchow, or a local ViT checkpoint, change
only the extractor construction and expected feature dimension. The WSI,
TFRecord, export, and validation steps remain the same. See the
:doc:`encoder Method Guides <../methods/encoders/index>` for the required
arguments and licenses.

Run the complete example
------------------------

The companion script performs the non-destructive checks and feature export.
It intentionally refuses to extract tiles automatically because that step may
consume substantial storage.

:download:`Download feature_bag_tutorial.py <../../examples/feature_bag_tutorial.py>`

.. code-block:: bash

   python docs/examples/feature_bag_tutorial.py \
       --project /path/to/my-histox-project \
       --tile-um 20x \
       --outdir /path/to/output/bags

Troubleshooting
---------------

No TFRecords found
   Run the explicit tile-extraction step and confirm that ``tile_px`` and
   ``tile_um`` match when rebuilding the dataset.

No ``.pt`` files created
   Check the extraction log, dataset filters, minimum tile count, and whether
   every selected slide has a TFRecord.

CUDA out of memory
   Lower ``batch_size``. For a quick functional check, process fewer slides
   before changing model settings.

Unexpected feature width
   Confirm the extractor and layer configuration in ``bags_config.json``.
   Different encoders intentionally produce different feature dimensions.

What you learned
----------------

You loaded a real HistoX project, prepared slide tiles, inspected an encoder's
output width, generated feature bags, and checked the tensor-to-coordinate
mapping. This task boundary is the hand-off from WSI preprocessing to MIL.
