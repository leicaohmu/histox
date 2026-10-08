.. _method-dinov2:

DINOv2
========

``"dinov2"`` is HistoX's tile feature extractor for DINOv2 checkpoints. Use
this page to choose and configure the extractor. For a complete public-SVS
workflow with real output, open the :doc:`DINOv2 whole-slide example
<../../auto_examples/feature_extraction/plot_dinov2_wsi>`.

What this implementation does
-----------------------------

HistoX accepts either a Hugging Face model ID or an original DINOv2
configuration-and-weights pair. For Hugging Face models, the current adapter
uses a 256-pixel resize, 224-pixel center crop, bicubic interpolation, and
ImageNet normalization unless transform overrides are supplied. It does not
read preprocessing values from ``AutoImageProcessor``. The extractor runs in
evaluation mode and returns one embedding per tile.
``facebook/dinov2-small`` returns 384 values per tile.

.. list-table:: Implementation contract
   :header-rows: 1
   :widths: 25 75

   * - Item
     - Current behavior
   * - Registry name
     - ``"dinov2"``
   * - Public factory
     - ``histox.build_feature_extractor("dinov2", ...)``
   * - Tile input
     - uint8 channel-last batch, shape ``(N, H, W, 3)``
   * - Tile output
     - float tensor, shape ``(N, D)``
   * - Whole-slide output
     - masked spatial feature grid, shape ``(grid_y, grid_x, D)``
   * - Hugging Face license
     - determined by the selected checkpoint; verify before use
   * - HistoX adapter license
     - Apache-2.0

Install
-------

Install HistoX with the PyTorch and Hugging Face integrations:

.. code-block:: bash

   python -m pip install "histox[torch,huggingface]"

The original DINOv2 configuration-and-weights path additionally requires the
``facebookresearch/dinov2`` source package and OmegaConf.

Quickstart
----------

Build a pinned Hugging Face checkpoint and verify it on a small tile batch
before processing a whole slide:

.. code-block:: python

   import torch
   import histox as hx

   encoder = hx.build_feature_extractor(
       "dinov2",
       model_id="facebook/dinov2-small",
       revision="ed25f3a31f01632728cabb09d1542f84ab7b0056",
       device="cuda",
       mixed_precision=False,
   )

   tiles = torch.randint(
       0,
       256,
       size=(2, 224, 224, 3),
       dtype=torch.uint8,
   )
   features = encoder(tiles)

   print(features.shape)  # torch.Size([2, 384])

Use a mirror only when needed
-----------------------------

Set the standard Hugging Face endpoint **before** importing HistoX or creating
the extractor:

.. code-block:: python

   import os
   os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

   import histox as hx

The model ID and pinned revision remain unchanged. For an offline workflow,
download the snapshot in advance and pass either its local path as ``model_id``
or set ``local_files_only=True``.

Constructor options
-------------------

``model_id``
   Hugging Face model ID or local snapshot path. Use this path for official
   checkpoints such as ``facebook/dinov2-small``.

``revision``
   Immutable Hugging Face commit or revision. Pin this in reproducible work.

``local_files_only``
   When ``True``, do not contact Hugging Face. Loading fails if the snapshot is
   not already available locally.

``cfg`` and ``weights``
   Original DINOv2 YAML configuration and weights. They must be provided
   together and cannot be combined with ``model_id``.

``device``
   PyTorch device used for inference, for example ``"cuda"`` or ``"cpu"``.

The common feature-extractor options, including mixed precision and transform
overrides, are passed through the HistoX extractor factory.

Whole-slide example
-------------------

The executable gallery example performs all of the following steps from one
maintained Python source file:

#. downloads and checksum-verifies a public CC0 SVS;
#. opens the slide at a defined tile size and field of view;
#. verifies one tile and one embedding;
#. extracts a masked spatial feature grid;
#. maps PCA scores back to tile coordinates;
#. saves and reloads the feature bag with model provenance.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Run the complete example
      :link: ../../auto_examples/feature_extraction/plot_dinov2_wsi
      :link-type: doc

      View real output, figures, runtime information, and downloadable Python
      and Jupyter versions.

   .. grid-item-card:: Learn the feature workflow
      :link: ../../features
      :link-type: doc

      Understand feature grids, bags, coordinates, and downstream use.

Limitations and research checks
-------------------------------

* DINOv2 is a general vision foundation model, not a clinical device and not a
  pathology-specific validation claim.
* Results depend on magnification, tile size, tissue filtering, color
  variation, and checkpoint revision.
* Check the selected checkpoint's license and intended use separately from the
  HistoX adapter license.
* Split cohorts by patient before training downstream models. Do not allow
  tiles or slides from one patient to cross data splits.
* Record the model ID, immutable revision, HistoX version, and slide-processing
  parameters with every generated feature bag.

Troubleshooting
---------------

``Provide model_id, or provide both cfg and weights``
   Choose exactly one loading route. For the simplest setup, pass a Hugging
   Face ``model_id``.

``Loading DINOv2 from Hugging Face requires transformers``
   Install ``histox[huggingface]`` in the environment running extraction.

Network or endpoint errors
   Confirm ``HF_ENDPOINT`` is set before model construction, or use a local
   snapshot with ``local_files_only=True``.

Unexpected output shape
   Check the checkpoint variant. ``dinov2-small`` has 384 features; larger
   variants use different embedding dimensions.

API and source
--------------

* :func:`histox.build_feature_extractor`
* :class:`histox.model.extractors.dinov2.DinoV2Features`
* ``histox/model/extractors/dinov2.py`` in the HistoX repository

The DINOv2 citation stored by the adapter is available through
``encoder.cite()``.

See also
--------

* :doc:`Examples gallery <../../auto_examples/index>`
* :doc:`Image encoders <index>`
* :doc:`Feature extraction <../../features>`
