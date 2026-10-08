.. _method-vit:

Vision Transformer
==================

**Method Guide.** Use this page to configure a local ViT correctly. For the
shared WSI and feature-bag interfaces, start with the :doc:`feature extraction
overview <../../features>`.

The registered ``"vit"`` extractor is a configurable Vision Transformer for
loading local checkpoints. Choose the architecture size that matches the
checkpoint; HistoX constructs that architecture and uses its class token as the
tile feature vector.

Choose a size
-------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - ``size``
     - Feature dimension
     - Default patch size
     - Transformer depth
   * - ``"tiny"``
     - 192
     - 16
     - 12
   * - ``"small"``
     - 384
     - 16
     - 12
   * - ``"base"``
     - 768
     - 16
     - 12

Before you begin
----------------

Install HistoX with the PyTorch dependencies:

.. code-block:: bash

   python -m pip install "histox[torch]"

Prepare a checkpoint for the selected architecture. The loader accepts either a
plain state dictionary or a mapping containing a ``teacher`` state dictionary.
It removes leading ``module.`` and ``backbone.`` prefixes before loading.

Quickstart: encode a tile batch
-------------------------------

.. code-block:: python

   import torch
   import histox as hx

   encoder = hx.build_feature_extractor(
       "vit",
       size="base",
       weights="/path/to/vit_base.pth",
   )

   tiles = torch.randint(
       0,
       256,
       size=(2, 224, 224, 3),
       dtype=torch.uint8,
   )
   features = encoder(tiles)

   print(features.shape)  # torch.Size([2, 768])
   print(features.dtype)  # torch.float32

Use ``size="tiny"`` or ``size="small"`` only with a checkpoint trained for the
matching architecture.

Generate whole-slide feature bags
---------------------------------

Once the encoder loads correctly on a small tile batch, reuse it for a prepared
dataset:

.. code-block:: python

   dataset.generate_feature_bags(
       encoder,
       outdir="/path/to/features/vit-base",
       batch_size=32,
   )

The resulting rows have 192, 384, or 768 values according to the selected
model size.

Checkpoint and license responsibility
-------------------------------------

HistoX does not provide a weight license for the generic ViT adapter. Record
the checkpoint source, checksum, training data, and license in your project.
The current loader uses ``strict=False``; a checkpoint may therefore load while
leaving some parameters unmatched. Validate the checkpoint architecture and
downstream output before using generated features in an experiment.

Common problems
---------------

``Unrecognized size``
   Use exactly ``"tiny"``, ``"small"``, or ``"base"``.

Shape or positional-embedding errors
   Confirm that the checkpoint uses the selected size, patch size, and image
   dimensions. Start with 224 x 224 tiles and the default patch size of 16.

Unexpected model behavior after loading
   Checkpoint loading uses ``strict=False``, and the current public extractor
   does not expose the missing/unexpected-key report. Treat successful
   construction as insufficient evidence of compatibility: compare the
   checkpoint architecture independently and verify embeddings on a known test
   set before generating a cohort.

Source and citation
-------------------

The adapter and architecture are implemented in
``histox.model.extractors.vit``. The stored Vision Transformer citation can be
printed with ``encoder.cite()``. The license for the checkpoint remains the
responsibility of its provider.

See also
--------

* :doc:`Image encoders <index>`
* :doc:`Feature extraction overview <../../features>`
* :func:`histox.build_feature_extractor`
