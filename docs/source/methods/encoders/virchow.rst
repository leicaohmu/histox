.. _method-virchow:

Virchow
========

**Method Guide.** Use this page to configure Virchow correctly. For the shared
WSI and feature-bag interfaces, start with the :doc:`feature extraction
overview <../../features>`.

Virchow is a pathology foundation-model encoder developed by Paige. The HistoX
integration builds the Virchow architecture, loads a compatible local PyTorch
state dictionary, and combines its class token with the mean of its patch
tokens to produce a 2,560-dimensional tile representation.

At a glance
-----------

.. list-table::
   :widths: 30 70

   * - Registry name
     - ``"virchow"``
   * - Constructor
     - :func:`histox.build_feature_extractor`
   * - Required file
     - Compatible local Virchow PyTorch state dictionary
   * - Batch input
     - 224 x 224 ``torch.uint8`` tiles
   * - Batch output
     - ``torch.float32`` tensor with shape ``(batch, 2560)``
   * - Weight license
     - Apache-2.0; gated access and the repository's current terms apply

Before you begin
----------------

Install the PyTorch runtime:

.. code-block:: bash

   python -m pip install "histox[torch]"

Obtain the model from the `upstream Virchow repository
<https://huggingface.co/paige-ai/Virchow>`_ and review its access and license
terms. Access currently requires a Hugging Face account, approval, and
acceptance of the repository terms. The HistoX loader uses ``torch.load`` and
therefore expects a compatible PyTorch state dictionary rather than a model
identifier.

Quickstart: encode a tile batch
-------------------------------

.. code-block:: python

   import torch
   import histox as hx

   encoder = hx.build_feature_extractor(
       "virchow",
       weights="/path/to/virchow_weights.pth",
   )

   tiles = torch.randint(
       0,
       256,
       size=(2, 224, 224, 3),
       dtype=torch.uint8,
   )
   features = encoder(tiles)

   print(features.shape)  # torch.Size([2, 2560])
   print(features.dtype)  # torch.float32

HistoX applies the registered bicubic interpolation and normalization pipeline.
Supplying 224 x 224 tiles avoids changing the spatial scale expected by the
upstream model.

Generate whole-slide feature bags
---------------------------------

Prepare a dataset with 224-pixel tiles and a physical field of view consistent
with the model and your study, then run:

.. code-block:: python

   dataset.generate_feature_bags(
       encoder,
       outdir="/path/to/features/virchow",
       batch_size=16,
   )

Virchow has a large ViT-H backbone. Start with a smaller batch size than for a
compact encoder, then increase it only after checking GPU memory use.

License boundary
----------------

The current upstream model card identifies Virchow as Apache-2.0 and separately
requires users to accept gated-access terms. HistoX does not redistribute the
weights or replace those terms. Recheck the model repository before obtaining
or redistributing weights, derived artifacts, or a dependent workflow.

Common problems
---------------

Missing or unexpected state-dictionary keys
   Confirm that the file is the PyTorch state dictionary expected by the
   upstream Virchow architecture. The loader uses strict key matching.

``Expected input to be a uint8 tensor``
   Keep image batches as ``torch.uint8``. The extractor performs scaling and
   normalization internally.

GPU memory exhaustion
   Reduce ``batch_size``. For debugging, ``mixed_precision=False`` disables
   automatic mixed precision but generally uses more memory.

Source and citation
-------------------

The adapter is implemented by
``histox.model.extractors.virchow.VirchowFeatures``. Use ``encoder.cite()`` to
print the citation stored with the implementation and
``encoder.print_license()`` to display its registered license statement.

See also
--------

* :doc:`Image encoders <index>`
* :doc:`Feature extraction overview <../../features>`
* `Virchow model repository <https://huggingface.co/paige-ai/Virchow>`_
* :func:`histox.build_feature_extractor`
