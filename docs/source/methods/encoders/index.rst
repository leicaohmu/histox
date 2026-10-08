.. _methods-encoders:

Image encoders
==============

Image encoders convert pathology tiles into feature vectors. Those vectors can
be used for multiple-instance learning, retrieval, clustering, visualization,
or other downstream models. These Method Guides document the exact constructor,
weights contract, output shape, and limitations of encoders implemented in the
current HistoX source tree.

For a continuous public-SVS-to-feature-bag workflow with captured output,
follow the executable :doc:`DINOv2 whole-slide example
<../../auto_examples/feature_extraction/plot_dinov2_wsi>`.

Choose an encoder
-----------------

.. grid:: 1 1 3 3
   :gutter: 2

   .. grid-item-card:: DINOv2
      :link: dinov2
      :link-type: doc

      Load a Hugging Face checkpoint, or an original configuration and its
      matching teacher checkpoint.

      **Output:** checkpoint-dependent; 384 for ``dinov2-small``

      **Source:** model ID, or YAML + teacher checkpoint

   .. grid-item-card:: Virchow
      :link: virchow
      :link-type: doc

      Use Paige's pathology foundation-model representation.

      **Output:** 2,560 values per tile

      **License:** Apache-2.0; gated access terms apply

   .. grid-item-card:: Vision Transformer
      :link: vit
      :link-type: doc

      Load a local checkpoint into a configurable HistoX ViT.

      **Output:** 192, 384, or 768 values

      **Sizes:** tiny, small, or base

Install the common runtime
--------------------------

All three encoders require the PyTorch dependencies:

.. code-block:: bash

   python -m pip install "histox[torch]"

For an editable checkout, use ``python -m pip install -e ".[torch]"`` from the
repository root. DINOv2 has one additional dependency described in its
:doc:`dedicated guide <dinov2>`.

Common input contract
---------------------

The registered encoders accept a ``torch.uint8`` tensor in either channel-last
``(batch, height, width, 3)`` or channel-first
``(batch, 3, height, width)`` layout. They return a ``torch.float32`` feature
tensor shaped ``(batch, feature_dimension)``.

The same extractor object can be called on a :class:`histox.WSI` or a TFRecord,
and can be passed to :meth:`histox.Dataset.generate_feature_bags` for a prepared
dataset. See the :doc:`feature extraction overview <../../features>` for the
shared API, or run the :doc:`DINOv2 whole-slide example
<../../auto_examples/feature_extraction/plot_dinov2_wsi>` for an executed task.

.. note::

   HistoX does not make third-party model weights part of the Python package.
   Keep model files in project storage, record their source and checksum, and
   follow the upstream license.

.. toctree::
   :hidden:
   :maxdepth: 1

   dinov2
   virchow
   vit
