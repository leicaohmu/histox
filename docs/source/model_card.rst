.. currentmodule:: histox.model

Model cards
===========

A :class:`ModelCard` describes a model: name, license, citation, where its
weights come from, the input size and normalization its preprocessing
applies, and the feature dimension it outputs. Fields that are not known stay
``None``; values come from the model's own code and configuration.

.. code-block:: python

    import histox as hx

    # Before building (no weights are downloaded): terms only
    hx.model.extractor_card('dinov2')

    # After building: everything that is known
    extractor = hx.build_feature_extractor('dinov2', model_id='facebook/dinov2-small')
    print(extractor.card.summary())
    # dinov2 (tile_encoder, torch)
    #   features: 384
    #   input: 224 px
    #   normalization: mean=[...] std=[...]
    #   weights: model_id=facebook/dinov2-small
    #   license: Apache-2.0
    #   citation: available (.citation)

    # The underlying PyTorch network, e.g. for fine-tuning
    extractor.module

Check the license of a model and of its weights before using it. A card
records the license stated in HistoX's code; the provider's current terms are
authoritative.

.. autoclass:: ModelCard
   :members:

.. autofunction:: extractor_card
