.. _mil_api:

.. currentmodule:: histox.mil

histox.mil
==============

This submodule contains tools for multiple-instance learning (MIL) model training and evaluation. See :ref:`mil` for more information. A summary of the API is given below.

**Training:**
    - :func:`train_mil()`: Train an MIL model, using an MIL configuration, Datasets, and a directory of bags.
    - :func:`build_fastai_learner()`: Build and return the FastAI Learner, but do not execute training. Useful for customizing training.
    - :func:`build_multimodal_learner()`: Build and return a FastAI Learner designed for multi-modal/multi-magnification input.

**Evaluation/Inference:**
    - :func:`eval_mil()`: Evaluate an MIL model using a path to a saved model, a Dataset, and path to bags. Generates metrics.
    - :func:`predict_mil()`: Generate predictions from an MIL model and saved bags. Returns a pandas dataframe.
    - :func:`predict_multimodal_mil()`: Generate predictions from a multimodal MIL model. Returns a dataframe.
    - :func:`predict_slide()`: Generate MIL predictions for a single slide. Returns a 2D array of predictions and attention.
    - :func:`predict_from_bags()`: Low-level interface for generating predictions from a loaded MIL model and pre-loaded bag Tensors.
    - :func:`predict_from_multimodal_bags()`: Low-level interface for generating multimodal predictions from a loaded MIL model and bag Tensors.
    - :func:`get_mil_tile_predictions()`: Get tile-level predictions and attention from a saved MIL model for a given Dataset and saved bags.
    - :func:`generate_attention_heatmaps()`: Generate and save attention heatmaps.
    - :func:`generate_mil_features()`: Get last-layer activations from an MIL model. Returns an MILFeatures object.


Main functions
**************

.. autofunction:: mil_config
.. autofunction:: train_mil
.. autofunction:: build_fastai_learner
.. autofunction:: build_multimodal_learner
.. autofunction:: eval_mil
.. autofunction:: predict_mil
.. autofunction:: predict_multimodal_mil
.. autofunction:: predict_from_bags
.. autofunction:: predict_from_multimodal_bags
.. autofunction:: predict_slide
.. autofunction:: get_mil_tile_predictions
.. autofunction:: generate_attention_heatmaps
.. autofunction:: generate_mil_features

TrainerConfig
*************

.. autoclass:: histox.mil.TrainerConfig
.. autosummary::

    TrainerConfig.model_fn
    TrainerConfig.loss_fn
    TrainerConfig.is_multimodal
    TrainerConfig.model_type

.. autofunction:: histox.mil.TrainerConfig.to_dict
.. autofunction:: histox.mil.TrainerConfig.json_dump
.. autofunction:: histox.mil.TrainerConfig.is_classification
.. autofunction:: histox.mil.TrainerConfig.get_metrics
.. autofunction:: histox.mil.TrainerConfig.prepare_training
.. autofunction:: histox.mil.TrainerConfig.build_model
.. autofunction:: histox.mil.TrainerConfig.predict
.. autofunction:: histox.mil.TrainerConfig.batched_predict
.. autofunction:: histox.mil.TrainerConfig.train
.. autofunction:: histox.mil.TrainerConfig.eval
.. autofunction:: histox.mil.TrainerConfig.build_train_dataloader
.. autofunction:: histox.mil.TrainerConfig.build_val_dataloader
.. autofunction:: histox.mil.TrainerConfig.inspect_batch
.. autofunction:: histox.mil.TrainerConfig.run_metrics

MILModelConfig
**************

.. autoclass:: MILModelConfig
.. autosummary::

    MILModelConfig.apply_softmax
    MILModelConfig.loss_fn
    MILModelConfig.model_fn
    MILModelConfig.model_type
    MILModelConfig.is_multimodal

.. autofunction:: histox.mil.MILModelConfig.is_classification
.. autofunction:: histox.mil.MILModelConfig.to_dict
.. autofunction:: histox.mil.MILModelConfig.inspect_batch
.. autofunction:: histox.mil.MILModelConfig.build_model
.. autofunction:: histox.mil.MILModelConfig.predict
.. autofunction:: histox.mil.MILModelConfig.batched_predict
.. autofunction:: histox.mil.MILModelConfig.run_metrics

CLAMModelConfig
***************

The CLAM model configuration class requires ``slideflow-gpl``, which can be installed with:

.. code-block:: bash

    pip install slideflow-gpl

Once installed, the class is available at ``histox.clam.CLAMModelConfig``.

.. autoclass:: histox.clam.CLAMModelConfig

