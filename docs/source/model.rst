.. currentmodule:: histox.model

histox.model
===============

This module provides the :class:`ModelParams` class to organize model and training
parameters/hyperparameters and assist with model building, as well as the :class:`Trainer` class that
executes model training and evaluation. :class:`RegressionTrainer` and :class:`SurvivalTrainer`
are extensions of this class, supporting regression and Cox Proportional Hazards outcomes, respectively. The function
:func:`build_trainer` can choose and return the correct model instance based on the provided
hyperparameters.

.. note::
    In order to support both Tensorflow and PyTorch backends, the :mod:`histox.model` module will import either
    :mod:`histox.model.tensorflow` or :mod:`histox.model.torch` according to the currently active backend,
    indicated by the environmental variable ``SF_BACKEND``.

See :ref:`training` for a detailed look at how to train models.

Trainer
*******
.. autoclass:: Trainer
.. autofunction:: histox.model.Trainer.load
.. autofunction:: histox.model.Trainer.evaluate
.. autofunction:: histox.model.Trainer.predict
.. autofunction:: histox.model.Trainer.train

RegressionTrainer
*****************
.. autoclass:: RegressionTrainer

SurvivalTrainer
***************
.. autoclass:: SurvivalTrainer

Features
********
.. autoclass:: Features
.. autofunction:: histox.model.Features.from_model
.. autofunction:: histox.model.Features.__call__

Other functions
***************
.. autofunction:: build_trainer
.. autofunction:: build_feature_extractor
.. autofunction:: list_extractors
.. autofunction:: load
.. autofunction:: is_tensorflow_model
.. autofunction:: is_tensorflow_tensor
.. autofunction:: is_torch_model
.. autofunction:: is_torch_tensor
.. autofunction:: read_hp_sweep
.. autofunction:: rebuild_extractor