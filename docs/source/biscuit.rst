.. currentmodule:: histox.biscuit

histox.biscuit
=================

This module contains an official implementation of `BISCUIT <https://www.nature.com/articles/s41467-022-34025-x>`__, an uncertainty quantification and confidence thresholding algorithm for whole-slide images. The original implementation, which includes instructions for reproducing experimental results reported in the manuscript, is available on `GitHub <https://github.com/jamesdolezal/biscuit>`__.

This module is requires the ``slideflow-noncommercial`` package, which can be installed with:

.. code-block:: bash

    pip install slideflow-noncommercial

See :ref:`uncertainty` for more information.

.. autofunction:: find_cv
.. autofunction:: get_model_results

biscuit.Experiment
******************
.. autoclass:: Experiment
.. autofunction:: biscuit.Experiment.display
.. autofunction:: biscuit.Experiment.plot_uq_calibration
.. autofunction:: biscuit.Experiment.results
.. autofunction:: biscuit.Experiment.thresholds_from_nested_cv
.. autofunction:: biscuit.Experiment.train
.. autofunction:: biscuit.Experiment.train_nested_cv

biscuit.hp
**********

.. autofunction:: biscuit.hp.nature2022

biscuit.threshold
*****************
.. autofunction:: biscuit.threshold.apply
.. autofunction:: biscuit.threshold.detect
.. autofunction:: biscuit.threshold.from_cv
.. autofunction:: biscuit.threshold.plot_uncertainty
.. autofunction:: biscuit.threshold.process_group_predictions
.. autofunction:: biscuit.threshold.process_tile_predictions

biscuit.utils
*************

.. autofunction:: biscuit.utils.auc
.. autofunction:: biscuit.utils.auc_and_threshold
.. autofunction:: biscuit.utils.df_from_cv
.. autofunction:: biscuit.utils.eval_exists
.. autofunction:: biscuit.utils.find_cv
.. autofunction:: biscuit.utils.find_cv_early_stop
.. autofunction:: biscuit.utils.find_eval
.. autofunction:: biscuit.utils.find_model
.. autofunction:: biscuit.utils.get_model_results
.. autofunction:: biscuit.utils.get_eval_results
.. autofunction:: biscuit.utils.model_exists
.. autofunction:: biscuit.utils.prediction_metrics
.. autofunction:: biscuit.utils.read_group_predictions
.. autofunction:: biscuit.utils.truncate_colormap

biscuit.delong
**************

.. autofunction:: biscuit.delong.fastDeLong
.. autofunction:: biscuit.delong.delong_roc_variance
.. autofunction:: biscuit.delong.delong_roc_test
