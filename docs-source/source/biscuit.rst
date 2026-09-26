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
.. autofunction:: histox.biscuit.Experiment.display
.. autofunction:: histox.biscuit.Experiment.plot_uq_calibration
.. autofunction:: histox.biscuit.Experiment.results
.. autofunction:: histox.biscuit.Experiment.thresholds_from_nested_cv
.. autofunction:: histox.biscuit.Experiment.train
.. autofunction:: histox.biscuit.Experiment.train_nested_cv

biscuit.hp
**********

.. autofunction:: histox.biscuit.hp.nature2022

biscuit.threshold
*****************
.. autofunction:: histox.biscuit.threshold.apply
.. autofunction:: histox.biscuit.threshold.detect
.. autofunction:: histox.biscuit.threshold.from_cv
.. autofunction:: histox.biscuit.threshold.plot_uncertainty
.. autofunction:: histox.biscuit.threshold.process_group_predictions
.. autofunction:: histox.biscuit.threshold.process_tile_predictions

biscuit.utils
*************

.. autofunction:: histox.biscuit.utils.auc
.. autofunction:: histox.biscuit.utils.auc_and_threshold
.. autofunction:: histox.biscuit.utils.df_from_cv
.. autofunction:: histox.biscuit.utils.eval_exists
.. autofunction:: histox.biscuit.utils.find_cv
.. autofunction:: histox.biscuit.utils.find_cv_early_stop
.. autofunction:: histox.biscuit.utils.find_eval
.. autofunction:: histox.biscuit.utils.find_model
.. autofunction:: histox.biscuit.utils.get_model_results
.. autofunction:: histox.biscuit.utils.get_eval_results
.. autofunction:: histox.biscuit.utils.model_exists
.. autofunction:: histox.biscuit.utils.prediction_metrics
.. autofunction:: histox.biscuit.utils.read_group_predictions
.. autofunction:: histox.biscuit.utils.truncate_colormap

biscuit.delong
**************

.. autofunction:: histox.biscuit.delong.fastDeLong
.. autofunction:: histox.biscuit.delong.delong_roc_variance
.. autofunction:: histox.biscuit.delong.delong_roc_test
