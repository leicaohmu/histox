.. currentmodule:: histox

.. _project:

histox.Project
=================

.. autoclass:: Project

Attributes
----------

.. autosummary::

    Project.annotations
    Project.dataset_config
    Project.eval_dir
    Project.models_dir
    Project.name
    Project.neptune_api
    Project.neptune_workspace
    Project.sources

Methods
-------

.. autofunction:: Project.add_source

.. autofunction:: Project.associate_slide_names

.. autofunction:: Project.cell_segmentation

.. autofunction:: Project.create_blank_annotations

.. autofunction:: Project.create_hp_sweep

.. autofunction:: Project.evaluate

.. autofunction:: Project.evaluate_mil

.. autofunction:: Project.extract_cells

.. autofunction:: Project.extract_tiles

.. autofunction:: Project.gan_train

.. autofunction:: Project.gan_generate

.. autofunction:: Project.generate_features

.. autofunction:: Project.generate_feature_bags

.. autofunction:: Project.generate_heatmaps

.. autofunction:: Project.generate_mosaic

.. autofunction:: Project.generate_mosaic_from_annotations

.. autofunction:: Project.generate_tfrecord_heatmap

.. autofunction:: Project.dataset

.. autofunction:: Project.predict

.. autofunction:: Project.predict_ensemble

.. autofunction:: Project.predict_wsi

.. autofunction:: Project.save

.. autofunction:: Project.smac_search

.. autofunction:: Project.train

.. autofunction:: Project.train_ensemble

.. autofunction:: Project.train_mil

.. autofunction:: Project.train_simclr
