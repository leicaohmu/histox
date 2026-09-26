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

.. autofunction:: histox.Project.add_source

.. autofunction:: histox.Project.associate_slide_names

.. autofunction:: histox.Project.cell_segmentation

.. autofunction:: histox.Project.create_blank_annotations

.. autofunction:: histox.Project.create_hp_sweep

.. autofunction:: histox.Project.evaluate

.. autofunction:: histox.Project.evaluate_mil

.. autofunction:: histox.Project.extract_cells

.. autofunction:: histox.Project.extract_tiles

.. autofunction:: histox.Project.gan_train

.. autofunction:: histox.Project.gan_generate

.. autofunction:: histox.Project.generate_features

.. autofunction:: histox.Project.generate_feature_bags

.. autofunction:: histox.Project.generate_heatmaps

.. autofunction:: histox.Project.generate_mosaic

.. autofunction:: histox.Project.generate_mosaic_from_annotations

.. autofunction:: histox.Project.generate_tfrecord_heatmap

.. autofunction:: histox.Project.dataset

.. autofunction:: histox.Project.predict

.. autofunction:: histox.Project.predict_ensemble

.. autofunction:: histox.Project.predict_wsi

.. autofunction:: histox.Project.save

.. autofunction:: histox.Project.smac_search

.. autofunction:: histox.Project.train

.. autofunction:: histox.Project.train_ensemble

.. autofunction:: histox.Project.train_mil

.. autofunction:: histox.Project.train_simclr
