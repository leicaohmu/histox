.. currentmodule:: histox

.. _dataset:

histox.Dataset
=================

.. autoclass:: Dataset

Attributes
----------

.. autosummary::

    Dataset.annotations

    Dataset.filters
    Dataset.filter_blank
    Dataset.filtered_annotations
    Dataset.img_format
    Dataset.min_tiles
    Dataset.num_tiles

Methods
-------

.. autofunction:: Dataset.balance
.. autofunction:: Dataset.build_index
.. autofunction:: Dataset.cell_segmentation
.. autofunction:: Dataset.check_duplicates
.. autofunction:: Dataset.clear_filters
.. autofunction:: Dataset.clip
.. autofunction:: Dataset.convert_xml_rois
.. autofunction:: Dataset.extract_cells
.. autofunction:: Dataset.extract_tiles
.. autofunction:: Dataset.extract_tiles_from_tfrecords
.. autofunction:: Dataset.filter
.. autofunction:: Dataset.find_slide
.. autofunction:: Dataset.find_tfrecord
.. autofunction:: Dataset.generate_feature_bags
.. autofunction:: Dataset.get_tfrecord_locations
.. autofunction:: Dataset.get_tile_dataframe
.. autofunction:: Dataset.harmonize_labels
.. autofunction:: Dataset.is_float
.. autofunction:: Dataset.kfold_split
.. autofunction:: Dataset.labels
.. autofunction:: Dataset.load_annotations
.. autofunction:: Dataset.load_indices
.. autofunction:: Dataset.manifest
.. autofunction:: Dataset.manifest_histogram
.. autofunction:: Dataset.patients
.. autofunction:: Dataset.get_bags
.. autofunction:: Dataset.read_tfrecord_by_location
.. autofunction:: Dataset.remove_filter
.. autofunction:: Dataset.rebuild_index
.. autofunction:: Dataset.resize_tfrecords
.. autofunction:: Dataset.rois
.. autofunction:: Dataset.slide_manifest
.. autofunction:: Dataset.slide_paths
.. autofunction:: Dataset.slides
.. autofunction:: Dataset.split
.. autofunction:: Dataset.split_tfrecords_by_roi
.. autofunction:: Dataset.summary
.. autofunction:: Dataset.tensorflow
.. autofunction:: Dataset.tfrecord_report
.. autofunction:: Dataset.tfrecord_heatmap
.. autofunction:: Dataset.tfrecords
.. autofunction:: Dataset.tfrecords_by_subfolder
.. autofunction:: Dataset.tfrecords_folders
.. autofunction:: Dataset.tfrecords_from_tiles
.. autofunction:: Dataset.tfrecords_have_locations
.. autofunction:: Dataset.transform_tfrecords
.. autofunction:: Dataset.thumbnails
.. autofunction:: Dataset.torch
.. autofunction:: Dataset.unclip
.. autofunction:: Dataset.update_manifest
.. autofunction:: Dataset.update_annotations_with_slidenames
.. autofunction:: Dataset.verify_annotations_slides
.. autofunction:: Dataset.verify_img_format
.. autofunction:: Dataset.verify_slide_names
