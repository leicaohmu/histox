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

.. autofunction:: histox.Dataset.balance
.. autofunction:: histox.Dataset.build_index
.. autofunction:: histox.Dataset.cell_segmentation
.. autofunction:: histox.Dataset.check_duplicates
.. autofunction:: histox.Dataset.clear_filters
.. autofunction:: histox.Dataset.clip
.. autofunction:: histox.Dataset.convert_xml_rois
.. autofunction:: histox.Dataset.extract_cells
.. autofunction:: histox.Dataset.extract_tiles
.. autofunction:: histox.Dataset.extract_tiles_from_tfrecords
.. autofunction:: histox.Dataset.filter
.. autofunction:: histox.Dataset.find_slide
.. autofunction:: histox.Dataset.find_tfrecord
.. autofunction:: histox.Dataset.generate_feature_bags
.. autofunction:: histox.Dataset.get_tfrecord_locations
.. autofunction:: histox.Dataset.get_tile_dataframe
.. autofunction:: histox.Dataset.harmonize_labels
.. autofunction:: histox.Dataset.is_float
.. autofunction:: histox.Dataset.kfold_split
.. autofunction:: histox.Dataset.labels
.. autofunction:: histox.Dataset.load_annotations
.. autofunction:: histox.Dataset.load_indices
.. autofunction:: histox.Dataset.manifest
.. autofunction:: histox.Dataset.manifest_histogram
.. autofunction:: histox.Dataset.patients
.. autofunction:: histox.Dataset.get_bags
.. autofunction:: histox.Dataset.read_tfrecord_by_location
.. autofunction:: histox.Dataset.remove_filter
.. autofunction:: histox.Dataset.rebuild_index
.. autofunction:: histox.Dataset.resize_tfrecords
.. autofunction:: histox.Dataset.rois
.. autofunction:: histox.Dataset.slide_manifest
.. autofunction:: histox.Dataset.slide_paths
.. autofunction:: histox.Dataset.slides
.. autofunction:: histox.Dataset.split
.. autofunction:: histox.Dataset.split_tfrecords_by_roi
.. autofunction:: histox.Dataset.summary
.. autofunction:: histox.Dataset.tensorflow
.. autofunction:: histox.Dataset.tfrecord_report
.. autofunction:: histox.Dataset.tfrecord_heatmap
.. autofunction:: histox.Dataset.tfrecords
.. autofunction:: histox.Dataset.tfrecords_by_subfolder
.. autofunction:: histox.Dataset.tfrecords_folders
.. autofunction:: histox.Dataset.tfrecords_from_tiles
.. autofunction:: histox.Dataset.tfrecords_have_locations
.. autofunction:: histox.Dataset.transform_tfrecords
.. autofunction:: histox.Dataset.thumbnails
.. autofunction:: histox.Dataset.torch
.. autofunction:: histox.Dataset.unclip
.. autofunction:: histox.Dataset.update_manifest
.. autofunction:: histox.Dataset.update_annotations_with_slidenames
.. autofunction:: histox.Dataset.verify_annotations_slides
.. autofunction:: histox.Dataset.verify_img_format
.. autofunction:: histox.Dataset.verify_slide_names
