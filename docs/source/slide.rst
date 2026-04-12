.. currentmodule:: histox.slide

histox.slide
===============

This module contains classes to load slides and extract tiles. For optimal performance, tile extraction should
generally not be performed by instancing these classes directly, but by calling either
:func:`histox.Project.extract_tiles` or :func:`histox.Dataset.extract_tiles`, which include performance
optimizations and additional functionality.

histox.WSI
*************

.. autoclass:: WSI

Attributes
----------

.. autosummary::

    WSI.dimensions
    WSI.qc_mask
    WSI.levels
    WSI.level_dimensions
    WSI.level_downsamples
    WSI.level_mpp
    WSI.properties
    WSI.slide
    WSI.vendor

Methods
-------

.. autofunction:: WSI.align_to
.. autofunction:: WSI.align_tiles_to
.. autofunction:: WSI.apply_qc_mask
.. autofunction:: WSI.apply_segmentation
.. autofunction:: WSI.area
.. autofunction:: WSI.build_generator
.. autofunction:: WSI.dim_to_mpp
.. autofunction:: WSI.get_tile_mask
.. autofunction:: WSI.get_tile_dataframe
.. autofunction:: WSI.extract_cells
.. autofunction:: WSI.extract_tiles
.. autofunction:: WSI.export_rois
.. autofunction:: WSI.has_rois
.. autofunction:: WSI.load_csv_roi
.. autofunction:: WSI.load_json_roi
.. autofunction:: WSI.load_roi_array
.. autofunction:: WSI.mpp_to_dim
.. autofunction:: WSI.predict
.. autofunction:: WSI.preview
.. autofunction:: WSI.process_rois
.. autofunction:: WSI.show_alignment
.. autofunction:: WSI.square_thumb
.. autofunction:: WSI.qc
.. autofunction:: WSI.remove_qc
.. autofunction:: WSI.remove_roi
.. autofunction:: WSI.tensorflow
.. autofunction:: WSI.torch
.. autofunction:: WSI.thumb
.. autofunction:: WSI.verify_alignment
.. autofunction:: WSI.view

Other functions
***************

.. autofunction:: slide.predict