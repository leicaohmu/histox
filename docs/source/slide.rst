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

.. autofunction:: histox.WSI.align_to
.. autofunction:: histox.WSI.align_tiles_to
.. autofunction:: histox.WSI.apply_qc_mask
.. autofunction:: histox.WSI.apply_segmentation
.. autofunction:: histox.WSI.area
.. autofunction:: histox.WSI.build_generator
.. autofunction:: histox.WSI.dim_to_mpp
.. autofunction:: histox.WSI.get_tile_mask
.. autofunction:: histox.WSI.get_tile_dataframe
.. autofunction:: histox.WSI.extract_cells
.. autofunction:: histox.WSI.extract_tiles
.. autofunction:: histox.WSI.export_rois
.. autofunction:: histox.WSI.has_rois
.. autofunction:: histox.WSI.load_csv_roi
.. autofunction:: histox.WSI.load_json_roi
.. autofunction:: histox.WSI.load_roi_array
.. autofunction:: histox.WSI.mpp_to_dim
.. autofunction:: histox.WSI.predict
.. autofunction:: histox.WSI.preview
.. autofunction:: histox.WSI.process_rois
.. autofunction:: histox.WSI.show_alignment
.. autofunction:: histox.WSI.square_thumb
.. autofunction:: histox.WSI.qc
.. autofunction:: histox.WSI.remove_qc
.. autofunction:: histox.WSI.remove_roi
.. autofunction:: histox.WSI.tensorflow
.. autofunction:: histox.WSI.torch
.. autofunction:: histox.WSI.thumb
.. autofunction:: histox.WSI.verify_alignment
.. autofunction:: histox.WSI.view

Other functions
***************

.. autofunction:: histox.slide.predict