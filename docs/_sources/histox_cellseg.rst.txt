.. currentmodule:: histox.cellseg

histox.cellseg
=================

This module contains utility functions for performing whole-slide image cell segmentation with Cellpose.

See :ref:`cellseg` for more information.

.. autofunction:: segment_slide

Segmentation
************
.. autoclass:: Segmentation
.. autofunction:: cellseg.Segmentation.apply_rois
.. autofunction:: cellseg.Segmentation.calculate_centroids
.. autofunction:: cellseg.Segmentation.calculate_outlines
.. autofunction:: cellseg.Segmentation.centroids
.. autofunction:: cellseg.Segmentation.centroid_to_image
.. autofunction:: cellseg.Segmentation.extract_centroids
.. autofunction:: cellseg.Segmentation.mask_to_image
.. autofunction:: cellseg.Segmentation.outline_to_image
.. autofunction:: cellseg.Segmentation.save