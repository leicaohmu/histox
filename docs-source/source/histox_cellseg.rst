.. currentmodule:: histox.cellseg

histox.cellseg
=================

This module contains utility functions for performing whole-slide image cell segmentation with Cellpose.

See :ref:`cellseg` for more information.

.. autofunction:: segment_slide

Segmentation
************
.. autoclass:: Segmentation
.. autofunction:: histox.cellseg.Segmentation.apply_rois
.. autofunction:: histox.cellseg.Segmentation.calculate_centroids
.. autofunction:: histox.cellseg.Segmentation.calculate_outlines
.. autofunction:: histox.cellseg.Segmentation.centroids
.. autofunction:: histox.cellseg.Segmentation.centroid_to_image
.. autofunction:: histox.cellseg.Segmentation.extract_centroids
.. autofunction:: histox.cellseg.Segmentation.mask_to_image
.. autofunction:: histox.cellseg.Segmentation.outline_to_image
.. autofunction:: histox.cellseg.Segmentation.save