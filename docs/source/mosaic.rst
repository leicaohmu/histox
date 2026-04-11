.. currentmodule:: slideflow

.. _mosaic:

histox.Mosaic
================

:class:`histox.Mosaic` plots tile images onto a map of slides, generating a mosaic map.

The idea of a mosaic map is to visualize image feature variation across slides and among categories, in an attempt
to better understand the kinds of image features discriminative models might be using to generate class predictions.
They are created by first generating whole-dataset layer features (using
:class:`histox.DatasetFeatures`), which are then mapped into two-dimensional space using UMAP
dimensionality reduction (:class:`histox.SlideMap`). This resulting SlideMap is then passed to
:class:`histox.Mosaic`, which overlays tile images onto the dimensionality-reduced slide map.

An example of a mosaic map can be found in Figure 4 of `this paper <https://doi.org/10.1038/s41379-020-00724-3>`_.
It bears some resemblence to the Activation Atlases created by
`Google and OpenAI <https://distill.pub/2019/activation-atlas/>`_, without the use of feature inversion.

See :ref:`mosaic_map` for an example of how a mosaic map can be used in the context of a project.

.. autoclass:: Mosaic

Methods
-------

.. autofunction:: histox.Mosaic.generate_grid
.. autofunction:: histox.Mosaic.plot
.. autofunction:: histox.Mosaic.points_at_grid_index
.. autofunction:: histox.Mosaic.save
.. autofunction:: histox.Mosaic.save_report
.. autofunction:: histox.Mosaic.view