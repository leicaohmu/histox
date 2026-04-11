.. currentmodule:: histox

histox.SlideMap
==================

:class:`histox.SlideMap` assists with visualizing tiles and slides in two-dimensional space.

Once a model has been trained, tile-level predictions and intermediate layer activations can be calculated
across an entire dataset with :class:`histox.DatasetFeatures`.
The :class:`histox.SlideMap` class can then perform dimensionality reduction on these dataset-wide
activations, plotting tiles and slides in two-dimensional space. Visualizing the distribution and clustering
of tile-level and slide-level layer activations can help reveal underlying structures in the dataset and shared
visual features among classes.

The primary method of use is first generating an :class:`histox.DatasetFeatures` from a trained
model, then using :meth:`histox.DatasetFeatures.map_activations`, which returns an instance of
:class:`histox.SlideMap`.

.. code-block:: python

    ftrs = sf.DatasetFeatures(model='/path/', ...)
    slide_map = ftrs.map_activations()

Alternatively, if you would like to map slides from a dataset in two-dimensional space using pre-calculated *x* and *y*
coordinates, you can use the :meth:`sldieflow.SlideMap.from_xy` class method. In addition to X and Y, this method
requires supplying tile-level metadata in the form of a list of dicts. Each dict must contain the name of the origin
slide and the tile index in the slide TFRecord.

.. code-block:: python

    x = np.array(...)
    y = np.array(...)
    slides = ['slide1', 'slide1', 'slide5', ...]
    slide_map = sf.SlideMap.from_xy(x=x, y=y, slides=slides)

.. autoclass:: SlideMap

Methods
-------

.. autofunction:: histox.SlideMap.activations
.. autofunction:: histox.SlideMap.build_mosaic
.. autofunction:: histox.SlideMap.cluster
.. autofunction:: histox.SlideMap.neighbors
.. autofunction:: histox.SlideMap.filter
.. autofunction:: histox.SlideMap.umap_transform
.. autofunction:: histox.SlideMap.label
.. autofunction:: histox.SlideMap.label_by_preds
.. autofunction:: histox.SlideMap.label_by_slide
.. autofunction:: histox.SlideMap.label_by_uncertainty
.. autofunction:: histox.SlideMap.load
.. autofunction:: histox.SlideMap.load_coordinates
.. autofunction:: histox.SlideMap.load_umap
.. autofunction:: histox.SlideMap.plot
.. autofunction:: histox.SlideMap.plot_3d
.. autofunction:: histox.SlideMap.save
.. autofunction:: histox.SlideMap.save_3d
.. autofunction:: histox.SlideMap.save_plot
.. autofunction:: histox.SlideMap.save_coordinates
.. autofunction:: histox.SlideMap.save_umap
.. autofunction:: histox.SlideMap.save_encoder
