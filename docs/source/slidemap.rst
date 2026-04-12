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

.. autofunction:: SlideMap.activations
.. autofunction:: SlideMap.build_mosaic
.. autofunction:: SlideMap.cluster
.. autofunction:: SlideMap.neighbors
.. autofunction:: SlideMap.filter
.. autofunction:: SlideMap.umap_transform
.. autofunction:: SlideMap.label
.. autofunction:: SlideMap.label_by_preds
.. autofunction:: SlideMap.label_by_slide
.. autofunction:: SlideMap.label_by_uncertainty
.. autofunction:: SlideMap.load
.. autofunction:: SlideMap.load_coordinates
.. autofunction:: SlideMap.load_umap
.. autofunction:: SlideMap.plot
.. autofunction:: SlideMap.plot_3d
.. autofunction:: SlideMap.save
.. autofunction:: SlideMap.save_3d
.. autofunction:: SlideMap.save_plot
.. autofunction:: SlideMap.save_coordinates
.. autofunction:: SlideMap.save_umap
.. autofunction:: SlideMap.save_encoder
