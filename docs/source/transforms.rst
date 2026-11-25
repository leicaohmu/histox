.. currentmodule:: histox.transforms

histox.transforms
=================

HistoX has three families of image transforms, which act at two levels.

**Slide transforms** look at a whole slide and mark regions to exclude.
Quality-control (QC) methods such as :class:`Otsu` and :class:`GaussianV2` are
called as ``qc(wsi)`` and return a boolean mask (``True`` = excluded).
``wsi.qc(...)`` applies them to the tile grid; :func:`qc_spatial_map` returns
the mask as a :class:`histox.SpatialMap` without changing the slide.

**Tile transforms** map one tile to another: they take a ``uint8`` tensor of
shape ``(C, H, W)`` or ``(B, C, H, W)`` and return the same kind of tensor,
like ``torchvision.transforms``. Augmentations such as
:class:`RandomColorDistortion` are tile transforms, and a stain normalizer
becomes one with :class:`TileNormalizer`.

.. code-block:: python

    from histox import transforms as T

    tissue = T.qc_spatial_map('otsu', wsi)

    pipeline = T.Compose([
        T.TileNormalizer('macenko'),
        T.RandomColorDistortion(0.5),
    ])
    tile = pipeline(tile)

    T.describe(pipeline)
    # {'type': 'histox.transforms.Compose', 'transforms': [
    #     {'type': 'histox.transforms.TileNormalizer', 'method': 'macenko',
    #      'augment': False, 'fit': {...}},
    #     {'type': 'histox.io.torch.augment.RandomColorDistortion', 'params': {}}]}

:func:`describe` returns a JSON-serializable record of any transform, so the
preprocessing behind a set of features or results can be saved alongside
them.

QC methods and stain normalizers are listed in the registry
(:mod:`histox.registry`, kinds ``'qc'`` and ``'normalizer'``):

.. code-block:: python

    T.QC_METHODS.names()      # ['blur', 'deepfocus', 'gaussian', 'otsu', ...]
    T.NORMALIZERS.names()     # ['augment', 'macenko', 'reinhard', ...]

Pipelines
*********

.. autoclass:: Compose
   :members:

.. autoclass:: TileNormalizer
   :members:

.. autofunction:: normalizer
.. autofunction:: qc_spatial_map
.. autofunction:: describe

Quality control
***************

The QC classes are documented in :doc:`slide_qc`; they are available here as
``T.Otsu``, ``T.Gaussian``, ``T.GaussianV2``, ``T.DeepFocus``,
``T.StridedDL``, ``T.Segment``, and ``T.StridedSegment``.

Augmentations
*************

``T.RandomCardinalRotation``, ``T.RandomGaussianBlur``,
``T.RandomJPEGCompression``, ``T.RandomColorProfile``, and
``T.RandomColorDistortion`` are the augmentations from ``histox.io.torch``.
