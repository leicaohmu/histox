.. currentmodule:: histox

histox.Bag
==========

A feature bag holds one feature vector per tile of a slide, the tile
coordinates, and the settings used to extract the features. Bags are the
standard input for multiple-instance learning (:doc:`mil`).

.. code-block:: python

    import histox as hx

    bag = hx.Bag.load('bags/', 'TCGA-3L-AA1B-01Z-00-DX1')
    bag.features.shape      # torch.Size([N, F]), float32
    bag.coords[:3]          # tile centers, (x, y) in level-0 pixels
    bag.config['tile_um']   # extraction settings from bags_config.json

    # Keep only some tiles, and save the result as a new bag
    left_half = bag.subset(bag.coords[:, 0] < 40_000)
    left_half.save('bags_left/')

On disk, a bag directory keeps the layout HistoX has always used, so existing
bags load unchanged:

* ``<slide>.pt``: the ``(N, F)`` feature tensor;
* ``<slide>.index.npz``: the ``(N, 2)`` tile centers (see :doc:`coordinates`);
* ``bags_config.json``: one per directory, the extraction settings.

Storage precision
*****************

New bags are stored as **float16** and loaded as **float32**. Features are
computed in mixed precision, so float32 storage doubled disk use and I/O
without adding information. A bag with values outside the float16 range is
stored as float32 automatically, with a warning. To store full precision,
pass ``dtype=torch.float32`` to :meth:`Bag.save` or
:meth:`DatasetFeatures.to_torch`.

All HistoX readers (:class:`Bag`, ``histox.mil``, :meth:`DatasetFeatures.from_bags`)
return float32 whatever the storage precision. Code that calls
``torch.load()`` on a bag file directly gets the stored dtype; convert with
``.float()``, or use :func:`histox.bag.load_features`.

API reference
*************

.. autoclass:: Bag
   :members:

.. autofunction:: histox.bag.list_bags
.. autofunction:: histox.bag.load_features
.. autofunction:: histox.bag.load_coords
