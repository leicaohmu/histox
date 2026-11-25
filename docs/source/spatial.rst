.. currentmodule:: histox

histox.SpatialMap
=================

QC masks, model heatmaps, cell segmentations, and MIL attention scores all
assign values to regions of a slide. A :class:`SpatialMap` holds any of them:
a ``(rows, cols)`` or ``(rows, cols, channels)`` array plus where it sits on
the slide, in the level-0 coordinates described in :doc:`coordinates`.

.. code-block:: python

    import histox as hx

    # MIL attention scores of one slide, from its bag's tile coordinates
    bag = hx.Bag.load('bags/', slide_name)
    att = hx.SpatialMap.from_bag(bag, attention, stride=wsi.full_stride, mpp=wsi.mpp)
    att.value_at(bag.coords[:3])            # attention of those tiles

    # Existing outputs convert directly
    pred = heatmap.to_spatial_map()         # histox.Heatmap
    cells = segmentation.to_spatial_map()   # histox.cellseg.Segmentation
    tissue = wsi.qc_masks[0].to_spatial_map(wsi)

    # Overlay on a thumbnail
    ds = wsi.dimensions[0] / thumb.size[0]
    plt.imshow(thumb)
    plt.imshow(att.values, extent=att.extent(downsample=ds), alpha=0.5)

    att.save('attention.npz')
    att = hx.SpatialMap.load('attention.npz')

Cell ``values[r, c]`` covers level-0 ``x`` from ``origin_x + c * stride_x``
to ``origin_x + (c + 1) * stride_x``, and likewise ``y`` for rows. Maps built
from tiles (:meth:`SpatialMap.from_points`, :meth:`SpatialMap.from_heatmap`)
center each cell on its tile, so :meth:`SpatialMap.value_at` at a tile center
returns that tile's value.

``kind`` is ``'mask'`` (boolean; for QC masks ``True`` means excluded),
``'labels'`` (integer IDs, 0 = background), or ``'scores'``.

.. autoclass:: SpatialMap
   :members:
