.. _coordinates:

Coordinates
===========

Everything HistoX computes from a slide uses one coordinate system, so tiles,
feature bags, TFRecords, heatmaps, masks, and cell segmentations can be
overlaid on each other without guesswork.

The convention
**************

1. **Points are** ``(x, y)``. ``x`` is the column (left to right) and ``y``
   the row (top to bottom).
2. **Units are level-0 pixels.** Level 0 is the highest-resolution level of
   the slide file. Multiply by the slide's ``mpp`` (microns per pixel at
   level 0) to get microns.
3. **Tiles are referenced by their center.** ``loc_x`` / ``loc_y`` in
   TFRecords, tile dataframes from :meth:`histox.WSI.get_tile_dataframe`, and
   feature-bag indices (``<slide>.index.npz``) are tile centers. Top-left
   corners are used only inside :class:`histox.WSI` (``WSI.coord``), and
   methods that accept either take an ``anchor='center' | 'topleft'``
   argument.
4. **Arrays laid over a slide are indexed** ``[row, col]``. A mask or
   prediction grid is mapped to the slide through a downsample factor
   (level-0 pixels per array pixel) and the level-0 ``(x, y)`` of the
   array's origin.

A tile's size in level-0 pixels is ``WSI.full_extract_px``. For a tile with
top-left corner ``(x, y)``, its center is ``(x + full_extract_px // 2,
y + full_extract_px // 2)``. Odd sizes round down, as all existing TFRecords
and feature bags do.

.. code-block:: python

    import histox as hx

    wsi = hx.WSI('slide.svs', tile_px=256, tile_um=128)
    wsi.mpp                     # e.g. 0.25 microns per level-0 pixel
    wsi.full_extract_px         # tile size in level-0 pixels, e.g. 512

    # Tile centers of all tiles that passed QC, as (x, y, grid_x, grid_y) rows.
    centers = wsi.get_tile_coord(anchor='center')

Conversion helpers
******************

:mod:`histox.geometry` contains the only implementation of these conversions.
Use it rather than writing the arithmetic by hand, so that rounding stays
consistent.

.. code-block:: python

    from histox import geometry

    # Tile corners <-> centers (level-0 pixels)
    centers = geometry.tile_center(topleft_xy, wsi.full_extract_px)
    topleft = geometry.tile_topleft(centers, wsi.full_extract_px)

    # Mask or thumbnail pixels <-> level-0 pixels
    xy_level0 = geometry.to_level0(xy_mask, downsample, offset=(x0, y0))
    xy_mask = geometry.from_level0(xy_level0, downsample, offset=(x0, y0))

    # Array indices <-> points
    xy = geometry.rc_to_xy(rows_cols)
    rows_cols = geometry.xy_to_rc(xy)

Cell segmentations
******************

:class:`histox.cellseg.Segmentation` stores masks in their own pixel space,
together with ``wsi_dim`` (the level-0 size of the area the masks cover) and
``wsi_offset`` (the level-0 ``(x, y)`` of the mask's top-left corner).
``Segmentation.centroids(wsi_dim=True)`` returns cell centroids as level-0
``(x, y)`` points. A missing ``wsi_offset`` is treated as ``(0, 0)``.

API reference
*************

.. automodule:: histox.geometry
   :members:
