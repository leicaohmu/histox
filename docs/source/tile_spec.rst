.. currentmodule:: histox

histox.TileSpec
===============

A :class:`TileSpec` describes how a slide is cut into tiles: tile size in
pixels, physical size (microns or magnification), stride, and grid origin.
It can be passed wherever HistoX takes ``tile_px`` and ``tile_um``.

.. code-block:: python

    import histox as hx

    spec = hx.TileSpec(tile_px=224, tile_um=112)
    spec.label                          # '224px_112um'

    wsi = hx.WSI('slide.svs', spec)
    wsi.tile_spec == spec               # True

    dataset = project.dataset(spec)
    dataset.tile_spec                   # TileSpec(tile_px=224, tile_um=112, ...)

    # Overlapping tiles at 20x
    project.extract_tiles(hx.TileSpec(256, '20x', stride_div=2))

    # Feature bags record the tile size they were extracted with
    hx.Bag.load('bags/', slide).tile_spec

A :class:`Dataset` stores only ``tile_px`` and ``tile_um`` (they determine
where its TFRecords live). Stride and origin apply per extraction: pass them
to :meth:`Dataset.extract_tiles`, or pass a full TileSpec to
:meth:`Project.extract_tiles`.

The separate ``tile_px``, ``tile_um``, ``stride_div``, and ``origin``
arguments keep working everywhere.

.. autoclass:: TileSpec
   :members:
