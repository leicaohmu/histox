.. currentmodule:: histox

histox.WSI
==========

The :class:`WSI` class opens one whole-slide image lazily and defines the tile
grid used by preview, quality control, ROI filtering, extraction, and model
input pipelines. For cohort-scale work, prefer
:meth:`Project.extract_tiles` or :meth:`Dataset.extract_tiles`; they add
project configuration, parallelism, reporting, and output management around
the same slide operations.

Start with the executable :doc:`first-WSI tutorial
<auto_examples/basics/plot_open_wsi>` if you have not used this class before.
It downloads a 1.9 MB public SVS and verifies the coordinate system, thumbnail,
Otsu mask, and one RGB tile.

Minimal example
---------------

``tile_px`` is the returned tile width in pixels. ``tile_um`` is the physical
field of view represented by each tile.

.. code-block:: python

   import histox as hx

   slide = hx.WSI(
       "/path/to/slide.svs",
       tile_px=224,
       tile_um=112,
       roi_method="ignore",
   )

   print(slide.dimensions)       # full-resolution (width, height)
   print(slide.grid.shape)       # tile-grid (x, y)

   thumbnail = slide.thumb(width=600)
   tile = slide[4, 6]            # NumPy RGB array or None

Indexing uses grid coordinates, not base-level pixels. Use
:meth:`WSI.grid_to_coord` and :meth:`WSI.coord_to_grid` when moving between
the two coordinate systems.

DICOM Slide Microscopy
----------------------

Install the optional reader before opening DICOM whole-slide images:

.. code-block:: bash

   python -m pip install "histox[dicom]"

A DICOM WSI is usually a *series* of ``.dcm`` instances rather than one file.
Place one series in its own directory and pass the directory to :class:`WSI`:

.. code-block:: python

   import histox as hx

   slide = hx.WSI(
       "/data/idc/1.3.6.1.4.1.example/",
       tile_px=256,
       tile_um=256,
       roi_method="ignore",
   )

   print(slide.filetype)          # dcm
   print(slide.dimensions)        # base-level (width, height)
   print(slide.level_downsamples)
   tile = slide[0, 0]             # uint8 RGB NumPy array

Passing one ``.dcm`` file is also supported, but only the instances supplied
to the reader can contribute pyramid levels. Pass the series directory when a
complete pyramid is required. :func:`histox.util.get_slide_paths` groups a
directory containing ``.dcm`` files as one slide, so keep different series in
different directories.

HistoX reads DICOM pixel spacing as microns per pixel and uses the dedicated
``wsidicom`` adapter regardless of ``HX_SLIDE_BACKEND``. The current adapter
opens local files and directories; it does not treat an HTTP or DICOMweb URL
as a :class:`WSI` path. Cloud frame access can also be slow for older converted
slides that contain neither a Basic nor Extended Offset Table. Download those
series locally before repeated tile extraction.

Constructor
-----------

.. autoclass:: WSI

Core metadata
-------------

.. autosummary::

   WSI.dimensions
   WSI.width
   WSI.height
   WSI.shape
   WSI.levels
   WSI.level_dimensions
   WSI.level_downsamples
   WSI.level_mpp
   WSI.properties
   WSI.vendor
   WSI.qc_mask
   WSI.slide

Viewing and quality control
---------------------------

.. automethod:: WSI.thumb
.. automethod:: WSI.square_thumb
.. automethod:: WSI.preview
.. automethod:: WSI.masked_thumb
.. automethod:: WSI.qc
.. automethod:: WSI.get_qc_mask
.. automethod:: WSI.apply_qc_mask
.. automethod:: WSI.remove_qc
.. automethod:: WSI.remove_roi_qc

Coordinates, tiles, and loaders
-------------------------------

.. automethod:: WSI.get_tile_coord
.. automethod:: WSI.get_tile_dataframe
.. automethod:: WSI.grid_to_coord
.. automethod:: WSI.coord_to_grid
.. automethod:: WSI.get_masked_coord
.. automethod:: WSI.get_tile_mask
.. automethod:: WSI.extract_tiles
.. automethod:: WSI.build_generator
.. automethod:: WSI.torch
.. automethod:: WSI.tensorflow

Regions of interest
-------------------

.. automethod:: WSI.get_rois
.. automethod:: WSI.get_artifacts
.. automethod:: WSI.get_roi_by_name
.. automethod:: WSI.get_roi_label_grid
.. automethod:: WSI.get_tile_roi
.. automethod:: WSI.get_tile_roi_mask
.. automethod:: WSI.has_rois
.. automethod:: WSI.load_csv_roi
.. automethod:: WSI.load_json_roi
.. automethod:: WSI.load_roi_array
.. automethod:: WSI.export_rois
.. automethod:: WSI.remove_roi
.. automethod:: WSI.set_artifacts

Alignment and segmentation
--------------------------

.. automethod:: WSI.align_to
.. automethod:: WSI.align_tiles_to
.. automethod:: WSI.apply_alignment
.. automethod:: WSI.load_alignment
.. automethod:: WSI.show_alignment
.. automethod:: WSI.verify_alignment
.. automethod:: WSI.apply_segmentation
.. automethod:: WSI.extract_cells

Advanced operations
-------------------

.. automethod:: WSI.area
.. automethod:: WSI.dim_to_mpp
.. automethod:: WSI.mpp_to_dim
.. automethod:: WSI.predict
.. automethod:: WSI.view

Whole-slide prediction helper
-----------------------------

.. currentmodule:: histox.slide

.. autofunction:: predict
