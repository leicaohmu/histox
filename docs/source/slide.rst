.. currentmodule:: histox.slide

histox.slide
===============

This module contains classes to load slides and extract tiles. For optimal performance, tile extraction should
generally not be performed by instancing these classes directly, but by calling either
:func:`histox.Project.extract_tiles` or :func:`histox.Dataset.extract_tiles`, which include performance
optimizations and additional functionality.

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

histox.WSI
*************

.. autoclass:: WSI

Attributes
----------

.. autosummary::

    WSI.dimensions
    WSI.qc_mask
    WSI.levels
    WSI.level_dimensions
    WSI.level_downsamples
    WSI.level_mpp
    WSI.properties
    WSI.slide
    WSI.vendor

Methods
-------

.. autofunction:: WSI.align_to
.. autofunction:: WSI.align_tiles_to
.. autofunction:: WSI.apply_qc_mask
.. autofunction:: WSI.apply_segmentation
.. autofunction:: WSI.area
.. autofunction:: WSI.build_generator
.. autofunction:: WSI.dim_to_mpp
.. autofunction:: WSI.get_tile_mask
.. autofunction:: WSI.get_tile_dataframe
.. autofunction:: WSI.extract_cells
.. autofunction:: WSI.extract_tiles
.. autofunction:: WSI.export_rois
.. autofunction:: WSI.has_rois
.. autofunction:: WSI.load_csv_roi
.. autofunction:: WSI.load_json_roi
.. autofunction:: WSI.load_roi_array
.. autofunction:: WSI.mpp_to_dim
.. autofunction:: WSI.predict
.. autofunction:: WSI.preview
.. autofunction:: WSI.process_rois
.. autofunction:: WSI.show_alignment
.. autofunction:: WSI.square_thumb
.. autofunction:: WSI.qc
.. autofunction:: WSI.remove_qc
.. autofunction:: WSI.remove_roi
.. autofunction:: WSI.tensorflow
.. autofunction:: WSI.torch
.. autofunction:: WSI.thumb
.. autofunction:: WSI.verify_alignment
.. autofunction:: WSI.view

Other functions
***************

.. autofunction:: slide.predict