"""
Download, open, and inspect your first WSI
==========================================

This beginner tutorial downloads a 1.9 MB public Aperio ``.svs`` file, opens
it lazily with :class:`histox.WSI`, and inspects both a whole-slide thumbnail
and one model-sized tile. You can run it on a CPU; no GPU or model weights are
required.

By the end, you will be able to:

* download a registered public dataset into a local cache;
* distinguish slide dimensions, physical resolution, and the tile grid;
* retrieve a tile by grid coordinate; and
* connect that tile to its location in the full-resolution slide.

The example is a data-access check, not a diagnostic or training workflow.

.. container:: histox-example-meta

   **Level:** Beginner

   **Input:** One public 1.9 MB SVS (CC0-1.0)

   **Output:** A slide thumbnail and one ``(224, 224, 3)`` RGB tile

   **Hardware:** CPU; typically under one minute after installation

.. container:: histox-example-flow

   ``public registry -> local SVS -> WSI metadata -> thumbnail + tile``

.. raw:: html

   <div class="histox-example-actions" data-histox-example-actions aria-label="Example actions">
     <a class="histox-example-action histox-example-action--github" href="https://github.com/leicaohmu/histox/blob/master/docs/gallery/basics/plot_open_wsi.py">View on GitHub</a>
   </div>
"""

# %%
# Before you start
# ----------------
#
# Install HistoX in a Python 3.9 environment and make sure ``libvips`` is
# available on the host. A development checkout can be installed with:
#
# .. code-block:: bash
#
#    python -m pip install -e .
#
# The download location is explicit so you always know where the example put
# its data. Set ``HISTOX_EXAMPLE_DIR`` to keep it somewhere else.

import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import histox as hx
from histox import data

# Use the verified public-slide thumbnail as the gallery card image.
# sphinx_gallery_thumbnail_path = '_static/examples/openslide-cmu-small-thumbnail.png'


WORK_DIR = Path(
    os.environ.get(
        "HISTOX_EXAMPLE_DIR",
        Path(tempfile.gettempdir()) / "histox-first-wsi",
    )
).expanduser().resolve()
WORK_DIR.mkdir(parents=True, exist_ok=True)

print("HistoX:", hx.__version__)
print("Slide backend:", hx.slide_backend())
print("Working directory:", WORK_DIR)

# %%
# The slide backend depends on the packages installed on your system. The
# verified server run used cuCIM:
#
# .. only:: histox_gallery_precomputed
#
#    .. code-block:: text
#
#       HistoX: 0.2.1
#       Slide backend: cucim


# %%
# 1. Download the public slide
# ----------------------------
#
# :func:`histox.data.download_dataset` looks up the dataset by name, streams
# the asset into the cache, and validates it against the registry. Running the
# same cell again reuses the valid local file instead of downloading it twice.

download = data.download_dataset(
    "openslide-cmu-small",
    path=WORK_DIR / "data",
)
slide_path = download.destination / "CMU-1-Small-Region.svs"
cache_status = "downloaded" if download.downloaded_count else "reused"

print("File:", slide_path.name)
print("Size:", f"{slide_path.stat().st_size / 1_000_000:.2f} MB")
print("License:", download.dataset.license)
print("Cache status:", cache_status)

# %%
# On the verified first run, the output was:
#
# .. only:: histox_gallery_precomputed
#
#    .. code-block:: text
#
#       File: CMU-1-Small-Region.svs
#       Size: 1.94 MB
#       License: CC0-1.0
#       Cache status: downloaded


# %%
# 2. Open the WSI and read its coordinate system
# ------------------------------------------------
#
# ``tile_px`` is the returned array size. ``tile_um`` is the physical field of
# view represented by each tile. Keeping both explicit prevents image pixels
# from being confused with tissue distance.

slide = hx.WSI(
    str(slide_path),
    tile_px=224,
    tile_um=112,
)

print("Dimensions (width, height):", slide.dimensions)
print("Microns per pixel:", slide.mpp)
print("Pyramid levels:", len(slide.level_dimensions))
print("Tile grid (x, y):", slide.grid.shape)
print("Available grid positions:", int(slide.grid.sum()))

# %%
# These values describe the same image at three scales: the full-resolution
# pixel canvas, its physical resolution, and HistoX's model-input grid.
#
# .. only:: histox_gallery_precomputed
#
#    .. code-block:: text
#
#       Dimensions (width, height): (2220, 2967)
#       Microns per pixel: 0.499
#       Pyramid levels: 1
#       Tile grid (x, y): (9, 13)
#       Available grid positions: 117


# %%
# 3. View the complete slide
# ---------------------------
#
# ``thumb`` requests a low-resolution view without loading the full-resolution
# WSI into memory. It is the quickest way to confirm orientation, tissue
# coverage, and whether the correct slide was opened.

thumbnail = np.asarray(slide.thumb(width=600))

fig, axis = plt.subplots(figsize=(5.2, 6.8), constrained_layout=True)
axis.imshow(thumbnail)
axis.set_title("CMU-1-Small-Region: whole-slide thumbnail")
axis.axis("off")
plt.show()

# %%
# Read the Docs uses the output captured by the verified server run. An
# executing gallery build replaces this fallback with the figure above.
#
# .. only:: histox_gallery_precomputed
#
#    .. figure:: ../../_static/examples/openslide-cmu-small-thumbnail.png
#       :alt: Thumbnail of the public OpenSlide CMU small Aperio region
#       :width: 480px
#
#       A 600-pixel-wide thumbnail read from the public SVS.


# %%
# 4. Preview extraction and remove background
# --------------------------------------------
#
# :meth:`histox.WSI.preview` performs a dry run of tile extraction and draws
# the tile locations that pass HistoX's default pixel-level filters. It does
# not change the WSI grid. The Otsu QC step is different: it builds a
# slide-level background mask and updates the grid so later operations skip
# masked positions.

candidate_tiles = int(slide.grid.sum())
preview_before = np.asarray(
    slide.preview(rois=False, show_progress=False)
)

qc_mask = np.asarray(slide.qc("otsu"))
tiles_after_otsu = int(slide.grid.sum())
preview_after = np.asarray(
    slide.preview(rois=False, show_progress=False)
)

print("Candidate grid positions:", candidate_tiles)
print("Tiles after Otsu QC:", tiles_after_otsu)
print("Filtered grid positions:", candidate_tiles - tiles_after_otsu)

fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.6), constrained_layout=True)
axes[0].imshow(preview_before)
axes[0].set_title("Default extraction preview")
axes[1].imshow(qc_mask, cmap="gray")
axes[1].set_title("Otsu background mask")
axes[2].imshow(preview_after)
axes[2].set_title(f"After Otsu QC: {tiles_after_otsu} tiles")
for axis in axes:
    axis.axis("off")
fig.savefig(WORK_DIR / "openslide-cmu-small-qc-workflow.png", dpi=180)
plt.show()

# %%
# White pixels in the middle panel are masked background. The numbers below
# are specific to this small demonstration slide; real cohorts require visual
# review and may need different QC settings.
#
# .. only:: histox_gallery_precomputed
#
#    .. code-block:: text
#
#       Candidate grid positions: 117
#       Tiles after Otsu QC: 45
#       Filtered grid positions: 72
#
#    .. figure:: ../../_static/examples/openslide-cmu-small-qc-workflow.png
#       :alt: HistoX extraction preview, Otsu mask, and post-QC tile grid
#       :width: 980px
#
#       HistoX first previews extractable tiles, then applies a slide-level
#       Otsu mask so subsequent operations skip background grid positions.


# %%
# 5. Retrieve one tile and locate it on the WSI
# ------------------------------------------------
#
# Indexing a :class:`histox.WSI` uses ``(grid_x, grid_y)``. The selected grid
# position ``(4, 6)`` begins at base-level pixel coordinate ``(1008, 1456)``.
# It remains available after QC. That coordinate distinction matters later
# when tile predictions are mapped back to the slide.

grid_xy = (4, 6)
base_xy = slide.grid_to_coord(*grid_xy)
tile = slide[grid_xy]
if tile is None:
    raise RuntimeError("The documented tile at grid (4, 6) is unavailable.")

assert tile.shape == (224, 224, 3)
assert tile.dtype == np.uint8

print("Grid coordinate (x, y):", grid_xy)
print("Base pixel coordinate (x, y):", base_xy)
print("Tile array:", tile.shape, tile.dtype)

fig, axis = plt.subplots(figsize=(4.8, 4.8), constrained_layout=True)
axis.imshow(tile)
axis.set_title("RGB tile at grid (4, 6)")
axis.axis("off")
plt.show()

# %%
# .. only:: histox_gallery_precomputed
#
#    .. code-block:: text
#
#       Grid coordinate (x, y): (4, 6)
#       Base pixel coordinate (x, y): (1008, 1456)
#       Tile array: (224, 224, 3) uint8
#
#    .. figure:: ../../_static/examples/openslide-cmu-small-tile-4-6.png
#       :alt: A 224 by 224 pixel RGB tile from the public WSI
#       :width: 320px
#
#       The exact RGB tile returned by ``slide[(4, 6)]``.


# %%
# Checkpoint and next steps
# -------------------------
#
# You have completed the basic WSI data path if your run produced:
#
# * a ``(2220, 2967)`` slide with MPP ``0.499``;
# * a ``(9, 13)`` tile grid; and
# * 45 grid positions remaining after the demonstrated Otsu QC; and
# * a uint8 RGB tile with shape ``(224, 224, 3)``.
#
# These checks confirm file access and coordinate handling. They do not assess
# stain quality, tissue segmentation, or model suitability. Continue with the
# :doc:`slide-processing guide </slide_processing>` for cohort-scale QC and
# tiling, :doc:`HistoX Studio </studio>` for interactive ROI annotation, or the
# :doc:`DINOv2 WSI example </auto_examples/feature_extraction/plot_dinov2_wsi>`
# when you are ready to convert tiles into feature vectors. Together, those
# stages form the broader HistoX path:
#
# ``data -> WSI -> QC / ROI -> tiles -> features -> slide-level models``.
