"""
DINOv2 features from a whole-slide image
========================================

This example starts with OpenSlide's public CC0 test slide and finishes with a
validated DINOv2 feature bag. It demonstrates the complete path through HistoX:
download and verify data, open an SVS lazily, inspect a tile, load a pinned
checkpoint, encode the whole slide, visualize the embedding grid, and save the
result for downstream modeling.

The verified reference run used Python 3.9.23, HistoX 0.2.1, PyTorch 2.2.1,
and an NVIDIA RTX 3090. The small public slide retained 50 tiles and produced a
``(50, 384)`` feature bag. Runtime varies by hardware and cache state; the
shapes and checksum are the reproducibility checks that matter.

``facebook/dinov2-small`` is a general-domain demonstration encoder. It has
not been validated as a diagnostic pathology model, and the embeddings shown
here must not be interpreted as clinical predictions.

.. container:: histox-example-meta

   **Level:** Intermediate

   **Input:** One public SVS

   **Output:** ``(50, 384)`` feature bag + ``(50, 2)`` tile coordinates

   **Verified:** About 8.4 s on an NVIDIA RTX 3090

.. container:: histox-example-flow

   ``SVS -> retained tiles -> DINOv2 -> feature bag + coordinates``

.. raw:: html

   <div class="histox-example-actions" data-histox-example-actions aria-label="Example actions">
     <a class="histox-example-action histox-example-action--github" href="https://github.com/leicaohmu/histox/blob/master/docs/gallery/feature_extraction/plot_dinov2_wsi.py">View on GitHub</a>
   </div>
"""

# %%
# Install the optional dependencies
# ---------------------------------
#
# The example needs HistoX's PyTorch and Hugging Face integrations. OpenSlide
# must also be available on the host operating system.
#
# .. code-block:: bash
#
#    python -m pip install "histox[torch,huggingface]"
#
# ``HF_ENDPOINT`` is set before importing HistoX. Remove this line to use the
# default Hugging Face endpoint instead of the mirror.

import hashlib
import os
import platform
import tempfile
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

import histox as hx
from histox import data

# Use the verified PCA map as the stable gallery card image, including on
# non-executing Read the Docs builds.
# sphinx_gallery_thumbnail_path = '_static/examples/dinov2-small-pca-map.png'


WORK_DIR = Path(
    os.environ.get(
        "HISTOX_EXAMPLE_DIR",
        Path(tempfile.gettempdir()) / "histox-dinov2-gallery",
    )
).expanduser().resolve()
WORK_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "facebook/dinov2-small"
REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
TILE_PX = 224
TILE_UM = 112
COORDINATE_ORDER = "yx"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Python:", platform.python_version())
print("HistoX:", hx.__version__)
print("PyTorch:", torch.__version__)
print("Device:", DEVICE)
if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# %%
# Download and verify a public SVS
# --------------------------------
#
# ``download_dataset`` resolves the registry entry, reuses a valid cached file,
# and records provenance. The explicit SHA-256 below makes the input visible in
# the example output as an independent check.


def sha256sum(path, chunk_size=1024 * 1024):
    """Return a SHA-256 digest without loading the entire file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


download = data.download_dataset(
    "openslide-cmu-small",
    path=WORK_DIR / "data",
)
slide_path = download.destination / "CMU-1-Small-Region.svs"
input_sha256 = sha256sum(slide_path)

print("File:", slide_path.name)
print("Size:", f"{slide_path.stat().st_size:,} bytes")
print("SHA-256:", input_sha256)
print("Provenance:", download.provenance_path.name)


# %%
# Open the slide and inspect a model input
# ----------------------------------------
#
# HistoX opens the WSI lazily. ``tile_px`` fixes the tensor size consumed by
# the encoder, while ``tile_um`` fixes the physical field of view.

slide = hx.WSI(
    str(slide_path),
    tile_px=TILE_PX,
    tile_um=TILE_UM,
)

possible_tiles = int(np.prod(slide.grid.shape))
tile_xy = (4, 6)
thumbnail = np.asarray(slide.thumb(width=600))
tile = slide[tile_xy]
if tile is None:
    raise RuntimeError("The reference tile at grid coordinate (4, 6) was filtered.")

print("Dimensions:", slide.dimensions)
print("MPP:", slide.mpp)
print("Grid (x, y):", slide.grid.shape)
print("Possible tiles:", possible_tiles)
print("Tile shape:", tile.shape)
print("Tile dtype:", tile.dtype)

fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.2), constrained_layout=True)
axes[0].imshow(thumbnail)
axes[0].set_title("Whole-slide thumbnail")
axes[1].imshow(tile)
axes[1].set_title(f"Tile at grid {tile_xy}")
for axis in axes:
    axis.axis("off")
fig.savefig(WORK_DIR / "dinov2-inputs.png", dpi=180)
plt.show()

# %%
# The public documentation build does not download the SVS, so it displays the
# output captured by the verified server run. An executing build replaces this
# fallback with the figure produced directly above.
#
# .. only:: histox_gallery_precomputed
#
#    .. figure:: ../../_static/examples/dinov2-inputs.png
#       :alt: Whole-slide thumbnail and the tile at grid coordinate 4, 6
#       :width: 760px
#
#       The exact slide thumbnail and input tile used by the verified run.


# %%
# Load a pinned DINOv2 checkpoint
# --------------------------------
#
# Pinning the Hugging Face revision prevents a future upstream update from
# silently changing this example. HistoX owns the image preprocessing and
# returns one embedding per tile.

encoder = hx.build_feature_extractor(
    "dinov2",
    model_id=MODEL_ID,
    revision=REVISION,
    device=DEVICE,
    mixed_precision=False,
)

parameter_count = sum(parameter.numel() for parameter in encoder.model.parameters())
print("Extractor:", type(encoder).__name__)
print("Parameters:", f"{parameter_count:,}")
print("Features per tile:", encoder.num_features)


# %%
# Check one tile before scaling to the slide
# -------------------------------------------
#
# The public HistoX extractor accepts a uint8, channel-last batch. Testing one
# tile first catches shape, preprocessing, and device errors cheaply.

tile_batch = torch.from_numpy(tile[None])
tile_embedding = encoder(tile_batch).cpu()

print("Input batch:", tuple(tile_batch.shape), tile_batch.dtype)
print("Embedding:", tuple(tile_embedding.shape), tile_embedding.dtype)
print("First five values:", tile_embedding[0, :5].numpy())


# %%
# Extract the whole-slide feature grid
# ------------------------------------
#
# Passing ``WSI`` streams retained tiles in batches. ``slide.grid.shape`` uses
# ``(grid_x, grid_y)``, while the returned feature array uses
# ``(grid_y, grid_x, feature)`` so it can be displayed directly as an image.
# Consequently, each row of ``grid_yx`` below is ordered as ``(grid_y, grid_x)``.

started = time.perf_counter()
feature_grid = encoder(
    slide,
    batch_size=16,
    show_progress=True,
    dtype=np.float32,
)
elapsed = time.perf_counter() - started
if feature_grid is None:
    raise RuntimeError("Whole-slide feature extraction returned no data.")

mask = np.ma.getmaskarray(feature_grid)[..., 0]
features = feature_grid.data[~mask]
grid_yx = np.argwhere(~mask)

print("Retained tiles:", len(features))
print("Feature grid:", feature_grid.shape)
print("Feature bag:", features.shape)
print("Coordinates:", grid_yx.shape)
print("Feature grid order: (grid_y, grid_x, feature)")
print("Coordinate order: (grid_y, grid_x)")
print("Elapsed:", f"{elapsed:.2f} s")
print("Feature mean:", f"{features.mean():.7f}")
print("Feature std:", f"{features.std():.7f}")
print("Mean L2 norm:", f"{np.linalg.norm(features, axis=1).mean():.5f}")


# %%
# Sanity-check spatial correspondence with PCA
# --------------------------------------------
#
# PCA reduces each 384-value embedding to one score solely to make the spatial
# layout visible. Mapping those scores back to ``grid_yx`` checks that features
# and coordinates stayed aligned. PCA is not part of DINOv2 inference or
# downstream training, and its colors are not tumor probabilities, class
# labels, or biological evidence. In this verified run, the first component
# explains 18.30% of the feature variance.

pca = PCA(n_components=1)
pca_scores = pca.fit_transform(features).ravel()
pca_grid = np.full(mask.shape, np.nan, dtype=np.float32)
pca_grid[~mask] = pca_scores

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.8, 4.8),
    constrained_layout=True,
    gridspec_kw={"width_ratios": (1.25, 1)},
)
axes[0].imshow(thumbnail)
axes[0].set_title("Source whole-slide image")
axes[0].axis("off")

image = axes[1].imshow(pca_grid, cmap="magma", interpolation="nearest")
axes[1].set(
    title="DINOv2 features (PCA component 1)",
    xlabel="Tile grid x",
    ylabel="Tile grid y",
)
fig.colorbar(image, ax=axes[1], label="PC1 score")
fig.savefig(WORK_DIR / "dinov2-small-pca-map.png", dpi=180)
plt.show()

print("PCA explained variance:", f"{pca.explained_variance_ratio_[0]:.4f}")

# %%
# .. only:: histox_gallery_precomputed
#
#    .. figure:: ../../_static/examples/dinov2-small-pca-map.png
#       :alt: Source whole-slide image beside the spatial PCA map of DINOv2 embeddings
#       :width: 760px
#
#       The source WSI and PCA component 1 in the same spatial orientation.
#       PCA is shown only as a pipeline sanity check, not as a diagnostic map.


# %%
# Save and validate the feature bag
# ---------------------------------
#
# Store the dense features, tile coordinates, model ID, and immutable revision
# together with the preprocessing scale, input checksum, coordinate convention,
# and HistoX version. Reloading the artifact immediately verifies that
# serialization did not change the features or their spatial correspondence.

bag_path = WORK_DIR / "CMU-1-Small-Region.dinov2-small.npz"
np.savez_compressed(
    bag_path,
    features=features,
    grid_yx=grid_yx,
    coordinate_order=COORDINATE_ORDER,
    slide_grid_xy=np.asarray(slide.grid.shape, dtype=np.int64),
    histox_version=hx.__version__,
    model_id=MODEL_ID,
    revision=REVISION,
    tile_px=TILE_PX,
    tile_um=TILE_UM,
    input_sha256=input_sha256,
)

with np.load(bag_path, allow_pickle=False) as saved:
    np.testing.assert_array_equal(saved["features"], features)
    np.testing.assert_array_equal(saved["grid_yx"], grid_yx)
    assert saved["coordinate_order"].item() == COORDINATE_ORDER
    assert saved["histox_version"].item() == hx.__version__
    assert saved["model_id"].item() == MODEL_ID
    assert saved["revision"].item() == REVISION
    assert saved["tile_px"].item() == TILE_PX
    assert saved["tile_um"].item() == TILE_UM
    assert saved["input_sha256"].item() == input_sha256
    saved_arrays = sorted(saved.files)
    saved_feature_shape = tuple(saved["features"].shape)
    saved_coordinate_order = saved["coordinate_order"].item()

print("Saved feature bag:", bag_path)
print("Saved arrays:", saved_arrays)
print("Saved feature shape:", saved_feature_shape)
print("Saved coordinate order:", saved_coordinate_order)
print("Saved HistoX version:", hx.__version__)
print("Saved tile scale:", f"{TILE_PX} px / {TILE_UM} um")
print("Saved input SHA-256:", input_sha256)
print("Validation: passed")


# %%
# Interpreting the result
# -----------------------
#
# The output is a slide-level bag with one 384-dimensional row per retained
# tile and a matching ``(y, x)`` coordinate for every row. It is suitable for
# MIL, clustering, retrieval, or spatial visualization. Before using a new
# cohort, verify the model license, slide magnification, tissue filtering, and
# patient-level train/validation/test split.
#
# The workflow above uses :func:`histox.data.download_dataset`,
# :doc:`histox.WSI </slide>`, and :func:`histox.build_feature_extractor`. See the
# :doc:`feature extraction guide </features>` for the broader HistoX workflow
# and extractor limitations.
#
# Next steps
# ----------
#
# 1. Repeat the workflow with a second slide and compare the saved metadata.
# 2. Load the validated ``(N, 384)`` bag in a slide-level MIL experiment.
# 3. Split cohorts by patient before fitting or evaluating any downstream model.
