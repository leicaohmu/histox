"""Coordinate conventions and conversions for slide-aligned data.

HistoX places everything it computes from a slide, including tiles, feature
bags, TFRecords, heatmaps, masks, and cell segmentations, in one coordinate
system:

* Coordinates are ``(x, y)``: ``x`` is the column, ``y`` the row.
* Units are **pixels at level 0** (the highest-resolution level) of the source
  slide. The slide's ``mpp`` (microns per pixel at level 0) converts them to
  physical units.
* A tile's position in public APIs (``loc_x`` / ``loc_y`` in TFRecords, tile
  dataframes, and feature-bag indices) is its **center**. Top-left corners
  appear only inside :class:`histox.WSI` (``WSI.coord``).
* Arrays laid over a slide (masks, prediction grids) are indexed
  ``[row, col]``. Converting between array space and slide space goes through
  :func:`rc_to_xy` / :func:`xy_to_rc` and a downsample factor.

The functions in this module are the only place where these conversions are
written out, so that every component rounds the same way.

See RFC 0001 (``rfcs/0001-core-abstractions.md``), proposal P2, and the
user guide page *Coordinates*.
"""

from typing import Sequence, Union

import numpy as np

__all__ = [
    'half_tile',
    'tile_center',
    'tile_topleft',
    'to_level0',
    'from_level0',
    'rc_to_xy',
    'xy_to_rc',
]

ArrayLike = Union[np.ndarray, Sequence]


def half_tile(tile_size_level0: int) -> int:
    """Distance from a tile's top-left corner to its center, in level-0 pixels.

    Rounds down for odd tile sizes. This matches how tile records have
    always been written, so centers in existing TFRecords and feature bags
    stay valid.

    Args:
        tile_size_level0 (int): Tile width (= height) in level-0 pixels, i.e.
            ``WSI.full_extract_px``.

    Returns:
        int
    """
    return int(tile_size_level0 / 2)


def tile_center(topleft: ArrayLike, tile_size_level0: int) -> np.ndarray:
    """Convert tile top-left corners to tile centers (level-0 pixels).

    Args:
        topleft (array-like): One ``(x, y)`` pair or an ``(N, 2)`` array.
        tile_size_level0 (int): ``WSI.full_extract_px``.

    Returns:
        np.ndarray of the same shape, with the input's integer dtype if it had
        one.
    """
    arr = np.asarray(topleft)
    return arr + half_tile(tile_size_level0)


def tile_topleft(center: ArrayLike, tile_size_level0: int) -> np.ndarray:
    """Convert tile centers to tile top-left corners (level-0 pixels).

    The exact inverse of :func:`tile_center`.
    """
    arr = np.asarray(center)
    return arr - half_tile(tile_size_level0)


def to_level0(
    points: ArrayLike,
    downsample: float,
    offset: ArrayLike = (0, 0)
) -> np.ndarray:
    """Map ``(x, y)`` points from a downsampled space to level 0.

    ``level0 = points * downsample + offset``

    Args:
        points (array-like): One ``(x, y)`` pair or an ``(N, 2)`` array, in
            the downsampled space (e.g. mask or thumbnail pixels).
        downsample (float): Level-0 pixels per pixel of the downsampled space.
        offset (array-like): Level-0 ``(x, y)`` of the downsampled space's
            origin. Defaults to ``(0, 0)``.

    Returns:
        np.ndarray (float unless all inputs are integers).
    """
    return np.asarray(points) * downsample + np.asarray(offset)


def from_level0(
    points: ArrayLike,
    downsample: float,
    offset: ArrayLike = (0, 0)
) -> np.ndarray:
    """Map ``(x, y)`` points from level 0 to a downsampled space.

    The inverse of :func:`to_level0`:
    ``points = (level0 - offset) / downsample``.
    """
    return (np.asarray(points) - np.asarray(offset)) / downsample


def rc_to_xy(points: ArrayLike) -> np.ndarray:
    """Convert ``(row, col)`` array indices to ``(x, y)`` points.

    Args:
        points (array-like): One ``(row, col)`` pair or an ``(N, 2)`` array.

    Returns:
        np.ndarray with the two columns swapped.
    """
    return np.asarray(points)[..., ::-1]


def xy_to_rc(points: ArrayLike) -> np.ndarray:
    """Convert ``(x, y)`` points to ``(row, col)`` array indices."""
    return np.asarray(points)[..., ::-1]
