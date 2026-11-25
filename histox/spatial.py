"""SpatialMap: an array laid over a slide.

QC masks, model heatmaps, cell segmentations, and MIL attention scores all
assign values to regions of a slide. :class:`SpatialMap` is one type for all
of them: a 2-D (or 2-D x channels) array plus where it sits on the slide.

Cell ``values[r, c]`` covers, in level-0 slide pixels (see
:mod:`histox.geometry`)::

    x in [origin_x + c * stride_x, origin_x + (c + 1) * stride_x)
    y in [origin_y + r * stride_y, origin_y + (r + 1) * stride_y)

.. code-block:: python

    import histox as hx

    att = hx.SpatialMap.from_points(bag.coords, attention, stride=wsi.full_stride,
                                    slide=wsi.name, mpp=wsi.mpp, kind='scores')
    att.value_at(bag.coords[:3])          # the attention of those tiles
    tissue = hx.SpatialMap.from_qc_mask(wsi.qc_masks[0], wsi)
    att.save('attention.npz'); hx.SpatialMap.load('attention.npz')

    plt.imshow(att.values, extent=att.extent(downsample=thumb_downsample))

See RFC 0001 (``rfcs/0001-core-abstractions.md``), proposal P5.
"""

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from histox import geometry

if TYPE_CHECKING:
    import histox as hx

__all__ = ['SpatialMap']

_KINDS = ('mask', 'labels', 'scores')


def _pair(value: Union[float, Sequence[float]], name: str) -> Tuple[float, float]:
    if np.isscalar(value):
        value = (value, value)
    value = tuple(float(v) for v in value)  # type: ignore
    if len(value) != 2:
        raise ValueError(f"{name} must be a number or an (x, y) pair")
    return value  # type: ignore


@dataclass
class SpatialMap:
    """An array laid over a slide.

    Args:
        values (np.ndarray): ``(H, W)`` or ``(H, W, C)`` array, indexed
            ``[row, col]``. Masked arrays are kept; NaN marks missing values
            in float maps.
        origin (tuple): Level-0 ``(x, y)`` of the top-left corner of
            ``values[0, 0]``.
        stride (float or tuple): Level-0 pixels per cell, as one number or an
            ``(x, y)`` pair.
        slide (str, optional): Slide name.
        mpp (float, optional): Microns per level-0 pixel of the slide.
        kind (str): ``'mask'`` (boolean), ``'labels'`` (integer IDs, 0 =
            background), or ``'scores'`` (continuous values).
        channels (list(str), optional): Names of the channels of a
            ``(H, W, C)`` map, e.g. class names.
        metadata (dict): Anything else to keep with the map (JSON-serializable
            values only, if it will be saved).
    """

    values: np.ndarray
    origin: Tuple[float, float] = (0.0, 0.0)
    stride: Union[float, Tuple[float, float]] = 1.0
    slide: Optional[str] = None
    mpp: Optional[float] = None
    kind: str = 'scores'
    channels: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.values, np.ndarray):
            self.values = np.asarray(self.values)
        if self.values.ndim not in (2, 3):
            raise ValueError(f"values must be 2-D or 3-D, got shape {self.values.shape}")
        if self.kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}, got {self.kind!r}")
        self.origin = _pair(self.origin, 'origin')
        self.stride = _pair(self.stride, 'stride')
        if min(self.stride) <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")
        if self.channels is not None:
            self.channels = list(self.channels)
            n = self.values.shape[2] if self.values.ndim == 3 else 1
            if len(self.channels) != n:
                raise ValueError(f"{len(self.channels)} channel names for {n} channels")

    # --- Geometry -------------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.values.shape

    @property
    def size_level0(self) -> Tuple[float, float]:
        """Width and height covered, in level-0 pixels."""
        h, w = self.values.shape[:2]
        return (w * self.stride[0], h * self.stride[1])

    def extent(self, downsample: float = 1.0) -> Tuple[float, float, float, float]:
        """``(left, right, bottom, top)`` for ``matplotlib.pyplot.imshow``.

        Args:
            downsample (float): Divide level-0 coordinates by this, e.g. to
                overlay on a thumbnail of the slide. Defaults to 1 (level 0).
        """
        w, h = self.size_level0
        x0, y0 = self.origin
        return ((x0) / downsample, (x0 + w) / downsample,
                (y0 + h) / downsample, (y0) / downsample)

    def cell_of(self, xy: Any) -> np.ndarray:
        """``(row, col)`` indices of the cells containing level-0 points.

        Points outside the map get indices outside ``[0, H) x [0, W)``.
        """
        pts = np.asarray(xy, dtype=float)
        cells = np.floor(geometry.from_level0(pts, np.asarray(self.stride), self.origin))
        return geometry.xy_to_rc(cells).astype(np.int64)

    def value_at(self, xy: Any, default: Any = np.nan) -> np.ndarray:
        """Values at level-0 points.

        Args:
            xy: One ``(x, y)`` point or an ``(N, 2)`` array.
            default: Value for points outside the map. Defaults to NaN.

        Returns:
            ``(N,)`` or ``(N, C)`` array (or one value / ``(C,)`` for a single
            point). Masked cells of a masked array return ``default``.
        """
        pts = np.asarray(xy, dtype=float)
        single = pts.ndim == 1
        rc = self.cell_of(pts.reshape(-1, 2))
        h, w = self.values.shape[:2]
        inside = (rc[:, 0] >= 0) & (rc[:, 0] < h) & (rc[:, 1] >= 0) & (rc[:, 1] < w)
        data = np.ma.getdata(self.values)
        mask = np.ma.getmaskarray(self.values)
        out_shape = (len(rc),) + self.values.shape[2:]
        dtype = np.result_type(data.dtype, np.asarray(default).dtype)
        out = np.full(out_shape, default, dtype=dtype)
        r, c = rc[inside, 0], rc[inside, 1]
        out[inside] = data[r, c]
        if mask.any():
            m = mask[r, c]
            if m.ndim > 1:
                m = m.all(axis=-1)
            idx = np.flatnonzero(inside)[m]
            out[idx] = default
        return out[0] if single else out

    # --- Construction ---------------------------------------------------------

    @classmethod
    def from_points(
        cls,
        coords: Any,
        values: Any,
        *,
        stride: Union[float, Tuple[float, float]],
        slide: Optional[str] = None,
        mpp: Optional[float] = None,
        kind: str = 'scores',
        channels: Optional[List[str]] = None,
        fill: float = np.nan,
    ) -> "SpatialMap":
        """Rasterize per-tile values onto a grid of cells, one cell per tile.

        Use this for values attached to tile centers, such as MIL attention
        scores or tile predictions with a :class:`histox.Bag`'s ``coords``.

        Args:
            coords: ``(N, 2)`` tile centers, level-0 pixels.
            values: ``(N,)`` or ``(N, C)`` values.
            stride: Spacing of the tile grid in level-0 pixels
                (``WSI.full_stride``).
            fill: Value of cells without a tile. Defaults to NaN.

        The map is centered on the tiles: each tile center lies in the middle
        of its cell, so :meth:`value_at` at a tile center returns its value.
        """
        pts = np.asarray(coords, dtype=float).reshape(-1, 2)
        vals = np.asarray(values)
        if len(vals) != len(pts):
            raise ValueError(f"{len(pts)} coords but {len(vals)} values")
        sx, sy = _pair(stride, 'stride')
        if not len(pts):
            raise ValueError("No coordinates given")
        origin = (pts[:, 0].min() - sx / 2, pts[:, 1].min() - sy / 2)
        cols = np.floor((pts[:, 0] - origin[0]) / sx).astype(np.int64)
        rows = np.floor((pts[:, 1] - origin[1]) / sy).astype(np.int64)
        shape = (rows.max() + 1, cols.max() + 1) + vals.shape[1:]
        grid = np.full(shape, fill, dtype=np.result_type(vals.dtype, np.asarray(fill).dtype))
        grid[rows, cols] = vals
        return cls(grid, origin, (sx, sy), slide=slide, mpp=mpp, kind=kind, channels=channels)

    @classmethod
    def from_bag(cls, bag: "hx.Bag", values: Any, *, stride: Union[float, Tuple[float, float]],
                 mpp: Optional[float] = None, **kwargs) -> "SpatialMap":
        """:meth:`from_points` using a bag's tile coordinates and slide name."""
        if bag.coords is None:
            raise ValueError(f"Bag {bag.slide} has no tile coordinates")
        return cls.from_points(bag.coords, values, stride=stride, slide=bag.slide,
                               mpp=mpp, **kwargs)

    @classmethod
    def from_qc_mask(cls, qc_mask: Any, wsi: "hx.WSI") -> "SpatialMap":
        """Wrap a QC mask of a slide (``True`` = excluded by QC).

        Args:
            qc_mask: A :class:`histox.slide.QCMask` or boolean array covering
                the whole slide (e.g. ``wsi.qc_mask``).
            wsi: The slide the mask belongs to.
        """
        mask = getattr(qc_mask, 'mask', qc_mask)
        mask = np.asarray(mask, dtype=bool)
        # Same scale WSI.apply_qc_mask uses: slide width / mask width.
        downsample = wsi.dimensions[0] / mask.shape[1]
        meta = {}
        if hasattr(qc_mask, 'filter_threshold'):
            meta = {'filter_threshold': float(qc_mask.filter_threshold),
                    'is_roi': bool(qc_mask.is_roi)}
        return cls(mask, (0, 0), downsample, slide=wsi.name, mpp=wsi.mpp,
                   kind='mask', metadata=meta)

    @classmethod
    def from_heatmap(cls, heatmap: Any, *, uncertainty: bool = False) -> "SpatialMap":
        """Wrap the predictions (or uncertainty) grid of a :class:`histox.Heatmap`.

        Each cell is centered on its tile; cells without a tile are masked.
        """
        wsi = heatmap.slide
        grid = heatmap.uncertainty if uncertainty else heatmap.predictions
        if grid is None:
            raise ValueError("Heatmap has no " + ("uncertainty" if uncertainty else "predictions"))
        return cls(grid, _tile_grid_origin(wsi), wsi.full_stride, slide=wsi.name,
                   mpp=wsi.mpp, kind='scores',
                   metadata={'tile_px': int(wsi.tile_px), 'tile_um': wsi.tile_um})

    @classmethod
    def from_segmentation(cls, segmentation: Any, *, slide: Optional[str] = None,
                          mpp: Optional[float] = None) -> "SpatialMap":
        """Wrap the label masks of a :class:`histox.cellseg.Segmentation`.

        Requires ``segmentation.wsi_dim`` (the level-0 size the masks cover).
        """
        ratio = segmentation.wsi_ratio
        if ratio is None:
            raise ValueError("Segmentation.wsi_dim is not set; cannot place masks on the slide")
        if slide is None and getattr(segmentation, 'slide', None) is not None:
            slide = segmentation.slide.name
            mpp = mpp or segmentation.slide.mpp
        return cls(segmentation.masks, segmentation._offset, ratio, slide=slide,
                   mpp=mpp, kind='labels')

    # --- I/O ------------------------------------------------------------------

    def save(self, path: str) -> str:
        """Save to ``.npz`` (values, plus geometry and metadata as JSON)."""
        meta = {
            'origin': list(self.origin), 'stride': list(self.stride),
            'slide': self.slide, 'mpp': self.mpp, 'kind': self.kind,
            'channels': self.channels, 'metadata': self.metadata,
            'format': 'histox.SpatialMap/1',
        }
        arrays = {'values': np.ma.getdata(self.values),
                  'meta': np.array(json.dumps(meta))}
        if np.ma.is_masked(self.values):
            arrays['mask'] = np.ma.getmaskarray(self.values)
        if not path.endswith('.npz'):
            path += '.npz'
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path: str) -> "SpatialMap":
        """Load a map saved with :meth:`save`."""
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f['meta']))
            values = f['values']
            if 'mask' in f.files:
                values = np.ma.masked_array(values, mask=f['mask'])
        return cls(values, tuple(meta['origin']), tuple(meta['stride']),
                   slide=meta['slide'], mpp=meta['mpp'], kind=meta['kind'],
                   channels=meta['channels'], metadata=meta.get('metadata') or {})

    def __repr__(self) -> str:
        return (f"SpatialMap(kind={self.kind!r}, shape={self.values.shape}, "
                f"origin={self.origin}, stride={self.stride}, slide={self.slide!r})")


def _tile_grid_origin(wsi: "hx.WSI") -> Tuple[float, float]:
    """Level-0 (x, y) of the top-left of the cell around grid tile (0, 0).

    Cells are ``full_stride`` wide and centered on tile centers, as in
    :func:`histox.heatmap.calculate_heatmap_extent`.
    """
    coord = np.asarray(wsi.coord)
    if len(coord):
        x, y, gx, gy = coord[0][:4]
        start = (x - gx * wsi.full_stride, y - gy * wsi.full_stride)
    else:
        start = wsi.origin if isinstance(wsi.origin, tuple) else (0, 0)
    center0 = geometry.tile_center(np.asarray(start), wsi.full_extract_px)
    return (float(center0[0]) - wsi.full_stride / 2, float(center0[1]) - wsi.full_stride / 2)
