"""Feature bags: tile features of one slide, with coordinates and provenance.

A feature bag holds one feature vector per tile of a slide. It is the
standard input for multiple-instance learning (MIL). On disk, a directory of
bags contains, for each slide:

* ``<slide>.pt``: a ``torch.Tensor`` of shape ``(N, F)``;
* ``<slide>.index.npz`` (or ``.index.npy``): an ``(N, 2)`` array of tile
  centers, ``(x, y)`` in level-0 slide pixels (see :mod:`histox.geometry`);

and, once per directory, ``bags_config.json``, describing how the features
were extracted (extractor, stain normalizer, tile size).

:class:`Bag` reads and writes this layout, so bags written before this class
existed load unchanged.

.. code-block:: python

    import histox as hx

    bag = hx.Bag.load('bags/', 'TCGA-3L-AA1B-01Z-00-DX1')
    bag.features.shape          # torch.Size([N, F]), float32
    bag.coords[:3]              # tile centers, level-0 pixels
    bag.config['tile_um']

    inside = bag.subset(bag.coords[:, 0] > 50_000)
    inside.save('bags_subset/')

See RFC 0001 (``rfcs/0001-core-abstractions.md``), proposal P3.
"""

import json
import os
from os.path import exists, join
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    import torch
    from histox.slide.tile_spec import TileSpec

__all__ = ['Bag', 'list_bags', 'load_features', 'load_coords', 'FLOAT16_MAX']

#: Largest finite float16 value. Features beyond it are stored as float32.
FLOAT16_MAX = 65504.0

_CONFIG_NAME = 'bags_config.json'


def list_bags(directory: str) -> List[str]:
    """Return the slide names of the bags in a directory, sorted."""
    return sorted(f[:-3] for f in os.listdir(directory) if f.endswith('.pt'))


class Bag:
    """Tile features of one slide, with tile coordinates and provenance.

    Args:
        features (torch.Tensor or np.ndarray): ``(N, F)`` features, one row
            per tile.
        coords (np.ndarray, optional): ``(N, 2)`` tile centers, ``(x, y)`` in
            level-0 slide pixels. ``None`` if unknown.
        slide (str): Slide name.
        config (dict, optional): Extraction settings, in the format of
            ``bags_config.json`` (keys such as ``extractor``,
            ``normalizer``, ``num_features``, ``tile_px``, ``tile_um``).
    """

    def __init__(
        self,
        features: Union["torch.Tensor", np.ndarray],
        coords: Optional[np.ndarray] = None,
        *,
        slide: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        import torch
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be a torch.Tensor or np.ndarray")
        if features.ndim != 2:
            raise ValueError(
                f"features must have shape (N, F), got {tuple(features.shape)}")
        if coords is not None:
            coords = np.asarray(coords)
            if coords.shape != (features.shape[0], 2):
                raise ValueError(
                    f"coords must have shape ({features.shape[0]}, 2), "
                    f"got {coords.shape}")
        if not isinstance(slide, str) or not slide:
            raise ValueError("slide must be a non-empty string")
        self.features = features
        self.coords = coords
        self.slide = slide
        self.config = dict(config) if config else {}

    # --- Properties -----------------------------------------------------------

    def __len__(self) -> int:
        return int(self.features.shape[0])

    @property
    def num_features(self) -> int:
        """Number of features per tile."""
        return int(self.features.shape[1])

    @property
    def tile_px(self) -> Optional[int]:
        """Tile size in pixels, from the extraction config."""
        return self.config.get('tile_px')

    @property
    def tile_um(self) -> Optional[Union[int, str]]:
        """Tile size in microns (or magnification), from the extraction config."""
        return self.config.get('tile_um')

    @property
    def tile_spec(self) -> Optional["TileSpec"]:
        """Tile size as a :class:`histox.TileSpec`, from the extraction config.

        ``None`` if the config does not record ``tile_px`` and ``tile_um``.
        """
        from histox.slide.tile_spec import TileSpec
        if self.tile_px is None or self.tile_um is None:
            return None
        return TileSpec.from_dict(self.config)

    @property
    def extractor(self) -> Optional[str]:
        """Class path of the feature extractor, from the extraction config."""
        ext = self.config.get('extractor')
        if isinstance(ext, dict):
            return ext.get('class')
        return ext

    def __repr__(self) -> str:
        return (f"<Bag slide={self.slide!r} tiles={len(self)} "
                f"features={self.num_features} dtype={self.features.dtype}>")

    # --- Operations -----------------------------------------------------------

    def subset(self, index: Union[np.ndarray, Sequence[int], Sequence[bool]]) -> "Bag":
        """Return a new bag with the selected tiles.

        Args:
            index: Integer indices, or a boolean mask of length ``len(self)``.
        """
        import torch
        idx = np.asarray(index)
        if idx.dtype == bool:
            if idx.shape != (len(self),):
                raise ValueError(
                    f"Boolean mask must have length {len(self)}, got {idx.shape}")
            idx = np.flatnonzero(idx)
        idx_t = torch.as_tensor(idx, dtype=torch.long)
        return Bag(
            self.features[idx_t],
            None if self.coords is None else self.coords[idx],
            slide=self.slide,
            config=self.config,
        )

    def to(self, *args, **kwargs) -> "Bag":
        """Return a bag whose features are moved/cast with ``Tensor.to``."""
        return Bag(self.features.to(*args, **kwargs), self.coords,
                   slide=self.slide, config=self.config)

    # --- I/O ------------------------------------------------------------------

    def save(
        self,
        directory: str,
        *,
        dtype: Optional["torch.dtype"] = None,
        write_config: bool = True,
    ) -> str:
        """Save the bag in the standard bag-directory layout.

        Writes ``<slide>.pt`` and, if coordinates are known,
        ``<slide>.index.npz`` (``.index.npy`` when ZIP output is disabled).
        Writes ``bags_config.json`` if the directory does not have one yet;
        if it has a different one, logs a warning and leaves it unchanged.

        Args:
            directory (str): Destination directory; created if needed.
            dtype (torch.dtype, optional): Storage precision. Defaults to
                ``torch.float16``, which halves disk use. Bags with values
                outside the float16 range are stored as float32 instead, with
                a warning. Pass ``torch.float32`` to keep full precision.
            write_config (bool): Write ``bags_config.json``. Defaults to True.

        Returns:
            str: Path to the ``.pt`` file.
        """
        import torch
        from histox import log
        from histox.util import tfrecord2idx

        os.makedirs(directory, exist_ok=True)
        if dtype is None:
            dtype = torch.float16
        features = self.features.detach().cpu()
        if dtype == torch.float16 and features.numel():
            peak = float(features.abs().max())
            if not np.isfinite(peak) or peak > FLOAT16_MAX:
                log.warning(
                    f"Bag {self.slide}: features exceed the float16 range "
                    f"(max |value| = {peak:.4g}); storing as float32.")
                dtype = torch.float32
        path = join(directory, f'{self.slide}.pt')
        torch.save(features.to(dtype).contiguous(), path)
        if self.coords is not None:
            tfrecord2idx.save_index(self.coords, join(directory, f'{self.slide}.index'))
        if write_config and self.config:
            _write_config(directory, self.config)
        return path

    @classmethod
    def load(
        cls,
        directory: str,
        slide: str,
        *,
        dtype: Optional["torch.dtype"] = None,
        device: Any = 'cpu',
    ) -> "Bag":
        """Load the bag of ``slide`` from a bag directory.

        Args:
            directory (str): Bag directory.
            slide (str): Slide name.
            dtype (torch.dtype, optional): Precision of the returned features.
                Defaults to ``torch.float32``, whatever the storage precision.
            device: Device for the returned features. Defaults to ``'cpu'``.
        """
        import torch
        path = join(directory, f'{slide}.pt')
        if not exists(path):
            raise FileNotFoundError(f"No bag for slide {slide!r} in {directory}")
        features = torch.load(path, map_location=device)
        features = features.to(torch.float32 if dtype is None else dtype)
        from histox.util import tfrecord2idx
        # Same lookup as TFRecord indices: <slide>.index[.npz|.npy]
        coords = (tfrecord2idx.load_index(path)
                  if tfrecord2idx.find_index(path) else None)
        config_path = join(directory, _CONFIG_NAME)
        config = None
        if exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
        return cls(features, coords, slide=slide, config=config)

    @classmethod
    def from_path(cls, path: str, **kwargs) -> "Bag":
        """Load a bag from the path of its ``.pt`` file."""
        directory, name = os.path.split(path)
        if not name.endswith('.pt'):
            raise ValueError(f"Expected a .pt file, got {path}")
        return cls.load(directory or '.', name[:-3], **kwargs)


def load_features(path: str, *, dtype=None, device: Any = 'cpu') -> "torch.Tensor":
    """Load just the feature tensor of a bag file, as float32 by default.

    Accepts bags stored as float16 or float32.
    """
    import torch
    features = torch.load(path, map_location=device)
    return features.to(torch.float32 if dtype is None else dtype)


def load_coords(path: str) -> Optional[np.ndarray]:
    """Load the tile centers stored next to a bag file, or ``None`` if absent.

    Args:
        path (str): Path to the bag's ``.pt`` file. Looks for
            ``<slide>.index.npz`` or ``<slide>.index.npy`` beside it.
    """
    from histox.util import tfrecord2idx
    if not tfrecord2idx.find_index(path):
        return None
    return tfrecord2idx.load_index(path)


def _write_config(directory: str, config: Dict[str, Any]) -> None:
    from histox import log
    path = join(directory, _CONFIG_NAME)
    if exists(path):
        with open(path) as f:
            old = json.load(f)
        if old != _jsonable(config):
            log.warning(
                "Feature extraction configuration does not match the "
                "configuration used to generate the existing bags at "
                f"{directory}. Current configuration will not be saved. "
                f"Current configuration:\n{config}")
        return
    from histox.util import write_json
    write_json(config, path)


def _jsonable(obj: Any) -> Any:
    # Round-trip through JSON so tuples compare equal to lists, etc.
    from histox.util import ValidJSONEncoder
    return json.loads(json.dumps(obj, cls=ValidJSONEncoder))
