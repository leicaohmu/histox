"""TileSpec: how a slide is cut into tiles.

A tiling is defined by four values:

* ``tile_px``: tile width (= height) in pixels, as given to models;
* ``tile_um``: physical tile width, in microns (``int``) or as a
  magnification (``str``, e.g. ``'20x'``);
* ``stride_div``: stride divisor (1 = adjacent tiles, 2 = 50% overlap, ...);
* ``origin``: level-0 ``(x, y)`` where the tile grid starts, or ``'random'``.

:class:`TileSpec` groups them in one immutable value that can be passed
wherever a tiling is needed:

.. code-block:: python

    import histox as hx

    spec = hx.TileSpec(tile_px=224, tile_um=112)
    spec.label                    # '224px_112um' (TFRecord folder name)

    wsi = hx.WSI('slide.svs', spec)
    dataset = project.dataset(spec)
    wsi.tile_spec == spec         # True

The separate ``tile_px`` / ``tile_um`` / ``stride_div`` / ``origin``
arguments keep working everywhere.

See RFC 0001 (``rfcs/0001-core-abstractions.md``), proposal P4.
"""

import dataclasses
import re
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

__all__ = ['TileSpec']

_LABEL_RE = re.compile(r'^(\d+)px_(\d+um|\d+(?:\.\d+)?x)$', re.IGNORECASE)


def _is_mag(value: str) -> bool:
    from histox.util import is_mag
    return is_mag(value)


@dataclass(frozen=True)
class TileSpec:
    """How a slide is cut into tiles.

    Args:
        tile_px (int): Tile width (= height) in pixels.
        tile_um (int or str): Physical tile width in microns (int), or a
            magnification (str, e.g. ``'20x'``; case-insensitive).
        stride_div (int): Stride divisor. 1 = adjacent tiles, 2 = 50% overlap.
            Defaults to 1.
        origin (tuple(int, int) or str): Level-0 ``(x, y)`` where the tile
            grid starts, or ``'random'`` for a random offset. Defaults to
            ``(0, 0)``.
    """

    tile_px: int
    tile_um: Union[int, str]
    stride_div: int = 1
    origin: Union[Tuple[int, int], str] = (0, 0)

    def __post_init__(self) -> None:
        # tile_px
        if isinstance(self.tile_px, bool) or not _is_int(self.tile_px) or int(self.tile_px) <= 0:
            raise ValueError(f"tile_px must be a positive integer, got {self.tile_px!r}")
        object.__setattr__(self, 'tile_px', int(self.tile_px))
        # tile_um
        tile_um = self.tile_um
        if isinstance(tile_um, str):
            if not _is_mag(tile_um):
                raise ValueError(
                    f"tile_um must be microns (int) or a magnification such as "
                    f"'20x', got {tile_um!r}")
            tile_um = tile_um.lower()
        elif isinstance(tile_um, bool) or not _is_int(tile_um) or int(tile_um) <= 0:
            raise ValueError(
                f"tile_um must be a positive integer (microns) or a "
                f"magnification such as '20x', got {tile_um!r}")
        else:
            tile_um = int(tile_um)
        object.__setattr__(self, 'tile_um', tile_um)
        # stride_div
        if isinstance(self.stride_div, bool) or not _is_int(self.stride_div) or int(self.stride_div) <= 0:
            raise ValueError(f"stride_div must be a positive integer, got {self.stride_div!r}")
        object.__setattr__(self, 'stride_div', int(self.stride_div))
        # origin
        origin = self.origin
        if isinstance(origin, str):
            if origin != 'random':
                raise ValueError(
                    f"origin must be 'random' or an (x, y) pair, got {origin!r}")
        else:
            try:
                origin = tuple(int(v) for v in origin)
            except TypeError:
                raise ValueError(
                    f"origin must be 'random' or an (x, y) pair, got {origin!r}") from None
            if len(origin) != 2:
                raise ValueError(f"origin must have two values, got {origin!r}")
        object.__setattr__(self, 'origin', origin)

    # --- Derived values -------------------------------------------------------

    @property
    def is_magnification(self) -> bool:
        """True if ``tile_um`` is a magnification (e.g. ``'20x'``)."""
        return isinstance(self.tile_um, str)

    @property
    def label(self) -> str:
        """Short label, e.g. ``'224px_112um'``, used for TFRecord folders.

        Depends only on ``tile_px`` and ``tile_um``.
        """
        from histox.util import tile_size_label
        return tile_size_label(self.tile_px, self.tile_um)

    def replace(self, **changes: Any) -> "TileSpec":
        """Return a copy with some fields changed."""
        return dataclasses.replace(self, **changes)

    # --- Conversion -----------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            'tile_px': self.tile_px,
            'tile_um': self.tile_um,
            'stride_div': self.stride_div,
            'origin': self.origin if isinstance(self.origin, str) else list(self.origin),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TileSpec":
        """Build from a dictionary with at least ``tile_px`` and ``tile_um``.

        Other keys are ignored, so extraction configs such as
        ``bags_config.json`` can be passed directly.
        """
        kw = {k: d[k] for k in ('tile_px', 'tile_um', 'stride_div', 'origin')
              if k in d and d[k] is not None}
        return cls(**kw)

    @classmethod
    def from_label(cls, label: str) -> "TileSpec":
        """Parse a label such as ``'224px_112um'`` or ``'256px_20x'``."""
        m = _LABEL_RE.match(label.strip())
        if not m:
            raise ValueError(
                f"Not a tile label: {label!r} (expected e.g. '224px_112um' or '256px_20x')")
        px, size = m.groups()
        size = size.lower()
        tile_um: Union[int, str] = int(size[:-2]) if size.endswith('um') else size
        return cls(int(px), tile_um)


def _is_int(value: Any) -> bool:
    try:
        return int(value) == value
    except (TypeError, ValueError):
        return False


def resolve_tile_args(
    tile_px: Any,
    tile_um: Any,
    stride_div: Any = None,
    origin: Any = None,
    *,
    caller: str = 'This function',
) -> Tuple[Any, Any, Any, Any]:
    """Unpack a TileSpec passed in place of ``tile_px``.

    Returns ``(tile_px, tile_um, stride_div, origin)``. If ``tile_px`` is a
    :class:`TileSpec`, its values are returned and the other arguments must be
    left unset (``None``). Otherwise the arguments are returned unchanged.
    """
    if isinstance(tile_px, TileSpec):
        given = [name for name, value in (('tile_um', tile_um),
                                          ('stride_div', stride_div),
                                          ('origin', origin))
                 if value is not None]
        if given:
            raise ValueError(
                f"{caller}: pass either a TileSpec or {', '.join(given)}, not both "
                f"(use spec.replace(...) to change a field)")
        return tile_px.tile_px, tile_px.tile_um, tile_px.stride_div, tile_px.origin
    return tile_px, tile_um, stride_div, origin
