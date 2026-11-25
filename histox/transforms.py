"""Transforms: quality control, stain normalization, and augmentation.

HistoX has three families of image transforms, which act at two levels:

* **Slide transforms** look at a whole slide and mark regions to exclude.
  Quality-control (QC) methods such as :class:`Otsu` and :class:`GaussianV2`
  are called as ``qc(wsi)`` and return a boolean mask (``True`` = excluded).
  :func:`qc_spatial_map` returns that mask as a :class:`histox.SpatialMap`.
* **Tile transforms** map one tile to another. They take a ``uint8`` tensor of
  shape ``(C, H, W)`` (or a batch ``(B, C, H, W)``) and return the same kind
  of tensor, like ``torchvision.transforms``: augmentations such as
  :class:`RandomColorDistortion`, and stain normalizers wrapped with
  :class:`TileNormalizer`.

Every transform can be described with :func:`describe`, which returns a
JSON-serializable record of its class and parameters, so a preprocessing
pipeline can be saved with the features and results it produced.

.. code-block:: python

    import histox as hx
    from histox import transforms as T

    tissue = T.qc_spatial_map('otsu', wsi)              # SpatialMap of the QC mask
    pipeline = T.Compose([
        T.TileNormalizer(T.normalizer('macenko')),
        T.RandomColorDistortion(0.5),
    ])
    tile = pipeline(tile)                               # uint8 (C, H, W)
    T.describe(pipeline)                                # provenance record

QC methods and stain normalizers are registered in :mod:`histox.registry`
(kinds ``'qc'`` and ``'normalizer'``).

See RFC 0001 (``rfcs/0001-core-abstractions.md``), proposal P6.
"""

from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional,
                    Union)

from histox.registry import Registry
from histox.slide.qc import (Otsu, Gaussian, GaussianV2, DeepFocus, StridedDL,  # noqa: F401
                             Segment, StridedSegment)
from histox.io.torch.augment import (RandomCardinalRotation, RandomGaussianBlur,  # noqa: F401
                                     RandomJPEGCompression, RandomColorProfile,
                                     RandomColorDistortion)

if TYPE_CHECKING:
    import torch
    import histox as hx
    from histox.norm import StainNormalizer

__all__ = [
    'Compose', 'TileNormalizer', 'describe', 'normalizer', 'qc_spatial_map',
    'QC_METHODS', 'NORMALIZERS',
    'Otsu', 'Gaussian', 'GaussianV2', 'DeepFocus', 'StridedDL', 'Segment',
    'StridedSegment', 'RandomCardinalRotation', 'RandomGaussianBlur',
    'RandomJPEGCompression', 'RandomColorProfile', 'RandomColorDistortion',
]


# --- Registries ---------------------------------------------------------------

def _lower(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Names must be non-empty strings")
    return name.strip().lower()


#: QC methods, by name. ``'otsu'`` and ``'blur'`` build the same objects
#: ``WSI.qc('otsu')`` and ``WSI.qc('blur')`` use.
QC_METHODS = Registry('qc', normalize=_lower, label='QC method')
QC_METHODS.add('otsu', Otsu, summary='Tissue detection by Otsu thresholding of saturation')
QC_METHODS.add('blur', GaussianV2, summary='Out-of-focus and pen-mark detection (Gaussian, strided)')
QC_METHODS.add('gaussian', Gaussian, summary='Out-of-focus detection (Gaussian, whole thumbnail)')
QC_METHODS.add('gaussian_v2', GaussianV2, summary='Out-of-focus detection (Gaussian, strided)')
QC_METHODS.add('deepfocus', DeepFocus, summary='Out-of-focus detection with the DeepFocus model')
QC_METHODS.add('strided_dl', StridedDL, summary='Any tile classifier applied across the slide')
QC_METHODS.add('segment', Segment, summary='Tissue segmentation model')
QC_METHODS.add('strided_segment', StridedSegment, summary='Tissue segmentation model, strided')


def normalizer(method: str, source: Optional[str] = None, **kwargs) -> "StainNormalizer":
    """Build a stain normalizer (see :func:`histox.norm.autoselect`).

    Args:
        method (str): e.g. ``'macenko'``, ``'reinhard'``, ``'reinhard_fast'``.
        source (str, optional): Image or preset to fit to.
    """
    from histox.norm import autoselect
    return autoselect(method, source=source, **kwargs)


def _normalizer_factory(method: str) -> Callable[..., "StainNormalizer"]:
    def build(source: Optional[str] = None, **kwargs):
        return normalizer(method, source=source, **kwargs)
    build.__name__ = method
    build.__doc__ = f"Build a '{method}' stain normalizer."
    return build


#: Stain normalization methods, by name. Calling an entry builds a
#: :class:`histox.norm.StainNormalizer`.
NORMALIZERS = Registry('normalizer', normalize=_lower, label='stain normalizer')


def _register_normalizers() -> None:
    from histox.norm import StainNormalizer
    for method, cls in StainNormalizer.normalizers.items():
        NORMALIZERS.add(method, _normalizer_factory(method),
                        **{'class': f"{cls.__module__}.{cls.__qualname__}"})


_register_normalizers()


# --- Tile transforms ------------------------------------------------------------

class TileNormalizer:
    """A stain normalizer as a tile transform.

    Takes a ``uint8`` tensor ``(C, H, W)`` or ``(B, C, H, W)`` and returns the
    normalized ``uint8`` tensor, like a ``torchvision`` transform.

    Args:
        normalizer (StainNormalizer or str): A normalizer, or a method name
            for :func:`normalizer`.
        augment (bool): Apply the normalizer's stain augmentation instead of
            (or in addition to, depending on the method) normalization.
            Defaults to False.
    """

    def __init__(self, normalizer: Union["StainNormalizer", str], *, augment: bool = False) -> None:
        if isinstance(normalizer, str):
            normalizer = globals()['normalizer'](normalizer)
        self.normalizer = normalizer
        self.augment = augment

    def __call__(self, image: "torch.Tensor") -> "torch.Tensor":
        return self.normalizer.torch_to_torch(image, augment=self.augment)

    def config(self) -> Dict[str, Any]:
        return {
            'type': 'histox.transforms.TileNormalizer',
            'method': self.normalizer.method,
            'augment': self.augment,
            'fit': self.normalizer.get_fit(as_list=True),
        }

    def __repr__(self) -> str:
        return f"TileNormalizer(method={self.normalizer.method!r}, augment={self.augment})"


class Compose:
    """Apply tile transforms in sequence, and describe them together.

    Works like ``torchvision.transforms.Compose``; :meth:`config` returns the
    :func:`describe` record of every step.
    """

    def __init__(self, transforms: Iterable[Callable]) -> None:
        self.transforms = list(transforms)

    def __call__(self, image: Any) -> Any:
        for t in self.transforms:
            image = t(image)
        return image

    def config(self) -> Dict[str, Any]:
        return {'type': 'histox.transforms.Compose',
                'transforms': [describe(t) for t in self.transforms]}

    def __repr__(self) -> str:
        inner = ',\n    '.join(repr(t) for t in self.transforms)
        return f"Compose([\n    {inner}\n])"


# --- Slide transforms ------------------------------------------------------------

def qc_spatial_map(method: Union[str, Callable], wsi: "hx.WSI", **kwargs) -> "hx.SpatialMap":
    """Run a QC method on a slide and return its mask as a SpatialMap.

    The slide itself is not modified; use ``wsi.qc(method)`` to apply QC to
    the tile grid.

    Args:
        method (str or callable): A QC method object, or a name registered in
            :data:`QC_METHODS` (built with ``**kwargs``).
        wsi (histox.WSI): The slide.

    Returns:
        histox.SpatialMap: ``kind='mask'``, ``True`` = excluded. Its metadata
        holds the :func:`describe` record of the method.
    """
    from histox.spatial import SpatialMap
    if isinstance(method, str):
        method = QC_METHODS.get(method)(**kwargs)
    mask = method(wsi)
    if mask is None:
        raise ValueError(f"QC method {method!r} returned no mask")
    smap = SpatialMap.from_qc_mask(getattr(mask, 'mask', mask), wsi)
    smap.metadata['qc'] = describe(method)
    return smap


# --- Provenance --------------------------------------------------------------------

_SIMPLE = (str, int, float, bool, type(None))


def _jsonable(value: Any) -> bool:
    if isinstance(value, _SIMPLE):
        return True
    if isinstance(value, (list, tuple)):
        return all(_jsonable(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _jsonable(v) for k, v in value.items())
    return False


def describe(transform: Any) -> Dict[str, Any]:
    """A JSON-serializable record of a transform's class and parameters.

    Uses the transform's own ``config()`` if it has one. Stain normalizers are
    described by method and fit; ``Compose`` objects (HistoX's or
    torchvision's) by their steps; other objects by their public attributes
    that are plain values (numbers, strings, booleans, lists of those).
    """
    config = getattr(transform, 'config', None)
    if callable(config):
        try:
            return config()
        except TypeError:
            pass
    try:
        from histox.norm import StainNormalizer
        if isinstance(transform, StainNormalizer):
            return {'type': 'histox.norm.StainNormalizer', 'method': transform.method,
                    'fit': transform.get_fit(as_list=True)}
    except ImportError:
        pass
    steps = getattr(transform, 'transforms', None)
    if isinstance(steps, (list, tuple)) and type(transform).__name__ == 'Compose':
        return {'type': f"{type(transform).__module__}.Compose",
                'transforms': [describe(t) for t in steps]}
    cls = type(transform)
    params = {k: (list(v) if isinstance(v, tuple) else v)
              for k, v in sorted(vars(transform).items()) if not k.startswith('_')
              and _jsonable(v)} if hasattr(transform, '__dict__') else {}
    return {'type': f"{cls.__module__}.{cls.__qualname__}", 'params': params}
