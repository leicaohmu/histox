"""Model cards: what a model is, where its weights come from, and what it expects.

A :class:`ModelCard` records, for one model:

* identity: name, kind, backend;
* terms: license and citation;
* weights: where the pretrained weights come from (e.g. a Hugging Face ID);
* input contract: tile size the preprocessing produces, normalization;
* output: number of features.

Fields that are not known stay ``None``. Values are taken from the model's
own code and configuration, never filled in by hand.

.. code-block:: python

    import histox as hx

    hx.model.extractor_card('dinov2')        # before building: terms only
    ex = hx.build_feature_extractor('dinov2', model_id='facebook/dinov2-small')
    ex.card                                  # after building: everything known
    print(ex.card.summary())

See RFC 0001 (``rfcs/0001-core-abstractions.md``), proposal P7.
"""

import importlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

__all__ = ['ModelCard', 'extractor_card']

_WEIGHT_KEYS = ('model_id', 'revision', 'weights', 'cfg', 'ckpt', 'checkpoint')


@dataclass(frozen=True)
class ModelCard:
    """Description of a model: identity, terms, weights, inputs, and outputs.

    Args:
        name (str): Registry name or tag of the model.
        kind (str): ``'tile_encoder'`` or ``'mil_model'``.
        backend (str, optional): ``'torch'`` or ``'tensorflow'``.
        license (str, optional): License of the model and/or its weights.
        citation (str, optional): Citation (BibTeX or text).
        weights (dict, optional): Where the weights come from, e.g.
            ``{'model_id': 'facebook/dinov2-small', 'revision': None}``.
        num_features (int, optional): Output feature dimension.
        input_px (int, optional): Tile size, in pixels, after the model's
            preprocessing (resize / center crop). ``None`` if tiles are used
            at their extracted size.
        norm_mean (tuple, optional): Per-channel mean used for normalization.
        norm_std (tuple, optional): Per-channel standard deviation.
        extras (str, optional): Extras package that provides the model, if
            it is not part of HistoX itself.
        installed (bool): Whether the model can be built in this environment.
        details (dict): Other model-specific facts.
    """

    name: str
    kind: str = 'tile_encoder'
    backend: Optional[str] = None
    license: Optional[str] = None
    citation: Optional[str] = None
    weights: Optional[Dict[str, Any]] = None
    num_features: Optional[int] = None
    input_px: Optional[int] = None
    norm_mean: Optional[Tuple[float, ...]] = None
    norm_std: Optional[Tuple[float, ...]] = None
    extras: Optional[str] = None
    installed: bool = True
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        d = asdict(self)
        for key in ('norm_mean', 'norm_std'):
            if d[key] is not None:
                d[key] = list(d[key])
        return d

    def summary(self) -> str:
        """Return a short, human-readable description."""
        lines = [f"{self.name} ({self.kind}"
                 + (f", {self.backend}" if self.backend else "") + ")"]
        if not self.installed:
            lines.append(f"  not installed: pip install {self.extras}"
                         if self.extras else "  not installed")
        if self.num_features is not None:
            lines.append(f"  features: {self.num_features}")
        if self.input_px is not None:
            lines.append(f"  input: {self.input_px} px")
        if self.norm_mean is not None:
            lines.append(f"  normalization: mean={list(self.norm_mean)} "
                         f"std={list(self.norm_std) if self.norm_std else None}")
        if self.weights:
            lines.append("  weights: " + ", ".join(
                f"{k}={v}" for k, v in self.weights.items() if v is not None))
        lines.append(f"  license: {_first_line(self.license) or 'not recorded'}")
        lines.append("  citation: " + ("available (.citation)" if self.citation
                                       else "not recorded"))
        return "\n".join(lines)


def _first_line(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    text = text.strip()
    return text.splitlines()[0] if text else None


def _none_if_empty(value: Any) -> Any:
    if isinstance(value, str):
        value = value.strip()
    return value or None


def _transform_contract(transform: Any) -> Dict[str, Any]:
    """Read input size and normalization from a torchvision Compose."""
    out: Dict[str, Any] = {}
    steps = getattr(transform, 'transforms', None)
    if not steps:
        return out
    try:
        from torchvision import transforms as T
    except ImportError:
        return out
    for step in steps:
        if isinstance(step, (T.Resize, T.CenterCrop)):
            size = step.size
            if isinstance(size, (list, tuple)):
                size = size[0] if len(set(size)) == 1 else None
            if isinstance(size, int):
                out['input_px'] = size
        elif isinstance(step, T.Normalize):
            out['norm_mean'] = tuple(float(v) for v in step.mean)
            out['norm_std'] = tuple(float(v) for v in step.std)
    return out


def card_from_extractor(extractor: Any, name: Optional[str] = None) -> ModelCard:
    """Build a card from a constructed feature extractor."""
    info: Dict[str, Any] = dict(
        name=name or getattr(extractor, 'tag', None) or type(extractor).__name__,
        kind='tile_encoder',
        backend=getattr(extractor, 'backend', None),
        license=_none_if_empty(getattr(extractor, 'license', None)),
        citation=_none_if_empty(getattr(extractor, 'citation', None)),
        num_features=getattr(extractor, 'num_features', None) or None,
    )
    try:
        cfg = extractor.dump_config()
        kwargs = cfg.get('kwargs', {}) if isinstance(cfg, dict) else {}
        weights = {k: kwargs[k] for k in _WEIGHT_KEYS if k in kwargs}
        info['weights'] = weights or None
        if isinstance(cfg, dict) and cfg.get('class'):
            info['details'] = {'class': cfg['class']}
    except Exception:
        pass
    info.update(_transform_contract(getattr(extractor, 'transform', None)))
    return ModelCard(**info)


def extractor_card(name: str) -> ModelCard:
    """Return the card of a registered feature extractor, without building it.

    Building an extractor may download large weights; this does not. The card
    therefore holds what is known before construction (name, backend,
    license, citation, extras package). Build the extractor and use its
    ``.card`` for the input contract and feature dimension.

    Args:
        name (str): Registered extractor name (see
            :func:`histox.model.list_extractors`).

    Raises:
        histox.errors.InvalidFeatureExtractor: If ``name`` is unknown.
    """
    from histox import errors
    from histox.model.extractors import _registry as reg

    for backend, registry in (('torch', reg._torch_extractors),
                              ('tensorflow', reg._tf_extractors)):
        for key in (name, name + '_imagenet'):
            if key in registry:
                meta = dict(registry.metadata(key))
                info: Dict[str, Any] = dict(name=key, backend=backend)
                cls_path = meta.pop('card_cls', None)
                if cls_path:
                    module_name, cls_name = cls_path.rsplit('.', 1)
                    try:
                        cls = getattr(importlib.import_module(module_name), cls_name)
                        info['license'] = _none_if_empty(getattr(cls, 'license', None))
                        info['citation'] = _none_if_empty(getattr(cls, 'citation', None))
                        info['details'] = {'class': cls_path}
                    except Exception:
                        info['installed'] = False
                for k in ('license', 'citation'):
                    if meta.get(k):
                        info[k] = meta[k]
                return ModelCard(**info)
    package = reg._extras_extractors.get(name)
    if package is not None:
        return ModelCard(name=name, extras=package, installed=False)
    raise errors.InvalidFeatureExtractor(f"Unrecognized feature extractor: {name}")
