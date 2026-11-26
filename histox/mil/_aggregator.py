"""One calling convention for MIL models (slide aggregators).

MIL models in HistoX and in extension packages take their inputs in
different ways:

* ``forward(bags, lens)``, declared with a class attribute ``use_lens = True``
  (``Attention_MIL``);
* ``forward(bags)`` (``TransMIL``, CLAM models);
* ``forward(*[(bags, lens), ...])`` for multimodal models
  (``MultiModal_Attention_MIL``, which also sets ``use_lens = True``).

Some expose ``calculate_attention`` and ``get_last_layer_activations``, some
accept ``return_attention=`` or ``uq=`` in ``forward``, and some return a tuple
whose first element is the logits.

:class:`SlideAggregator` wraps any of them and offers one interface, so
inference, evaluation, and feature analysis do not branch on model classes.
Models are not modified.

.. code-block:: python

    from histox.mil import SlideAggregator

    agg = SlideAggregator(model)
    logits = agg.predict(bags)                       # bags: (B, N, F)
    logits, att = agg.predict(bags, attention=True)  # att: (B, N) or (B, N, C)
    h = agg.activations(bags)

Writing a new MIL model: subclass :class:`histox.nn.Aggregator`, whose
``forward(features, mask)`` returns logits, attention, and the slide embedding
in one :class:`histox.nn.AggregatorOutput`. :class:`SlideAggregator` handles
these models and the older conventions below in the same way.

Older convention, still supported: subclass ``torch.nn.Module`` and implement
``forward(bags)`` or, if it needs bag lengths, ``forward(bags, lens)`` with
``use_lens = True``. Optional: ``calculate_attention(...)`` (same arguments
as ``forward``) and ``get_last_layer_activations(...)``.

See RFC 0001 (``rfcs/0001-core-abstractions.md``), proposal P7.
"""

import inspect
from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    import torch

__all__ = ['SlideAggregator', 'uses_lens', 'is_aggregator']


def uses_lens(model: Any) -> bool:
    """Whether ``model.forward`` expects bag lengths after the bags.

    Uses the model's ``use_lens`` attribute if it has one; otherwise checks
    whether the second parameter of ``forward`` is named ``lens``.
    """
    flag = getattr(model, 'use_lens', None)
    if flag is not None:
        return bool(flag)
    try:
        params = [p for p in inspect.signature(model.forward).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    except (TypeError, ValueError):
        return False
    return len(params) >= 2 and params[1].name == 'lens'


def _accepts(fn: Any, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def is_aggregator(model: Any) -> bool:
    """Whether ``model`` is a :class:`histox.nn.Aggregator`."""
    try:
        from histox.nn import Aggregator
    except ImportError:
        return False
    return isinstance(model, Aggregator)


def _logits(out: Any) -> Any:
    # Some models (e.g. CLAM) return (logits, ...) from forward().
    return out[0] if isinstance(out, (tuple, list)) else out


class SlideAggregator:
    """Uniform interface to a MIL model.

    Args:
        model (torch.nn.Module): A MIL model.
        use_lens (bool, optional): Whether ``forward`` takes bag lengths.
            Defaults to :func:`uses_lens` of the model.
    """

    def __init__(self, model: Any, use_lens: Optional[bool] = None) -> None:
        self.model = model
        #: True if the model is a :class:`histox.nn.Aggregator`.
        self.is_aggregator = is_aggregator(model)
        if self.is_aggregator:
            # Aggregators take a mask; lengths given to the methods below are
            # converted to one.
            self.use_lens = False
        else:
            self.use_lens = uses_lens(model) if use_lens is None else bool(use_lens)

    def _run(self, bags, lens=None):
        """Call an Aggregator; ``lens`` becomes its mask."""
        from histox.nn import lens_to_mask
        mask = None if lens is None else lens_to_mask(lens, bags.shape[1]).to(bags.device)
        return self.model(bags, mask=mask), mask

    @staticmethod
    def _legacy_attention(att):
        # Aggregators return (B, N) or (B, heads, N); older models return
        # (B, N, 1) or (B, N, C). Callers expect the older layout.
        return att.unsqueeze(-1) if att.ndim == 2 else att.transpose(1, 2)

    # --- Capabilities ---------------------------------------------------------

    @property
    def is_multimodal(self) -> bool:
        """Whether the model takes several bags (one per modality) per sample."""
        if self.is_aggregator:
            return False
        return bool(getattr(self.model, 'is_multimodal', False)) or (
            type(self.model).__name__ in ('MultiModal_Attention_MIL',
                                          'UQ_MultiModal_Attention_MIL'))

    @property
    def supports_attention(self) -> bool:
        """Whether attention scores can be computed."""
        if self.is_aggregator:
            return bool(self.model.provides_attention)
        return (hasattr(self.model, 'calculate_attention')
                or _accepts(self.model.forward, 'return_attention'))

    @property
    def supports_uq(self) -> bool:
        """Whether ``forward`` can return uncertainty (``uq=True``)."""
        if self.is_aggregator:
            return hasattr(self.model, 'predict_uq')
        return _accepts(self.model.forward, 'uq')

    @property
    def supports_activations(self) -> bool:
        """Whether last-layer activations can be computed."""
        if self.is_aggregator:
            return bool(self.model.provides_embedding)
        return hasattr(self.model, 'get_last_layer_activations')

    # --- Arguments ------------------------------------------------------------

    def args(
        self,
        bags: Union["torch.Tensor", Sequence["torch.Tensor"]],
        lens: Optional[Union["torch.Tensor", Sequence["torch.Tensor"]]] = None,
    ) -> Tuple[Any, ...]:
        """Positional arguments for the wrapped model.

        Args:
            bags: ``(B, N, F)`` tensor, or one such tensor per modality for a
                multimodal model.
            lens: ``(B,)`` bag lengths (one tensor per modality for a
                multimodal model). Defaults to the full bag length ``N``.
        """
        import torch

        def full(t):
            return torch.full((t.shape[0],), t.shape[1], device=t.device)

        if self.is_multimodal:
            bags = list(bags)
            if not self.use_lens:
                return (bags,)
            lens = [full(b) for b in bags] if lens is None else list(lens)
            return tuple((b, l) for b, l in zip(bags, lens))
        if not self.use_lens:
            return (bags,)
        return (bags, full(bags) if lens is None else lens)

    # --- Calls ----------------------------------------------------------------

    def forward(self, bags, lens=None, **kwargs) -> Any:
        """Call the model and return its raw output.

        For a :class:`histox.nn.Aggregator` this is an
        :class:`~histox.nn.AggregatorOutput`.
        """
        if self.is_aggregator:
            return self._run(bags, lens)[0]
        return self.model(*self.args(bags, lens), **kwargs)

    def predict(
        self,
        bags,
        lens=None,
        *,
        attention: bool = False,
        uq: bool = False,
        **kwargs
    ) -> Any:
        """Return logits, plus attention and/or uncertainty if requested.

        Returns:
            ``logits``; ``(logits, attention)`` if ``attention``;
            ``(logits, uncertainty)`` if ``uq``; ``(logits, attention,
            uncertainty)`` if both. Attention is unpooled and unsqueezed, as
            the model returns it.
        """
        if self.is_aggregator:
            return self._predict_aggregator(bags, lens, attention=attention, uq=uq)
        args = self.args(bags, lens)
        if uq:
            if not self.supports_uq:
                raise RuntimeError(
                    f"Model {type(self.model).__name__} does not support UQ.")
            kwargs = dict(uq=True, **kwargs)
        att = None
        if attention and _accepts(self.model.forward, 'return_attention'):
            out, att = self.model(*args, return_attention=True, **kwargs)
        else:
            out = self.model(*args, **kwargs)
            if attention:
                if not hasattr(self.model, 'calculate_attention'):
                    raise RuntimeError(
                        f"Model {type(self.model).__name__} does not support attention.")
                att = self.model.calculate_attention(*args)
        if uq:
            logits, unc = out
        else:
            logits, unc = _logits(out), None
        if attention and uq:
            return logits, att, unc
        if attention:
            return logits, att
        if uq:
            return logits, unc
        return logits

    def _predict_aggregator(self, bags, lens, *, attention, uq):
        name = type(self.model).__name__
        if attention and not self.supports_attention:
            raise RuntimeError(f"Model {name} does not support attention.")
        if uq and not self.supports_uq:
            raise RuntimeError(f"Model {name} does not support UQ.")
        out, mask = self._run(bags, lens)
        logits, unc = out.logits, None
        if uq:
            logits, unc = self.model.predict_uq(bags, mask)
        att = self._legacy_attention(out.attention) if attention else None
        if attention and uq:
            return logits, att, unc
        if attention:
            return logits, att
        if uq:
            return logits, unc
        return logits

    def attention(self, bags, lens=None, **kwargs) -> Any:
        """Return attention scores, as the model computes them."""
        if self.is_aggregator:
            if not self.supports_attention:
                raise RuntimeError(
                    f"Model {type(self.model).__name__} does not support attention.")
            return self._legacy_attention(self._run(bags, lens)[0].attention)
        if hasattr(self.model, 'calculate_attention'):
            return self.model.calculate_attention(*self.args(bags, lens), **kwargs)
        if _accepts(self.model.forward, 'return_attention'):
            return self.model(*self.args(bags, lens), return_attention=True)[1]
        raise RuntimeError(
            f"Model {type(self.model).__name__} does not support attention.")

    def activations(self, bags, lens=None) -> Tuple[Any, Optional[Any]]:
        """Return ``(activations, attention_or_None)`` from the last layer.

        Most models return activations only; CLAM models also return their
        attention, which is passed through.
        """
        if not self.supports_activations:
            raise NotImplementedError(
                f"Layer activation support for model {type(self.model).__name__} "
                "is not implemented.")
        if self.is_aggregator:
            return self._run(bags, lens)[0].embedding, None
        out = self.model.get_last_layer_activations(*self.args(bags, lens))
        if isinstance(out, (tuple, list)):
            return out[0], out[1]
        return out, None
