"""The slide aggregator interface (RFC 0002, P3)."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

__all__ = ['Aggregator', 'AggregatorOutput', 'lens_to_mask', 'masked_softmax']


@dataclass
class AggregatorOutput:
    """Output of an :class:`Aggregator`.

    Attributes:
        logits (torch.Tensor): ``(B, C)`` model outputs, before any softmax.
        attention (torch.Tensor, optional): Per-tile weights, ``(B, N)`` or,
            for models with several attention heads, ``(B, heads, N)``.
            Padded tiles have weight 0. None if the model has no per-tile
            attention.
        embedding (torch.Tensor, optional): ``(B, D)`` slide-level
            representation that the logits are computed from. None if the
            model does not expose one.
    """

    logits: torch.Tensor
    attention: Optional[torch.Tensor] = None
    embedding: Optional[torch.Tensor] = None


class Aggregator(nn.Module):
    """Base class for slide-level models that pool a bag of tile features.

    Subclasses implement :meth:`forward`. Inputs come in the format of
    :func:`histox.data.collate_bags`: a zero-padded ``(B, N, F)`` feature
    tensor and a boolean ``(B, N)`` mask that is True for real tiles.

    Any subclass works with HistoX inference, evaluation, and attention
    heatmaps through :class:`histox.mil.SlideAggregator`, without registration.

    Class attributes:
        provides_attention (bool): Whether :meth:`forward` returns per-tile
            ``attention``. Defaults to True.
        provides_embedding (bool): Whether :meth:`forward` returns the slide
            ``embedding``. Defaults to True.

    Example:
        .. code-block:: python

            import torch
            from torch import nn
            import histox as hx

            class MeanPool(hx.nn.Aggregator):
                provides_attention = False

                def __init__(self, in_features, num_classes):
                    super().__init__()
                    self.head = nn.Linear(in_features, num_classes)

                def forward(self, features, mask=None, *, coords=None):
                    mask = self.full_mask(features) if mask is None else mask
                    w = mask.unsqueeze(-1).to(features.dtype)
                    pooled = (features * w).sum(1) / w.sum(1).clamp(min=1)
                    return hx.nn.AggregatorOutput(self.head(pooled), embedding=pooled)
    """

    provides_attention: bool = True
    provides_embedding: bool = True

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        *,
        coords: Optional[torch.Tensor] = None,
    ) -> AggregatorOutput:
        """Pool the tiles of each bag.

        Args:
            features (torch.Tensor): ``(B, N, F)`` tile features, zero-padded.
            mask (torch.Tensor, optional): ``(B, N)`` boolean mask, True for
                real tiles. None means every tile is real.
            coords (torch.Tensor, optional): ``(B, N, 2)`` tile centers in
                level-0 pixels, for models that use tile positions.

        Returns:
            AggregatorOutput
        """
        raise NotImplementedError

    @staticmethod
    def full_mask(features: torch.Tensor) -> torch.Tensor:
        """A mask that marks every tile of ``features`` as real."""
        return torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)


def lens_to_mask(lens: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """Convert bag lengths ``(B,)`` to a boolean mask ``(B, max_len)``."""
    lens = torch.as_tensor(lens)
    n = int(lens.max()) if max_len is None else int(max_len)
    return torch.arange(n, device=lens.device).unsqueeze(0) < lens.unsqueeze(-1)


def masked_softmax(scores: torch.Tensor, mask: Optional[torch.Tensor], dim: int = -1) -> torch.Tensor:
    """Softmax over ``dim`` that gives masked-out positions a weight of 0.

    Args:
        scores (torch.Tensor): Scores, e.g. ``(B, N)``.
        mask (torch.Tensor, optional): Boolean mask broadcastable to
            ``scores``, True where scores are valid. None means all valid.
        dim (int): Dimension to normalize over.
    """
    if mask is None:
        return torch.softmax(scores, dim=dim)
    scores = scores.masked_fill(~mask, float('-inf'))
    weights = torch.softmax(scores, dim=dim)
    # Rows with no valid entries would be NaN; make them 0.
    return torch.nan_to_num(weights, nan=0.0)
