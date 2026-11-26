"""TransMIL as a histox.nn.Aggregator."""

import numpy as np
import torch
from torch import nn

from histox.nn import Aggregator, AggregatorOutput

__all__ = ['TransMIL']


class TransMIL(Aggregator):
    """Transformer-based multiple-instance learning.

    Shao et al., "TransMIL: Transformer based Correlated Multiple Instance
    Learning for Whole Slide Image Classification", NeurIPS 2021. Same
    architecture and parameter names as :class:`histox.mil.models.TransMIL`.
    Requires the ``nystrom_attention`` package.

    Args:
        in_features (int): Tile feature dimension ``F``.
        num_classes (int): Number of outputs.

    Returns from :meth:`forward`: logits ``(B, C)`` and the class-token
    embedding ``(B, 512)``. TransMIL has no per-tile attention weights, so
    ``attention`` is None.

    The model reshapes the tiles of a bag into a square grid, so tiles from
    padding would change the result. When ``mask`` marks padded tiles, each
    bag is processed on its real tiles alone, which gives the same output as
    an unpadded batch of one.
    """

    provides_attention = False

    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        from histox.mil.models.transmil import PPEG, TransLayer
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(in_features, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, num_classes)

    def _embed(self, h: torch.Tensor) -> torch.Tensor:
        h = self._fc1(h)
        n = h.shape[1]
        side = int(np.ceil(np.sqrt(n)))
        h = torch.cat([h, h[:, :side * side - n, :]], dim=1)
        h = torch.cat((self.cls_token.expand(h.shape[0], -1, -1), h), dim=1)
        h = self.layer1(h)
        h = self.pos_layer(h, side, side)
        h = self.layer2(h)
        return self.norm(h)[:, 0]

    def forward(self, features, mask=None, *, coords=None) -> AggregatorOutput:
        if mask is None or bool(mask.all()):
            embedding = self._embed(features)
        else:
            embedding = torch.cat([
                self._embed(features[i][mask[i].to(torch.bool)].unsqueeze(0))
                for i in range(features.shape[0])
            ])
        return AggregatorOutput(logits=self._fc2(embedding), embedding=embedding)

    @classmethod
    def from_legacy(cls, model: nn.Module) -> 'TransMIL':
        """Build a TransMIL with the weights of a ``histox.mil.models.TransMIL``."""
        new = cls.__new__(cls)
        Aggregator.__init__(new)
        for name in ('pos_layer', '_fc1', 'layer1', 'layer2', 'norm', '_fc2'):
            setattr(new, name, getattr(model, name))
        new.cls_token = model.cls_token
        new.train(model.training)
        return new
