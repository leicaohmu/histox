"""Attention-based multiple-instance learning as a histox.nn.Aggregator."""

from typing import Optional, Tuple

import torch
from torch import nn

from histox.nn import Aggregator, AggregatorOutput

__all__ = ['ABMIL']


def _attention_net(n_in: int, n_latent: Optional[int] = None) -> nn.Module:
    if n_latent == 0:
        return nn.Linear(n_in, 1)
    n_latent = n_latent or (n_in + 1) // 2
    return nn.Sequential(nn.Linear(n_in, n_latent), nn.Tanh(), nn.Linear(n_latent, 1))


class ABMIL(Aggregator):
    """Attention-based multiple-instance learning.

    Tile features are embedded, scored by an attention network, and pooled
    by the softmax-normalized scores; a head predicts from the pooled
    embedding (Ilse et al., "Attention-based Deep Multiple Instance Learning",
    ICML 2018). Same architecture and parameter names as
    :class:`histox.mil.models.Attention_MIL`.

    Args:
        in_features (int): Tile feature dimension ``F``.
        num_classes (int): Number of outputs.
        z_dim (int): Embedding dimension. Defaults to 256.
        dropout_p (float): Dropout before the final layer. Defaults to 0.5.
        encoder, attention, head (torch.nn.Module, optional): Replace the
            default embedding network ``F -> z_dim``, attention network
            ``z_dim -> 1``, or prediction head ``z_dim -> num_classes``.
        attention_gate (float): If > 0, only tiles above this quantile of
            attention scores are pooled. Defaults to 0 (disabled).
        temperature (float): Attention softmax temperature. Defaults to 1.

    Returns from :meth:`forward`: logits ``(B, C)``, attention ``(B, N)``
    (padded tiles get 0), and the pooled embedding ``(B, z_dim)``.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        z_dim: int = 256,
        *,
        dropout_p: float = 0.5,
        encoder: Optional[nn.Module] = None,
        attention: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        attention_gate: float = 0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder or nn.Sequential(nn.Linear(in_features, z_dim), nn.ReLU())
        self.attention = attention or _attention_net(z_dim)
        self.head = head or nn.Sequential(
            nn.Flatten(), nn.BatchNorm1d(z_dim), nn.Dropout(dropout_p), nn.Linear(z_dim, num_classes))
        self.attention_gate = attention_gate
        self.temperature = temperature

    def _pool(self, features, mask):
        mask = self.full_mask(features) if mask is None else mask.to(torch.bool)
        embeddings = self.encoder(features)                     # (B, N, z)
        scores = self.attention(embeddings).squeeze(-1)         # (B, N)
        scores = scores.masked_fill(~mask, float('-inf'))
        if self.attention_gate and features.shape[1] > 1:
            threshold = torch.quantile(scores, q=self.attention_gate, dim=1, keepdim=True)
            scores = torch.where(scores > threshold, scores, torch.full_like(scores, float('-inf')))
        weights = torch.softmax(scores / self.temperature, dim=1)
        weights = torch.nan_to_num(weights, nan=0.0)            # bags with no tiles
        pooled = (weights.unsqueeze(-1) * embeddings).sum(1)    # (B, z)
        return weights, pooled

    def forward(self, features, mask=None, *, coords=None) -> AggregatorOutput:
        weights, pooled = self._pool(features, mask)
        return AggregatorOutput(logits=self.head(pooled), attention=weights, embedding=pooled)

    def predict_uq(
        self, features, mask=None, *, samples: int = 30, softmax: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Monte Carlo dropout: mean and standard deviation of predictions.

        Applies the head's dropout ``samples`` times to the pooled embedding.
        Requires the default head (flatten, batch norm, dropout, linear).

        Returns:
            tuple: ``(mean, std)``, each ``(B, C)``.
        """
        _, pooled = self._pool(features, mask)
        norm = self.head[1](self.head[0](pooled))
        expanded = norm.unsqueeze(1).expand(-1, samples, -1)
        was_training = self.head[2].training
        self.head[2].train()
        preds = self.head[3](self.head[2](expanded))
        self.head[2].train(was_training)
        if softmax:
            preds = torch.softmax(preds, dim=2)
        return preds.mean(dim=1), preds.std(dim=1)

    @classmethod
    def from_legacy(cls, model: nn.Module) -> 'ABMIL':
        """Build an ABMIL with the weights of a ``histox.mil.models.Attention_MIL``.

        The networks are reused, so custom encoder, attention, or head
        modules are kept.
        """
        new = cls.__new__(cls)
        Aggregator.__init__(new)
        new.encoder = model.encoder
        new.attention = model.attention
        new.head = model.head
        new.attention_gate = getattr(model, 'attention_gate', 0)
        new.temperature = getattr(model, 'temperature', 1.0)
        new.train(model.training)
        return new
