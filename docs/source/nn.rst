.. currentmodule:: histox.nn

.. _nn:

histox.nn
=========

:mod:`histox.nn` defines the interfaces that HistoX models implement. A model
that follows them works with the rest of HistoX without being registered or
configured.

.. note::

   This module is **beta**. It implements RFC 0002, proposal P3.

Slide aggregators
-----------------

A slide aggregator pools a bag of tile features into a slide-level
prediction, as in multiple-instance learning. Subclass :class:`Aggregator` and
implement ``forward``:

.. code-block:: python

    import torch
    from torch import nn
    import histox as hx

    class GatedAttentionMIL(hx.nn.Aggregator):
        def __init__(self, in_features, num_classes, hidden=256):
            super().__init__()
            self.embed = nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU())
            self.score = nn.Linear(hidden, 1)
            self.head = nn.Linear(hidden, num_classes)

        def forward(self, features, mask=None, *, coords=None):
            h = self.embed(features)                              # (B, N, hidden)
            weights = hx.nn.masked_softmax(self.score(h).squeeze(-1), mask)
            pooled = (weights.unsqueeze(-1) * h).sum(1)           # (B, hidden)
            return hx.nn.AggregatorOutput(self.head(pooled), attention=weights,
                                          embedding=pooled)

The inputs follow :func:`histox.data.collate_bags`: zero-padded features
``(B, N, F)`` and a boolean ``mask`` ``(B, N)`` that is True for real tiles.
Such a model can be:

* trained with a PyTorch loop over :class:`histox.data.BagDataset`;
* used for MIL inference and evaluation, which call models through
  :class:`histox.mil.SlideAggregator`;
* turned into an attention map with :meth:`histox.SpatialMap.from_bag`.

Set ``provides_attention = False`` or ``provides_embedding = False`` on the
class if ``forward`` does not return per-tile attention or a slide embedding.

Reference models built this way are in :doc:`models`.

API
---

.. autoclass:: Aggregator
   :members: forward, full_mask

.. autoclass:: AggregatorOutput

.. autofunction:: lens_to_mask

.. autofunction:: masked_softmax
