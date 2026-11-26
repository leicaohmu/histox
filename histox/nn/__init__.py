"""Base classes and layers for pathology models.

``histox.nn`` defines the interfaces that HistoX models implement, so that a
model written by a user works with the rest of HistoX -- datasets, inference,
evaluation, and heatmaps -- without being registered or configured.

* :class:`Aggregator`: slide-level models that pool a bag of tile features
  (multiple-instance learning). ``forward(features, mask)`` returns an
  :class:`AggregatorOutput` with logits, per-tile attention, and the pooled
  slide embedding.
* Helpers: :func:`lens_to_mask` and :func:`masked_softmax`.

Reference models built on these classes are in :mod:`histox.models`.

See RFC 0002 (``rfcs/0002-framework-building-blocks.md``), proposal P3.
"""

from ._aggregator import Aggregator, AggregatorOutput, lens_to_mask, masked_softmax

__all__ = ['Aggregator', 'AggregatorOutput', 'lens_to_mask', 'masked_softmax']
