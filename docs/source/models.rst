.. currentmodule:: histox.models

.. _models:

histox.models
=============

:mod:`histox.models` contains complete reference models built only on the
public interfaces in :doc:`nn`, as ``torchvision.models`` does for
``torch.nn``.

.. note::

   This module is **beta**. It implements RFC 0002, proposals P3 and P8.

Each model is a :class:`histox.nn.Aggregator`. It has the same architecture and
parameter names as the corresponding class in :mod:`histox.mil.models`, which
is used by :func:`histox.mil.train_mil`. ``from_legacy`` converts a trained
model of the older class; the converted model gives the same outputs.

.. code-block:: python

    import histox as hx
    from histox.mil.utils import load_model_weights

    legacy, config = load_model_weights("mil/00000-attention_mil-subtype")
    model = hx.models.ABMIL.from_legacy(legacy).eval()
    out = model(features, mask)     # out.logits, out.attention, out.embedding

.. autoclass:: ABMIL
   :members: forward, predict_uq, from_legacy

.. autoclass:: TransMIL
   :members: forward, from_legacy
