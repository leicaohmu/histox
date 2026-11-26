"""Reference models built on :mod:`histox.nn`.

These are complete models that use only the public HistoX interfaces, as
``torchvision.models`` does for ``torch.nn``. Each one is a
:class:`histox.nn.Aggregator`, so it can be trained with a plain PyTorch loop
over :class:`histox.data.BagDataset` and used with HistoX inference,
evaluation, and heatmaps.

* :class:`ABMIL`: attention-based MIL (Ilse et al., 2018).
* :class:`TransMIL`: transformer-based MIL (Shao et al., 2021).

Both load the weights of the corresponding ``histox.mil.models`` classes with
:meth:`from_legacy` and then give the same outputs.

See RFC 0002 (``rfcs/0002-framework-building-blocks.md``), proposals P3 and P8.
"""

from ._abmil import ABMIL
from ._transmil import TransMIL

__all__ = ['ABMIL', 'TransMIL']
