"""Datasets: public cohort discovery and downloads, and PyTorch datasets.

* Public datasets: discovery, storage planning, verified downloads, and
  provenance (:func:`list_datasets`, :func:`download_dataset`, ...).
* PyTorch datasets that return one ``dict`` per item (RFC 0002, P2):
  :class:`BagDataset` and :func:`collate_bags`. They are imported on first use,
  so ``import histox.data`` does not load PyTorch.
"""

from ._cache import plan_download, resolve_cache_root
from ._download import (
    DatasetAccessError,
    DatasetConflictError,
    DatasetDownloadError,
    DatasetIntegrityError,
    download_dataset,
)
from ._models import (
    AssetRecord,
    DatasetRecord,
    DownloadItem,
    DownloadPlan,
    DownloadResult,
    ProviderRecord,
)
from ._registry import DatasetNotFoundError, get_dataset, list_datasets

__all__ = [
    "AssetRecord",
    "BagDataset",
    "DatasetNotFoundError",
    "DatasetRecord",
    "DatasetAccessError",
    "DatasetConflictError",
    "DatasetDownloadError",
    "DatasetIntegrityError",
    "DownloadItem",
    "DownloadPlan",
    "DownloadResult",
    "ProviderRecord",
    "download_dataset",
    "get_dataset",
    "list_datasets",
    "plan_download",
    "collate_bags",
    "resolve_cache_root",
]

_TORCH_DATASETS = {"BagDataset", "collate_bags"}


def __getattr__(name):
    if name in _TORCH_DATASETS:
        from . import _bags
        value = getattr(_bags, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'histox.data' has no attribute {name!r}")
