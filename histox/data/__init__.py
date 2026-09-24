"""Dataset discovery, storage planning, verified downloads, and provenance."""

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
    "resolve_cache_root",
]
