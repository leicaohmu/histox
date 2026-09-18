"""Offline dataset registry and download planning.

This module describes datasets and determines where provider-managed assets
would be stored.  It deliberately performs no network transfers; provider
downloads are introduced separately so that large transfers remain explicit.
"""

from ._cache import plan_download, resolve_cache_root
from ._models import (
    AssetRecord,
    DatasetRecord,
    DownloadItem,
    DownloadPlan,
    ProviderRecord,
)
from ._registry import DatasetNotFoundError, get_dataset, list_datasets

__all__ = [
    "AssetRecord",
    "DatasetNotFoundError",
    "DatasetRecord",
    "DownloadItem",
    "DownloadPlan",
    "ProviderRecord",
    "get_dataset",
    "list_datasets",
    "plan_download",
    "resolve_cache_root",
]
