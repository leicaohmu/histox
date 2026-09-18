"""Typed records shared by the HistoX data registry and providers."""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional, Tuple


_ASSET_STATES = {"conflict", "missing", "present", "size-mismatch"}


def _validate_component(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a non-empty string".format(field))
    if value != value.strip():
        raise ValueError("{} cannot start or end with whitespace".format(field))
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError("{} must be a single safe path component".format(field))


def _validate_asset_path(value: str) -> None:
    if not isinstance(value, str) or not value.strip() or "\\" in value:
        raise ValueError("asset path must be a non-empty POSIX relative path")
    if value in {".", ".."}:
        raise ValueError("asset path must name a file inside the dataset directory")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        raise ValueError("asset path must stay inside the dataset directory")


@dataclass(frozen=True)
class ProviderRecord:
    """Authoritative source for a dataset."""

    name: str
    dataset_id: str
    url: str

    def __post_init__(self) -> None:
        for field, value in (
            ("provider name", self.name),
            ("provider dataset_id", self.dataset_id),
            ("provider url", self.url),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("{} must be a non-empty string".format(field))


@dataclass(frozen=True)
class AssetRecord:
    """One provider-managed file described by a dataset record."""

    path: str
    url: Optional[str] = None
    size_bytes: Optional[int] = None
    checksum: Optional[str] = None
    checksum_algorithm: Optional[str] = None

    def __post_init__(self) -> None:
        _validate_asset_path(self.path)
        if self.size_bytes is not None:
            if isinstance(self.size_bytes, bool) or not isinstance(
                self.size_bytes, int
            ):
                raise TypeError("asset size_bytes must be an integer or null")
            if self.size_bytes < 0:
                raise ValueError("asset size_bytes cannot be negative")
        if self.url is not None and (
            not isinstance(self.url, str) or not self.url.strip()
        ):
            raise ValueError("asset url must be null or a non-empty string")
        if (self.checksum is None) != (self.checksum_algorithm is None):
            raise ValueError(
                "asset checksum and checksum_algorithm must be provided together"
            )
        for field, value in (
            ("asset checksum", self.checksum),
            ("asset checksum_algorithm", self.checksum_algorithm),
        ):
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError("{} must be a non-empty string".format(field))


@dataclass(frozen=True)
class DatasetRecord:
    """Versioned metadata for a dataset known to HistoX."""

    name: str
    version: str
    title: str
    description: str
    provider: ProviderRecord
    access: str
    license: Optional[str]
    terms_url: Optional[str]
    citations: Tuple[str, ...] = ()
    tasks: Tuple[str, ...] = ()
    assets: Tuple[AssetRecord, ...] = ()

    def __post_init__(self) -> None:
        _validate_component(self.name, "dataset name")
        _validate_component(self.version, "dataset version")
        for field, value in (
            ("dataset title", self.title),
            ("dataset description", self.description),
            ("dataset access", self.access),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("{} must be a non-empty string".format(field))
        if not isinstance(self.provider, ProviderRecord):
            raise TypeError("dataset provider must be a ProviderRecord")
        for field, value in (
            ("dataset license", self.license),
            ("dataset terms_url", self.terms_url),
        ):
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError("{} must be a non-empty string".format(field))
        if not all(
            isinstance(value, str) and value.strip() for value in self.citations
        ):
            raise ValueError("dataset citations must contain non-empty strings")
        if not all(
            isinstance(value, str) and value.strip() for value in self.tasks
        ):
            raise ValueError("dataset tasks must contain non-empty strings")
        if not all(isinstance(value, AssetRecord) for value in self.assets):
            raise TypeError("dataset assets must contain AssetRecord values")
        object.__setattr__(self, "citations", tuple(self.citations))
        object.__setattr__(self, "tasks", tuple(self.tasks))
        object.__setattr__(self, "assets", tuple(self.assets))

    @property
    def metadata_only(self) -> bool:
        """Return whether this record intentionally contains no data assets."""

        return not self.assets


@dataclass(frozen=True)
class DownloadItem:
    """Local state of one asset in a download plan."""

    asset: AssetRecord
    destination: Path
    status: str

    def __post_init__(self) -> None:
        if not isinstance(self.asset, AssetRecord):
            raise TypeError("download item asset must be an AssetRecord")
        if not isinstance(self.destination, Path):
            raise TypeError("download item destination must be a pathlib.Path")
        if self.status not in _ASSET_STATES:
            raise ValueError("unsupported download item status: {}".format(self.status))


@dataclass(frozen=True)
class DownloadPlan:
    """Read-only description of files needed for one dataset version."""

    dataset: DatasetRecord
    destination: Path
    items: Tuple[DownloadItem, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, DatasetRecord):
            raise TypeError("download plan dataset must be a DatasetRecord")
        if not isinstance(self.destination, Path):
            raise TypeError("download plan destination must be a pathlib.Path")
        if not all(isinstance(item, DownloadItem) for item in self.items):
            raise TypeError("download plan items must contain DownloadItem values")
        object.__setattr__(self, "items", tuple(self.items))

    @property
    def total_size_bytes(self) -> int:
        """Total known size of all assets."""

        return sum(item.asset.size_bytes or 0 for item in self.items)

    @property
    def download_size_bytes(self) -> int:
        """Known bytes needed for missing, conflicting, or invalid assets."""

        return sum(
            item.asset.size_bytes or 0
            for item in self.items
            if item.status != "present"
        )

    @property
    def unknown_size_count(self) -> int:
        """Number of non-present assets whose provider size is unknown."""

        return sum(
            1
            for item in self.items
            if item.status != "present" and item.asset.size_bytes is None
        )

    @property
    def present_count(self) -> int:
        """Number of assets already present with the expected size."""

        return sum(item.status == "present" for item in self.items)

    @property
    def pending_count(self) -> int:
        """Number of assets that still require provider action."""

        return len(self.items) - self.present_count

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation of this plan."""

        return {
            "schema_version": 1,
            "dataset": self.dataset.name,
            "version": self.dataset.version,
            "provider": self.dataset.provider.name,
            "provider_dataset_id": self.dataset.provider.dataset_id,
            "access": self.dataset.access,
            "license": self.dataset.license,
            "terms_url": self.dataset.terms_url,
            "destination": str(self.destination),
            "total_assets": len(self.items),
            "present_assets": self.present_count,
            "pending_assets": self.pending_count,
            "total_size_bytes": self.total_size_bytes,
            "download_size_bytes": self.download_size_bytes,
            "unknown_size_assets": self.unknown_size_count,
            "items": [
                {
                    "path": item.asset.path,
                    "url": item.asset.url,
                    "size_bytes": item.asset.size_bytes,
                    "checksum": item.asset.checksum,
                    "checksum_algorithm": item.asset.checksum_algorithm,
                    "destination": str(item.destination),
                    "status": item.status,
                }
                for item in self.items
            ],
        }
