"""Verified direct HTTP(S) downloads for open dataset assets."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

import requests

from ._cache import PathLike, plan_download
from ._models import AssetRecord, DatasetRecord, DownloadResult
from ._registry import get_dataset


PROVENANCE_FILENAME = ".histox-provenance.json"


class DatasetDownloadError(RuntimeError):
    """Base error raised while executing a dataset download."""


class DatasetAccessError(DatasetDownloadError):
    """Raised when a dataset requires a provider-specific access workflow."""


class DatasetConflictError(DatasetDownloadError):
    """Raised when an existing local path cannot be reused safely."""


class DatasetIntegrityError(DatasetDownloadError):
    """Raised when a downloaded or existing asset fails validation."""


def _resolve_record(
    dataset: Union[str, DatasetRecord], version: Optional[str]
) -> DatasetRecord:
    if isinstance(dataset, DatasetRecord):
        if version is not None:
            raise ValueError("version cannot be used with a DatasetRecord")
        return dataset
    if isinstance(dataset, str):
        return get_dataset(dataset, version=version)
    raise TypeError("dataset must be a registry name or DatasetRecord")


def _validate_url(asset: AssetRecord) -> str:
    if asset.url is None:
        raise DatasetDownloadError(
            "asset {!r} has no direct download URL".format(asset.path)
        )
    parsed = urlparse(asset.url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise DatasetDownloadError(
            "asset {!r} must use an absolute HTTP(S) URL".format(asset.path)
        )
    return asset.url


def _digest(path: Path, algorithm: str, chunk_size: int) -> str:
    try:
        hasher = hashlib.new(algorithm.lower())
    except ValueError as error:
        raise DatasetIntegrityError(
            "unsupported checksum algorithm {!r}".format(algorithm)
        ) from error
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify_asset(path: Path, asset: AssetRecord, chunk_size: int) -> str:
    if not path.is_file():
        raise DatasetIntegrityError("asset is not a regular file: {}".format(path))
    actual_size = path.stat().st_size
    if asset.size_bytes is not None and actual_size != asset.size_bytes:
        raise DatasetIntegrityError(
            "asset {!r} has {} bytes; expected {}".format(
                asset.path, actual_size, asset.size_bytes
            )
        )
    if asset.checksum is not None:
        actual = _digest(path, asset.checksum_algorithm or "", chunk_size)
        if actual.lower() != asset.checksum.lower():
            raise DatasetIntegrityError(
                "asset {!r} failed {} verification".format(
                    asset.path, asset.checksum_algorithm
                )
            )
        return "checksum"
    if asset.size_bytes is not None:
        return "size"
    return "unverified"


def _content_range_start(value: Optional[str]) -> Optional[int]:
    if not value or not value.startswith("bytes ") or "/" not in value:
        return None
    byte_range = value[6:].split("/", 1)[0]
    if "-" not in byte_range:
        return None
    try:
        return int(byte_range.split("-", 1)[0])
    except ValueError:
        return None


def _transfer_asset(
    asset: AssetRecord,
    target: Path,
    resume: bool,
    timeout: float,
    chunk_size: int,
) -> bool:
    url = _validate_url(asset)
    partial = target.with_name(target.name + ".part")
    if target.is_symlink() or partial.is_symlink():
        raise DatasetConflictError(
            "asset and partial destinations cannot be symbolic links: {}".format(
                target
            )
        )
    target.parent.mkdir(parents=True, exist_ok=True)

    if partial.is_file():
        try:
            _verify_asset(partial, asset, chunk_size)
        except DatasetIntegrityError:
            if (
                asset.size_bytes is not None
                and partial.stat().st_size >= asset.size_bytes
            ):
                partial.unlink()
        else:
            os.replace(str(partial), str(target))
            return True

    resume_from = partial.stat().st_size if resume and partial.is_file() else 0
    headers = {"Range": "bytes={}-".format(resume_from)} if resume_from else {}
    try:
        response = requests.get(
            url,
            headers=headers,
            stream=True,
            timeout=timeout,
            allow_redirects=True,
        )
    except requests.RequestException as error:
        raise DatasetDownloadError(
            "download failed for asset {!r}: {}".format(asset.path, error)
        ) from error
    try:
        response.raise_for_status()
        resumed = resume_from > 0 and response.status_code == 206
        if resumed:
            response_start = _content_range_start(
                response.headers.get("Content-Range")
            )
            if response_start != resume_from:
                raise DatasetDownloadError(
                    "provider returned an invalid Content-Range for {!r}".format(
                        asset.path
                    )
                )
        mode = "ab" if resumed else "wb"
        with partial.open(mode) as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
    except requests.RequestException as error:
        raise DatasetDownloadError(
            "download failed for asset {!r}: {}".format(asset.path, error)
        ) from error
    finally:
        response.close()

    try:
        _verify_asset(partial, asset, chunk_size)
    except DatasetIntegrityError:
        if partial.exists():
            partial.unlink()
        raise
    os.replace(str(partial), str(target))
    return resumed


def _write_provenance(
    record: DatasetRecord,
    destination: Path,
    actions: Dict[str, str],
    validation: Dict[str, str],
) -> Path:
    provenance_path = destination / PROVENANCE_FILENAME
    partial = provenance_path.with_name(provenance_path.name + ".part")
    document = {
        "schema_version": 1,
        "dataset": record.name,
        "version": record.version,
        "provider": record.provider.name,
        "provider_dataset_id": record.provider.dataset_id,
        "provider_url": record.provider.url,
        "access": record.access,
        "license": record.license,
        "terms_url": record.terms_url,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "assets": [
            {
                "path": asset.path,
                "url": asset.url,
                "size_bytes": (destination / asset.path).stat().st_size,
                "expected_size_bytes": asset.size_bytes,
                "checksum": asset.checksum,
                "checksum_algorithm": asset.checksum_algorithm,
                "validation": validation[asset.path],
                "action": actions[asset.path],
            }
            for asset in record.assets
        ],
    }
    with partial.open("w", encoding="utf-8") as file:
        json.dump(document, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(str(partial), str(provenance_path))
    return provenance_path


def download_dataset(
    dataset: Union[str, DatasetRecord],
    version: Optional[str] = None,
    path: Optional[PathLike] = None,
    *,
    overwrite: bool = False,
    resume: bool = True,
    timeout: float = 60.0,
    chunk_size: int = 1024 * 1024,
) -> DownloadResult:
    """Download and validate one open dataset into the HistoX cache.

    Assets are streamed into sibling ``.part`` files and moved atomically only
    after their expected size and provider checksum pass. Existing valid files
    are reused. A partial file is resumed when the provider honors HTTP Range
    requests; otherwise the transfer restarts safely.

    This generic adapter accepts only records with ``access="open"`` and direct
    HTTP(S) asset URLs. Controlled-access datasets require their own official
    provider client and user credentials.
    """

    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise TypeError("timeout must be a positive number")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be a positive integer")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    record = _resolve_record(dataset, version)
    if record.access.strip().lower() != "open":
        raise DatasetAccessError(
            "dataset {!r} requires provider-specific access; HistoX will not "
            "bypass provider authentication".format(record.name)
        )
    plan = plan_download(record, path=path)

    for item in plan.items:
        _validate_url(item.asset)
        partial = item.destination.with_name(item.destination.name + ".part")
        if item.destination.is_symlink() or partial.is_symlink():
            raise DatasetConflictError(
                "asset and partial destinations cannot be symbolic links: {}".format(
                    item.destination
                )
            )
        if item.status == "conflict":
            raise DatasetConflictError(
                "asset destination is not a regular file: {}".format(
                    item.destination
                )
            )
        if item.status == "size-mismatch" and not overwrite:
            raise DatasetConflictError(
                "asset {!r} has an unexpected size; pass overwrite=True to "
                "replace it".format(item.asset.path)
            )

    downloaded = []  # type: List[Path]
    reused = []  # type: List[Path]
    resumed = []  # type: List[Path]
    actions = {}  # type: Dict[str, str]
    validation = {}  # type: Dict[str, str]
    plan.destination.mkdir(parents=True, exist_ok=True)

    for item in plan.items:
        should_download = item.status != "present"
        if item.status == "present":
            try:
                validation[item.asset.path] = _verify_asset(
                    item.destination, item.asset, chunk_size
                )
            except DatasetIntegrityError:
                if not overwrite:
                    raise DatasetConflictError(
                        "asset {!r} failed validation; pass overwrite=True to "
                        "replace it".format(item.asset.path)
                    )
                should_download = True

        if should_download:
            was_resumed = _transfer_asset(
                item.asset,
                item.destination,
                resume=resume,
                timeout=float(timeout),
                chunk_size=chunk_size,
            )
            validation[item.asset.path] = _verify_asset(
                item.destination, item.asset, chunk_size
            )
            downloaded.append(item.destination)
            actions[item.asset.path] = "downloaded"
            if was_resumed:
                resumed.append(item.destination)
        else:
            reused.append(item.destination)
            actions[item.asset.path] = "reused"

    provenance_path = _write_provenance(
        record, plan.destination, actions, validation
    )
    return DownloadResult(
        dataset=record,
        destination=plan.destination,
        provenance_path=provenance_path,
        downloaded_files=tuple(downloaded),
        reused_files=tuple(reused),
        resumed_files=tuple(resumed),
    )
