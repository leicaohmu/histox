"""Loading and querying of the built-in HistoX dataset registry."""

import json
from functools import lru_cache
from importlib import resources
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ._models import AssetRecord, DatasetRecord, ProviderRecord


class DatasetNotFoundError(LookupError):
    """Raised when a dataset name or version is absent from the registry."""


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("registry field {!r} must be a non-empty string".format(key))
    return value


def _optional_string(payload: Mapping[str, Any], key: str) -> Optional[str]:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("registry field {!r} must be null or a string".format(key))
    return value


def _string_tuple(payload: Mapping[str, Any], key: str) -> Tuple[str, ...]:
    value = payload.get(key, [])
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError("registry field {!r} must be a list of strings".format(key))
    return tuple(value)


def _asset_record(payload: Mapping[str, Any]) -> AssetRecord:
    size_bytes = payload.get("size_bytes")
    if size_bytes is not None and (
        isinstance(size_bytes, bool) or not isinstance(size_bytes, int)
    ):
        raise ValueError("registry asset size_bytes must be an integer or null")
    return AssetRecord(
        path=_required_string(payload, "path"),
        url=_optional_string(payload, "url"),
        size_bytes=size_bytes,
        checksum=_optional_string(payload, "checksum"),
        checksum_algorithm=_optional_string(payload, "checksum_algorithm"),
    )


def _dataset_record(payload: Mapping[str, Any]) -> DatasetRecord:
    provider = payload.get("provider")
    if not isinstance(provider, dict):
        raise ValueError("registry field 'provider' must be an object")
    assets = payload.get("assets", [])
    if not isinstance(assets, list) or not all(
        isinstance(asset, dict) for asset in assets
    ):
        raise ValueError("registry field 'assets' must be a list of objects")
    return DatasetRecord(
        name=_required_string(payload, "name"),
        version=_required_string(payload, "version"),
        title=_required_string(payload, "title"),
        description=_required_string(payload, "description"),
        provider=ProviderRecord(
            name=_required_string(provider, "name"),
            dataset_id=_required_string(provider, "dataset_id"),
            url=_required_string(provider, "url"),
        ),
        access=_required_string(payload, "access"),
        license=_optional_string(payload, "license"),
        terms_url=_optional_string(payload, "terms_url"),
        citations=_string_tuple(payload, "citations"),
        tasks=_string_tuple(payload, "tasks"),
        assets=tuple(_asset_record(asset) for asset in assets),
    )


@lru_cache(maxsize=1)
def _load_registry() -> Tuple[
    Dict[Tuple[str, str], DatasetRecord], Dict[str, str]
]:
    with resources.open_text(
        "histox.data", "registry.json", encoding="utf-8"
    ) as file:
        document = json.load(file)
    if not isinstance(document, dict):
        raise ValueError("HistoX dataset registry must be a JSON object")
    if document.get("schema_version") != 1:
        raise ValueError("unsupported HistoX dataset registry schema")
    entries = document.get("datasets")
    if not isinstance(entries, list):
        raise ValueError("registry field 'datasets' must be a list")

    records = {}  # type: Dict[Tuple[str, str], DatasetRecord]
    defaults = {}  # type: Dict[str, str]
    versions = {}  # type: Dict[str, List[str]]
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each registry dataset must be an object")
        record = _dataset_record(entry)
        if record.name != record.name.lower():
            raise ValueError("registry dataset names must be lowercase")
        key = (record.name, record.version)
        if key in records:
            raise ValueError("duplicate dataset record: {} {}".format(*key))
        records[key] = record
        versions.setdefault(record.name, []).append(record.version)
        is_default = entry.get("default", False)
        if not isinstance(is_default, bool):
            raise ValueError("registry field 'default' must be a boolean")
        if is_default:
            if record.name in defaults:
                raise ValueError(
                    "dataset {!r} has more than one default version".format(record.name)
                )
            defaults[record.name] = record.version

    for name, available_versions in versions.items():
        if name not in defaults:
            if len(available_versions) != 1:
                raise ValueError(
                    "dataset {!r} requires an explicit default version".format(name)
                )
            defaults[name] = available_versions[0]
    return records, defaults


def list_datasets() -> Tuple[DatasetRecord, ...]:
    """Return all built-in dataset records, ordered by name and version."""

    records, _ = _load_registry()
    return tuple(records[key] for key in sorted(records))


def get_dataset(name: str, version: Optional[str] = None) -> DatasetRecord:
    """Return one built-in dataset record.

    If ``version`` is omitted, the registry's explicit default version is used.
    """

    if not isinstance(name, str) or not name.strip():
        raise ValueError("dataset name must be a non-empty string")
    normalized_name = name.strip().lower()
    records, defaults = _load_registry()
    if normalized_name not in defaults:
        available = ", ".join(sorted(defaults)) or "none"
        raise DatasetNotFoundError(
            "unknown dataset {!r}; available datasets: {}".format(name, available)
        )
    selected_version = defaults[normalized_name] if version is None else str(version)
    key = (normalized_name, selected_version)
    if key not in records:
        available_versions = sorted(
            record_version
            for record_name, record_version in records
            if record_name == normalized_name
        )
        raise DatasetNotFoundError(
            "unknown version {!r} for dataset {!r}; available versions: {}".format(
                selected_version, normalized_name, ", ".join(available_versions)
            )
        )
    return records[key]
