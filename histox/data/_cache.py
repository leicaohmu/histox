"""Cache resolution and read-only planning for dataset assets."""

import os
from pathlib import Path, PurePosixPath
from typing import Optional, Union

from ._models import DatasetRecord, DownloadItem, DownloadPlan
from ._registry import get_dataset


PathLike = Union[str, Path]


def _clean_path(value: PathLike, source: str) -> Path:
    if isinstance(value, str) and not value.strip():
        raise ValueError("{} cannot be empty".format(source))
    if not isinstance(value, (str, Path)):
        raise TypeError("{} must be a string or pathlib.Path".format(source))
    return Path(value).expanduser()


def resolve_cache_root(path: Optional[PathLike] = None) -> Path:
    """Resolve the HistoX dataset cache root without creating it.

    Resolution order is an explicit ``path``, ``HISTOX_CACHE_DIR``,
    ``XDG_CACHE_HOME/histox``, then ``~/.cache/histox``.

    Args:
        path: Optional explicit cache root. ``~`` is expanded, but the path is
            not created.

    Returns:
        pathlib.Path: Resolved cache root.

    Raises:
        TypeError: If ``path`` is not a string or :class:`pathlib.Path`.
        ValueError: If an explicit string path is empty.

    Examples:
        >>> from histox import data
        >>> data.resolve_cache_root("~/histox-data").name
        'histox-data'
    """

    if path is not None:
        return _clean_path(path, "path")
    histox_cache = os.environ.get("HISTOX_CACHE_DIR")
    if histox_cache:
        return _clean_path(histox_cache, "HISTOX_CACHE_DIR")
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return _clean_path(xdg_cache, "XDG_CACHE_HOME") / "histox"
    return Path.home() / ".cache" / "histox"


def _asset_status(target: Path, expected_size: Optional[int]) -> str:
    if not target.exists():
        return "missing"
    if not target.is_file():
        return "conflict"
    if expected_size is not None and target.stat().st_size != expected_size:
        return "size-mismatch"
    return "present"


def plan_download(
    dataset: Union[str, DatasetRecord],
    version: Optional[str] = None,
    path: Optional[PathLike] = None,
) -> DownloadPlan:
    """Build a read-only plan for dataset assets.

    ``dataset`` may be a built-in registry name or a provider-created
    :class:`DatasetRecord`. ``path`` is the storage root; the resolved dataset
    destination is ``<path>/<name>/<version>``. No directories are created.

    Args:
        dataset: Built-in registry name or a provider-created record.
        version: Optional version for a registry name. Cannot be combined with
            a :class:`DatasetRecord`.
        path: Optional storage root. See :func:`resolve_cache_root` for the
            default resolution order.

    Returns:
        DownloadPlan: Read-only local state and expected transfer size for all
        assets.

    Raises:
        TypeError: If ``dataset`` has an unsupported type.
        ValueError: If ``version`` is combined with a record.
        DatasetNotFoundError: If a registry name or version is unknown.

    Examples:
        >>> from histox import data
        >>> plan = data.plan_download(
        ...     "histox-download-fixture", path="histox-data"
        ... )
        >>> plan.destination.as_posix()
        'histox-data/histox-download-fixture/1.0'
        >>> len(plan.items), plan.total_size_bytes
        (1, 11357)
    """

    if isinstance(dataset, DatasetRecord):
        if version is not None:
            raise ValueError("version cannot be used with a DatasetRecord")
        record = dataset
    elif isinstance(dataset, str):
        record = get_dataset(dataset, version=version)
    else:
        raise TypeError("dataset must be a registry name or DatasetRecord")

    destination = resolve_cache_root(path) / record.name / record.version
    items = []
    for asset in record.assets:
        asset_path = PurePosixPath(asset.path)
        target = destination.joinpath(*asset_path.parts)
        items.append(
            DownloadItem(
                asset=asset,
                destination=target,
                status=_asset_status(target, asset.size_bytes),
            )
        )
    return DownloadPlan(record, destination, tuple(items))
