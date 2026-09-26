# histox.data

The `histox.data` module defines HistoX dataset metadata, cache
resolution, read-only download planning, verified direct downloads, and local
provenance.

Tip

New to this module? Start with the runnable
[data quickstart](data.html#data-quickstart) before using the reference below.

## Common workflow

The main calls are deliberately separate so a large transfer never starts
while you are only inspecting metadata:

```
import histox as hx

record = hx.data.get_dataset("histox-download-fixture")
plan = hx.data.plan_download(record, path="histox-data")

print(plan.pending_count, plan.download_size_bytes)

result = hx.data.download_dataset(record, path="histox-data")
print(result.downloaded_count, result.reused_count)
print(result.provenance_path)
```

## Registry

histox.data.list_datasets() → Tuple[DatasetRecord, ...][[source]](_modules/histox/data/_registry.html#list_datasets)

Return all built-in dataset records, ordered by name and version.

Returns:

Immutable dataset records in deterministic
name and version order.

Return type:

Tuple[DatasetRecord, ...]

Examples

```
>>> from histox import data
>>> for record in data.list_datasets():
... print(record.name, record.version)
histox-download-fixture 1.0
histox-metadata-fixture 1.0
```

histox.data.get_dataset(*name: str*, *version: str | None = None*) → DatasetRecord[[source]](_modules/histox/data/_registry.html#get_dataset)

Return one built-in dataset record.

If `version` is omitted, the registry's explicit default version is used.

Parameters:

- **name** - Case-insensitive built-in dataset name.
- **version** - Dataset version. Uses the registry default when omitted.

Returns:

Versioned provider, access, citation, task, and asset
metadata.

Return type:

DatasetRecord

Raises:

- **ValueError** - If `name` is empty.
- **DatasetNotFoundError** - If the name or selected version is not present.

Examples

```
>>> from histox import data
>>> record = data.get_dataset("HISTOX-DOWNLOAD-FIXTURE")
>>> record.version, record.access, record.assets[0].path
('1.0', 'open', 'LICENSE')
```

## Storage planning

histox.data.resolve_cache_root(*path: str | Path | None = None*) → Path[[source]](_modules/histox/data/_cache.html#resolve_cache_root)

Resolve the HistoX dataset cache root without creating it.

Resolution order is an explicit `path`, `HISTOX_CACHE_DIR`,
`XDG_CACHE_HOME/histox`, then `~/.cache/histox`.

Parameters:

**path** - Optional explicit cache root. `~` is expanded, but the path is
not created.

Returns:

Resolved cache root.

Return type:

pathlib.Path

Raises:

- **TypeError** - If `path` is not a string or `pathlib.Path`.
- **ValueError** - If an explicit string path is empty.

Examples

```
>>> from histox import data
>>> data.resolve_cache_root("~/histox-data").name
'histox-data'
```

histox.data.plan_download(*dataset: str | DatasetRecord*, *version: str | None = None*, *path: str | Path | None = None*) → DownloadPlan[[source]](_modules/histox/data/_cache.html#plan_download)

Build a read-only plan for dataset assets.

`dataset` may be a built-in registry name or a provider-created
`DatasetRecord`. `path` is the storage root; the resolved dataset
destination is `<path>/<name>/<version>`. No directories are created.

Parameters:

- **dataset** - Built-in registry name or a provider-created record.
- **version** - Optional version for a registry name. Cannot be combined with
a `DatasetRecord`.
- **path** - Optional storage root. See `resolve_cache_root()` for the
default resolution order.

Returns:

Read-only local state and expected transfer size for all
assets.

Return type:

DownloadPlan

Raises:

- **TypeError** - If `dataset` has an unsupported type.
- **ValueError** - If `version` is combined with a record.
- **DatasetNotFoundError** - If a registry name or version is unknown.

Examples

```
>>> from histox import data
>>> plan = data.plan_download(
... "histox-download-fixture", path="histox-data"
... )
>>> plan.destination.as_posix()
'histox-data/histox-download-fixture/1.0'
>>> len(plan.items), plan.total_size_bytes
(1, 11357)
```

## Download execution

histox.data.download_dataset(*dataset: str | DatasetRecord*, *version: str | None = None*, *path: str | Path | None = None*, ***, *overwrite: bool = False*, *resume: bool = True*, *timeout: float = 60.0*, *chunk_size: int = 1048576*) → DownloadResult[[source]](_modules/histox/data/_download.html#download_dataset)

Download and validate one open dataset into the HistoX cache.

Assets are streamed into sibling `.part` files and moved atomically only
after their expected size and provider checksum pass. Existing valid files
are reused. A partial file is resumed when the provider honors HTTP Range
requests; otherwise the transfer restarts safely.

This generic adapter accepts only records with `access="open"` and direct
HTTP(S) asset URLs. Controlled-access datasets require their own official
provider client and user credentials.

Parameters:

- **dataset** - Built-in registry name or a provider-created record.
- **version** - Optional version for a registry name. Cannot be combined with
a `DatasetRecord`.
- **path** - Optional storage root. See `resolve_cache_root()` for the
default resolution order.
- **overwrite** - Replace an existing asset that fails size or checksum
validation. Defaults to `False`.
- **resume** - Request the remaining bytes when a partial file exists.
Defaults to `True`; HistoX restarts safely if the provider does
not honor the Range request.
- **timeout** - Per-request timeout in seconds. Defaults to `60`.
- **chunk_size** - Streaming and checksum chunk size in bytes. Defaults to
one MiB.

Returns:

Downloaded, reused, and resumed paths plus the local
provenance path.

Return type:

DownloadResult

Raises:

- **DatasetAccessError** - If the record is not open access.
- **DatasetConflictError** - If an existing destination is unsafe or invalid
 and `overwrite` is false.
- **DatasetDownloadError** - If a direct URL is missing or the HTTP transfer
 fails.
- **DatasetIntegrityError** - If a transferred asset fails size or checksum
 validation.
- **TypeError** - If `dataset`, `timeout`, or `chunk_size` has an
 unsupported type.
- **ValueError** - If `version` conflicts with a record or a numeric option
 is not positive.

Examples

Download the built-in 11 KB smoke fixture into a local cache:

```
>>> from histox import data
>>> result = data.download_dataset(
... "histox-download-fixture", path="histox-data"
... )
>>> result.provenance_path.name
'.histox-provenance.json'
```

Calling the function again verifies and reuses the cached file rather
than transferring it again.

## Records

*class*histox.data.ProviderRecord(*name: str*, *dataset_id: str*, *url: str*)[[source]](_modules/histox/data/_models.html#ProviderRecord)

Authoritative source for a dataset.

*class*histox.data.AssetRecord(*path: str*, *url: str | None = None*, *size_bytes: int | None = None*, *checksum: str | None = None*, *checksum_algorithm: str | None = None*)[[source]](_modules/histox/data/_models.html#AssetRecord)

One provider-managed file described by a dataset record.

*class*histox.data.DatasetRecord(*name: str*, *version: str*, *title: str*, *description: str*, *provider: ProviderRecord*, *access: str*, *license: str | None*, *terms_url: str | None*, *citations: Tuple[str, ...] = ()*, *tasks: Tuple[str, ...] = ()*, *assets: Tuple[AssetRecord, ...] = ()*)[[source]](_modules/histox/data/_models.html#DatasetRecord)

Versioned metadata for a dataset known to HistoX.

*property*metadata_only*: bool*

Return whether this record intentionally contains no data assets.

*class*histox.data.DownloadItem(*asset: AssetRecord*, *destination: Path*, *status: str*)[[source]](_modules/histox/data/_models.html#DownloadItem)

Local state of one asset in a download plan.

*class*histox.data.DownloadPlan(*dataset: DatasetRecord*, *destination: Path*, *items: Tuple[DownloadItem, ...]*)[[source]](_modules/histox/data/_models.html#DownloadPlan)

Read-only description of files needed for one dataset version.

*property*download_size_bytes*: int*

Known bytes needed for missing, conflicting, or invalid assets.

*property*pending_count*: int*

Number of assets that still require provider action.

*property*present_count*: int*

Number of assets already present with the expected size.

to_dict() → Dict[str, Any][[source]](_modules/histox/data/_models.html#DownloadPlan.to_dict)

Return a JSON-serializable representation of this plan.

*property*total_size_bytes*: int*

Total known size of all assets.

*property*unknown_size_count*: int*

Number of non-present assets whose provider size is unknown.

*class*histox.data.DownloadResult(*dataset: DatasetRecord*, *destination: Path*, *provenance_path: Path*, *downloaded_files: Tuple[Path, ...]*, *reused_files: Tuple[Path, ...]*, *resumed_files: Tuple[Path, ...] = ()*)[[source]](_modules/histox/data/_models.html#DownloadResult)

Completed dataset transfer and its local provenance record.

dataset

Versioned dataset metadata used for the transfer.

Type:

histox.data._models.DatasetRecord

destination

Version-specific local dataset directory.

Type:

pathlib.Path

provenance_path

Machine-readable JSON record written after success.

Type:

pathlib.Path

downloaded_files

Assets transferred during this call.

Type:

Tuple[pathlib.Path, ...]

reused_files

Existing assets that passed validation.

Type:

Tuple[pathlib.Path, ...]

resumed_files

Transferred assets that continued from `.part` files.

Type:

Tuple[pathlib.Path, ...]

*property*downloaded_count*: int*

Number of assets transferred during this call.

*property*resumed_count*: int*

Number of transfers resumed from a partial file.

*property*reused_count*: int*

Number of existing assets that passed validation.

## Errors

*class*histox.data.DatasetDownloadError[[source]](_modules/histox/data/_download.html#DatasetDownloadError)

Base error raised while executing a dataset download.

*class*histox.data.DatasetAccessError[[source]](_modules/histox/data/_download.html#DatasetAccessError)

Raised when a dataset requires a provider-specific access workflow.

*class*histox.data.DatasetConflictError[[source]](_modules/histox/data/_download.html#DatasetConflictError)

Raised when an existing local path cannot be reused safely.

*class*histox.data.DatasetIntegrityError[[source]](_modules/histox/data/_download.html#DatasetIntegrityError)

Raised when a downloaded or existing asset fails validation.