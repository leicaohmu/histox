# Data quickstart

HistoX keeps dataset metadata in the Python package while the actual files stay
in storage controlled by you. This guide walks through the complete public-data
workflow: discover a record, review the transfer, download a small fixture,
inspect provenance, and reuse the verified cache.

The example downloads HistoX's 11 KB license file, not pathology data. You can
therefore run it before configuring WSI storage.

Note

This page documents the current development version. Until the next package
release includes A4b2, install the repository checkout in editable mode to
run these examples: `python -m pip install -e .`

[`Download the complete Python example`](_downloads/87b3cc011d46b65a5243debfc8556ce9/data_quickstart.py) or follow the steps below interactively.

## What you will learn

- how to inspect provider, access, license, and asset metadata;
- how to see the destination and expected transfer size before networking;
- how to download with checksum verification and atomic file publication;
- how to find the machine-readable provenance record;
- how HistoX reuses an existing valid asset.

## 1. Discover available datasets

Start by listing the versioned records shipped with HistoX:

```
import histox as hx

for record in hx.data.list_datasets():
 print(record.name, record.version, record.access)
```

Expected output:

```
histox-download-fixture 1.0 open
histox-metadata-fixture 1.0 open
```

Load one record and inspect its source and terms before downloading:

```
record = hx.data.get_dataset("histox-download-fixture")

print(record.title)
print(record.provider.name, record.provider.dataset_id)
print(record.license, record.terms_url)
print(len(record.assets), record.assets[0].path)
```

Expected output begins with:

```
HistoX direct-download smoke fixture
HistoX histox:download-fixture:25c06216
Apache-2.0 ...
1 LICENSE
```

The other built-in record, `histox-metadata-fixture`, deliberately has no
assets. It exercises discovery and planning without creating files or using the
network.

## 2. Review the download plan

Planning is read-only. It does not create the cache directory or start a
network request.

```
from pathlib import Path

cache_root = Path("histox-data")
plan = hx.data.plan_download(record, path=cache_root)

print("destination:", plan.destination)
print("pending assets:", plan.pending_count)
print("known download bytes:", plan.download_size_bytes)
```

For a fresh cache, the output is:

```
destination: histox-data/histox-download-fixture/1.0
pending assets: 1
known download bytes: 11357
```

For programmatic inspection or a command-line preview, the complete plan is
JSON serializable:

```
import json

print(json.dumps(plan.to_dict(), indent=2))
```

The `path` argument is a storage root. HistoX places assets beneath
`<path>/<dataset name>/<version>`. When `path` is omitted, the cache root
is resolved in this order:

1. `HISTOX_CACHE_DIR`;
2. `XDG_CACHE_HOME/histox`;
3. `~/.cache/histox`.

## 3. Download and verify

Downloading is a separate, explicit call:

```
result = hx.data.download_dataset(record, path=cache_root)

print("downloaded:", result.downloaded_count)
print("reused:", result.reused_count)
print("provenance:", result.provenance_path)
```

For the first run:

```
downloaded: 1
reused: 0
provenance: histox-data/histox-download-fixture/1.0/.histox-provenance.json
```

Each asset is streamed into a sibling `.part` file. HistoX resumes that file
only when the provider honors HTTP Range requests, verifies its expected size
and checksum, and then atomically moves it to the final path. A failed checksum
does not publish the asset or write a completed provenance record.

## 4. Inspect provenance

The provenance JSON records the dataset version, provider, source URL, access
and license metadata, expected integrity values, validation level, and whether
each asset was downloaded or reused.

```
provenance = json.loads(result.provenance_path.read_text())
asset = provenance["assets"][0]

print(provenance["dataset"], provenance["version"])
print(asset["path"], asset["validation"], asset["action"])
```

Expected output after the first run:

```
histox-download-fixture 1.0
LICENSE checksum downloaded
```

Keep this file with downstream experiment records. It identifies what HistoX
validated locally; it does not replace the provider's citation or terms.

## 5. Reuse the verified cache

Call the same function again to verify and reuse the existing file without a
second network transfer:

```
second = hx.data.download_dataset(record, path=cache_root)
print("downloaded:", second.downloaded_count)
print("reused:", second.reused_count)
```

Expected output:

```
downloaded: 0
reused: 1
```

If an existing destination has the wrong size or checksum, HistoX raises
[`DatasetConflictError`](data_api.html#histox.data.DatasetConflictError). Use `overwrite=True` only when
replacement is intentional.

## Access-controlled datasets

The generic downloader accepts only `access="open"` records with direct
HTTP(S) asset URLs. It does not accept credentials or bypass access controls.
Controlled datasets, including authenticated GDC workflows, require an
official provider client and the user's own account.

The inherited [`histox.create_project()`](histox.html#histox.create_project) TCGA download path is unchanged and
is not yet connected to `histox.data`.

## API reference

See [histox.data](data_api.html) for parameters, return types, exceptions, record classes,
and shorter examples for each public function.