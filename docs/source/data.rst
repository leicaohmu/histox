Dataset registry, downloads, and provenance
===========================================

The :mod:`histox.data` module provides a registry, a read-only download
planner, and an explicit downloader for open assets with direct HTTP(S) URLs.
It describes authoritative providers, access terms, citations, supported
tasks, assets, checksums, and expected sizes before any network transfer.

Listing datasets
----------------

.. code-block:: python

    import histox as hx

    for record in hx.data.list_datasets():
        print(record.name, record.version, record.provider.name)

The first registry entry is deliberately metadata-only. It validates the
registry and planning contract without bundling pathology data in the package.

Planning local storage
----------------------

.. code-block:: python

    plan = hx.data.plan_download(
        "histox-metadata-fixture",
        path="/shared/pathology-data",
    )

    print(plan.destination)
    print(plan.pending_count, plan.download_size_bytes)

The ``path`` argument is a storage root. Dataset files are planned beneath
``<path>/<dataset name>/<version>``. Planning does not create directories.

When ``path`` is omitted, HistoX resolves the cache root in this order:

1. ``HISTOX_CACHE_DIR``;
2. ``XDG_CACHE_HOME/histox``;
3. ``~/.cache/histox``.

Downloading open assets
-----------------------

Review the plan before starting a large transfer, then explicitly download the
same record:

.. code-block:: python

    record = hx.data.get_dataset("histox-download-fixture")
    plan = hx.data.plan_download(record, path="/shared/pathology-data")

    # Start network transfer only after reviewing plan.to_dict().
    result = hx.data.download_dataset(record, path="/shared/pathology-data")
    print(result.downloaded_count, result.reused_count)
    print(result.provenance_path)

The built-in ``histox-download-fixture`` contains only HistoX's 11 KB license
file. It provides a reproducible smoke test without transferring pathology
data. Real datasets use the same API and store their assets outside the Python
package. The downloader streams each asset into a sibling ``.part`` file. It
resumes a partial file only when the provider honors HTTP Range requests,
validates the expected size and checksum, and atomically publishes the final
file. A ``.histox-provenance.json`` record is written after the complete
dataset succeeds.

Existing valid files are reused. A conflicting or invalid destination raises
:class:`~histox.data.DatasetConflictError`; pass ``overwrite=True`` only when
replacement is intentional.

Access boundary
---------------

The generic downloader accepts only ``access="open"`` records with direct
HTTP(S) asset URLs. It does not accept credentials or bypass provider access
controls. Controlled datasets, including authenticated GDC workflows, require
an official provider client and the user's own account. Existing
:func:`histox.create_project` TCGA behavior is unchanged and is not yet bridged
to this API.

Read-only planning remains fast: local asset status is based on file presence
and expected byte size. Checksums are enforced when
:func:`~histox.data.download_dataset` validates an existing or newly downloaded
asset.

API reference
-------------

See :doc:`data_api` for the complete :mod:`histox.data` API reference.
