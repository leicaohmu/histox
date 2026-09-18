Dataset registry and storage planning
=====================================

The :mod:`histox.data` module provides an offline registry and a read-only
download planner. It describes authoritative providers, access terms,
citations, supported tasks, assets, checksums, and expected sizes without
starting a network transfer.

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

Provider downloads are not part of A4b1. A later provider adapter will execute
plans, handle authentication and resumption, enforce checksums, and write
provenance records. Existing :func:`histox.create_project` TCGA behavior is
unchanged.

In A4b1, local asset status is based on file presence and expected byte size.
Stored checksums are descriptive metadata; checksum execution is added with the
provider download layer.

API
---

.. autofunction:: histox.data.list_datasets

.. autofunction:: histox.data.get_dataset

.. autofunction:: histox.data.resolve_cache_root

.. autofunction:: histox.data.plan_download

.. autoclass:: histox.data.ProviderRecord
   :members:

.. autoclass:: histox.data.AssetRecord
   :members:

.. autoclass:: histox.data.DatasetRecord
   :members:

.. autoclass:: histox.data.DownloadItem
   :members:

.. autoclass:: histox.data.DownloadPlan
   :members:
