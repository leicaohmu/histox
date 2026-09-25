.. currentmodule:: histox.data

.. _data-api:

histox.data
===========

The :mod:`histox.data` module defines HistoX dataset metadata, cache
resolution, read-only download planning, verified direct downloads, and local
provenance.

.. tip::

   New to this module? Start with the runnable
   :ref:`data quickstart <data-quickstart>` before using the reference below.

Common workflow
---------------

The main calls are deliberately separate so a large transfer never starts
while you are only inspecting metadata:

.. code-block:: python

    import histox as hx

    record = hx.data.get_dataset("histox-download-fixture")
    plan = hx.data.plan_download(record, path="histox-data")

    print(plan.pending_count, plan.download_size_bytes)

    result = hx.data.download_dataset(record, path="histox-data")
    print(result.downloaded_count, result.reused_count)
    print(result.provenance_path)

Registry
--------

.. autofunction:: list_datasets

.. autofunction:: get_dataset

Storage planning
----------------

.. autofunction:: resolve_cache_root

.. autofunction:: plan_download

Download execution
------------------

.. autofunction:: download_dataset

Records
-------

.. autoclass:: ProviderRecord
   :members:

.. autoclass:: AssetRecord
   :members:

.. autoclass:: DatasetRecord
   :members:

.. autoclass:: DownloadItem
   :members:

.. autoclass:: DownloadPlan
   :members:

.. autoclass:: DownloadResult
   :members:

Errors
------

.. autoclass:: DatasetDownloadError

.. autoclass:: DatasetAccessError

.. autoclass:: DatasetConflictError

.. autoclass:: DatasetIntegrityError
