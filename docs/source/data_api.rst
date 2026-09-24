.. currentmodule:: histox.data

.. _data-api:

histox.data
===========

The :mod:`histox.data` module defines HistoX dataset metadata, cache
resolution, read-only download planning, verified direct downloads, and local
provenance. See :doc:`data` for concepts and usage examples.

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
