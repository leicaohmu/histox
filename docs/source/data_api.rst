.. currentmodule:: histox.data

.. _data-api:

histox.data
===========

The :mod:`histox.data` module defines HistoX dataset metadata, cache
resolution, and read-only download planning. See :doc:`data` for concepts and
usage examples.

Registry
--------

.. autofunction:: list_datasets

.. autofunction:: get_dataset

Storage planning
----------------

.. autofunction:: resolve_cache_root

.. autofunction:: plan_download

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
