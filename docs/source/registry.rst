.. currentmodule:: histox.registry

histox.registry
===============

Every pluggable HistoX component (feature extractors, MIL models and trainers,
workflow tasks and methods) is looked up by name in a :class:`Registry`. All
registries share the same lookup rules, error messages, per-entry metadata,
and hints for components that live in optional extras packages.

.. code-block:: python

    import histox as hx

    hx.registry.list_registries()
    # ['extractor.tensorflow', 'extractor.torch', 'method', 'mil_model',
    #  'mil_trainer', 'task']

    extractors = hx.registry.get_registry('extractor.torch')
    'resnet50_imagenet' in extractors     # True
    extractors.names()                    # sorted list of names

The usual way to add a component is the registration function of the module
that owns the registry, for example :func:`histox.workflows.register_task` or
``histox.model.extractors.register_torch``. A :class:`Registry` can also be used
directly:

.. code-block:: python

    from histox.registry import get_registry

    extractors = get_registry('extractor.torch')

    @extractors.register('my_encoder', license='Apache-2.0')
    def my_encoder(**kwargs):
        ...

    extractors.metadata('my_encoder')     # {'license': 'Apache-2.0'}

Packages that extend HistoX usually register their components from a
``histox.plugins`` entry point, which HistoX calls on import.

The design is described in RFC 0001, proposal P1, in the repository's
``rfcs/`` folder.

Registries
----------

.. autoclass:: Registry
   :members: register, add, unregister, get, metadata, names, extras_package, extras

.. autofunction:: get_registry
.. autofunction:: list_registries

Exceptions
----------

.. autoexception:: RegistryError
.. autoexception:: UnknownEntryError
.. autoexception:: DuplicateEntryError
