.. currentmodule:: histox.workflows

histox.workflows
================

Composable experiment workflows: a :class:`Task` describes what to predict, a
:class:`Method` describes how, and an :class:`Experiment` runs and records the
lifecycle stages. See :doc:`workflows` for a guided introduction.

Core types
----------

.. autoclass:: DataSpec
   :members:

.. autoclass:: Task
   :members:

.. autoclass:: SurvivalTask
   :members:

.. autoclass:: ClassificationTask
   :members:

.. autoclass:: RegressionTask
   :members:

.. autoclass:: Method
   :members:

.. autoclass:: Experiment
   :members:

.. autoclass:: RunResult
   :members:

.. autoclass:: Artifact
   :members:

Registries
----------

.. autofunction:: register_task
.. autofunction:: register_method
.. autofunction:: build_task
.. autofunction:: build_method
.. autofunction:: list_tasks
.. autofunction:: list_methods

Engines and pretrained bundles
------------------------------

.. autoclass:: histox.workflows.engines.SurvivalEngine
   :members:

.. autoclass:: PS3PretrainedBundle
   :members:

Methods
-------

.. autoclass:: histox.workflows.MILMethod
   :members: trainer_config, build_model, stage_fit, stage_evaluate

.. autoclass:: histox.workflows.TileClassifierMethod
   :members: project, model_params, build_model, stage_fit, stage_evaluate

Exceptions
----------

.. autoexception:: WorkflowError
.. autoexception:: RegistryError
.. autoexception:: DuplicateRegistrationError
.. autoexception:: UnknownTaskError
.. autoexception:: UnknownMethodError
.. autoexception:: IncompatibleWorkflowError
.. autoexception:: InvalidDataSpecError
.. autoexception:: UnsupportedStageError

.. minigallery:: histox.workflows
   :add-heading: Examples using ``histox.workflows``
