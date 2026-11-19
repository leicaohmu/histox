Unified workflows
=================

HistoX separates an experiment into four public objects:

``DataSpec``
   Declares the modalities and semantic fields available in a cohort.

``Task``
   Defines the prediction target, required fields, and default metrics.

``Method``
   Adapts one algorithm to one or more tasks. It declares required
   modalities, constructs its model, and implements the lifecycle stages it
   supports.

``Experiment``
   Validates task and method compatibility, constructs the model, runs
   lifecycle stages, and retains their ``RunResult`` objects.

This division keeps cohort preparation, algorithm code, and result files
separate. A new method does not need to copy the project layout or invent a
new result format.

Build PS3 through the common interface
--------------------------------------

The PS3 adapter constructs the existing :class:`~histox.mil.models.PS3SurvivalModel`;
it does not contain a second implementation of the model.

.. code-block:: python

   import histox as hx

   data_spec = hx.DataSpec(
       modalities=("histology", "report", "rna"),
       fields={
           "patient_id": "case_id",
           "duration": "dss_survival_days",
           "event": "dss_event",
       },
   )

   experiment = hx.Experiment(
       task="survival",
       method="ps3",
       data_spec=data_spec,
       output="outputs/ps3",
       method_config={
           "n_genes": 4241,
           "pathway_gene_indices": pathway_gene_indices,
           "histology_dim": 1025,
           "report_dim": 512,
           "n_histology_prototypes": 16,
           "n_report_prototypes": 10,
           "report_selection_mode": "paper",
       },
   )

   model = experiment.build_model()

Validation happens before model construction. Missing report data, missing
survival fields, or an incompatible task therefore raises a focused error
before HistoX loads a large model.

Train, evaluate, and explain PS3
--------------------------------

The PS3 method delegates reusable optimization and evaluation to the shared
survival engine. The loader uses the batch keys ``patient_id``, ``histology``,
``rna``, ``report``, ``report_mask``, ``duration``, and ``event``.

.. code-block:: python

   fitted = experiment.fit(
       train_loader=train_loader,
       epochs=50,
       evaluation_loaders={
           "train": train_eval_loader,
           "test": test_loader,
       },
       learning_rate=1e-4,
       weight_decay=1e-5,
       scheduler="cosine",
   )

   print(fitted.metrics["test_c_index"])
   print(fitted.artifacts["checkpoint"].path)

   evaluated = experiment.evaluate(loader=test_loader)
   predicted = experiment.predict(loader=test_loader)
   explained = experiment.explain(loader=test_loader)

``explain`` writes ``joint_attention.npz``. It contains the joint prototype
attention used by PS3 together with patient identifiers and log-risk. When a
pretrained bundle includes the fold-specific PANTHER centers, use
``bundle.spatial_attention(...)`` with the original patch features and their
aligned coordinates to project histology-prototype attention back onto the
WSI. ``bundle.save_spatial_heatmap(...)`` renders the result over the slide.

Lifecycle stages
----------------

The stable stage names are:

.. code-block:: python

   experiment.prepare()
   experiment.fit()
   experiment.evaluate()
   experiment.predict()
   experiment.explain()

A method implements only the stages it supports. For example, an embedding
extractor may implement ``prepare`` and ``predict`` without ``fit``; a
retrieval method may build its index during ``prepare``. Calling an
unsupported stage raises ``UnsupportedStageError``.

The PS3 adapter implements ``fit``, ``evaluate``, ``predict``, and ``explain``
through the shared survival engine. Cohort-specific file preparation remains
an explicit step because its slide, report, RNA, and outcome sources differ
between studies. Other methods may implement only the stages they support.

Register a custom method
------------------------

Custom methods use the same registry as built-in workflow adapters. Optional
framework imports should stay inside ``build_model`` so importing HistoX
remains lightweight.

.. code-block:: python

   @hx.register_method("my_survival_model")
   class MySurvivalMethod(hx.Method):
       name = "my_survival_model"
       supported_tasks = ("survival",)
       required_modalities = ("histology",)

       def build_model(self, data_spec):
           from my_package import MyModel
           return MyModel(**self.config)

       def stage_predict(self, experiment, batches):
           model = experiment.build_model()
           predictions = [model(batch) for batch in batches]
           return hx.RunResult(
               stage="predict",
               predictions=predictions,
           )

   experiment = hx.Experiment(
       task="survival",
       method="my_survival_model",
       data_spec=hx.DataSpec(
           modalities=("histology",),
           fields={"duration": "os_days", "event": "os_event"},
       ),
       method_config={"hidden_dim": 256},
   )

Tasks are extensible in the same way with ``@hx.register_task``. This allows
retrieval and question-answering tasks to define different required fields and
metrics without changing ``Experiment``.

Results and visual evidence
---------------------------

Every lifecycle stage returns a ``RunResult``. Metrics, predictions, and
typed artifacts stay together:

.. code-block:: python

   result = hx.RunResult(
       stage="evaluate",
       metrics={"c_index": 0.81},
   )

   result.add_artifact(
       "patient_risk",
       kind="heatmap",
       path="outputs/patient_risk.png",
       metadata={"slide": "patient.svs"},
   )

   result.add_artifact(
       "lymphocyte_concept",
       kind="concept_heatmap",
       path="outputs/lymphocyte_concept.png",
   )

   result.save_json("outputs/result.json")

Artifact kinds are open to extensions. Common values include ``checkpoint``,
``predictions``, ``table``, ``figure``, ``heatmap``, ``concept_heatmap``, and
``retrieved_patches``. The method computes model-specific evidence; shared
renderers and Studio can consume the resulting artifacts.

Registry inspection
-------------------

.. code-block:: python

   print(hx.list_tasks())
   print(hx.list_methods())

Registrations are protected from accidental replacement. Reusing an existing
name raises ``DuplicateRegistrationError`` unless the extension explicitly
requests replacement.

API reference
-------------

.. autoclass:: histox.workflows.DataSpec
   :members:

.. autoclass:: histox.workflows.Task
   :members:

.. autoclass:: histox.workflows.Method
   :members:

.. autoclass:: histox.workflows.Experiment
   :members:

.. autoclass:: histox.workflows.RunResult
   :members:

.. autoclass:: histox.workflows.Artifact
   :members:

.. autoclass:: histox.workflows.engines.SurvivalEngine
   :members:
