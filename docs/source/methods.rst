Methods
========

The HistoX Methods Hub organizes pathology algorithms by research task. It is
the stable entry point for methods reproduced from published work, adaptations
maintained by HistoX, and methods developed by the HistoX team.

This page distinguishes available documentation from planned method families.
An entry is not considered implemented until its code, tests, usage example,
and provenance are present in the repository.

Browse by task
--------------

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Classification
      :link: training
      :link-type: doc

      Tile- and slide-level prediction with PyTorch-first training workflows.

      **Available**

   .. grid-item-card:: MIL & weak supervision
      :link: mil
      :link-type: doc

      Learn slide-level targets from bags of image features.

      **Available**

   .. grid-item-card:: Segmentation & detection
      :link: segmentation
      :link-type: doc

      Tissue segmentation, cell analysis, and spatial target detection.

      **Available**

   .. grid-item-card:: Normalization & quality control
      :link: norm
      :link-type: doc

      Standardize stain appearance and identify unsuitable slide regions.

      **Available**

   .. grid-item-card:: Generative methods
      :link: gan
      :link-type: doc

      Generative modeling and representation-learning workflows.

      **Available**

   .. grid-item-card:: Explainability & uncertainty
      :link: heatmap
      :link-type: doc

      Heatmaps, saliency, gradients, and uncertainty analysis.

      **Available**

.. _methods-foundation:

Foundation models and VLM
-------------------------

This family is reserved for pathology foundation encoders, zero-shot
classification, image-text retrieval, visual question answering, and other
vision-language workflows. Implementations will appear here only after their
weights, preprocessing contract, license, and evaluation example are verified.

**Status:** Planned catalog area. No unified HistoX foundation-model API is
claimed on this page yet.

.. _methods-survival:

Survival and prognosis
----------------------

This family will cover time-to-event prediction, censoring-aware objectives,
risk stratification, and multimodal prognosis. Each implementation must state
its endpoint definition, censoring assumptions, split unit, and reported
metrics.

**Status:** Planned catalog area. Existing general training utilities may be
used experimentally, but a stable survival interface is not documented yet.

.. _methods-multimodal:

Multimodal and spatial methods
------------------------------

This family is reserved for methods that combine histology with text, clinical
variables, genomics, spatial transcriptomics, or other molecular measurements.
Entries must document modality alignment, missing-data behavior, and cohort
requirements.

**Status:** Planned catalog area. Methods will be added as tested workflows
rather than as a list of unsupported paper names.

Method records
--------------

Every method added to HistoX should expose the same minimum record:

.. list-table:: Required method metadata
   :header-rows: 1
   :widths: 24 76

   * - Field
     - What it records
   * - Origin
     - ``reproduced``, ``adapted``, or ``histox-original``.
   * - Maturity
     - ``experimental``, ``beta``, or ``stable``.
   * - Provenance
     - Paper citation, upstream code, license, and material deviations.
   * - Inputs and outputs
     - Accepted slide, tile, feature, label, and metadata formats.
   * - Evidence
     - Tests, example configuration, supported datasets, and known limits.

Adding a method
---------------

Before a method is listed as available, its contribution should include:

#. a focused implementation under the appropriate HistoX module;
#. a public configuration or constructor with documented defaults;
#. unit tests plus a small executable example;
#. a method page stating provenance, expected inputs, and limitations; and
#. an entry in this hub that links to the implementation and example.

See :doc:`developer` for repository and contribution guidance, or browse the
:doc:`api` for the interfaces that are currently part of HistoX.
