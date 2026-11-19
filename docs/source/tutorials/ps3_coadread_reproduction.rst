Reproduce PS3 on TCGA COADREAD
==============================

This page records the complete five-fold, three-seed experiment used to check
the HistoX PS3 implementation. It is a reproduction protocol, not the best
place to learn the API for the first time.

Start with the :doc:`single-fold training tutorial
</auto_examples/multimodal/plot_ps3_crc>` if you want to understand the
``Dataset``, ``Experiment``, training, evaluation, and explanation objects.

Experiment definition
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Item
     - Value
   * - Cohort
     - 269 TCGA COADREAD patients
   * - Endpoint
     - Disease-specific survival (DSS)
   * - Observed events
     - 35 across the five held-out folds
   * - Evaluation
     - Five patient-level folds, repeated with seeds 1, 2, and 3
   * - Training
     - 50 epochs per model, batch size 64
   * - Inputs
     - 16 PANTHER histology prototypes, 50 Hallmark pathway prototypes, and
       10 PLIP report prototypes
   * - Summary metric
     - Mean of five held-out C-indices within each seed, followed by the mean
       and sample standard deviation across the three seed means

Required files
--------------

.. code-block:: text

   data/ps3_crc/
   |-- rna_clean.csv
   |-- hallmarks_signatures.csv
   |-- report_embeddings/{patient_id}.pt
   `-- splits/
       |-- TCGA_COADREAD_dss_survival_k=0/
       |   |-- train.csv
       |   |-- test.csv
       |   `-- embeddings/*.pkl
       |-- ...
       `-- TCGA_COADREAD_dss_survival_k=4/...

The retained PANTHER and report tensors are not distributed with HistoX.
Obtain source data under its original access terms. Pickle and PyTorch tensor
files can execute code while loading; inspect their provenance before passing
``--trust-input-files``.

Run the experiment
------------------

Download :download:`ps3_crc_train.py <../../examples/ps3_crc_train.py>`, then
run:

.. code-block:: bash

   python ps3_crc_train.py \
       --split-root data/ps3_crc/splits \
       --fold-pattern 'TCGA_COADREAD_dss_survival_k={fold}' \
       --rna data/ps3_crc/rna_clean.csv \
       --signatures data/ps3_crc/hallmarks_signatures.csv \
       --report-root data/ps3_crc/report_embeddings \
       --folds 0,1,2,3,4 \
       --seeds 1,2,3 \
       --epochs 50 \
       --batch-size 64 \
       --learning-rate 1e-4 \
       --weight-decay 1e-5 \
       --report-selection-mode crc_reproduction \
       --reference-profile coadread \
       --output runs/ps3_crc \
       --trust-input-files

``--mode`` controls the program's purpose. Its default, ``train``, runs the
full experiment above. ``overfit`` fits a small training subset to diagnose
data, loss, and gradient connections; it is not used to report performance.
The command above therefore omits ``--mode`` and uses the default.

``--report-selection-mode crc_reproduction`` preserves the report-selection
behavior used by the retained CRC implementation. For a new cancer cohort,
use ``paper`` unless the study protocol specifies another validated rule.

The recorded CRC checkpoints also use
``PS3HistologyPrototypeEncoder.unpack_panther_allcat(...,
layout="crc_reproduction")``. This preserves the direct serialized-vector
reshape used during their training. New studies should keep the method's
default ``layout="prototype_aligned"`` so each token contains one PANTHER
component's weight, mean, and covariance. Changing this layout requires
retraining; it cannot be changed only at inference time.

Output contract
---------------

.. code-block:: text

   runs/ps3_crc/
   |-- seed_1/
   |   |-- summary.json
   |   |-- fold_0/
   |   |   |-- history.csv
   |   |   |-- metrics.json
   |   |   |-- model_state.pt
   |   |   |-- train_predictions.csv
   |   |   `-- test_predictions.csv
   |   `-- fold_1/...fold_4/
   |-- seed_2/...
   |-- seed_3/...
   `-- summary.json

Recorded result
---------------

.. list-table:: Held-out DSS C-index
   :header-rows: 1
   :widths: 12 12 12 12 12 12 28

   * - Seed
     - Fold 0
     - Fold 1
     - Fold 2
     - Fold 3
     - Fold 4
     - Five-fold mean
   * - 1
     - 0.890909
     - 0.907216
     - 0.686747
     - 0.686667
     - 0.880435
     - 0.810395
   * - 2
     - 0.836364
     - 0.890034
     - 0.698795
     - 0.663333
     - 0.864130
     - 0.790531
   * - 3
     - 0.872727
     - 0.890034
     - 0.694779
     - 0.706667
     - 0.896739
     - 0.812189

The three seed-level means give ``0.804372 +/- 0.012020`` using the sample
standard deviation. This is not the score of one selected model.

For context, Raza et al. report ``0.826 +/- 0.101`` for the full PS3 model on
TCGA CRC in their own five-fold site-stratified evaluation (ICCV 2025,
Table 1). That figure is a reference point, not a matched comparison: the
original tissue segmentation, features and prototypes were unavailable, so
the inputs used here differ. The matched comparison is the retained
reproduction on identical splits and inputs (``0.817``), described in
:doc:`/methods/multimodal/ps3`.

.. image:: /_static/examples/ps3_crc_multiseed.png
   :alt: Fold-level C-index for the three HistoX PS3 seeds
   :width: 780px
   :align: center

Generate the report-phrase figure
---------------------------------

The paper-style report comparison must use one held-out prediction and one
held-out model per patient. Download :download:`ps3_crc_report_wordclouds.py
<../../examples/ps3_crc_report_wordclouds.py>` and run it after the five
seed-1 folds are complete:

.. code-block:: bash

   python -m pip install "wordcloud>=1.9,<2"

   python ps3_crc_report_wordclouds.py \
       --validation-root runs/ps3_crc/seed_1 \
       --split-root data/ps3_crc/splits \
       --rna data/ps3_crc/rna_clean.csv \
       --signatures data/ps3_crc/hallmarks_signatures.csv \
       --reports-csv data/ps3_crc/TCGA_Reports.csv \
       --report-embeddings data/ps3_crc/report_embeddings \
       --output runs/ps3_crc/seed_1/report_words \
       --trust-inputs

The script writes:

.. code-block:: text

   report_words/
   |-- ps3_report_segment_evidence.csv
   |-- ps3_report_phrase_frequencies.csv
   |-- ps3_report_wordcloud_summary.json
   `-- ps3_report_wordclouds.png

The patient-level table records the fold, predicted risk, selected report
prototype, source segment index, attention score, and source text. This makes
every displayed phrase traceable to one held-out patient and one original
report segment. The source-text CSV is a local result and is not included in
the HistoX package or documentation build.

The executed seed-1 cohort used median log-risk ``-7.5766287`` and contained
135 low-risk and 134 high-risk patients. Reloaded-model predictions matched
the saved fold predictions within ``2.42e-6`` at worst.

.. image:: /_static/examples/ps3_report_wordclouds.png
   :alt: PS3 report phrases for low- and high-risk TCGA COADREAD patients
   :width: 1050px
   :align: center

See the :doc:`single-fold tutorial
</auto_examples/multimodal/plot_ps3_crc>` for the model call, source-segment
mapping, phrase extraction, counts, and interpretation limits.

Interpretation boundary
-----------------------

The result shows that the independent HistoX implementation reaches the same
0.8 performance range as the retained CRC reproduction. It does not establish
clinical validity, external-cohort performance, or numerical identity with a
different implementation.

For another cancer type, redefine the endpoint, patient-level splits, slide
and report inclusion rules, and fold-specific preprocessing. Do not transfer
the COADREAD score or ``crc_reproduction`` setting as if they were universal
defaults.

Related pages
-------------

* :doc:`PS3 method and equations </methods/multimodal/ps3>`
* :doc:`Prepare multimodal inputs for PS3 <ps3_data_preparation>`
* :doc:`Train PS3 with the HistoX workflow
  </auto_examples/multimodal/plot_ps3_crc>`
