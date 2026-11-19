Prepare multimodal inputs for PS3
=================================

This guide turns three patient-level source modalities into the tensors used
by the :doc:`PS3 training tutorial
</auto_examples/multimodal/plot_ps3_crc>`. It stops before model training.

Use this page when you have legally obtained whole-slide images,
machine-readable de-identified pathology reports, gene-expression data, and
survival outcomes. If the three prepared feature types already exist, go
directly to the training tutorial.

Inputs and outputs
------------------

One manifest row identifies one patient and the files belonging to that
patient:

.. code-block:: text

   patient_id,slide_path,report_path
   PATIENT-001,slides/PATIENT-001.svs,reports/PATIENT-001.txt
   PATIENT-002,slides/PATIENT-002.svs,reports/PATIENT-002.txt

The preparation program produces three feature directories. RNA is transformed
inside each training fold by the training program so that validation patients
never influence gene scaling.

.. code-block:: text

   ps3_inputs/
   |-- histology_bags/
   |   |-- PATIENT-001.pt
   |   `-- PATIENT-002.pt
   |-- histology_coordinates/
   |   |-- PATIENT-001.pt
   |   `-- PATIENT-002.pt
   `-- report_embeddings/
       |-- PATIENT-001.pt
       `-- PATIENT-002.pt

   data/
   |-- rna_clean.csv
   |-- hallmarks_signatures.csv
   `-- splits/
       `-- MY_COHORT_survival_k=0/
           |-- train.csv
           `-- test.csv

Every table and tensor must use the same patient identifiers. Split files are
patient-level, not slide-level.

Install the required components
-------------------------------

.. code-block:: bash

   python -m pip install -e ".[torch,huggingface]"

PLIP weights are obtained at runtime and are not distributed with HistoX. Pin
a reviewed model revision when preparing data for a study that must be
reproducible.

Prepare WSI patch features
--------------------------

HistoX reads each WSI, removes background, divides retained tissue into
non-overlapping 256-pixel patches at 0.5 microns per pixel, and encodes those
patches with PLIP. The saved feature tensor contains one 512-dimensional row
per retained tissue patch. A second tensor stores aligned tile coordinates as
``[x, y, grid_x, grid_y]``. Row ``i`` in the feature tensor and row ``i`` in
the coordinate tensor always refer to the same WSI patch. PS3 uses the
features for prediction and the coordinates to place patch scores back onto
the slide.

The reusable preparation program performs that process for every manifest
row:

.. code-block:: bash

   python ps3_crc_prepare_inputs.py \
       --manifest data/manifest.csv \
       --rna data/rna_clean.csv \
       --split-root data/splits \
       --fold-pattern 'MY_COHORT_survival_k={fold}' \
       --output ps3_inputs

Download the program before running it:

* :download:`ps3_crc_prepare_inputs.py
  <../../examples/ps3_crc_prepare_inputs.py>`

The filename records the cohort in which the program was validated. Its
manifest and split arguments can point to another cancer cohort.

PANTHER tissue prototypes are **not** fitted at this stage. During training,
HistoX learns the 16 tissue centers from the current fold's training patch
bags and applies those fixed centers to both training and held-out patients.

Verify one prepared slide
~~~~~~~~~~~~~~~~~~~~~~~~~

Load the paired files before starting a long training run:

.. code-block:: python

   import torch

   patch_features = torch.load(
       "ps3_inputs/histology_bags/PATIENT-001.pt",
       map_location="cpu",
       weights_only=True,
   )
   coordinates = torch.load(
       "ps3_inputs/histology_coordinates/PATIENT-001.pt",
       map_location="cpu",
       weights_only=True,
   )

   assert patch_features.ndim == 2
   assert patch_features.shape[1] == 512
   assert coordinates.shape == (len(patch_features), 4)
   print(patch_features.shape, coordinates[:3])

Do not recreate coordinates later from a different tiling configuration.
Changing ``tile_px``, ``tile_um``, quality control, ROI filtering, or stride
changes which feature row belongs at each location.

Prepare report embeddings
-------------------------

The report input is de-identified text. PDF parsing, OCR, and
de-identification must already be complete. HistoX divides a report into
consecutive PLIP-compatible segments without silently truncating it and saves
one 512-dimensional embedding per segment.

For one report, the underlying API is:

.. code-block:: python

   from histox.mil import PS3ReportEncoder

   encoder = PS3ReportEncoder(
       model_id="vinid/plip",
       revision=None,  # Replace with a reviewed revision for a real study.
   )
   segments = encoder.encode(report_text)
   print(segments.shape)  # [number of report segments, 512]

The preparation program applies the same operation to every manifest row and
writes ``report_embeddings/{patient_id}.pt``.

Prepare RNA and pathway membership
----------------------------------

``rna_clean.csv`` contains one patient per row and one gene symbol per column.
``hallmarks_signatures.csv`` contains one pathway per column and its member
gene symbols in the non-empty rows.

Gene standardization is fitted separately inside each training fold:

.. code-block:: python

   import pandas as pd
   from histox.mil import PS3PathwayPreprocessor

   rna = pd.read_csv("data/rna_clean.csv", index_col=0)
   signatures = pd.read_csv("data/hallmarks_signatures.csv")

   pathway = PS3PathwayPreprocessor().fit(
       rna.loc[train_patient_ids],
       signatures,
   )
   train_rna = pathway.transform(rna.loc[train_patient_ids])
   valid_rna = pathway.transform(rna.loc[valid_patient_ids])

Do not fit a second scaler on validation or test patients.

Define the survival endpoint
----------------------------

Every split CSV needs a patient identifier, an observed follow-up duration,
and an event indicator. For example:

.. code-block:: text

   case_id,survival_days,event
   PATIENT-001,531,1
   PATIENT-002,842,0

``event=1`` means that the endpoint was observed; ``event=0`` means that the
observation was censored. If a source table stores censorship instead, convert
it explicitly with ``event = 1 - censorship`` and document that definition.

Check the prepared cohort
-------------------------

Before training, verify all four conditions:

#. training and validation patient identifiers do not overlap;
#. every split patient has RNA, histology, and report inputs;
#. every report tensor has shape ``[segments, 512]``;
#. every training batch can contain at least one observed event for the Cox
   loss.

The training tutorial performs these checks again before constructing the
model.

Next step
---------

Continue with :doc:`Train PS3 with the HistoX workflow
</auto_examples/multimodal/plot_ps3_crc>`. For the exact COADREAD multi-seed
experiment, use :doc:`Reproduce PS3 on TCGA COADREAD
<ps3_coadread_reproduction>`.
