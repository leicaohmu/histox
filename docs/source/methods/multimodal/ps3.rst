PS3 multimodal survival
=======================

PS3 (Predicting Survival from Three Modalities) combines histological
prototypes, biological-pathway prototypes, and diagnostic prototypes derived
from pathology reports. HistoX provides an independent implementation of the
three input pipelines and the fusion equations described in the ICCV 2025
paper: ``WSI -> PLIP -> PANTHER``, ``RNA -> Hallmark pathways``, and
``report text -> PLIP segments``.

.. warning::

   This interface is **experimental and intended for research**. It is not a
   clinical model, does not ship pretrained weights, and is not claimed to be
   checkpoint-compatible with the authors' code. The current HistoX
   implementation has been evaluated on CRC inputs from one paper cohort;
   the other five paper cohorts and external clinical validation remain
   untested.

Method record
-------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Record
   * - Origin
     - Independent implementation from the published PS3 equations; no
       upstream PS3 source was copied.
   * - Maturity
     - ``experimental``
   * - Task
     - Patient-level disease-specific survival risk prediction.
   * - Inputs
     - Histological prototypes, gene-expression vectors plus pathway gene
       membership, and embedded pathology-report segments.
   * - Output
     - One unconstrained log-risk score per patient.
   * - Evidence
     - Shape, masking, attention, gradient, and Cox-loss unit tests, plus 15
       independently trained CRC models across three seeds and five folds.
   * - Missing from this release
     - Pretrained PS3 survival weights, a distributed prepared cohort, and
       external-cohort validation.

Install
-------

PS3 uses PyTorch, scikit-learn, and Hugging Face Transformers. Install the
PyTorch and Hugging Face extras:

.. code-block:: bash

   python -m pip install -e ".[torch,huggingface]"

PLIP weights are downloaded at runtime and are not distributed by HistoX.
The current upstream model repository may serve ``pytorch_model.bin``. Recent
Transformers versions require PyTorch 2.6 or newer to load that format safely;
alternatively pass a trusted local ``safetensors`` snapshot as ``model_id``.

Choose the right guide
----------------------

The method page explains the model and tensor contract. Continue with the page
that matches your current starting point:

* :doc:`Prepare multimodal inputs for PS3
  </tutorials/ps3_data_preparation>` starts with WSI files, report text, RNA,
  and patient-level outcomes.
* :doc:`Train PS3 with the HistoX workflow
  </auto_examples/multimodal/plot_ps3_crc>` trains and evaluates one real
  patient-level fold from prepared multimodal features.
* :doc:`Reproduce PS3 on TCGA COADREAD
  </tutorials/ps3_coadread_reproduction>` runs the complete five-fold,
  three-seed experiment and reports the measured results.

What the model does
-------------------

The implementation follows five explicit stages:

#. :class:`~histox.mil.models.DiagnosticPrototypeSelector` applies masked
   self-attention to report-segment embeddings, ranks valid segments by mean
   incoming attention, and retains a fixed number of diagnostic prototypes.
#. :class:`~histox.mil.models.PathwayPrototypeEncoder` selects genes belonging
   to each pathway and applies an independent self-normalizing MLP.
#. :class:`~histox.mil.PS3HistologyPrototypeEncoder` fits training-fold tissue
   centers and constructs compact PANTHER-style mixture representations from
   WSI patch features produced by the native ``plip`` extractor.
#. :class:`~histox.mil.models.PS3PrototypeFusion` projects all three modalities
   to one dimension, appends learned modality embeddings, and computes joint
   self- and cross-modal attention.
#. Independent post-fusion MLPs transform prototype positions. The model
   averages tokens within each modality, concatenates the three summaries, and
   predicts a patient-level log-risk score.

Core equations, step by step
----------------------------

This section rewrites the paper's equations in the order that data travels
through HistoX. Superscripts ``t``, ``h``, and ``p`` mean text, histology, and
pathway. ``N_T``, ``N_H``, and ``N_P`` are their prototype counts, ``d`` is a
feature dimension, and :math:`\sigma` denotes a row-wise softmax unless stated
otherwise.

Diagnostic prototypes from reports (paper equations 1--3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let :math:`\widetilde{\mathbf{H}}^t \in
\mathbb{R}^{b\times m\times d_t}` contain the padded embeddings of ``b``
reports, each with at most ``m`` segments. Three learned projections produce
queries, keys, and values:

.. math::

   \mathbf{Q}^t = \mathbf{W}_Q\widetilde{\mathbf{H}}^t,\qquad
   \mathbf{K}^t = \mathbf{W}_K\widetilde{\mathbf{H}}^t,\qquad
   \mathbf{V}^t = \mathbf{W}_V\widetilde{\mathbf{H}}^t. \tag{1}

Each report segment then asks how strongly it should attend to every other
segment:

.. math::

   \mathbf{A}^t = \operatorname{softmax}\!\left(
   \frac{\mathbf{Q}^t(\mathbf{K}^t)^\top}{\sqrt{d_t}} + \mathbf{M}
   \right),\qquad \mathbf{Z}^t = \mathbf{A}^t\mathbf{V}^t. \tag{2}

Here :math:`\mathbf{M}` is ``0`` at valid keys and :math:`-\infty` at padded
keys. Therefore padding receives zero attention after the softmax. The
importance of valid segment ``j`` in report ``n`` is the mean attention it
receives from all valid query segments:

.. math::

   s^t_{n,j} = \frac{1}{m_n}
   \sum_{i:\,M_{n,i}=1} A^t_{n,i,j},\qquad
   \mathbf{Z}^t_{n,\mathrm{proto}} =
   \operatorname{TopK}_{j}\!\left(\mathbf{Z}^t_n, s^t_{n,j}, N_T\right).
   \tag{3}

In plain language, a segment scores highly when many other report segments
attend to it. HistoX selects post-attention vectors, not the original text
vectors. :class:`~histox.mil.models.DiagnosticPrototypeSelector` exposes two
explicit behaviors because the paper does not define the case
:math:`m_n<N_T`:

* ``mode="paper"`` divides by the number of real query segments and linearly
  resizes the valid post-attention sequence only when the report is shorter
  than ``N_T``. Padding cannot change its result.
* ``mode="crc_reproduction"`` matches the released CRC run: reports are
  padded to the training-fold maximum, attention is averaged over every padded
  query position, and top-k may select the post-attention output of a padded
  query. The verified CRC configuration used ``N_T=10`` and maximum length 32
  in fold 0 and 42 in the other recorded folds.

The second mode is retained for numerical reproduction. The first mode is the
recommended default for a new study because its output does not depend on
irrelevant padding length.

Histological prototypes from a WSI (paper equations 4--7)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a slide with patch embeddings :math:`\mathbf{h}^h_j`, the paper models the
patch distribution with ``N_H`` Gaussian components. Component ``c`` has
weight :math:`\pi_c`, mean :math:`\boldsymbol{\mu}_c`, and diagonal covariance
:math:`\boldsymbol{\Sigma}_c`:

.. math::

   \boldsymbol{\mu}_c \sim \mathcal{N}(0, 0.1^2\mathbf{I}),\qquad
   \boldsymbol{\Sigma}_c=\mathbf{I},\qquad
   \pi_c=\frac{1}{N_H}. \tag{4}

The likelihood of one patch is:

.. math::

   p(\mathbf{h}^h_j)=\sum_{c=1}^{N_H}\pi_c
   \mathcal{N}(\mathbf{h}^h_j;\boldsymbol{\mu}_c,\boldsymbol{\Sigma}_c).
   \tag{5}

Assuming patches are conditionally independent under the mixture, the slide
likelihood becomes:

.. math::

   p(\mathbf{H}^h)=\prod_{j=1}^{N_h}
   \sum_{c=1}^{N_H}\pi_c
   \mathcal{N}(\mathbf{h}^h_j;\boldsymbol{\mu}_c,\boldsymbol{\Sigma}_c).
   \tag{6}

After expectation-maximization, each Gaussian component becomes one compact
histological prototype:

.. math::

   \mathbf{Z}^h_{\mathrm{proto}} =
   \left[(\pi_c,\boldsymbol{\mu}_c,\boldsymbol{\Sigma}_c)
   \right]_{c=1}^{N_H}
   \in \mathbb{R}^{N_H\times(1+2d_h)}. \tag{7}

The weight describes how common a morphology is, the mean describes its
typical appearance, and the diagonal covariance describes within-pattern
variation. :class:`~histox.mil.PS3HistologyPrototypeEncoder` performs this
construction. Fit its 16 centers on training-fold patch bags only, then call
``transform`` for every training, validation, and test slide. With 512-wide
PLIP features the output shape is ``[16, 1025]``.

When reading retained ``allcat`` vectors,
:meth:`~histox.mil.PS3HistologyPrototypeEncoder.unpack_panther_allcat` defaults
to ``layout="prototype_aligned"`` and reconstructs equation 7 explicitly.
The recorded CRC checkpoints predate that helper and were trained with a
direct vector reshape; use ``layout="crc_reproduction"`` only with those
checkpoints. The two layouts have the same shape but different value order,
so changing layout requires retraining the model.

Pathway prototypes (paper equations 8--9)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let :math:`\mathbf{x}^p\in\mathbb{R}^{N_G}` be one patient's gene-expression
vector and :math:`\mathbf{B}_i\in\{0,1\}^{N_G}` indicate membership in pathway
``i``. Masking the vector isolates that pathway, after which a pathway-specific
self-normalizing network maps its non-zero gene positions to a shared size:

.. math::

   \mathbf{Z}^p_{\mathrm{proto},i}
   = \mathbf{x}^p\odot\mathbf{B}_i. \tag{8}

.. math::

   \mathbf{Z}^p_{i,\alpha}
   = f^p_{\alpha,i}(\mathbf{Z}^p_{\mathrm{proto},i})
   \in\mathbb{R}^{d_e}. \tag{9}

Every pathway has its own :math:`f^p_{\alpha,i}` because different pathways
contain different genes. HistoX stores the member columns as
``pathway_gene_indices`` and gathers them directly; this is equivalent to
applying the binary mask and then removing its zero positions. The equations
are implemented by :class:`~histox.mil.models.PathwayPrototypeEncoder`.
:class:`~histox.mil.PS3PathwayPreprocessor` first fits gene-wise
standardization on training patients only and resolves the pathway membership
against the expression columns.

Joint three-modal attention (paper equations 10--13)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, each modality is projected to :math:`d_e` dimensions and receives a
learned modality embedding. The three prototype sequences are concatenated in
the fixed order ``pathway -> histology -> report``:

.. math::

   \mathbf{Z}^{mm} =
   [\mathbf{Z}^p_\alpha\,\|\,\mathbf{Z}^h_\alpha\,\|\,
   \mathbf{Z}^t_\alpha]
   \in\mathbb{R}^{(N_P+N_H+N_T)\times d},
   \qquad d=d_e+d_r.

Shared projections produce :math:`\mathbf{Q}`, :math:`\mathbf{K}`, and
:math:`\mathbf{V}`. Splitting each tensor by modality exposes the nine
attention blocks: three within-modality blocks and six cross-modal blocks.

.. math::

   \mathbf{Q}=\mathbf{W}_Q\mathbf{Z}^{mm},\qquad
   \mathbf{K}=\mathbf{W}_K\mathbf{Z}^{mm},\qquad
   \mathbf{V}=\mathbf{W}_V\mathbf{Z}^{mm}. \tag{10}

The projections are partitioned back into their three modality blocks:

.. math::

   \mathbf{Q}=[\mathbf{Q}^p\|\mathbf{Q}^h\|\mathbf{Q}^t],\quad
   \mathbf{K}=[\mathbf{K}^p\|\mathbf{K}^h\|\mathbf{K}^t],\quad
   \mathbf{V}=[\mathbf{V}^p\|\mathbf{V}^h\|\mathbf{V}^t]. \tag{11}

Each block of attention logits is
:math:`\mathbf{A}^{m\rightarrow n}=\mathbf{Q}^{m}(\mathbf{K}^{n})^\top`
for :math:`m,n\in\{p,h,t\}`.

All nine blocks are assembled before applying one row-wise softmax:

.. math::
   :nowrap:

   \begin{equation}
   \begin{aligned}
   \begin{bmatrix}
   \mathbf{Z}^p_{\mathrm{fusion}}\\
   \mathbf{Z}^h_{\mathrm{fusion}}\\
   \mathbf{Z}^t_{\mathrm{fusion}}
   \end{bmatrix}
   &= \operatorname{softmax}\!\left(
   \frac{1}{\sqrt d}
   \begin{bmatrix}
   \mathbf{A}^{p\rightarrow p} & \mathbf{A}^{p\rightarrow h} & \mathbf{A}^{p\rightarrow t}\\
   \mathbf{A}^{h\rightarrow p} & \mathbf{A}^{h\rightarrow h} & \mathbf{A}^{h\rightarrow t}\\
   \mathbf{A}^{t\rightarrow p} & \mathbf{A}^{t\rightarrow h} & \mathbf{A}^{t\rightarrow t}
   \end{bmatrix}
   \right)\\
   &\quad\times
   \begin{bmatrix}\mathbf{V}^p\\\mathbf{V}^h\\\mathbf{V}^t\end{bmatrix}.
   \end{aligned} \tag{12}
   \end{equation}

The full post-fusion sequence is the concatenation of the three output blocks:

.. math::

   \mathbf{Z}^{mm}_{\mathrm{fusion}}=
   [\mathbf{Z}^p_{\mathrm{fusion}}\|\mathbf{Z}^h_{\mathrm{fusion}}
   \|\mathbf{Z}^t_{\mathrm{fusion}}]. \tag{13}

For example, :math:`\mathbf{A}^{h\rightarrow t}` asks which report prototypes
are relevant to each histological prototype. HistoX computes the same block
matrix as one concatenated matrix multiplication rather than nine separate
operations. :class:`~histox.mil.models.PS3PrototypeFusion` additionally wraps
attention with residual connections, layer normalization, and a feed-forward
network, following a standard Transformer layer. These operations are not
explicitly shown in paper equation 12 and are documented here so that the
implementation boundary is visible.

Patient-level risk (paper equation 14)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each fused prototype position has an independent MLP :math:`f^x_{\beta,i}`.
The transformed tokens are normalized and averaged inside each modality; the
three modality summaries are concatenated for the final linear risk head:

.. math::

   \mathbf{u}^x = \frac{1}{N_x}\sum_{i=1}^{N_x}
   \operatorname{LN}\!\left(f^x_{\beta,i}
   (\mathbf{Z}^x_{i,\mathrm{fusion}})\right),\qquad
   r = f_{\mathrm{risk}}([\mathbf{u}^p\|\mathbf{u}^h\|\mathbf{u}^t]).
   \tag{14}

The scalar ``r`` is a **log-risk**, not a survival probability. Larger values
mean a larger modeled hazard relative to the other patients in the risk set.
During training, :class:`~histox.mil.models.CoxPHLoss` minimizes the negative
Cox partial log-likelihood with Breslow handling of tied event times:

.. math::

   \mathcal{L}_{\mathrm{Cox}} = -\frac{1}{\sum_i\delta_i}
   \sum_{\tau}\left[
   \sum_{i\in D_\tau}r_i
   - |D_\tau|\log\sum_{j:\,T_j\ge\tau}\exp(r_j)
   \right].

Here :math:`D_\tau` contains patients with an observed event at time
:math:`\tau`, :math:`T_j` is observed follow-up time, and :math:`\delta_i`
marks an event. The denominator's risk set depends on other patients, which is
why arbitrary shuffled minibatches do not reproduce a full-cohort Cox
objective.

Tensor contract
---------------

.. list-table::
   :header-rows: 1
   :widths: 28 32 40

   * - Argument
     - Shape
     - Meaning
   * - ``histology_prototypes``
     - ``[batch, N_h, D_h]``
     - Precomputed slide-level histological prototypes.
   * - ``gene_expression``
     - ``[batch, N_g]``
     - Gene-expression values in the same column order used to define pathway
       membership.
   * - ``report_embeddings``
     - ``[batch, segments, D_t]``
     - Frozen text-encoder embeddings for pathology-report segments.
   * - ``report_mask``
     - ``[batch, segments]``
     - ``True`` for real report segments and ``False`` for padding.
   * - return value
     - ``[batch, 1]``
     - Unconstrained log-risk; larger values indicate larger modeled hazard.

Run a complete forward and loss step
------------------------------------

The following small example exercises every implemented PS3 stage. The
dimensions are deliberately small so it runs on a CPU; replace these tensors
with cohort-derived values for real experiments.

.. code-block:: python

   import torch

   from histox.mil.models import CoxPHLoss, PS3SurvivalModel

   # Each inner list contains column indices for one biological pathway.
   pathway_gene_indices = [
       [0, 1, 2],
       [2, 3, 4, 5],
       [5, 6, 7],
   ]

   model = PS3SurvivalModel(
       n_genes=8,
       pathway_gene_indices=pathway_gene_indices,
       histology_dim=12,
       report_dim=16,
       n_histology_prototypes=4,
       n_report_prototypes=3,
       embedding_dim=32,
       modality_embedding_dim=8,
       feedforward_dim=64,
       dropout=0.0,
   )

   histology = torch.randn(3, 4, 12)
   genes = torch.randn(3, 8)
   reports = torch.randn(3, 6, 16)
   report_mask = torch.tensor([
       [1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 0, 0],
       [1, 1, 1, 0, 0, 0],
   ], dtype=torch.bool)

   log_risk, attention = model(
       histology_prototypes=histology,
       gene_expression=genes,
       report_embeddings=reports,
       report_mask=report_mask,
       return_attention=True,
   )

   # One risk value per patient; 3 pathway + 4 histology + 3 report tokens.
   assert log_risk.shape == (3, 1)
   assert attention.shape == (3, 10, 10)

   duration = torch.tensor([1200.0, 760.0, 410.0])
   event = torch.tensor([1, 0, 1])  # 1=event, 0=censored
   loss = CoxPHLoss()(log_risk, duration, event)
   loss.backward()

``CoxPHLoss`` constructs risk sets from the samples passed to one call. A
faithful training loop should therefore use full-risk-set batches or an
explicitly validated risk-set sampling method. Ordinary shuffled minibatches
change the objective and should not be treated as equivalent.

The attention matrix uses the fixed token order ``pathway -> histology ->
report``. Attention is useful for debugging interactions, but it is not by
itself a causal or clinical explanation.

Trace report evidence to source text
------------------------------------

Use :meth:`~histox.mil.models.PS3SurvivalModel.report_segment_evidence` when
the question is: *which original pathology-report segment received the most
attention from the histology prototypes?* The method normalizes the final
histology-to-report logits within the report block, averages over histology
queries, rejects report prototypes that point to padding, and returns the
source-segment index.

.. code-block:: python

   from histox.mil import PS3ReportEncoder

   evidence = model.report_segment_evidence(
       histology_prototypes=histology,
       gene_expression=genes,
       report_embeddings=reports,
       report_mask=report_mask,
   )

   patient = 0
   source_index = int(evidence["selected_source_segment"][patient])
   report_prototype = int(
       evidence["selected_report_prototype"][patient]
   )
   attention_score = float(evidence["selected_score"][patient])

   source_segments = PS3ReportEncoder.segment_text(
       raw_report_text,
       tokenizer=plip_processor.tokenizer,
       max_tokens=77,
   )
   selected_text = source_segments[source_index]

The returned ``report_prototype_source_valid`` mask is important in
``crc_reproduction`` mode because reports shorter than ten segments use a
padded sequence to reproduce the retained model. A padded prototype can
participate in the checkpoint computation but must not be presented as
source report text. If no prototype maps to a real segment, the method raises
``ValueError`` instead of generating an explanation.

For cohort-level low/high-risk word clouds and the accompanying count tables,
continue to :doc:`Train PS3 with the HistoX workflow
</auto_examples/multimodal/plot_ps3_crc>`.

Provenance and licensing boundary
---------------------------------

The method is described in:

   Raza, M., Azam, A., Qaiser, T. & Rajpoot, N. PS3: A Multimodal Transformer
   Integrating Pathology Reports with Histology Images and Biological Pathways
   for Cancer Survival Prediction. *ICCV* (2025).
   `Paper <https://openaccess.thecvf.com/content/ICCV2025/html/Raza_PS3_A_Multimodal_Transformer_Integrating_Pathology_Reports_with_Histology_Images_ICCV_2025_paper.html>`__.

The `authors' repository <https://github.com/manahilr/PS3>`_ and the CRC
reproduction repository did not contain a detected source-code license at the
time of integration. HistoX therefore implements the published equations
independently and does not bundle those repositories, their weights, or their
data. Users remain responsible for the terms of external code and datasets.

API
---

Input preparation
~~~~~~~~~~~~~~~~~

.. currentmodule:: histox.mil

.. autoclass:: PS3ReportEncoder
   :members:

.. autoclass:: PS3HistologyPrototypeEncoder
   :members:

.. autoclass:: PS3PathwayPreprocessor
   :members:

Model and loss
~~~~~~~~~~~~~~

.. currentmodule:: histox.mil.models

.. autoclass:: DiagnosticPrototypeSelector
   :members:

.. autoclass:: PathwayPrototypeEncoder
   :members:

.. autoclass:: PS3PrototypeFusion
   :members:

.. autoclass:: PS3SurvivalModel
   :members:

.. autoclass:: CoxPHLoss
   :members:
