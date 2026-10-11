Methods
========

The HistoX Methods Hub organizes pathology algorithms by research task. It is
the stable entry point for methods reproduced from published work, adaptations
maintained by HistoX, and methods developed by the HistoX team.

This page distinguishes callable implementations from planned method families.
``Available`` means that the current HistoX source contains an implementation
and corresponding documentation; it does not imply clinical validation.

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

.. _method-registry:

Method registry
---------------

Search the implementations that are currently exposed by HistoX. The registry
links to usage documentation rather than duplicating API details. Paper-level
provenance, licenses, weight requirements, and known limitations remain part of
each method record as the catalog matures.

.. raw:: html

   <section class="histox-method-registry" data-method-registry aria-labelledby="histox-method-registry-title">
     <h2 id="histox-method-registry-title" class="visually-hidden">Available HistoX methods</h2>
     <div class="histox-method-registry__toolbar">
       <label class="histox-method-registry__search" for="histox-method-search">
         <span>Search methods</span>
         <input id="histox-method-search" type="search" placeholder="Try DINOv2, MIL, normalization…" autocomplete="off" data-method-search>
       </label>
       <div class="histox-method-registry__filters" role="group" aria-label="Filter methods by task">
         <button type="button" data-method-filter="all" aria-pressed="true">All</button>
         <button type="button" data-method-filter="encoders" aria-pressed="false">Encoders</button>
         <button type="button" data-method-filter="mil" aria-pressed="false">MIL</button>
         <button type="button" data-method-filter="segmentation" aria-pressed="false">Segmentation</button>
         <button type="button" data-method-filter="preprocessing" aria-pressed="false">Preprocessing</button>
         <button type="button" data-method-filter="interpretation" aria-pressed="false">Interpretation</button>
       </div>
     </div>
     <p class="histox-method-registry__results" data-method-results role="status" aria-live="polite"></p>
     <div class="histox-method-registry__list">
       <article class="histox-method-entry" data-method-entry data-task="encoders">
         <div class="histox-method-entry__task">Encoder</div>
         <div class="histox-method-entry__body">
           <h3>DINOv2</h3>
           <p>Build a DINOv2 feature extractor for tiles, tensors, and whole-slide workflows.</p>
           <p class="histox-method-entry__symbol"><code>histox.model.build_feature_extractor("dinov2", cfg="config.yaml", weights="teacher.pth")</code></p>
         </div>
         <div class="histox-method-entry__meta"><span>Available</span><small>External integration</small><a href="methods/encoders/dinov2.html">Open guide <span aria-hidden="true">→</span></a></div>
       </article>
       <article class="histox-method-entry" data-method-entry data-task="encoders">
         <div class="histox-method-entry__task">Encoder</div>
         <div class="histox-method-entry__body">
           <h3>Virchow</h3>
           <p>Use Virchow as a pathology feature encoder with an explicit weights contract.</p>
           <p class="histox-method-entry__symbol"><code>histox.model.build_feature_extractor("virchow", weights="virchow.pth")</code></p>
         </div>
         <div class="histox-method-entry__meta"><span>Available</span><small>External integration</small><a href="methods/encoders/virchow.html">Open guide <span aria-hidden="true">→</span></a></div>
       </article>
       <article class="histox-method-entry" data-method-entry data-task="encoders">
         <div class="histox-method-entry__task">Encoder</div>
         <div class="histox-method-entry__body">
           <h3>Vision Transformer</h3>
           <p>Construct a configurable ViT extractor through the common HistoX feature interface.</p>
           <p class="histox-method-entry__symbol"><code>histox.model.build_feature_extractor("vit", size="base", weights="model.pth")</code></p>
         </div>
         <div class="histox-method-entry__meta"><span>Available</span><small>External integration</small><a href="methods/encoders/vit.html">Open guide <span aria-hidden="true">→</span></a></div>
       </article>
       <article class="histox-method-entry" data-method-entry data-task="mil">
         <div class="histox-method-entry__task">MIL</div>
         <div class="histox-method-entry__body">
           <h3>Attention MIL</h3>
           <p>Train attention-based multiple-instance models from slide-level feature bags.</p>
           <p class="histox-method-entry__symbol"><code>histox.mil.mil_config("attention_mil")</code></p>
         </div>
         <div class="histox-method-entry__meta"><span>Available</span><small>Published method</small><a href="mil.html">Documentation <span aria-hidden="true">→</span></a></div>
       </article>
       <article class="histox-method-entry" data-method-entry data-task="mil">
         <div class="histox-method-entry__task">MIL</div>
         <div class="histox-method-entry__body">
           <h3>Multimodal Attention MIL</h3>
           <p>Combine multiple feature bags in the HistoX multimodal MIL training path.</p>
           <p class="histox-method-entry__symbol"><code>histox.mil.mil_config("mm_attention_mil")</code></p>
         </div>
         <div class="histox-method-entry__meta"><span>Available</span><small>Integrated workflow</small><a href="mil.html">Documentation <span aria-hidden="true">→</span></a></div>
       </article>
       <article class="histox-method-entry" data-method-entry data-task="mil">
         <div class="histox-method-entry__task">MIL</div>
         <div class="histox-method-entry__body">
           <h3>TransMIL</h3>
           <p>Model correlations between instances with the registered transformer MIL architecture.</p>
           <p class="histox-method-entry__symbol"><code>histox.mil.mil_config("transmil")</code></p>
         </div>
         <div class="histox-method-entry__meta"><span>Available</span><small>Published method</small><a href="mil.html">Documentation <span aria-hidden="true">→</span></a></div>
       </article>
       <article class="histox-method-entry" data-method-entry data-task="segmentation">
         <div class="histox-method-entry__task">Segmentation</div>
         <div class="histox-method-entry__body">
           <h3>Whole-slide tissue segmentation</h3>
           <p>Train binary, multiclass, or multilabel tissue models and run tiled slide inference.</p>
           <p class="histox-method-entry__symbol"><code>histox.segment.SegmentConfig</code></p>
         </div>
         <div class="histox-method-entry__meta"><span>Available</span><small>HistoX workflow</small><a href="segmentation.html">Documentation <span aria-hidden="true">→</span></a></div>
       </article>
       <article class="histox-method-entry" data-method-entry data-task="segmentation">
         <div class="histox-method-entry__task">Segmentation</div>
         <div class="histox-method-entry__body">
           <h3>Cell segmentation</h3>
           <p>Segment cells from whole-slide tiles and export compressed masks for downstream analysis.</p>
           <p class="histox-method-entry__symbol"><code>histox.cellseg.Segmentation</code></p>
         </div>
         <div class="histox-method-entry__meta"><span>Available</span><small>External integration</small><a href="cellseg.html">Documentation <span aria-hidden="true">→</span></a></div>
       </article>
       <article class="histox-method-entry" data-method-entry data-task="preprocessing">
         <div class="histox-method-entry__task">Normalization</div>
         <div class="histox-method-entry__body">
           <h3>Macenko</h3>
           <p>Normalize and augment stain appearance with NumPy, TensorFlow, or PyTorch implementations.</p>
           <p class="histox-method-entry__symbol"><code>histox.norm.autoselect("macenko")</code></p>
         </div>
         <div class="histox-method-entry__meta"><span>Available</span><small>Published method</small><a href="norm.html">Documentation <span aria-hidden="true">→</span></a></div>
       </article>
       <article class="histox-method-entry" data-method-entry data-task="preprocessing">
         <div class="histox-method-entry__task">Normalization</div>
         <div class="histox-method-entry__body">
           <h3>Reinhard</h3>
           <p>Apply reference-based color normalization, including fast and masked variants.</p>
           <p class="histox-method-entry__symbol"><code>histox.norm.autoselect("reinhard")</code></p>
         </div>
         <div class="histox-method-entry__meta"><span>Available</span><small>Published method</small><a href="norm.html">Documentation <span aria-hidden="true">→</span></a></div>
       </article>
       <article class="histox-method-entry" data-method-entry data-task="preprocessing">
         <div class="histox-method-entry__task">Normalization</div>
         <div class="histox-method-entry__body">
           <h3>Vahadane</h3>
           <p>Perform sparse stain separation through SPAMS- or scikit-learn-based implementations.</p>
           <p class="histox-method-entry__symbol"><code>histox.norm.autoselect("vahadane_sklearn")</code></p>
         </div>
         <div class="histox-method-entry__meta"><span>Available</span><small>Published method</small><a href="norm.html">Documentation <span aria-hidden="true">→</span></a></div>
       </article>
       <article class="histox-method-entry" data-method-entry data-task="preprocessing">
         <div class="histox-method-entry__task">Quality control</div>
         <div class="histox-method-entry__body">
           <h3>Slide quality control</h3>
           <p>Compose Otsu, Gaussian blur, DeepFocus, saved-mask, and segmentation-based QC.</p>
           <p class="histox-method-entry__symbol"><code>histox.slide.qc</code></p>
         </div>
         <div class="histox-method-entry__meta"><span>Available</span><small>Integrated workflow</small><a href="slide_qc.html">Documentation <span aria-hidden="true">→</span></a></div>
       </article>
       <article class="histox-method-entry" data-method-entry data-task="interpretation">
         <div class="histox-method-entry__task">Interpretation</div>
         <div class="histox-method-entry__body">
           <h3>Whole-slide heatmaps</h3>
           <p>Inspect spatial predictions and model attention across a whole-slide image.</p>
           <p class="histox-method-entry__symbol"><code>histox.Heatmap</code></p>
         </div>
         <div class="histox-method-entry__meta"><span>Available</span><small>HistoX workflow</small><a href="heatmap.html">Documentation <span aria-hidden="true">→</span></a></div>
       </article>
       <article class="histox-method-entry" data-method-entry data-task="interpretation">
         <div class="histox-method-entry__task">Interpretation</div>
         <div class="histox-method-entry__body">
           <h3>Saliency maps</h3>
           <p>Explain model predictions with gradients and integrated-gradient methods.</p>
           <p class="histox-method-entry__symbol"><code>histox.grad.SaliencyMap</code></p>
         </div>
         <div class="histox-method-entry__meta"><span>Available</span><small>HistoX workflow</small><a href="saliency.html">Documentation <span aria-hidden="true">→</span></a></div>
       </article>
     </div>
     <p class="histox-method-registry__empty" data-method-empty hidden>No methods match the current filters.</p>
   </section>

Planned method families
-----------------------

.. _methods-foundation:

Foundation models and VLM
~~~~~~~~~~~~~~~~~~~~~~~~~

HistoX currently exposes selected encoder integrations such as DINOv2 and
Virchow in the registry above. This planned area is for a unified foundation
model catalog, zero-shot classification, image-text retrieval, visual question
answering, and other vision-language workflows.

**Status:** Planned catalog expansion. Available encoder integrations do not
yet constitute a unified HistoX foundation-model or VLM API.

.. _methods-survival:

Survival and prognosis
~~~~~~~~~~~~~~~~~~~~~~

This family will cover time-to-event prediction, censoring-aware objectives,
risk stratification, and multimodal prognosis. Each implementation must state
its endpoint definition, censoring assumptions, split unit, and reported
metrics.

**Status:** Planned catalog area. Existing general training utilities may be
used experimentally, but a stable survival interface is not documented yet.

.. _methods-multimodal:

Multimodal and spatial methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HistoX currently includes multimodal Attention MIL for combining multiple
feature bags. This planned area covers broader methods that combine histology
with text, clinical variables, genomics, spatial transcriptomics, or other
molecular measurements. Entries must document modality alignment, missing-data
behavior, and cohort requirements.

**Status:** Planned catalog expansion beyond the available feature-bag MIL
workflow. Methods will be added as tested workflows rather than as a list of
unsupported paper names.

Method records
--------------

Every method added to HistoX should expose the same minimum record:

.. list-table:: Required method metadata
   :header-rows: 1
   :widths: 24 76

   * - Field
     - What it records
   * - Origin
     - ``reproduced``, ``adapted``, ``integration``, or ``histox-original``.
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

.. toctree::
   :hidden:
   :maxdepth: 2

   methods/encoders/index
