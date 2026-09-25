HistoX documentation
====================

.. raw:: html

   <section class="histox-hero" aria-labelledby="histox-hero-title">
     <div class="histox-hero__eyebrow">PyTorch-first deep pathology</div>
     <h2 id="histox-hero-title">Build reproducible pathology AI workflows.</h2>
     <p>HistoX connects whole-slide image processing, verified public data,
     model training, MIL, evaluation, and interpretability through one Python API.</p>
     <div class="histox-hero__actions">
       <a class="histox-button histox-button--primary" href="quickstart.html">Start the quickstart</a>
       <a class="histox-button histox-button--secondary" href="data.html">Explore data workflows</a>
     </div>
   </section>

Start with a workflow
---------------------

.. grid:: 1 2 2 4
   :gutter: 3
   :class-container: histox-home-grid

   .. grid-item-card:: Install
      :link: installation
      :link-type: doc

      Prepare a Python environment and install the development or released package.

   .. grid-item-card:: Load data
      :link: data
      :link-type: doc

      Discover datasets, plan storage, verify downloads, and preserve provenance.

   .. grid-item-card:: Process slides
      :link: slide_processing
      :link-type: doc

      Tile, filter, normalize, and quality-control whole-slide images.

   .. grid-item-card:: Train models
      :link: training
      :link-type: doc

      Build PyTorch-first classification, MIL, and representation workflows.

HistoX is under active development. The current interface prioritizes PyTorch
workflows while retaining selected compatibility APIs inherited from Slideflow.
Pages that have not yet completed the HistoX migration are being revised in
stages; check each guide's notes before using it in a new project.

.. toctree::
   :maxdepth: 1
   :caption: Get started

   installation
   overview
   quickstart
   project_setup

.. toctree::
   :maxdepth: 1
   :caption: Data and slides

   data
   datasets_and_val
   slide_processing
   features
   segmentation
   cellseg

.. toctree::
   :maxdepth: 1
   :caption: Modeling

   training
   evaluation
   posthoc
   uq
   mil
   ssl
   stylegan
   saliency
   custom_loops
   studio

.. toctree::
   :maxdepth: 1
   :caption: API reference

   slideflow
   project
   dataset
   data_api
   dataset_features
   heatmap
   model_params
   mosaic
   slidemap
   biscuit
   slideflow_cellseg
   io
   io_torch
   gan
   grad
   mil_module
   model
   model_torch
   norm
   simclr
   slide
   slide_qc
   stats
   util
   studio_module

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorial1
   tutorial2
   tutorial3
   tutorial4
   tutorial5
   tutorial6
   tutorial7
   tutorial8

.. toctree::
   :maxdepth: 1
   :caption: Developer guide

   extensions
   tfrecords
   dataloaders
   custom_extractors
   tile_labels
   plugins
   troubleshooting

.. toctree::
   :maxdepth: 1
   :caption: Compatibility APIs

   io_tensorflow
   model_tensorflow
