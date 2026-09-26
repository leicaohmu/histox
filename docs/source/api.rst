HistoX API reference
====================

The API reference is organized by the object or workflow you are building.
For a guided introduction, begin with the :doc:`quickstart` instead.

.. raw:: html

   <div class="histox-api-index">
     <section><p class="histox-page-kicker">Core objects</p><h2>Projects and datasets</h2><ul><li><a href="project.html"><code>histox.Project</code><span>Manage study configuration, slides, models, and outputs.</span></a></li><li><a href="dataset.html"><code>histox.Dataset</code><span>Build, filter, split, and validate pathology cohorts.</span></a></li><li><a href="model_params.html"><code>histox.ModelParams</code><span>Describe model and training configuration.</span></a></li></ul></section>
     <section><p class="histox-page-kicker">Data</p><h2>Discovery and storage</h2><ul><li><a href="data_api.html"><code>histox.data</code><span>Discover, download, verify, and cache public datasets.</span></a></li><li><a href="tfrecords.html"><code>histox.io</code><span>Serialize tiles and records for model pipelines.</span></a></li><li><a href="dataset_features.html"><code>DatasetFeatures</code><span>Store and retrieve extracted feature representations.</span></a></li></ul></section>
     <section><p class="histox-page-kicker">Whole-slide imaging</p><h2>Slides and spatial outputs</h2><ul><li><a href="slide.html"><code>histox.WSI</code><span>Open, tile, and query whole-slide images.</span></a></li><li><a href="slide_qc.html"><code>histox.slide.qc</code><span>Apply built-in and custom quality-control methods.</span></a></li><li><a href="heatmap.html"><code>histox.Heatmap</code><span>Generate and export spatial model evidence.</span></a></li><li><a href="mosaic.html"><code>histox.Mosaic</code><span>Explore learned visual feature landscapes.</span></a></li></ul></section>
     <section><p class="histox-page-kicker">Learning</p><h2>PyTorch models</h2><ul><li><a href="model_torch.html"><code>histox.model.torch</code><span>Train and evaluate PyTorch pathology models.</span></a></li><li><a href="mil_module.html"><code>histox.mil</code><span>Multiple-instance learning models and utilities.</span></a></li><li><a href="simclr.html"><code>histox.simclr</code><span>Self-supervised representation learning.</span></a></li><li><a href="grad.html"><code>histox.grad</code><span>Gradient-based interpretation methods.</span></a></li></ul></section>
     <section><p class="histox-page-kicker">Pathology tools</p><h2>Normalization and segmentation</h2><ul><li><a href="norm.html"><code>histox.norm</code><span>Stain normalization and color transformation.</span></a></li><li><a href="histox_cellseg.html"><code>histox.cellseg</code><span>Cell segmentation workflows and model interfaces.</span></a></li><li><a href="biscuit.html"><code>histox.biscuit</code><span>Weakly supervised pathology utilities.</span></a></li></ul></section>
     <section><p class="histox-page-kicker">Utilities</p><h2>Analysis and extensions</h2><ul><li><a href="stats.html"><code>histox.stats</code><span>Statistical evaluation utilities.</span></a></li><li><a href="util.html"><code>histox.util</code><span>Shared helpers for project workflows.</span></a></li><li><a href="studio_module.html"><code>histox.studio</code><span>Interactive slide and model inspection.</span></a></li></ul></section>
   </div>

.. toctree::
   :maxdepth: 1
   :hidden:

   histox
   project
   dataset
   data_api
   dataset_features
   heatmap
   model_params
   mosaic
   slidemap
   biscuit
   histox_cellseg
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
