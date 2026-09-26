HistoX Tutorials
================

Learn HistoX through complete computational pathology workflows. Start with
project and data setup, then move to training, MIL, evaluation, and model
interpretation.

What's new in HistoX tutorials?
-------------------------------

* The :doc:`quickstart` now uses the HistoX package and PyTorch-first workflow.
* The :doc:`data` guide documents the public-data registry and local cache.
* Eight inherited workflow tutorials remain available while their examples are
  migrated and verified against the current HistoX API.

.. raw:: html

   <div class="histox-tutorial-entrypoints" aria-label="Tutorial starting points">
     <section>
       <p class="histox-page-kicker">Start here</p>
       <h2>Learn the basics</h2>
       <p>Install HistoX, create a project, inspect the data model, and run a minimal PyTorch-first workflow.</p>
       <a href="quickstart.html">Open the quickstart</a>
     </section>
     <section>
       <p class="histox-page-kicker">Task-oriented examples</p>
       <h2>HistoX recipes</h2>
       <p>Find focused examples for training, custom models, slide processing, evaluation, and MIL.</p>
       <a href="#tutorial-library">Browse the tutorial library</a>
     </section>
   </div>

Tutorial library
----------------

Filter the current tutorials by topic. Compatibility tutorials are clearly
marked until their code has been migrated and executed with HistoX.

.. raw:: html

   <div id="tutorial-library" class="histox-tutorial-library" data-histox-tutorial-library>
     <div class="histox-tutorial-filters" role="group" aria-label="Filter tutorials by topic" data-histox-tutorial-filter>
       <button id="topic-all" class="is-active" type="button" data-topic="all" aria-pressed="true">All</button>
       <button id="topic-getting-started" type="button" data-topic="getting-started" aria-pressed="false">Getting started</button>
       <button id="topic-training" type="button" data-topic="training" aria-pressed="false">Training</button>
       <button id="topic-models" type="button" data-topic="models" aria-pressed="false">Models</button>
       <button id="topic-slides" type="button" data-topic="slides" aria-pressed="false">Slides</button>
       <button id="topic-mil" type="button" data-topic="mil" aria-pressed="false">MIL</button>
       <button id="topic-evaluation" type="button" data-topic="evaluation" aria-pressed="false">Evaluation</button>
       <button id="topic-interpretation" type="button" data-topic="interpretation" aria-pressed="false">Interpretation</button>
       <button id="topic-extensions" type="button" data-topic="extensions" aria-pressed="false">Extensions</button>
     </div>
     <p class="histox-tutorial-results" aria-live="polite"><strong data-result-count>8</strong> tutorials shown</p>
     <div class="histox-tutorial-grid">
       <a class="histox-tutorial-card" href="tutorial1.html" data-topics="getting-started training"><span>Getting started · Training</span><h3>Train a first model</h3><p>Take a TCGA project from slides and labels to a trained classifier.</p><strong>Compatibility tutorial</strong></a>
       <a class="histox-tutorial-card" href="tutorial2.html" data-topics="training extensions"><span>Training · Extensions</span><h3>Control the training loop</h3><p>Work directly with datasets and trainers for a customized pipeline.</p><strong>Compatibility tutorial</strong></a>
       <a class="histox-tutorial-card" href="tutorial3.html" data-topics="models extensions"><span>Models · Extensions</span><h3>Use a custom architecture</h3><p>Integrate your own vision architecture with the HistoX model interface.</p><strong>Compatibility tutorial</strong></a>
       <a class="histox-tutorial-card" href="tutorial4.html" data-topics="evaluation interpretation"><span>Evaluation · Interpretation</span><h3>Evaluate and build heatmaps</h3><p>Run held-out evaluation and localize predictive regions on a slide.</p><strong>Compatibility tutorial</strong></a>
       <a class="histox-tutorial-card" href="tutorial5.html" data-topics="interpretation"><span>Interpretation</span><h3>Create a mosaic map</h3><p>Explore the visual feature landscape learned by a pathology model.</p><strong>Compatibility tutorial</strong></a>
       <a class="histox-tutorial-card" href="tutorial6.html" data-topics="slides extensions"><span>Slides · Extensions</span><h3>Write a custom slide filter</h3><p>Implement and preview a bespoke whole-slide quality-control method.</p><strong>Compatibility tutorial</strong></a>
       <a class="histox-tutorial-card" href="tutorial7.html" data-topics="training extensions"><span>Training · Extensions</span><h3>Add custom augmentations</h3><p>Build class-aware PyTorch augmentations for imbalanced outcomes.</p><strong>Compatibility tutorial</strong></a>
       <a class="histox-tutorial-card" href="tutorial8.html" data-topics="mil training"><span>MIL · Training</span><h3>Train a MIL model</h3><p>Learn slide-level outcomes from bags of extracted patch features.</p><strong>Compatibility tutorial</strong></a>
     </div>
   </div>

.. note::

   Compatibility tutorials still contain examples inherited from HistoX.
   Treat their APIs as migration references until each page is marked as
   verified for the current HistoX release.

Additional resources
--------------------

* :doc:`installation` — supported installation paths and dependency groups.
* :doc:`data` — discover, download, verify, and cache public datasets.
* :doc:`api` — package API organized by module and task.

.. toctree::
   :maxdepth: 1
   :hidden:

   tutorials_training
   tutorials_evaluation
   tutorials_slides
   tutorials_mil
