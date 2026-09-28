HistoX
======

.. raw:: html

   <div id="histox-home" class="histox-home">
     <section class="histox-home__hero">
       <div class="histox-home__hero-inner">
         <div class="histox-home__hero-copy">
           <p class="histox-home__kicker">Open-source deep pathology</p>
           <h2>Deep pathology,<br>from slides to models.</h2>
           <p class="histox-home__lead">An open-source Python library for whole-slide imaging, public pathology data, model development, evaluation, and interpretation.</p>
           <div class="histox-home__actions">
             <a class="histox-button histox-button--primary" href="api.html">View documentation</a>
             <a class="histox-button histox-button--secondary" href="tutorials.html">Browse tutorials</a>
           </div>
         </div>
       </div>
     </section>

     <section class="histox-home__pillars" aria-label="HistoX core modules">
       <div class="histox-home__pillars-inner">
         <a href="data.html"><strong>Data</strong><span>Discover and cache public cohorts</span></a>
         <a href="slide_processing.html"><strong>Whole-slide imaging</strong><span>Read, tile, normalize, and segment</span></a>
         <a href="training.html"><strong>Models</strong><span>Train classification, MIL, and survival pipelines</span></a>
         <a href="evaluation.html"><strong>Interpretation</strong><span>Evaluate models and inspect spatial evidence</span></a>
       </div>
     </section>

     <section class="histox-home__updates" aria-label="HistoX highlights">
       <div class="histox-home__updates-inner">
         <article>
           <p class="histox-home__update-label">Classification</p>
           <h2>Learn slide-level phenotypes from tiles and features</h2>
           <p>Build tile, slide, and MIL classifiers with traceable experiments.</p>
           <a href="training.html">Explore classification →</a>
         </article>
         <article>
           <p class="histox-home__update-label">Survival</p>
           <h2>Model time-to-event outcomes from pathology</h2>
           <p>Connect learned slide representations with prognosis and clinical endpoints.</p>
           <a href="methods.html">Explore survival methods →</a>
         </article>
         <article>
           <p class="histox-home__update-label">Multimodal &amp; VLM</p>
           <h2>Connect histology with text and other modalities</h2>
           <p>Develop vision-language and multimodal research workflows in one project structure.</p>
           <a href="methods.html">Explore multimodal methods →</a>
         </article>
       </div>
     </section>

     <section class="histox-home__section histox-home__section--white">
       <div class="histox-home__section-inner">
         <p class="histox-home__section-label">One workflow, end to end</p>
         <h2>Pathology infrastructure for research, not another isolated model.</h2>
         <div class="histox-home__capabilities">
           <article><h3>Data</h3><p>Discover public cohorts, verify files, cache locally, and keep dataset metadata reproducible.</p><a href="data.html">Explore data tools →</a></article>
           <article><h3>Slides</h3><p>Read, tile, normalize, segment, and quality-control whole-slide images through one interface.</p><a href="slide_processing.html">Process slides →</a></article>
           <article><h3>Models</h3><p>Train PyTorch models for classification, MIL, survival analysis, and multimodal research.</p><a href="training.html">Build models →</a></article>
           <article><h3>Evidence</h3><p>Evaluate cohorts and inspect spatial evidence with heatmaps, saliency, and uncertainty tools.</p><a href="evaluation.html">Evaluate results →</a></article>
         </div>
       </div>
     </section>

     <section class="histox-home__section histox-home__section--split">
       <div class="histox-home__section-inner histox-home__workflow">
         <div class="histox-home__workflow-copy">
           <p class="histox-home__section-label">Designed for real pathology projects</p>
           <h2>Use one project structure from cohort assembly to interpretation.</h2>
           <p>Keep slides, annotations, features, experiments, and model outputs connected without forcing large public datasets into the package or your server.</p>
           <a class="histox-text-link" href="project_setup.html">Set up a project →</a>
         </div>
         <aside class="histox-home__workflow-code" aria-label="HistoX project example">
           <div class="histox-home__code-head">
             <span>project.py</span>
             <span>Python</span>
           </div>
           <pre><code><span class="histox-code-keyword">import</span> histox <span class="histox-code-keyword">as</span> hx

   project = hx.create_project(
       root=<span class="histox-code-string">"study"</span>,
       annotations=<span class="histox-code-string">"annotations.csv"</span>,
       slides=<span class="histox-code-string">"/data/slides"</span>,
   )

   project.extract_tiles(
       tile_px=<span class="histox-code-number">256</span>,
       tile_um=<span class="histox-code-string">"20x"</span>,
   )</code></pre>
           <a href="project_setup.html">Open the project guide →</a>
         </aside>
       </div>
     </section>

     <section class="histox-home__resources">
       <div class="histox-home__resources-inner">
         <article><h2>Docs</h2><p>Install HistoX and learn the core project, dataset, and slide concepts.</p><a href="quickstart.html">Read the docs →</a></article>
         <article><h2>Tutorials</h2><p>Follow complete workflows for training, MIL, heatmaps, and custom pipelines.</p><a href="tutorials.html">Start a tutorial →</a></article>
         <article><h2>Methods</h2><p>Browse implemented and planned methods by pathology task, maturity, and origin.</p><a href="methods.html">Explore methods →</a></article>
       </div>
     </section>
   </div>

.. toctree::
   :maxdepth: 1
   :hidden:

   docs
   tutorials
