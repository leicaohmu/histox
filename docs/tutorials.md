# HistoX Tutorials

Learn HistoX through complete computational pathology workflows. Start with
project and data setup, then move to training, MIL, evaluation, and model
interpretation.

## What's new in HistoX tutorials?

- The [Quickstart](quickstart.html) now uses the HistoX package and PyTorch-first workflow.
- The [Data quickstart](data.html) guide documents the public-data registry and local cache.
- Eight inherited workflow tutorials remain available while their examples are
migrated and verified against the current HistoX API.

Start here

## Learn the basics

Install HistoX, create a project, inspect the data model, and run a minimal PyTorch-first workflow.

[Open the quickstart](quickstart.html)

Task-oriented examples

## HistoX recipes

Find focused examples for training, custom models, slide processing, evaluation, and MIL.

Browse the tutorial library

## Tutorial library

Filter the current tutorials by topic. Compatibility tutorials are clearly
marked until their code has been migrated and executed with HistoX.

All
Getting started
Training
Models
Slides
MIL
Evaluation
Interpretation
Extensions

**8** tutorials shown

[Getting started · Training

### Train a first model

Take a TCGA project from slides and labels to a trained classifier.

**Compatibility tutorial**](tutorial1.html)
[Training · Extensions

### Control the training loop

Work directly with datasets and trainers for a customized pipeline.

**Compatibility tutorial**](tutorial2.html)
[Models · Extensions

### Use a custom architecture

Integrate your own vision architecture with the HistoX model interface.

**Compatibility tutorial**](tutorial3.html)
[Evaluation · Interpretation

### Evaluate and build heatmaps

Run held-out evaluation and localize predictive regions on a slide.

**Compatibility tutorial**](tutorial4.html)
[Interpretation

### Create a mosaic map

Explore the visual feature landscape learned by a pathology model.

**Compatibility tutorial**](tutorial5.html)
[Slides · Extensions

### Write a custom slide filter

Implement and preview a bespoke whole-slide quality-control method.

**Compatibility tutorial**](tutorial6.html)
[Training · Extensions

### Add custom augmentations

Build class-aware PyTorch augmentations for imbalanced outcomes.

**Compatibility tutorial**](tutorial7.html)
[MIL · Training

### Train a MIL model

Learn slide-level outcomes from bags of extracted patch features.

**Compatibility tutorial**](tutorial8.html)

Note

Compatibility tutorials still contain examples inherited from HistoX.
Treat their APIs as migration references until each page is marked as
verified for the current HistoX release.

## Additional resources

- [Installation](installation.html) -- supported installation paths and dependency groups.
- [Data quickstart](data.html) -- discover, download, verify, and cache public datasets.
- [HistoX API reference](api.html) -- package API organized by module and task.