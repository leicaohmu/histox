# HistoX API reference

The API reference is organized by the object or workflow you are building.
For a guided introduction, begin with the [Quickstart](quickstart.html) instead.

Core objects

## Projects and datasets

- [`histox.Project`Manage study configuration, slides, models, and outputs.](project.html)
- [`histox.Dataset`Build, filter, split, and validate pathology cohorts.](dataset.html)
- [`histox.ModelParams`Describe model and training configuration.](model_params.html)

Data

## Discovery and storage

- [`histox.data`Discover, download, verify, and cache public datasets.](data_api.html)
- [`histox.io`Serialize tiles and records for model pipelines.](tfrecords.html)
- [`DatasetFeatures`Store and retrieve extracted feature representations.](dataset_features.html)

Whole-slide imaging

## Slides and spatial outputs

- [`histox.WSI`Open, tile, and query whole-slide images.](slide.html)
- [`histox.slide.qc`Apply built-in and custom quality-control methods.](slide_qc.html)
- [`histox.Heatmap`Generate and export spatial model evidence.](heatmap.html)
- [`histox.Mosaic`Explore learned visual feature landscapes.](mosaic.html)

Learning

## PyTorch models

- [`histox.model.torch`Train and evaluate PyTorch pathology models.](model_torch.html)
- [`histox.mil`Multiple-instance learning models and utilities.](mil_module.html)
- [`histox.simclr`Self-supervised representation learning.](simclr.html)
- [`histox.grad`Gradient-based interpretation methods.](grad.html)

Pathology tools

## Normalization and segmentation

- [`histox.norm`Stain normalization and color transformation.](norm.html)
- [`histox.cellseg`Cell segmentation workflows and model interfaces.](histox_cellseg.html)
- [`histox.biscuit`Weakly supervised pathology utilities.](biscuit.html)

Utilities

## Analysis and extensions

- [`histox.stats`Statistical evaluation utilities.](stats.html)
- [`histox.util`Shared helpers for project workflows.](util.html)
- [`histox.studio`Interactive slide and model inspection.](studio_module.html)