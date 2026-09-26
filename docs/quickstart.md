# Quickstart

This guide starts with a local project that does not download data. Project
creation and dataset planning can be tested on a laptop; tile extraction and
training require your own whole-slide images (WSIs), sufficient storage, and
the appropriate compute environment.

## 1. Check the active environment

```
import histox as hx

print("HistoX:", hx.__version__)
print("model backend:", hx.backend())
print("slide backend:", hx.slide_backend())
```

New workflows should use the PyTorch backend. If the import fails, return to
the [Installation](installation.html) guide before continuing.

## 2. Create an empty project

```
from pathlib import Path
import histox as hx

workspace = Path("histox-demo")
project = hx.create_project(
 root=str(workspace),
 name="my-cohort",
)

print(project)
print(project.annotations)
```

This call creates a project directory, an empty `annotations.csv`, a
`datasets.json` source configuration, and `models/` and `eval/` output
directories. It does not download slides. See [Setting up a Project](project_setup.html#project-setup) for the
complete project configuration interface.

## 3. Add slides and annotations

Place the source WSI files in the slide directory recorded in
`histox-demo/datasets.json`. Then populate `histox-demo/annotations.csv`.
A minimal classification table looks like this:

```
patient,slide,label
P001,S001,tumor
P002,S002,normal
```

The `slide` values must match the WSI filenames without their extensions.
Real cohorts should also carry the site, split, and provenance fields needed
for leakage-safe validation. Dataset construction validates the relationship
between the annotations and the files on disk.

## 4. Inspect the dataset

After adding WSIs and annotations, request a dataset at one physical scale:

```
dataset = project.dataset(
 tile_px=256,
 tile_um="20x",
)
dataset.summary()
```

This step reads project metadata and verifies the configured slide and tile
locations. It does not train a model. See [Datasets](datasets_and_val.html#datasets-and-validation) for
filtering, splitting, and validation options.

## 5. Extract tiles

```
project.extract_tiles(
 tile_px=256,
 tile_um="20x",
 qc="otsu",
)
```

Tile extraction reads every selected WSI and can create a large derived-data
store. Start with a small cohort, inspect tissue detection, and confirm the
output location before scaling up. Stain normalization is configured
separately; see [Slide Processing](slide_processing.html#filtering) for slide processing and quality control.

## 6. Configure and train a baseline

```
params = hx.ModelParams(
 tile_px=256,
 tile_um="20x",
 model="resnet50",
 epochs=1,
 batch_size=16,
)

results = project.train(
 "label",
 params=params,
)
```

This is a smoke-test configuration, not a recommended scientific benchmark.
Training requires extracted tiles, valid labels, and enough memory for the
selected batch size. Establish patient-level splits and an untouched test
cohort before reporting performance; see [Training](training.html#training).

## Plan public data without downloading it

The data registry separates dataset metadata from local storage. The bundled
metadata-only fixture can be inspected and planned without network transfer:

```
import histox as hx

record = hx.data.get_dataset("histox-metadata-fixture")
plan = hx.data.plan_download(record, path="/shared/pathology-data")

print(record.name, record.version, record.metadata_only)
print(plan.destination)
print("files requiring download:", plan.pending_count)
```

For this fixture, `record.metadata_only` is `True` and
`plan.pending_count` is `0`. Calling `plan_download` never downloads a
file. Provider access, license terms, checksums, and an explicit call to
`download_dataset` are required for records that contain assets. See
[histox.data](data_api.html) for the complete API and provenance model.

## Where to go next

- [Setting up a Project](project_setup.html#project-setup) -- configure a real cohort and storage locations.
- [Slide Processing](slide_processing.html#filtering) -- extract tiles and review quality control.
- [Training](training.html#training) -- define validation and training settings.
- [histox.data](data_api.html) -- discover, plan, and verify public data assets.
- [HistoX API reference](api.html) -- browse the Python reference.

HistoX began as a modified fork of HistoX. The `0.2.x` line still
contains inherited interfaces, but new examples use the public `histox`
namespace and HistoX environment variables.