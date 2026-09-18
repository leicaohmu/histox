# HistoX

<p align="center">
  <img src="histox/assets/branding/histox-wordmark.svg" alt="HistoX — Deep Pathology Python Library" width="760">
</p>

[![Package checks](https://github.com/leicaohmu/histox/actions/workflows/python-app.yml/badge.svg)](https://github.com/leicaohmu/histox/actions/workflows/python-app.yml)
[![PyPI](https://img.shields.io/pypi/v/histox)](https://pypi.org/project/histox/)
![Python](https://img.shields.io/badge/Python-%3E%3D3.7-blue)
[![License](https://img.shields.io/badge/License-Apache--2.0-green)](LICENSE)

HistoX is an open-source Python library for computational pathology. It is a
modified fork of [Slideflow](https://github.com/slideflow/slideflow) and is
being developed into a PyTorch-first toolkit for whole-slide image (WSI)
processing, model training, evaluation, and multimodal pathology research.

> [!IMPORTANT]
> HistoX is under active development. Version `0.2.x` retains a substantial
> Slideflow-derived API, and parts of the documentation are still being
> audited. Treat the current public API as pre-stable and pin the package
> version in reproducible projects.

## What works today

The current package provides a working baseline for:

- project and cohort configuration;
- WSI reading, tissue quality control, and tile extraction;
- stain normalization;
- PyTorch and legacy TensorFlow model backends;
- supervised training and multiple-instance learning workflows;
- feature extraction, heatmaps, mosaics, and the HistoX Studio viewer.

PyTorch is selected automatically when it is installed and is the target for
new model and task development. TensorFlow remains available for compatibility
with inherited workflows, but new HistoX capabilities will be designed for
PyTorch first. See [ROADMAP.md](ROADMAP.md) for the planned interfaces and
milestones.

## Requirements

- Package metadata currently declares Python `>=3.7`.
- Distribution builds are currently checked in CI with Python 3.9.
- WSI reading requires a working libvips installation, unless a compatible
  cuCIM backend is used.
- Training requires at least one deep-learning backend.

The Python-version declaration is not yet a full compatibility guarantee.
Python 3.9 is the current development and CI baseline.

## Installation

Install the published package with the PyTorch extras:

```bash
python -m pip install "histox[torch]"
```

For development from the current repository:

```bash
git clone https://github.com/leicaohmu/histox.git
cd histox
python -m pip install -e ".[torch]"
```

Optional installation groups are retained for existing workflows:

| Command | Purpose |
| --- | --- |
| `python -m pip install histox` | Base dependency set; install a backend separately |
| `python -m pip install "histox[torch]"` | PyTorch backend and related tools |
| `python -m pip install "histox[tf]"` | Legacy TensorFlow compatibility |
| `python -m pip install "histox[torch,cucim]"` | PyTorch plus cuCIM; manage CuPy separately |
| `python -m pip install "histox[torch,cucim-cuda12]"` | PyTorch, cuCIM, and CUDA 12 CuPy |
| `python -m pip install "histox[torch,cucim-cuda11]"` | PyTorch, cuCIM, and CUDA 11 CuPy |

Do not install multiple `cupy-*` variants in the same environment. Match the
CuPy package to the CUDA runtime reported by `nvidia-smi`.

### Verify the installation

```python
import histox as hx

print("HistoX:", hx.__version__)
print("model backend:", hx.backend())
print("slide backend:", hx.slide_backend())
```

With PyTorch installed, `hx.backend()` should report `torch` unless
`HX_BACKEND` explicitly selects another supported backend.

## Minimal project workflow

HistoX uses an annotation table to connect patient labels to slide names. A
minimal table looks like this:

```csv
patient,slide,label
P001,S001,tumor
P002,S002,normal
```

Create a project and configure a PyTorch model:

```python
from pathlib import Path

import histox as hx

workspace = Path("/path/to/workspace")

project = hx.create_project(
    root=str(workspace / "project"),
    name="my-cohort",
    annotations=str(workspace / "annotations.csv"),
    slides=str(workspace / "slides"),
    tfrecords=str(workspace / "tfrecords"),
)

params = hx.ModelParams(
    tile_px=256,
    tile_um="20x",
    model="resnet50",
    epochs=1,
)
```

Once the slide files named by the annotation table are available, extraction
and training use the following public methods:

```python
project.extract_tiles(
    tile_px=256,
    tile_um="20x",
    qc="otsu",
    normalizer="macenko",
)

results = project.train("label", params=params)
```

Tile extraction and training are intentionally separate from project creation:
they require real WSI files, sufficient local storage, and an environment
configured for the selected slide and model backends.

## Data and storage policy

HistoX does not intend to bundle public WSI datasets in the Python wheel, keep
them in the Git repository, or operate the laboratory server as a public data
mirror. Dataset providers remain the source of record.

The planned `histox.data` layer will provide small, versioned registry records
and download helpers. Those records will describe:

- the authoritative provider and dataset identifier;
- access and license requirements;
- file manifests and provider checksums when available;
- supported labels, cohorts, and HistoX adapters;
- the local cache layout and provenance metadata.

Downloads will go directly from the provider to storage controlled by the
user. The planned cache resolution order is:

1. `HISTOX_CACHE_DIR`, when set;
2. `$XDG_CACHE_HOME/histox`, when `XDG_CACHE_HOME` is set;
3. `~/.cache/histox` otherwise.

Controlled-access datasets will continue to require the user's own provider
account, approvals, and credentials. The unified registry and downloader are
planned for A4b and are not part of the current `0.2.1` API.

## Core modules

| Module | Current responsibility |
| --- | --- |
| `histox.project` | Project configuration and end-to-end workflow orchestration |
| `histox.dataset` | Cohort filtering, slide/tile records, and dataset operations |
| `histox.slide` | WSI reading, QC, ROI handling, and tile extraction |
| `histox.norm` | Stain-normalization algorithms |
| `histox.model` | Backend selection, model configuration, training, and features |
| `histox.mil` | Multiple-instance learning workflows |
| `histox.heatmap` | Spatial model visualization |
| `histox.studio` | Interactive desktop viewer |

## Documentation and support

- [Documentation](https://histox.readthedocs.io/)
- [Roadmap](ROADMAP.md)
- [GitHub issues](https://github.com/leicaohmu/histox/issues)
- [PyPI package](https://pypi.org/project/histox/)

The hosted documentation contains inherited material that is still being
migrated and audited. For installation and current project direction, this
README and [ROADMAP.md](ROADMAP.md) are the source of truth.

## Contributing

Small, reviewable pull requests are preferred. Before opening a pull request:

1. explain the user-facing behavior being changed;
2. add or update the smallest relevant test or check;
3. run the package checks used by GitHub Actions;
4. keep new APIs PyTorch-first unless compatibility work is explicitly scoped.

## License and provenance

HistoX is a modified fork of
[Slideflow](https://github.com/slideflow/slideflow). The top-level
[`LICENSE`](LICENSE) records the repository's Apache-2.0 license declaration.
Some bundled source files and optional integrations remain under their original
licenses.

See [`PROVENANCE.md`](PROVENANCE.md) for repository lineage and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for component-level notices
and unresolved review items. Model weights, datasets, and optional extensions
are not automatically covered by the repository license; consult each
artifact's terms before use or redistribution.

Do not publish a new release until the unresolved license-review items in
`THIRD_PARTY_NOTICES.md` are closed.

## Acknowledgments

- [Slideflow](https://github.com/slideflow/slideflow), the upstream project
  from which HistoX is derived.
- [The Cancer Genome Atlas](https://portal.gdc.cancer.gov/) and other data
  providers that make computational pathology research possible.
- The PyTorch, TensorFlow, libvips, and cuCIM communities.

Contact: [caolei@hrbmu.edu.cn](mailto:caolei@hrbmu.edu.cn)
