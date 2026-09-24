# HistoX roadmap

This roadmap separates the capabilities available in the current package from
the interfaces HistoX plans to add. It is a direction document, not a promise
of release dates or backward compatibility during the pre-stable phase.

## Design principles

1. **PyTorch first.** New training, inference, and model integrations should
   use PyTorch unless a compatibility task explicitly requires another
   backend.
2. **WSI data stays outside the package.** HistoX distributes code, metadata,
   manifests, and adapters—not whole-slide datasets.
3. **One task contract, multiple models.** Classification, prognosis,
   segmentation, retrieval, and vision-language workflows should share common
   dataset, prediction, and evaluation records.
4. **Reproducibility is part of the API.** Dataset versions, source checksums,
   preprocessing parameters, splits, model identifiers, and software versions
   should be recordable and recoverable.
5. **Provider terms remain authoritative.** HistoX must not bypass access
   controls or imply redistribution rights for datasets or model weights.
6. **Incremental migration.** Useful inherited Slideflow behavior remains
   available while HistoX-owned interfaces are introduced and tested.

## Current baseline: `0.2.x`

Available today:

- project and annotation management;
- WSI reading, QC, ROI handling, tiling, and stain normalization;
- PyTorch and legacy TensorFlow backends;
- supervised training, feature extraction, and MIL workflows;
- heatmaps, mosaics, reports, and HistoX Studio;
- package build checks, provenance records, and HistoX identity assets.

Known limits:

- the public API remains largely inherited from Slideflow;
- the CI matrix currently validates packaging on Python 3.9, not every Python
  version accepted by package metadata;
- documentation and examples are still being migrated;
- A4b2 covers verified direct HTTP(S) downloads for open assets, but
  authenticated provider clients and inherited project presets are not yet
  unified;
- model/task outputs are not yet standardized across classification,
  prognosis, MIL, and vision-language workflows.

## A4b: dataset registry and local cache

Introduce `histox.data` with a small provider-neutral contract.

A4b1 establishes:

- `list_datasets()` and `get_dataset(name, version=...)`;
- registry records for source, access terms, citations, manifests, checksums,
  and supported tasks;
- explicit download planning before large transfers;
- configurable cache roots using `HISTOX_CACHE_DIR`, `XDG_CACHE_HOME`, or
  `~/.cache/histox`;
- a `path=` override for shared storage, object-storage mounts, and HPC
  filesystems.

The first implementation uses a metadata-only fixture. A4b2 adds explicit
direct HTTP(S) transfers for open records, safe Range resumption, atomic file
publication, checksum verification, and a machine-readable local provenance
record. Subsequent A4b patches will add:

- authenticated and provider-specific download clients;
- compatibility bridging for inherited project presets.

TCGA/GDC support should use official provider tools and the user's own
credentials rather than a HistoX mirror.

## A5: PyTorch model and task contracts

Define stable HistoX-owned abstractions for:

- model registration and weight metadata;
- patch encoders and slide-level aggregators;
- supervised classification and regression;
- survival/prognosis modeling;
- multiple-instance learning;
- segmentation and spatial prediction;
- pathology vision-language encoders and generation models.

Each task should expose consistent configuration, prediction, checkpoint, and
provenance records without forcing every model into the same internal
architecture.

## A6: evaluation and reproducibility

Add reusable evaluation components for:

- patient-level, slide-level, and patch-level splits;
- leakage checks and cohort stratification;
- classification, calibration, survival, retrieval, and segmentation metrics;
- external-cohort evaluation;
- confidence intervals and uncertainty reporting;
- exportable experiment manifests and result tables.

## A7: reference integrations and benchmarks

After the data and task contracts stabilize:

- add selected modern pathology encoders and MIL methods;
- publish reference recipes for representative cancer tasks;
- provide small CI fixtures and larger reproducibility manifests;
- compare methods using fixed splits, preprocessing records, and explicit
  licensing constraints;
- release reusable weights only when training-data and weight licenses permit
  redistribution.

## Non-goals for the current phase

- hosting public WSI collections on the laboratory server;
- bundling datasets or large model weights in the Python distribution;
- claiming support for a model before its loading, inference, and evaluation
  path is tested;
- removing all inherited Slideflow APIs in one breaking rewrite;
- promising production or clinical use during the pre-stable phase.

## Definition of done for a new integration

A dataset, model, or task is not considered supported until it has:

1. a documented public entry point;
2. an explicit license and provenance record;
3. a minimal automated test or reproducible smoke check;
4. versioned configuration and outputs;
5. one runnable example that uses only existing public APIs;
6. clear failure messages for missing data, weights, or optional dependencies.
