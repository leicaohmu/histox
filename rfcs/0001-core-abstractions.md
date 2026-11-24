# RFC 0001: Core abstractions

| | |
|---|---|
| Status | Draft |
| Author | leicaohmu |
| Created | 2026-11-24 |
| Applies to | `0.2.x` → `1.0` |
| Related | [`ROADMAP.md`](../ROADMAP.md) (A4b, A5, A6) |

## 1. Summary

HistoX already contains most of the building blocks of a computational
pathology library, but several of them exist two to five times with different
interfaces: five registries, three families of image transforms, four kinds of
slide-aligned outputs, and three MIL `forward()` signatures. This RFC proposes a
small set of core abstractions, in the spirit of PyTorch's `Tensor`,
`Dataset`, and `nn.Module`, and a migration path that **unifies existing code
instead of rewriting it**.

The proposals, in implementation order:

| # | Proposal | Kind of change |
|---|---|---|
| P1 | One registry for every pluggable component | Unify 5 registries |
| P2 | One coordinate convention for everything spatial | Document + helpers |
| P3 | `Bag`: a feature bag object with coordinates and provenance | Wrap existing file format |
| P4 | `TileSpec`: tiling parameters as a value object | Extract from `WSI` |
| P5 | `SpatialMap`: one type for slide-aligned outputs | Unify 4 output types |
| P6 | Transform protocols for QC, stain normalization, augmentation | Unify 3 interfaces |
| P7 | Model protocols: `TileEncoder`, `SlideAggregator`, `ModelCard` | Formalize existing classes |
| P8 | `Cohort`: patients, slides, and labels without storage concerns | Extract from `Dataset` |
| P9 | `histox.workflows` as the task layer for all training methods | Extend existing contract |

Nothing in this RFC removes a public API. Old entry points keep working and,
where they are superseded, emit a deprecation warning (see section 6).

## 2. Motivation

### 2.1 What already exists

An audit of `develop` (commit `b388e97`) found these building blocks:

| Area | Existing code | Notes |
|---|---|---|
| Slides | `histox/slide/wsi.py` `WSI` (3,541 lines); readers for libvips, cuCIM, DICOM, OME-TIFF in `histox/slide/backends/` | Level-0 coordinates internally |
| Annotations | `histox/slide/utils.py` `ROI` (polygon with holes and label); QuPath-format CSV ROIs, ImageScope XML conversion (`Dataset`) | Polygons only |
| Registration | `histox/slide/utils.py` `Alignment` (origin, scale) | Translation and scale only |
| QC | `histox/slide/qc/`: `Otsu`, `Gaussian`, `GaussianV2`, `DeepFocus`, `Segment`, `StridedDL`; `QCMask` | Callable on a `WSI` or array, returns a boolean mask |
| Stain normalization | `histox/norm/`: `StainNormalizer` with Reinhard, Macenko, Vahadane, CycleGAN; torch and TF variants | `fit` / `transform` / `rgb_to_rgb` / `torch_to_torch` |
| Augmentation | `histox/io/torch/augment.py`: `RandomColorDistortion`, `RandomGaussianBlur`, `RandomJPEGCompression`, … | torchvision-style callables |
| Tile storage | `histox/tfrecord/`, `histox/io/` | TFRecord read/write, interleaving |
| Cohorts | `histox/project.py` `Project` (3,953 lines), `histox/dataset.py` `Dataset` (4,544 lines) | Annotations, sources, tile parameters, TFRecord paths, filters, and splits in one class |
| Feature extractors | `histox/model/base.py` `BaseFeatureExtractor`; `histox/model/extractors/` | Wrappers that dispatch on input type; `.license`, `.citation` |
| MIL | `histox/mil/`: `Attention_MIL`, `TransMIL`, multimodal and UQ variants, PS3; `BagDataset`, `MultiBagDataset`; `train_mil` | Models are `nn.Module`s |
| Survival and multimodal | `CoxPHLoss`, `SurvivalTrainer`, `histox/stats/concordance.py`; PS3 (histology + RNA pathways + report text) | |
| Segmentation | `histox/segment/`: `SegmentModel` (Lightning), mask datasets; `histox/cellseg/` `Segmentation` (cell masks, outlines, centroids, zarr) | |
| Outputs | `histox/heatmap.py` `Heatmap`; `histox/grad` `SaliencyMap`; `histox/mosaic.py` `Mosaic`; `histox/stats` `SlideMap` | |
| Retrieval | `histox/experimental` `EmbeddingSearch` | |
| Dataset registry | `histox/data/` (A4b): `DatasetRecord`, download planning, cache | Name `histox.data` is taken |
| Task contract | `histox/workflows/`: `Task`, `Method`, `Experiment`, `DataSpec`, `Artifact`, `RunResult`; `SurvivalTask`, `PS3Method` | A5 has started here |
| Plugins | `histox/plugin.py`: entry point group `histox.plugins` | |

### 2.2 What is fragmented

1. **Five registries.**
   - `histox/model/extractors/_registry.py`: `register_torch` / `register_tf`, plus `_known_extras_packages` hints.
   - `histox/mil/_registry.py`: `register_trainer` / `register_model`, also with extras hints.
   - `histox/workflows/_registry.py`: `register_task` / `register_method`, with duplicate protection and name normalization.
   - `histox/data/registry.json`: dataset records.
   - `histox/plugin.py`: entry points.

   Each has its own naming rules and error messages, and none records license or citation metadata in a common form.
2. **Coordinates are mostly consistent, but the convention is implicit.**
   - Tiles: `WSI.coord` stores level-0 top-left corners. Tile records (`loc` in `histox/slide/backends/*.py`, `loc_x`/`loc_y` in TFRecords) and bag indices store level-0 tile *centers*.
   - `Heatmap.save_npz()` stores a prediction grid with no coordinates or resolution.
   - `cellseg.Segmentation` uses its own `wsi_dim` / `wsi_offset` / `wsi_ratio`.
   - MIL attention is saved with bare `np.savez` (`histox/mil/utils.py`).
   - `CoordinateAlignmentError` exists, but no shared helper does the conversion.
3. **Feature bags are files, not objects.** A bag is `<slide>.pt` (float32 N×F tensor), `<slide>.index.npz` (`arr_0`: N×2 tile centers), and a directory-wide `bags_config.json` (`histox/model/features.py`). Readers re-implement the pairing in `histox/mil/data.py`, `histox/mil/eval.py`, `histox/mil/features.py`, and `Dataset.filter_bags_by_roi`.
4. **Three transform families, three interfaces:** QC (`__call__(wsi, mask=None) -> bool ndarray`), normalizers (`fit` / `transform` / `torch_to_torch`), and augmentations (`__call__(tensor)`). They cannot be composed into one recorded preprocessing pipeline.
5. **Four slide-aligned output types:** `QCMask`, `Heatmap` predictions, `Segmentation` masks, and saliency/attention maps. Each has its own storage and overlay code.
6. **Three MIL call signatures:**
   - `Attention_MIL.forward(bags, lens)`
   - `MultiModal_Attention_MIL.forward(*bags_and_lens)`
   - `TransMIL.forward(h)`

   Training code has to special-case them.
7. **`Dataset` does too much.** One 4.5k-line class owns annotations, slide sources, tile parameters, TFRecord locations, filtering, clipping, splitting, and extraction.
8. **Tests are thin where these pieces meet.** `histox/test/` has no tests for `heatmap`, `segment`, `grad`, or the non-PS3 MIL core.

## 3. Goals and non-goals

Goals:

- One obvious type for each core concept, usable on its own and from plain PyTorch.
- Every spatial object can be placed on its slide without guesswork.
- Every artifact records how it was made (provenance).
- No breaking change without a deprecation period.

Non-goals for this RFC:

- Rewriting `Project`, `Dataset`, or `WSI` from scratch.
- Removing the TensorFlow backend (a separate decision; see section 8).
- Designing cell-graph models, multiplex imaging, spatial transcriptomics, or distributed training in detail. Section 5 only reserves room for them.
- Studio. It should depend only on public APIs and is migrated last.

## 4. Proposals

Each proposal lists the current code, the proposed API, the migration steps,
and open questions. API sketches are illustrative; names are settled in review.

### P1. One registry

**Current code:** the five registries listed in section 2.2.

**Proposal:** a generic registry in a new module `histox/registry.py`, built
from `histox/workflows/_registry.py`, which is the most complete one (name
normalization, duplicate protection, `replace=` flag). Each entry carries a
`ModelCard`-style metadata record (see P7).

```python
from histox.registry import Registry

EXTRACTORS = Registry("extractor")

@EXTRACTORS.register("uni", license="CC-BY-NC-ND-4.0", citation="...",
                     extras="histox-contrib")
def uni(**kwargs): ...

EXTRACTORS.list()          # names
EXTRACTORS.get("uni")      # factory; clear error naming the extras package
EXTRACTORS.card("uni")     # metadata
```

Registry kinds: `extractor`, `mil_model`, `mil_trainer`, `task`, `method`,
`normalizer`, `qc`. The dataset registry (`histox/data/registry.json`)
describes data, not factories, so it stays as is, but exposes the same
`list()` / `get()` vocabulary. Plugins (`histox.plugins` entry points) keep
their mechanism and register into these registries.

**Migration:**
1. Add `histox/registry.py` with tests.
2. Re-implement `histox/workflows/_registry.py` on top of it; the public functions are unchanged.
3. Re-implement the extractor and MIL registries the same way; `list_extractors()`, `register_torch()`, `get_model()`, and the other existing functions become thin wrappers.

**Open question:** the extras hints point at `slideflow-noncommercial` and
`slideflow-gpl`. Should HistoX publish its own extras packages under its own name?

### P2. Coordinate convention

**Current code:** described in section 2.2, item 2. Level 0 is already the de
facto convention; this proposal writes it down and enforces it.

**Proposal:**

- All spatial coordinates in HistoX are `(x, y)` in **level-0 pixels of the source slide**.
- Tile positions in public APIs refer to the **tile center** (as TFRecords and bags already do). Top-left corners stay internal to `WSI`.
- Every spatial object carries the slide name and the level-0 resolution (`mpp`) it was computed against.
- A small module `histox/geometry.py` provides the only conversion helpers, for example `to_level0(xy, downsample)`, `from_level0(xy, downsample)`, and `tile_center_to_corner(xy, tile_size_level0)`. Code that converts coordinates by hand migrates to these helpers.

**Migration:**
1. Document the convention in the user guide.
2. Add `histox/geometry.py` with tests.
3. Adopt it in `cellseg.Segmentation`, `Heatmap`, and MIL attention export, without changing their outputs.

### P3. `Bag`

**Current code:** the `.pt` + `.index.npz` + `bags_config.json` layout written
by `histox/model/features.py` and read in four places (section 2.2, item 3).

**Proposal:** a `Bag` object that wraps the same files.

```python
@dataclass
class Bag:
    features: torch.Tensor      # N x F
    coords: np.ndarray          # N x 2, int64, level-0 tile centers (P2)
    slide: str
    tile_spec: TileSpec         # P4
    extractor: Optional[str]    # registry name (P1)
    config: dict                # provenance, as in bags_config.json

    def save(self, directory): ...           # same .pt / .index.npz / bags_config.json
    @classmethod
    def load(cls, directory, slide): ...
    def subset(self, mask_or_indices): ...   # used by filter_bags_by_roi
```

`BagDataset` accepts `Bag` objects in addition to the inputs it accepts today.
The on-disk format does not change in this step.

**Later:** `Bag.to_h5()` / `Bag.from_h5()` for interoperability with CLAM and
TRIDENT feature files. Before writing this converter, verify whether those
tools store top-left or center coordinates.

**Migration:**
1. Add `Bag` with tests.
2. Make the writer in `histox/model/features.py` use it.
3. Replace the four hand-written readers.

### P4. `TileSpec`

**Current code:** tile size, physical size, stride, and origin are `WSI`
constructor arguments. The folder label comes from `hx.util.tile_size_label(tile_px, tile_um)`.

**Proposal:** a frozen dataclass `histox.slide.TileSpec(tile_px, tile_um, stride_div=1, origin=(0, 0))`.
`tile_um` accepts microns or a magnification string, as today. `TileSpec.label`
produces the existing folder label. `WSI`, `Dataset`, and `Bag` accept a
`TileSpec`; the existing keyword arguments keep working.

Reserved for later: multi-scale specs (one position at several
magnifications) and non-RGB channel selection. Today `histox/slide/backends/vips.py`
assumes RGB or RGBA.

### P5. `SpatialMap`

**Current code:** `QCMask` (`histox/slide/utils.py`), `Heatmap.predictions`
(`histox/heatmap.py`), `cellseg.Segmentation` masks, `SaliencyMap`
(`histox/grad`), and MIL attention arrays.

**Proposal:** one type for any array laid over a slide.

```python
@dataclass
class SpatialMap:
    values: np.ndarray          # H x W or H x W x C
    origin: tuple               # level-0 (x, y) of values[0, 0]  (P2)
    stride: float               # level-0 pixels per cell
    slide: str
    mpp: float
    kind: str                   # "mask" | "labels" | "scores"
    channels: Optional[list]    # e.g. class names

    def value_at(self, xy): ...
    def resample(self, stride): ...
    def save(self, path): ...   # .npz with metadata
    @classmethod
    def load(cls, path): ...
```

Existing classes gain `to_spatial_map()` (and `from_spatial_map()` where
useful). Overlay and plotting code moves onto `SpatialMap`, so Studio and
the static plots share one implementation.

### P6. Transform protocols

**Current code:** QC classes (`histox/slide/qc/`), `StainNormalizer`
(`histox/norm/`), and augmentations (`histox/io/torch/augment.py`).

**Proposal:** keep the three families, since they act at different levels,
and give them two shared protocols plus a common namespace:

- `SlideTransform`: `__call__(wsi) -> SpatialMap` (kind `"mask"`). QC algorithms implement it; the boolean array returned today stays available.
- `TileTransform`: `__call__(tile: uint8 CHW tensor) -> tensor`, torchvision-compatible, so it composes with `torchvision.transforms.Compose`. Augmentations already fit. Stain normalizers expose `as_transform()`.
- Every transform implements `config() -> dict`, so a preprocessing pipeline can be recorded in `Bag.config` and `RunResult.metadata`.
- New namespace `histox.transforms` re-exports all of them. Normalizers and QC methods register in P1's registry (kinds `normalizer`, `qc`).

### P7. Model protocols and `ModelCard`

**Current code:**
- Tile encoders: `BaseFeatureExtractor` subclasses, which wrap a model in `.model` and dispatch `__call__` on a WSI, a TFRecord path, or a tensor. `license` and `citation` are class attributes.
- MIL models: `nn.Module`s with the three `forward()` signatures listed in section 2.2.
- Configs: `MILModelConfig` and `TrainerConfig` in `histox/mil/_params.py`; `ModelParams` in `histox/model/`.

**Proposal:**

- `ModelCard`: name, version, license, citation, input `TileSpec` (tile size and resolution the model expects), normalization, output dimension. Registries (P1) store one per entry.
- `TileEncoder`: the extractor exposes its underlying `nn.Module` as `.module` and a `ModelCard` as `.card`. The current wrapper behavior (WSI in, feature grid out; TFRecord in, features plus coordinates out) is unchanged. A WSI input can also return a `Bag`.
- `SlideAggregator`: one MIL calling convention, `forward(bags: Tensor[B, N, F], lens: Tensor[B]) -> Tensor[B, out]`, with optional `return_attention=True`. Multimodal models accept a sequence of `(bags, lens)` pairs. `TransMIL` gains a `lens` argument (ignored, or used for masking). This lets `train_mil` drop its special cases.

**Open question:** should MIL attention be returned as a `SpatialMap` directly,
or as raw per-tile scores that callers convert with the bag's coordinates?

### P8. `Cohort`

**Current code:**
- `Project`: folder configuration.
- `Dataset`: annotations, sources, tile parameters, TFRecord paths, filters, and splits.
- `DataSpec`: semantic fields and modalities, with no data.
- Split functions in `histox/dataset.py`: `split_patients`, `split_patients_balanced`, `split_patients_preserved_site`.

**Proposal:** `Cohort` holds the annotation table (patient, slide, label fields),
resolves slide paths, and provides a `DataSpec`. It knows nothing about tile
sizes or storage. `Dataset` becomes "a `Cohort` plus a `TileSpec` plus storage";
its public methods are unchanged. Split functions move to `histox.splits` and
operate on a `Cohort`, guaranteeing patient-level splits. Survival labels are
declared as duration and event fields, which `SurvivalTask` already validates
through `required_fields`.

### P9. Tasks through `histox.workflows`

**Current code:** `Task`, `Method`, `Experiment` (stages `prepare`, `fit`,
`evaluate`, `predict`, `explain`), `RunResult`, `Artifact`. Registered so far:
`SurvivalTask` and `PS3Method`. Other training paths (`train_mil` in
`histox/mil/train/`, `Project.train` / `histox.model` trainers) bypass this contract.

**Proposal:** make `histox.workflows` the single task layer.
- Add `ClassificationTask` and `RegressionTask`, with default metrics from `histox.stats` (`ClassifierMetrics`, concordance).
- Wrap MIL training as `Method("mil")`, using P7's aggregator convention and P3's `Bag`.
- Wrap tile-level supervised training as `Method("tile_classifier")`.
- `RunResult.artifacts` carry `Bag`s, `SpatialMap`s, and model checkpoints with `ModelCard`s.

The existing functions stay as convenience entry points that build an
`Experiment` internally.

## 5. Extension points (reserved, not designed here)

These attach to the abstractions above and get their own RFCs:

| Area | Attaches to | Starting point in the code |
|---|---|---|
| Cell-level objects and cell graphs | P2 coordinates, P5 `SpatialMap` (labels), P8 `Cohort` | `histox/cellseg` `Segmentation` |
| Multiplex / non-RGB imaging | P4 `TileSpec` channels, slide backends | `histox/slide/backends/` |
| Slide registration (H&E ↔ IHC) | P2 coordinates | `histox/slide/utils.py` `Alignment` |
| Spatial transcriptomics | P2, P5, P8 (new modality in `DataSpec`) | none |
| Self-supervised and generative training | P1, P7 | `histox/simclr`, `histox/gan` |
| Feature interoperability (CLAM / TRIDENT h5) | P3 `Bag` | none |
| Multi-GPU feature extraction and MIL training | P7, P9 | `num_gpus` in `generate_feature_bags`; Lightning in `histox/segment` |
| De-identification of slide labels and metadata | slide backends | none |
| Retrieval | P3, P7 | `histox/experimental` `EmbeddingSearch` |
| Studio | public APIs only | `histox/studio` |

## 6. Stability and deprecation policy

Adapted from PyTorch's feature classification:

- **Stable:** covered by tests and documented. Breaking changes only after deprecation.
- **Beta:** API may still change, with a note in the changelog. New abstractions from this RFC start here.
- **Prototype:** experimental and may disappear. This covers `histox.experimental` and anything marked as prototype in its docstring.

Deprecation: a superseded API emits `HistoXDeprecationWarning` naming its
replacement, and stays for at least two minor releases (for example, deprecated
in `0.3`, removed no earlier than `0.5`). Every deprecation and removal is
listed in the changelog.

The documentation marks each public module with its level.

## 7. Implementation plan

Each step is one pull request, with tests, docs, and no breaking change.

1. Accept this RFC.
2. P1 registry; migrate the workflows registry, then the extractor and MIL registries.
3. P2 convention and `histox/geometry.py`.
4. P3 `Bag`, keeping today's file format.
5. P4 `TileSpec`.
6. P7 `ModelCard` and the MIL calling convention (depends on 2 and 5).
7. P5 `SpatialMap` with converters from `QCMask`, `Heatmap`, `Segmentation`, and attention.
8. P6 transform protocols and `histox.transforms`.
9. P8 `Cohort` and `histox.splits`.
10. P9 `ClassificationTask`, `Method("mil")`, `Method("tile_classifier")`.

Every step that touches `heatmap`, `segment`, `grad`, or MIL adds the tests
those modules lack today.

## 8. Decisions needed from maintainers

1. **TensorFlow backend.** The roadmap says "PyTorch first". Deprecate in `0.3` and remove in `0.5`? This affects the size of P6 and P7.
2. **Minimum Python version.** `setup.py` declares `>=3.7`, but CI tests only 3.9. Proposal: `>=3.10`, tested on 3.10–3.12.
3. **Extras packages.** Keep pointing at `slideflow-noncommercial` / `slideflow-gpl`, or publish HistoX-named packages?
4. **Naming.** Keep `WSI` (this RFC assumes so) rather than renaming it to `Slide`. Names for new modules: `histox.registry`, `histox.geometry`, `histox.transforms`, `histox.splits`; `Bag`, `SpatialMap`, `Cohort`, and `TileSpec` exported at top level.
5. **Bag precision.** Bags are stored as float32 today; extraction defaults to float16. Store float16 to halve disk use?
