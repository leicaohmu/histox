# RFC 0002: From application to framework

| | |
|---|---|
| Status | Draft |
| Author | leicaohmu |
| Created | 2026-11-25 |
| Applies to | `0.4` → `1.0` |
| Related | [RFC 0001](0001-core-abstractions.md), [`ROADMAP.md`](../ROADMAP.md) |

## 1. Summary

HistoX aims to be for computational pathology what PyTorch is for deep
learning. PyTorch does not solve any particular task. It provides the
primitives every deep-learning researcher needs (`Tensor`, autograd, GPU
execution) and a few standard interfaces (`Dataset`, `nn.Module`). Methods,
papers, and other libraries are then written on top of it, so it becomes a
common language for the field.

HistoX inherits the opposite design from Slideflow. It is an application
centered on `Project`, where users follow a fixed pipeline and the library
trains for them. RFC 0001 built the first half of a framework: pathology
primitives (`WSI`, `TileSpec`, `Bag`, `Cohort`, `SpatialMap`) and one registry.
This RFC proposes the second half:

- **Standard base classes** that users subclass: datasets, tile encoders, slide aggregators, slide readers, and transforms.
- **User-owned training loops.** Any model that follows the base classes can be trained with a plain PyTorch loop, and is accepted unchanged by HistoX's trainers, evaluation, heatmaps, and workflows.
- **`Project` becomes a convenience layer.** It composes public building blocks and owns no capability of its own.

The proposals, in implementation order:

| # | Proposal | Kind of change |
|---|---|---|
| P1 | Layering rules | Architecture policy |
| P2 | `histox.data`: datasets and a sample convention | Unify 5 dataset families |
| P3 | `histox.nn`: `TileEncoder` and `Aggregator` base classes | Replace duck typing with base classes |
| P4 | `SlideReader` and transform base classes | Make slide I/O and preprocessing extensible |
| P5 | Training with user-owned loops; one checkpoint format | Open the training path |
| P6 | Downstream tools accept modules | Remove the `params.json` requirement |
| P7 | `Project` as a convenience layer | Rule for existing and new code |
| P8 | Reference implementations and an acceptance test | Prove the interfaces |
| P9 | Ecosystem hooks | Model hub and third-party packages |

## 2. Motivation

### 2.1 What "the PyTorch of computational pathology" requires

Three properties make PyTorch a framework rather than a tool, and HistoX needs
each of them:

1. **Domain primitives everyone needs.** For pathology these are gigapixel, multi-resolution, physically calibrated slides; coordinates and magnification; tiles and feature bags; patients, slides, and leakage-safe splits; and slide-aligned outputs. RFC 0001 provides these.
2. **Extension by subclassing a documented base class.** A user writes `class MyData(Dataset)` or `class MyNet(nn.Module)`, and every other part of the framework accepts the result. Nothing has to be registered, configured, or wrapped.
3. **The user owns the workflow.** The framework provides building blocks. Higher-level trainers (PyTorch Lightning, Hugging Face `Trainer`) are optional layers built from the same public pieces.

### 2.2 Where the code stands today

The facts below are from `develop` after RFC 0001.

**Datasets: five families, none public as a base class.**

- `hx.Dataset` (`histox/dataset.py`, 4,577 lines, 76 public methods and properties) is a cohort and file manager. It filters slides, manages TFRecords, and extracts tiles. It has no `__getitem__` or `__len__`, so subclassing it cannot define custom data.
- Tile training data comes from `Dataset.torch()`, which interleaves TFRecords through `InterleaveIterator` and `IndexedInterleaver` (`histox/io/torch/`). Training a tile model without first exporting TFRecords requires `from_wsi=True` and a project dataset.
- Other `torch.utils.data.Dataset` subclasses live in four more places, each private to one feature, with different sample formats:

| Location | Classes | Yields |
|---|---|---|
| `histox/tfrecord/torch/dataset.py` | `TFRecordDataset`, `IndexedTFRecordDataset`, `MultiTFRecordDataset`, `IndexedMultiTFRecordDataset` | parsed TFRecord records |
| `histox/io/torch/` | `InterleaveIterator`, `StyleGAN2Interleaver`, `TileLabelInterleaver`, `IndexedInterleaver` | tuples: image, label, optional slide and location |
| `histox/mil/data.py` | `BagDataset`, `MultiBagDataset`, `MapDataset`, `EncodedDataset` | `(features, len)` tuples or fixed-size tensors |
| `histox/segment/data.py` | `ThumbMaskDataset`, `RandomCropDataset`, `BufferedMaskDataset`, `BufferedRandomCropDataset`, `TileMaskDataset` | dicts |
| `histox/model/extractors/_slide/_torch.py` | `_SlideIterator` | `(image, location)` tuples |

**Models: names and implicit attributes instead of base classes.**

- Tile models are built inside `Trainer` from `ModelParams.model`. A custom architecture is passed as a builder function, which the `model` setter stores in the class-level `ModelDict` under the fixed name `'custom'` (`histox/model/base.py`). This works for one run, but the dict is shared by all `ModelParams` instances, two custom models overwrite each other, and the saved `params.json` records only `"custom"`, so the model cannot be rebuilt from disk without the original function. An `nn.Module` instance cannot be passed.
- MIL models can be custom: `mil_config(model=Callable)` accepts a class that is called as `model_fn(n_in, n_out)`. Behavior then depends on optional attributes that are checked with `hasattr`/`getattr`: `use_lens`, `is_multimodal`, `calculate_attention`, `get_last_layer_activations` (`histox/mil/_params.py`, `histox/mil/_aggregator.py`). These are undocumented and invisible to type checkers.
- MIL `forward` signatures still differ: `Attention_MIL.forward(bags, lens, *, return_attention=False, uq=False, ...)` and `MultiModal_Attention_MIL.forward(*bags_and_lens)`. RFC 0001's `SlideAggregator` hides these differences behind a wrapper; the models themselves were not changed.
- Tile encoders subclass `BaseFeatureExtractor` (`histox/model/base.py`). Its constructor still takes `backend='tensorflow'|'torch'`, preprocessing is a `preprocess_kwargs` dict, and the class is not an `nn.Module`. `TorchFeatureExtractor.__call__` dispatches on its input type: a `WSI`, a `.tfrecords` path, or a tensor batch.
- Reusable pieces sit inside individual models. For example, `CoxPHLoss` is defined in `histox/mil/models/ps3.py`.

**Training: owned by the library.**

- Tile training runs in `histox.model.torch.Trainer`, whose `train()` takes `hx.Dataset` objects (`train_dts`, `val_dts`).
- MIL training runs through fastai only: `TrainerConfig` has `tag = 'fastai'`, and `mil_config` says that only `'fastai'` is available.
- Saved models are found through a `params.json` next to the weights (106 lines in the package outside tests). `Heatmap(slide, model: str)` reads it through `hx.util.get_model_config`. A model trained outside `Project` therefore cannot use the standard heatmap path; only the lower-level `ModelHeatmap` subclass accepts an `nn.Module`.

**Slide I/O and preprocessing: closed sets.**

- Slide readers are chosen by `if hx.slide_backend() == 'libvips' ... elif ... 'cucim'` chains in `histox/slide/backends/__init__.py`, with DICOM detected by file name. There is no reader base class: `_VIPSReader` and `_cuCIMReader` are private, and `hx.slide_backend()` is compared in 43 places. A new format requires editing HistoX.
- QC methods (`Otsu`, `Gaussian`, `StridedDL`, ...) share no base class; `WSI.qc()` accepts any callable that takes a `WSI`. Stain normalizers have a facade (`StainNormalizer`) but no base class for new methods.

**`Project` holds most capabilities.** `histox/project.py` has 3,973 lines and
48 public methods and properties, including `train`, `train_mil`, `evaluate`, `evaluate_mil`,
`extract_tiles`, `generate_feature_bags`, `generate_heatmaps`, `train_simclr`,
and `gan_train`. The Quickstart and the training tutorials start from it.

## 3. Goals and non-goals

**Goals**

- Every extension point has a documented base class, in the style of `Dataset` and `nn.Module`.
- Any method can be implemented, trained, evaluated, and visualized using only public building blocks, in a plain PyTorch loop.
- Existing code keeps working. Existing classes become implementations of the new base classes or thin adapters.

**Non-goals**

- Replacing PyTorch. HistoX builds on `torch.Tensor`, autograd, `nn.Module`, and `DataLoader`; it adds pathology-specific layers only.
- Its own trainer framework. HistoX keeps its trainers, but they become optional conveniences.
- TensorFlow variants of anything in this RFC (RFC 0001, decision 1).
- Studio changes. Studio keeps using public APIs only.

## 4. Proposals

### P1. Layering rules

HistoX is organized in four layers. A layer may import only from layers below it.

| Layer | Contents | PyTorch analogue |
|---|---|---|
| 1. Primitives | `WSI`, `TileSpec`, `geometry`, `Bag`, `Cohort`, `splits`, `SpatialMap`, `registry` | `torch.Tensor` |
| 2. Building blocks | `histox.data`, `histox.nn`, `histox.transforms`, slide readers, `histox.models` | `torch.utils.data`, `torch.nn`, `torchvision.transforms`, `torchvision.models` |
| 3. Workflows | `histox.workflows` (`Experiment`, tasks, methods), built-in trainers, `Project` | PyTorch Lightning, torchvision reference scripts |
| 4. Ecosystem | `histox.hub`, plugins, `histox-contrib`, third-party packages | Hugging Face Hub, timm |

Two rules apply to all new code:

1. **No private paths.** Anything a workflow does must be possible with public layer 1–2 APIs. If `Project` needs a capability, the capability is added to a lower layer first.
2. **Accept base classes, not names.** Functions that take a model, dataset, reader, or transform accept an instance of the base class. Names from the registry stay as shortcuts that resolve to instances.

### P2. `histox.data`: datasets and a sample convention

**Sample convention.** Every HistoX dataset returns a `dict`. Standard keys:

| Key | Type | Meaning |
|---|---|---|
| `image` | uint8 tensor `[C, H, W]` | One tile; transforms convert it to float |
| `features` | float tensor `[N, F]` | One bag |
| `coords` | int tensor `[2]` or `[N, 2]` | Tile centers in level-0 pixels (RFC 0001, P2) |
| `slide`, `patient` | `str` | Identifiers |
| `target` | tensor | Label or regression or survival target, as defined by the task |
| `mask` | bool tensor `[N]` | Valid instances in a padded bag |

Extra keys are allowed. Datasets for other modalities (for example gene
expression for PS3) add their own keys.

**Base classes**, all `torch.utils.data.Dataset` subclasses:

- `TileDataset`: one item per tile. Built from a `Cohort` plus either a `TileSpec` (tiles read from slides on the fly) or a `TileStore`.
- `BagDataset`: one item per slide or patient. Built from a `Cohort` and a bag directory (`Bag`, RFC 0001 P3). Replaces `histox/mil/data.py:BagDataset` and `MultiBagDataset`. A `collate_bags` function pads bags and returns `mask`.
- `SlideDataset`: one item per slide (`WSI` plus metadata), for slide-level processing such as QC and feature extraction.

Users customize by subclassing and overriding `__getitem__`, or a smaller hook (`load_sample`, `make_target`).

**`TileStore` protocol** for extracted tiles: `__len__`, `__getitem__(i) ->
sample`, `slides()`. The existing TFRecord readers become the first
implementation; a plain directory store can follow. TFRecords stop being a
requirement for training.

**Migration:** the five families in section 2.2 become implementations or
adapters. `Dataset.torch()` keeps its signature and builds on `TileDataset`
internally. `hx.Dataset` stays as the project-level cohort manager.

### P3. `histox.nn`: `TileEncoder` and `Aggregator`

Two base classes, both `nn.Module` subclasses:

```python
class TileEncoder(nn.Module):
    card: ModelCard                      # RFC 0001, P7
    transform: TileTransform             # uint8 [B, C, H, W] -> model input
    num_features: int
    def forward(self, images: Tensor) -> Tensor:   # [B, C, H, W] -> [B, F]
        ...

class Aggregator(nn.Module):
    def forward(self, features: Tensor, mask: Tensor | None = None,
                coords: Tensor | None = None) -> AggregatorOutput:
        ...                              # features [B, N, F], mask [B, N]

@dataclass
class AggregatorOutput:
    logits: Tensor                       # [B, out]
    attention: Tensor | None = None      # [B, N], or [B, heads, N]
    embedding: Tensor | None = None      # [B, D], slide-level representation
```

- `AggregatorOutput` replaces the implicit `return_attention`, `calculate_attention`, and `get_last_layer_activations` hooks. A model that has no attention leaves it `None`.
- `mask` replaces `lens`. `SlideAggregator` converts `lens` to a mask for models that still take `lens`, so existing models keep working during migration.
- Multimodal aggregators take one input per modality as keyword arguments (for example `forward(features, mask, *, expression)`), replacing `forward(*bags_and_lens)`.
- Tile encoders lose the input-type dispatch. Extracting from a `WSI` or a `TileStore` becomes a function, `hx.features.extract(encoder, source) -> Bag`. `BaseFeatureExtractor` and `TorchFeatureExtractor` remain as adapters around a `TileEncoder`, and their `backend` argument is deprecated.
- `histox.nn` also holds reusable layers and losses: gated attention pooling, masked mean and max pooling, and `CoxPHLoss` (moved from `histox/mil/models/ps3.py`).
- `histox.models` holds reference implementations (P8), as `torchvision.models` does for `torch.nn`.

### P4. `SlideReader` and transform base classes

**`SlideReader`** is the base class for slide formats:

- Properties: `dimensions`, `levels` (dimensions and downsample per level), `mpp`, `properties`.
- Methods: `read_region(xy_level0, level, size) -> uint8 array`, `thumbnail(max_size)`, and a class method `can_read(path) -> bool`.
- Readers register under the registry kind `slide_reader`, with a priority. `WSI` picks the first registered reader that can read the file and honors a user's explicit choice.
- `_VIPSReader`, `_cuCIMReader`, and the DICOM reader become implementations. The `hx.slide_backend()` comparisons move into the readers. `hx.slide_backend()` stays as the default preference.

**Transforms.** RFC 0001 (P6) defined protocols and the `histox.transforms`
namespace. This RFC adds the base classes behind them:

- `SlideTransform.__call__(wsi) -> SpatialMap`, used by QC. Plain callables that return a boolean mask keep working.
- `StainNormalizer` base class with `fit(target)`, `__call__(tile)`, and `config()`, so new methods plug into `TileNormalizer` and the `normalizer` registry.

### P5. Training with user-owned loops; one checkpoint format

**Plain loops first.** With P2 and P3, every trainable model in HistoX can be
trained like any PyTorch model:

```python
loader = DataLoader(hx.data.BagDataset(cohort, "bags/", target="subtype"),
                    batch_size=8, shuffle=True, collate_fn=hx.data.collate_bags)
model = MyAggregator(in_dim=768, n_classes=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
for batch in loader:
    out = model(batch["features"], batch["mask"])
    loss = F.cross_entropy(out.logits, batch["target"])
    loss.backward(); optimizer.step(); optimizer.zero_grad()
```

The documentation teaches this form first.

**Built-in trainers accept modules.**

- `ModelParams(model=...)` also accepts an `nn.Module` instance. Custom builders are kept on the instance instead of the class-level `ModelDict`, and the checkpoint format below records how to rebuild them.
- MIL trainers accept any `Aggregator`. fastai stays the default MIL trainer, but core HistoX no longer requires it. A minimal plain-PyTorch MIL trainer is added for environments without fastai.
- Workflow methods (`MILMethod`, `TileClassifierMethod`) accept `model=` as an instance.

**One checkpoint format.** `hx.save(model, path, *, card=None, tile_spec=None,
config=None)` writes the `state_dict`, the import path of the module class, its
constructor arguments, the `ModelCard`, the `TileSpec`, and HistoX and PyTorch
versions. `hx.load(path)` rebuilds the module. The weights use `safetensors`,
already used by `histox/workflows/pretrained.py`. Models saved by `Project`
with `params.json` keep loading through an adapter.

### P6. Downstream tools accept modules

Evaluation, heatmaps, mosaics, and attention maps accept base-class instances
directly, in addition to saved model paths:

- `Heatmap(slide, model)` accepts a `TileEncoder` with a head, or any `nn.Module` plus a `TileSpec`.
- `SpatialMap.from_bag(bag, output.attention)` turns aggregator attention into a map.
- MIL evaluation (`eval_mil`, `predict_from_bags`) accepts an `Aggregator` instance, as `SlideAggregator` already does.

### P7. `Project` as a convenience layer

- **Rule:** each `Project` method is a short composition of public layer 1–3 APIs. New features land in lower layers first; `Project` may then expose them.
- Existing `Project` methods are migrated one at a time as the building blocks they need arrive. Signatures and outputs stay the same and are checked against the current behavior on real data, as for RFC 0001.
- `hx.Dataset` is gradually reduced to project-level concerns: sources, filters, and extraction. Labels and splits are already in `Cohort` and `histox.splits`.

### P8. Reference implementations and an acceptance test

The interfaces are proven by rewriting existing methods with them:

- `histox.models`: `AttentionMIL` (from `Attention_MIL`), `TransMIL`, the PS3 survival model, and a tile classifier (encoder plus head). Each uses only public APIs.
- **Acceptance test (CI):** one file, `docs/examples/custom_method.py`, defines a new aggregator and a new dataset subclass. It trains with a plain loop, evaluates with `ClassificationTask`, saves and loads with `hx.save`/`hx.load`, and builds an attention `SpatialMap`, without touching private modules or `Project`. If a future change breaks this file, the framework has regressed.
- Documentation: the Concepts page is rewritten around the four layers, with two new tutorials, "Write your own dataset" and "Write your own aggregator".

### P9. Ecosystem hooks

- `histox.hub.load(name) -> TileEncoder`: loads registered encoders with their `ModelCard`. Hugging Face identifiers are accepted when the card allows it. License terms on the card are shown before download, following ROADMAP principle 5.
- Third-party packages register datasets, encoders, aggregators, readers, and transforms through the existing `histox.plugins` entry-point group (`histox/plugin.py`). The guide for writing such a package moves from "Plugins" to "Build on HistoX".
- The Methods hub in the docs lists external packages built on HistoX next to built-in methods.

## 5. Compatibility

- Everything in this RFC is added in `0.4` with **beta** status (RFC 0001, section 6). It becomes stable no earlier than `1.0`.
- Existing classes stay importable. Replaced behavior is deprecated with `HistoXDeprecationWarning` (introduced in `0.3` with the TensorFlow deprecation), naming the replacement, and kept for at least two minor releases. Candidates: the `backend` argument of `BaseFeatureExtractor`, MIL models without `AggregatorOutput`, `lens`-only aggregators, and `mil.data.BagDataset` (moved to `histox.data`).
- Each migration step is checked on real TCGA data against the previous behavior, as in RFC 0001.

## 6. Implementation plan

Each step is one pull request with tests and docs.

1. Accept this RFC.
2. P2 sample convention, `histox.data.BagDataset`, and `collate_bags`.
3. P3 `Aggregator` and `AggregatorOutput`; port `Attention_MIL` and `TransMIL`; `SlideAggregator` accepts both styles.
4. P3 `TileEncoder` and `hx.features.extract`; port the PyTorch extractors; move `CoxPHLoss` and pooling layers to `histox.nn`.
5. P2 `TileDataset`, `SlideDataset`, and `TileStore` with the TFRecord implementation.
6. P5 `hx.save`/`hx.load`.
7. P5 trainers and workflow methods accept modules; `ModelParams(model=module)`; plain-PyTorch MIL trainer.
8. P6 heatmaps and evaluation accept modules.
9. P4 `SlideReader` and the reader registry.
10. P4 `SlideTransform` and `StainNormalizer` base classes.
11. P8 reference implementations, acceptance example in CI, Concepts page and tutorials.
12. P9 `histox.hub` and the "Build on HistoX" guide.

P7 runs alongside: each step also moves the matching `Project` methods onto the
new blocks.

## 7. Open questions

Each has a proposed answer, to be confirmed in review.

1. **Sample format: `dict` or a dataclass?** Proposed: `dict`. It works with the default `collate_fn` and allows extra keys, as torchvision's v2 transforms and Hugging Face datasets do.
2. **`mask` or `lens` for padded bags?** Proposed: a boolean `mask`. It is what attention layers consume, and it also covers non-contiguous padding. `lens` is converted automatically during migration.
3. **Should core HistoX depend on fastai or Lightning?** Proposed: no. Both stay optional extras for the trainers that use them.
4. **Module names.** Proposed: `histox.nn` for base classes, layers, and losses; `histox.models` for complete reference models; `histox.data` for datasets. This mirrors `torch.nn` and `torchvision.models`. `histox.data` already exists for public-dataset discovery, and the dataset classes are added next to it.
5. **Where does `Aggregator` attention live for multi-head models?** Proposed: `attention` may be `[B, N]` or `[B, heads, N]`; `SpatialMap.from_bag` takes a `head` argument.
