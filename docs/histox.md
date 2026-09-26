# histox

histox.about(*console=None*) → None[[source]](_modules/histox/util.html#about)

Print a summary of the histox version and active backends.

Display a styled panel in the terminal containing the HistoX ASCII logo,
current version, deep learning backend, slide reading backend, documentation
link, and Github repository link.

Parameters:

**console** (*rich.console.Console**,**optional*) - An existing Rich Console instance to print to.
If `None`, a new Console will be created. Defaults to `None`.

Examples

```
>>> import histox as hx
>>> hx.about()
 ╭─────────────────────────────────────────────────────╮
 │ │
 │ _ _ _ _ __ __ │
 │ | | | (_)___| |_ ___ \ \/ / │
 │ | |_| | / __| __/ _ \ \ / │
 │ | _ | \__ \ || (_) |/ \ │
 │ |_| |_|_|___/\__\___//_/\_\ │
 │ │
 │ 🔬 Version 0.2.0 │
 │ 🔧 Backend torch │
 │ 🎨 Slide cucim │
 │ 🌐 Docs lei-lab-site.pages.dev/api │
 | 🐙 Github https://github.com/leicaohmu/histox |
 │ │
 ╰─────────── Computational Pathology Toolkit ─────────╯
```

histox.build_feature_extractor(*name: str*, *backend: str | None = None*, ***kwargs*) → BaseFeatureExtractor[[source]](_modules/histox/model/extractors/_factory.html#build_feature_extractor)

Build a feature extractor.

The returned feature extractor is a callable object, which returns
features (often layer activations) for either a batch of images or a
`histox.WSI` object.

If generating features for a batch of images, images are expected to be in
`(B, W, H, C)` format, non-standardized (scaled 0-255), with dtype
`uint8`. The feature extractor performs all needed preprocessing
(normalization, resizing, channel reordering) on the fly.

If generating features for a slide, the slide is expected to be a
`histox.WSI` object. The feature extractor will generate features
for each tile in the slide, returning a numpy array of shape `(W, H, F)`,
where `F` is the number of features.

Parameters:

- **name** (*str*) - Name of the feature extractor to build, or a path to a saved
histox model (Tensorflow or PyTorch). When a model path is
provided, a `Features` (or `UncertaintyInterface` if UQ is
enabled) object is returned directly from the saved model.
Available named extractors can be listed with [histox.model.list_extractors][].
- **backend** (*str**,**optional*) - Deep learning backend to use, one of `{'tensorflow', 'torch'}`.
If `None`, the backend is selected automatically based on the
`name` argument (see Notes). Defaults to `None`.
- ****kwargs** (*Any*) - 

Additional keyword arguments passed to the feature extractor factory
function. Common options include:

- **tile_px** (*int*) - Tile size (input image size), in pixels.
Required for ImageNet-pretrained models (e.g. `resnet50_imagenet`).
- **layers** (*str or list of str*) - Layer name(s) at which to
intercept activations during the forward pass. A `forward hook`
is registered on each specified layer; activations are captured
automatically at inference time and returned as the feature output.
The special value `'postconv'` registers a hook on the
post-convolutional layer, which is predefined for each supported
architecture (e.g. `avgpool` for ResNet, outputting a feature
vector of shape `(B, 2048)` for ResNet50). Any other string is
resolved by name against the model's module tree via
[histox.model.torch_utils.get_module_by_name][]; use
`print(extractor.ftrs._model)` to inspect available layer names.
When multiple layers are specified as a list, their outputs are
concatenated along the feature dimension, and `num_features`
equals the sum of all individual layer output sizes.
Defaults to `'postconv'`.
- **include_preds** (*bool*) - Whether to append model predictions
(final logits) to the output features. Defaults to `False`.
- **mixed_precision** (*bool*) - Use FP16 mixed precision.
Defaults to `True`.
- **pooling** (*str or Callable*) - Pooling applied to intermediate
feature maps before flattening. May be `'avg'`
(adaptive average pooling), `'max'` (adaptive max pooling),
or a custom callable that accepts and returns a `Tensor`.
Only applied to layers whose output has 4 dimensions `(B, C, H, W)`.
Has no effect on `'postconv'`, which uses its own pooling logic.
Defaults to `None` (no pooling).

The following preprocessing transform arguments are also accepted
for PyTorch ImageNet-pretrained extractors:

- **center_crop** (*int or bool*) - If an integer, center-crop
images to this size before inference. If `True`, crop to
`tile_px`. Defaults to `None` (no cropping).
- **resize** (*int or bool*) - If an integer, resize images to
this size before inference. If `True`, resize to `tile_px`.
Defaults to `None` (no resizing).
- **interpolation** (*str*) - Interpolation mode used when resizing.
One of `'bilinear'`, `'bicubic'`, `'nearest'`,
`'nearest_exact'`. Defaults to `'bilinear'`.
- **antialias** (*bool*) - Apply antialiasing filter when resizing.
Defaults to `False`.
- **norm_mean** (*tuple of float*) - Per-channel normalization mean
applied after scaling to `[0, 1]`.
Defaults to `(0.485, 0.456, 0.406)` (ImageNet mean).
- **norm_std** (*tuple of float*) - Per-channel normalization std
applied after scaling to `[0, 1]`.
Defaults to `(0.229, 0.224, 0.225)` (ImageNet std).

Returns:

A callable object which accepts either:

- A batch of images of shape `(B, W, H, C)`, dtype `uint8`,
and returns features of shape `(B, F)`, dtype `float32`.
- A `histox.WSI` object, and returns a spatially-mapped feature
array of shape `(W, H, F)`, dtype `float32`.

Return type:

BaseFeatureExtractor

Raises:

- **ValueError** - If `backend` is not one of `{'tensorflow', 'torch'}`.
- **InvalidFeatureExtractor** - If `name` is not a recognized feature extractor for the
 specified or active backend. If the extractor requires an optional
 package that is not installed, the error message will indicate
 the package name and the install command.

Notes

The `name` parameter supports two modes:

**1. Registered extractor name** (e.g. `'resnet50_imagenet'`,
`'ctranspath'`):
histox maintains an internal registry mapping extractor names to
their implementations, which may exist in one or both backends
(`'torch'` and `'tensorflow'`). When `name` is a registered
extractor name, the backend is resolved in the following order:

- Step 1: Use the manually specified `backend` argument, if provided.
- Step 2: If `backend=None` and `name` is only available in one
backend, that backend is used automatically.
- Step 3: If `backend=None` and `name` is available in both backends,
the currently active backend (`histox.backend()`) is used,
and a notice is logged.

For ImageNet-pretrained models, the `_imagenet` suffix in `name`
may be omitted; e.g. `'resnet50'` is automatically resolved to
`'resnet50_imagenet'`.

The models `'xception_imagenet'` and `'nasnet_large_imagenet'`
use a different normalization strategy (mean and std of
`[0.5, 0.5, 0.5]`) compared to the standard ImageNet normalization
used by other models. Custom `norm_mean` / `norm_std` kwargs are
ignored for these two models.

**2. Model path** (e.g. `'/path/to/saved/model'`):
When `name` is a path to a saved histox model, a `Features`
object (or `UncertaintyInterface` if UQ is enabled) is returned
directly, bypassing the extractor registry and backend resolution
entirely. The backend is inferred automatically from the saved
model format.

For PyTorch ImageNet-pretrained models, the underlying `nn.Module`
is accessible via `extractor.ftrs._model.model` after construction.
To inspect available layer names for use with the `layers` argument,
print the model wrapper:

```
print(extractor.ftrs._model)
```

Examples

Create an extractor using an ImageNet-pretrained ResNet50, extracting
from the default post-convolutional layer (`avgpool`, 2048-dim):

```
>>> import histox as hx
>>> extractor = hx.build_feature_extractor(
... 'resnet50_imagenet',
... tile_px=224
... )
>>> extractor.num_features # 2048
```

Equivalently, specify `'postconv'` explicitly:

```
>>> extractor = hx.build_feature_extractor(
... 'resnet50_imagenet',
... tile_px=224,
... layers='postconv'
... )
```

Extract activations from a specific intermediate layer by name
(use `print(extractor.ftrs._model)` to inspect available layer names):

```
>>> extractor = hx.build_feature_extractor(
... 'resnet50_imagenet',
... tile_px=224,
... layers='model.layer3'
... )
>>> extractor.num_features # 1024
```

Concatenate activations from multiple layers:

```
>>> extractor = hx.build_feature_extractor(
... 'resnet50_imagenet',
... tile_px=224,
... layers=['model.layer3', 'model.layer4']
... )
>>> extractor.num_features # 1024 + 2048 = 3072
```

Create a pretrained CTransPath extractor:

```
>>> extractor = hx.build_feature_extractor('ctranspath')
```

Load a feature extractor from a saved finetuned model:

```
>>> extractor = hx.build_feature_extractor('/path/to/saved/model')
```

Calculate features for an entire dataset:

```
>>> P = hx.load_project('/path/to/project')
>>> dataset = P.dataset(tile_px=224, tile_um=302)
>>> resnet = hx.build_feature_extractor('resnet50_imagenet', tile_px=224)
>>> features = hx.DatasetFeatures(resnet, dataset=dataset)
```

Generate a map of features across a whole-slide image:

```
>>> wsi = hx.WSI('/path/to/slide.svs', tile_px=224, tile_um=302)
>>> retccl = hx.build_feature_extractor('retccl', resize=True)
>>> features = retccl(wsi) # shape: (W, H, F)
```

histox.create_project(*root: str*, *cfg: Dict | str | None = None*, ***, *download: bool = False*, *md5: bool = False*, ***kwargs*) → [Project](project.html#histox.Project)

Create a project at the existing folder from a given configuration.

Supports both manual project creation via keyword arguments, and setting
up a project through a specified configuration. The configuration may be
a dictionary or a path to a JSON file containing a dictionary. It must
have the key 'annotations', which includes a path to an annotations file,
and may optionally have the following arguments:

- **name**: Name for the project and dataset.
- **rois**: Path to .tar.gz file containing compressed ROIs.
- **slides**: Path in which slides will be stored.
- **tiles**: Path in which extracted tiles will be stored.
- **tfrecords**: Path in which TFRecords will be stored.

```
import histox as hx

P = hx.create_project(
 root='path',
 annotations='file.csv',
 slides='path',
 tfrecords='path'
)
```

Annotations files are copied into the created project folder.

Alternatively, you can create a project using a prespecified configuration,
of which there are three available:

- `hx.project.LungAdenoSquam()`
- `hx.project.ThyroidBRS()`
- `hx.project.BreastER()`

When creating a project from a configuration, setting `download=True`
will download the annoations file and slides from The Cancer Genome Atlas
(TCGA).

```
import histox as hx

project = hx.create_project(
 root='path',
 cfg=hx.project.LungAdenoSquam(),
 download=True
)
```

Parameters:

- **root** (*str*) - Path at which the Project will be set up.
- **cfg** (*dict**,**str**,**optional*) - Path to configuration file (JSON), or a
dictionary, containing the key "annotations", and optionally with
the keys "name", "rois", "slides", "tiles", or "tfrecords".
Defaults to None.

Keyword Arguments:

- **download** (*bool*) - Download any missing slides from the Genomic Data
Commons (GDC) automatically, using slide names stored in the
annotations file.
- **md5** (*bool*) - Perform MD5 hash verification for all slides using
the GDC (TCGA) MD5 manifest, which will be downloaded.
- **name** (*str*) - Set the project name. This has higher priority than any
supplied configuration, which will be ignored.
- **slides** (*str*) - Set the destination folder for slides. This has higher
priority than any supplied configuration, which will be ignored.
- **tiles** (*str*) - Set the destination folder for tiles. This has higher
priority than any supplied configuration, which will be ignored.
- **tfrecords** (*str*) - Set the destination for TFRecords. This has higher
priority than any supplied configuration, which will be ignored.
- **roi_dest** (*str*) - Set the destination folder for ROIs.
- **dataset_config** (*str*) - Path to dataset configuration JSON file for the
project. Defaults to './datasets.json'.
- **sources** (*list**(**str**)*) - List of dataset sources to include in project.
Defaults to 'MyProject'.
- **models_dir** (*str*) - Path to directory in which to save models.
Defaults to './models'.
- **eval_dir** (*str*) - Path to directory in which to save evaluations.
Defaults to './eval'.

Returns:

histox.Project

histox.load_project(*root: str*, ***kwargs*) → [Project](project.html#histox.Project)

Load a project at the given root directory.

Parameters:

**root** (*str*) - Path to project.

Returns:

histox.Project

histox.getLoggingLevel()[[source]](_modules/histox/util.html#getLoggingLevel)

Return the current logging level.

histox.setLoggingLevel(*level*)[[source]](_modules/histox/util.html#setLoggingLevel)

Set the logging level.

Uses standard python logging levels:

- 50: CRITICAL
- 40: ERROR
- 30: WARNING
- 20: INFO
- 10: DEBUG
- 0: NOTSET

Parameters:

**level** (*int*) - Logging level numeric value.