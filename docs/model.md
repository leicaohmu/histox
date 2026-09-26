# histox.model

This module provides the `ModelParams` class to organize model and training
parameters/hyperparameters and assist with model building, as well as the `Trainer` class that
executes model training and evaluation. `RegressionTrainer` and `SurvivalTrainer`
are extensions of this class, supporting regression and Cox Proportional Hazards outcomes, respectively. The function
`build_trainer()` can choose and return the correct model instance based on the provided
hyperparameters.

Note

In order to support both Tensorflow and PyTorch backends, the `histox.model` module will import either
`histox.model.tensorflow` or `histox.model.torch` according to the currently active backend,
indicated by the environmental variable `HX_BACKEND`.

See [Training](training.html#training) for a detailed look at how to train models.

## Trainer

*class*histox.model.Trainer(*hp: [ModelParams](model_params.html#histox.ModelParams)*, *outdir: str*, *labels: Dict[str, Any]*, ***, *slide_input: Dict[str, Any] | None = None*, *name: str = 'Trainer'*, *feature_sizes: List[int] | None = None*, *feature_names: List[str] | None = None*, *outcome_names: List[str] | None = None*, *mixed_precision: bool = True*, *allow_tf32: bool = False*, *config: Dict[str, Any] | None = None*, *use_neptune: bool = False*, *neptune_api: str | None = None*, *neptune_workspace: str | None = None*, *load_method: str = 'weights'*, *custom_objects: Dict[str, Any] | None = None*, *device: str | None = None*, *transform: Callable | Dict[str, Callable] | None = None*, *pin_memory: bool = True*, *num_workers: int = 4*, *chunk_size: int = 8*)[[source]](_modules/histox/model/torch.html#Trainer)

Base trainer class containing functionality for model building, input
processing, training, and evaluation.

This base class requires categorical outcome(s). Additional outcome types
are supported by `histox.model.RegressionTrainer` and
`histox.model.SurvivalTrainer`.

Slide-level (e.g. clinical) features can be used as additional model input
by providing slide labels in the slide annotations dictionary, under
the key 'input'.

model.Trainer.load(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

model.Trainer.evaluate(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

model.Trainer.predict(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

model.Trainer.train(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

## RegressionTrainer

*class*histox.model.RegressionTrainer(**args*, ***kwargs*)[[source]](_modules/histox/model/torch.html#RegressionTrainer)

Extends the base `histox.model.Trainer` class to add support
for continuous outcomes. Requires that all outcomes be continuous, with appropriate
regression loss function. Uses R-squared as the evaluation metric, rather
than AUROC.

In this case, for the PyTorch backend, the continuous outcomes support is
already baked into the base Trainer class, so no additional modifications
are required. This class is written to inherit the Trainer class without
modification to maintain consistency with the Tensorflow backend.

## SurvivalTrainer

*class*histox.model.SurvivalTrainer(**args*, ***kwargs*)[[source]](_modules/histox/model/torch.html#SurvivalTrainer)

Cox proportional hazards (CPH) models are not yet implemented, but are
planned for a future update.

## Features

*class*histox.model.Features(*path: str | None*, *layers: str | List[str] | None = 'postconv'*, ***, *include_preds: bool = False*, *mixed_precision: bool = True*, *channels_last: bool = True*, *device: torch.device | None = None*, *apply_softmax: bool | None = None*, *pooling: Any | None = None*, *load_method: str = 'weights'*)[[source]](_modules/histox/model/torch.html#Features)

Interface for obtaining predictions and features from intermediate layer
activations from HistoX models.

Use by calling on either a batch of images (returning outputs for a single
batch), or by calling on a `histox.WSI` object, which will
generate an array of spatially-mapped activations matching the slide.

Examples

*Calling on batch of images:*

```
interface = Features('/model/path', layers='postconv')
for image_batch in train_data:
 # Return shape: (batch_size, num_features)
 batch_features = interface(image_batch)
```

*Calling on a slide:*

```
slide = hx.slide.WSI(...)
interface = Features('/model/path', layers='postconv')
# Return shape:
# (slide.grid.shape[0], slide.grid.shape[1], num_features)
activations_grid = interface(slide)
```

Note

When this interface is called on a batch of images, no image processing
or stain normalization will be performed, as it is assumed that
normalization will occur during data loader image processing. When the
interface is called on a histox.WSI, the normalization strategy
will be read from the model configuration file, and normalization will
be performed on image tiles extracted from the WSI. If this interface
was created from an existing model and there is no model configuration
file to read, a histox.norm.StainNormalizer object may be passed
during initialization via the argument wsi_normalizer.

model.Features.from_model(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

model.Features.__call__(**args*, ***kwargs*)

Call self as a function.

## Other functions

histox.model.build_trainer(*hp: [ModelParams](model_params.html#histox.ModelParams)*, *outdir: str*, *labels: Dict[str, Any]*, ***kwargs*) → Trainer[[source]](_modules/histox/model.html#build_trainer)

From the given [`histox.ModelParams`](model_params.html#histox.ModelParams) object, returns
the appropriate instance of `histox.model.Trainer`.

Parameters:

- **hp** ([`histox.ModelParams`](model_params.html#histox.ModelParams)) - ModelParams object.
- **outdir** (*str*) - Path for event logs and checkpoints.
- **labels** (*dict*) - Dict mapping slide names to outcome labels (int or
float format).

Keyword Arguments:

- **slide_input** (*dict*) - Dict mapping slide names to additional
slide-level input, concatenated after post-conv.
- **name** (*str**,**optional*) - Optional name describing the model, used for
model saving. Defaults to 'Trainer'.
- **feature_sizes** (*list**,**optional*) - List of sizes of input features.
Required if providing additional input features as input to
the model.
- **feature_names** (*list**,**optional*) - List of names for input features.
Used when permuting feature importance.
- **outcome_names** (*list**,**optional*) - Name of each outcome. Defaults to
"Outcome {X}" for each outcome.
- **mixed_precision** (*bool**,**optional*) - Use FP16 mixed precision (rather
than FP32). Defaults to True.
- **allow_tf32** (*bool*) - Allow internal use of Tensorfloat-32 format.
Defaults to False.
- **config** (*dict**,**optional*) - Training configuration dictionary, used
for logging. Defaults to None.
- **use_neptune** (*bool**,**optional*) - Use Neptune API logging.
Defaults to False
- **neptune_api** (*str**,**optional*) - Neptune API token, used for logging.
Defaults to None.
- **neptune_workspace** (*str**,**optional*) - Neptune workspace.
Defaults to None.
- **load_method** (*str*) - Either 'full' or 'weights'. Method to use
when loading a Tensorflow model. If 'full', loads the model with
`tf.keras.models.load_model()`. If 'weights', will read the
`params.json` configuration file, build the model architecture,
and then load weights from the given model with
`Model.load_weights()`. Loading with 'full' may improve
compatibility across HistoX versions. Loading with 'weights'
may improve compatibility across hardware & environments.
- **custom_objects** (*dict**,**Optional*) - Dictionary mapping names
(strings) to custom classes or functions. Defaults to None.
- **num_workers** (*int*) - Number of dataloader workers. Only used for PyTorch.
Defaults to 4.

histox.model.build_feature_extractor(*name: str*, *backend: str | None = None*, ***kwargs*) → BaseFeatureExtractor[[source]](_modules/histox/model/extractors/_factory.html#build_feature_extractor)

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

histox.model.list_extractors()[[source]](_modules/histox/model/extractors/_registry.html#list_extractors)

Return a list of all available feature extractors.

Scans both the Tensorflow and PyTorch extractor registries and returns
a deduplicated list of all registered extractor names.

Extractors are registered automatically via the `@register_tf` and
`@register_torch` decorators when the corresponding backend module
(`histox.model.tensorflow` or `histox.model.torch`) is imported.
Therefore, the returned list reflects only the extractors available
under the **currently active backend**.

Returns:

list[str], a deduplicated list of extractor names. Note that the

order of the returned list is not guaranteed.

Return type:

extractor_lists

See also

- list_tensorflow_extractors: List only Tensorflow extractors.
- list_torch_extractors: List only PyTorch extractors.
- build_feature_extractor: Build an extractor by name.

Examples

```python
import histox as hx

# List all available extractors
extractors = hx.model.list_extractors()
print(extractors)
# ['vgg19_imagenet', 'resnet101_v2_imagenet', 'virchow', ...]
```

histox.model.load(*path: str*) → torch.nn.Module[[source]](_modules/histox/model/torch.html#load)

Load a model trained with HistoX.

Parameters:

**path** (*str*) - Path to saved model. Must be a model trained in HistoX.

Returns:

Loaded model.

Return type:

torch.nn.Module

histox.model.is_tensorflow_model(*arg: Any*) → bool[[source]](_modules/histox/model.html#is_tensorflow_model)

Checks if the object is a Tensorflow Model or path to Tensorflow model.

histox.model.is_tensorflow_tensor(*arg: Any*) → bool[[source]](_modules/histox/model.html#is_tensorflow_tensor)

Checks if the given object is a Tensorflow Tensor.

histox.model.is_torch_model(*arg: Any*) → bool[[source]](_modules/histox/model.html#is_torch_model)

Checks if the object is a PyTorch Module or path to PyTorch model.

histox.model.is_torch_tensor(*arg: Any*) → bool[[source]](_modules/histox/model.html#is_torch_tensor)

Checks if the given object is a Tensorflow Tensor.

histox.model.read_hp_sweep(*filename: str*, *models: List[str] | None = None*) → Dict[str, [ModelParams](model_params.html#histox.ModelParams)][[source]](_modules/histox/model.html#read_hp_sweep)

Organizes a list of hyperparameters ojects and associated models names.

Parameters:

- **filename** (*str*) - Path to hyperparameter sweep JSON file.
- **models** (*list**(**str**)*) - List of model names. Defaults to None.
If not supplied, returns all valid models from batch file.

Returns:

List of (Hyperparameter, model_name) for each HP combination

histox.model.rebuild_extractor(*bags_or_model: str*, *allow_errors: bool = False*, *native_normalizer: bool = True*) → Tuple[BaseFeatureExtractor | None, [StainNormalizer](norm.html#histox.norm.StainNormalizer) | None][[source]](_modules/histox/model/extractors/_factory.html#rebuild_extractor)

Recreate the extractor used to generate features stored in bags.

Parameters:

- **bags_or_model** (*str*) - Either a path to directory containing feature bags,
or a path to a trained MIL model. If a path to a trained MIL model,
the extractor used to generate features will be recreated.
- **allow_errors** (*bool*) - If True, return None if the extractor
cannot be rebuilt. If False, raise an error. Defaults to False.
- **native_normalizer** (*bool**,**optional*) - Whether to use PyTorch/Tensorflow-native
stain normalization, if applicable. If False, will use the OpenCV/Numpy
implementations. Defaults to True.

Returns:

Extractor function, or None if `allow_errors` is

True and the extractor cannot be rebuilt.

Optional[StainNormalizer]: Stain normalizer used when generating

feature bags, or None if no stain normalization was used.

Return type:

Optional[BaseFeatureExtractor]