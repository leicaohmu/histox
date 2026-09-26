# histox.mil

This submodule contains tools for multiple-instance learning (MIL) model training and evaluation. See [Multiple-Instance Learning (MIL)](mil.html#mil) for more information. A summary of the API is given below.

**Training:**

- `train_mil()`: Train an MIL model, using an MIL configuration, Datasets, and a directory of bags.
- `build_fastai_learner()`: Build and return the FastAI Learner, but do not execute training. Useful for customizing training.
- `build_multimodal_learner()`: Build and return a FastAI Learner designed for multi-modal/multi-magnification input.

**Evaluation/Inference:**

- `eval_mil()`: Evaluate an MIL model using a path to a saved model, a Dataset, and path to bags. Generates metrics.
- `predict_mil()`: Generate predictions from an MIL model and saved bags. Returns a pandas dataframe.
- `predict_multimodal_mil()`: Generate predictions from a multimodal MIL model. Returns a dataframe.
- `predict_slide()`: Generate MIL predictions for a single slide. Returns a 2D array of predictions and attention.
- `predict_from_bags()`: Low-level interface for generating predictions from a loaded MIL model and pre-loaded bag Tensors.
- `predict_from_multimodal_bags()`: Low-level interface for generating multimodal predictions from a loaded MIL model and bag Tensors.
- `get_mil_tile_predictions()`: Get tile-level predictions and attention from a saved MIL model for a given Dataset and saved bags.
- `generate_attention_heatmaps()`: Generate and save attention heatmaps.
- `generate_mil_features()`: Get last-layer activations from an MIL model. Returns an MILFeatures object.

## Main functions

histox.mil.mil_config(*model: str | Callable*, *trainer: str = 'fastai'*, ***kwargs*)[[source]](_modules/histox/mil/_params.html#mil_config)

Create a multiple-instance learning (MIL) training configuration.

All models by default are trained with the FastAI trainer. Additional
trainers and additional models can be installed with `histox-extras`.

Parameters:

- **model** (*str**,**Callable*) - Either the name of a model, or a custom torch
module. Valid model names include `"attention_mil"`,
`"transmil"`, and `"bistro.transformer"`.
- **trainer** (*str*) - Type of MIL trainer to use. Only 'fastai' is available,
unless additional trainers are installed.
- ****kwargs** - All additional keyword arguments are passed to
`histox.mil.TrainerConfig`

histox.mil.train_mil(*config: TrainerConfig*, *train_dataset: [Dataset](dataset.html#histox.Dataset)*, *val_dataset: [Dataset](dataset.html#histox.Dataset) | None*, *outcomes: str | List[str]*, *bags: str | List[str]*, ***, *outdir: str = 'mil'*, *exp_label: str | None = None*, ***kwargs*) → Learner[[source]](_modules/histox/mil/train.html#train_mil)

Train a multiple-instance learning (MIL) model.

This high-level trainer facilitates training from a given MIL configuration,
using Datasets as input and with input features taken from a given directory
of bags.

Parameters:

- **config** (`histox.mil.TrainerConfig`) - Trainer and model configuration.
- **train_dataset** ([`histox.Dataset`](dataset.html#histox.Dataset)) - Training dataset.
- **val_dataset** ([`histox.Dataset`](dataset.html#histox.Dataset)) - Validation dataset.
- **outcomes** (*str*) - Outcome column (annotation header) from which to
derive category labels.
- **bags** (*str*) - Either a path to directory with *.pt files, or a list
of paths to individual *.pt files. Each file should contain
exported feature vectors, with each file containing all tile
features for one patient.

Keyword Arguments:

- **outdir** (*str*) - Directory in which to save model and results.
- **exp_label** (*str*) - Experiment label, used for naming the subdirectory
in the `{project root}/mil` folder, where training history
and the model will be saved.
- **attention_heatmaps** (*bool*) - Generate attention heatmaps for slides.
Not available for multi-modal MIL models. Defaults to False.
- **interpolation** (*str**,**optional*) - Interpolation strategy for smoothing
attention heatmaps. Defaults to 'bicubic'.
- **cmap** (*str**,**optional*) - Matplotlib colormap for heatmap. Can be any
valid matplotlib colormap. Defaults to 'inferno'.
- **norm** (*str**,**optional*) - Normalization strategy for assigning heatmap
values to colors. Either 'two_slope', or any other valid value
for the `norm` argument of `matplotlib.pyplot.imshow`.
If 'two_slope', normalizes values less than 0 and greater than 0
separately. Defaults to None.

histox.mil.build_fastai_learner(*config: TrainerConfig*, *train_dataset: [Dataset](dataset.html#histox.Dataset)*, *val_dataset: [Dataset](dataset.html#histox.Dataset)*, *outcomes: str | List[str]*, *bags: str | ndarray | List[str]*, ***, *outdir: str = 'mil'*, *return_shape: bool = False*, ***kwargs*) → Learner[[source]](_modules/histox/mil/train.html#build_fastai_learner)

Build a FastAI Learner for training an MIL model.

Does not execute training. Useful for customizing a Learner object
prior to training.

Parameters:

- **train_dataset** ([`histox.Dataset`](dataset.html#histox.Dataset)) - Training dataset.
- **val_dataset** ([`histox.Dataset`](dataset.html#histox.Dataset)) - Validation dataset.
- **outcomes** (*str*) - Outcome column (annotation header) from which to
derive category labels.
- **bags** (*str*) - list of paths to individual *.pt files. Each file should
contain exported feature vectors, with each file containing all tile
features for one patient.

Keyword Arguments:

- **outdir** (*str*) - Directory in which to save model and results.
- **return_shape** (*bool*) - Return the input and output shapes of the model.
Defaults to False.
- **exp_label** (*str*) - Experiment label, used for naming the subdirectory
in the `outdir` folder, where training history
and the model will be saved.
- **lr** (*float*) - Learning rate, or maximum learning rate if
`fit_one_cycle=True`.
- **epochs** (*int*) - Maximum epochs.
- ****kwargs** - Additional keyword arguments to pass to the FastAI learner.

Returns:

fastai.learner.Learner, and optionally a tuple of input and output shapes
if `return_shape=True`.

histox.mil.build_multimodal_learner(*config: TrainerConfig*, *train_dataset: [Dataset](dataset.html#histox.Dataset)*, *val_dataset: [Dataset](dataset.html#histox.Dataset)*, *outcomes: str | List[str]*, *bags: ndarray | List[str]*, ***, *outdir: str = 'mil'*, *return_shape: bool = False*) → Learner[[source]](_modules/histox/mil/train.html#build_multimodal_learner)

Build a multi-magnification FastAI Learner for training an MIL model.

Does not execute training. Useful for customizing a Learner object
prior to training.

Parameters:

- **train_dataset** ([`histox.Dataset`](dataset.html#histox.Dataset)) - Training dataset.
- **val_dataset** ([`histox.Dataset`](dataset.html#histox.Dataset)) - Validation dataset.
- **outcomes** (*str*) - Outcome column (annotation header) from which to
derive category labels.
- **bags** (*list**(**str**)*) - List of bag directories containing *.pt files, one
directory for each mode.

Keyword Arguments:

- **outdir** (*str*) - Directory in which to save model and results.
- **return_shape** (*bool*) - Return the input and output shapes of the model.
Defaults to False.
- **exp_label** (*str*) - Experiment label, used for naming the subdirectory
in the `outdir` folder, where training history
and the model will be saved.
- **lr** (*float*) - Learning rate, or maximum learning rate if
`fit_one_cycle=True`.
- **epochs** (*int*) - Maximum epochs.
- ****kwargs** - Additional keyword arguments to pass to the FastAI learner.

Returns:

fastai.learner.Learner, and optionally a tuple of input and output shapes
if `return_shape=True`.

histox.mil.eval_mil(*weights: str*, *dataset: [Dataset](dataset.html#histox.Dataset)*, *outcomes: str | List[str]*, *bags: str | List[str]*, *config: TrainerConfig | None = None*, ***, *outdir: str = 'mil'*, *attention_heatmaps: bool = False*, *uq: bool = False*, *aggregation_level: str | None = None*, ***heatmap_kwargs*) → DataFrame[[source]](_modules/histox/mil/eval.html#eval_mil)

Evaluate a multiple-instance learning model.

Saves results for the evaluation in the target folder, including
predictions (parquet format), attention (Numpy format for each slide),
and attention heatmaps (if `attention_heatmaps=True`).

Logs classifier metrics (AUROC and AP) to the console.

Parameters:

- **weights** (*str*) - Path to model weights to load.
- **dataset** (*hx.Dataset*) - Dataset to evaluation.
- **outcomes** (*str**,**list**(**str**)*) - Outcomes.
- **bags** (*str**,**list**(**str**)*) - Path to bags, or list of bag file paths.
Each bag should contain PyTorch array of features from all tiles in
a slide, with the shape `(n_tiles, n_features)`.
- **config** (`histox.mil.TrainerConfig`) - Configuration for building model. If `weights` is a path to a
model directory, will attempt to read `mil_params.json` from this
location and load saved configuration. Defaults to None.

Keyword Arguments:

- **outdir** (*str*) - Path at which to save results.
- **attention_heatmaps** (*bool*) - Generate attention heatmaps for slides.
Not available for multi-modal MIL models. Defaults to False.
- **interpolation** (*str**,**optional*) - Interpolation strategy for smoothing
attention heatmaps. Defaults to 'bicubic'.
- **aggregation_level** (*str**,**optional*) - Aggregation level for predictions.
Either 'slide' or 'patient'. Defaults to None (uses the model
configuration).
- **cmap** (*str**,**optional*) - Matplotlib colormap for heatmap. Can be any
valid matplotlib colormap. Defaults to 'inferno'.
- **norm** (*str**,**optional*) - Normalization strategy for assigning heatmap
values to colors. Either 'two_slope', or any other valid value
for the `norm` argument of `matplotlib.pyplot.imshow`.
If 'two_slope', normalizes values less than 0 and greater than 0
separately. Defaults to None.

histox.mil.predict_mil(*model: str | Callable*, *dataset: [Dataset](dataset.html#histox.Dataset)*, *outcomes: str | List[str]*, *bags: str | ndarray | List[str]*, ***, *config: TrainerConfig | None = None*, *attention: bool = False*, *aggregation_level: str | None = None*, ***kwargs*) → DataFrame | Tuple[DataFrame, List[ndarray]][[source]](_modules/histox/mil/eval.html#predict_mil)

Generate predictions for a dataset from a saved MIL model.

Parameters:

- **model** (*torch.nn.Module*) - Model from which to generate predictions.
- **dataset** (*hx.Dataset*) - Dataset from which to generation predictions.
- **outcomes** (*str**,**list**(**str**)*) - Outcomes.
- **bags** (*str**,**list**(**str**)*) - Path to bags, or list of bag file paths.
Each bag should contain PyTorch array of features from all tiles in
a slide, with the shape `(n_tiles, n_features)`.

Keyword Arguments:

- **config** (`histox.mil.TrainerConfig`) - Configuration for the MIL model. Required if model is a loaded `torch.nn.Module`.
Defaults to None.
- **attention** (*bool*) - Whether to calculate attention scores. Defaults to False.
- **uq** (*bool*) - Whether to generate uncertainty estimates. Experimental. Defaults to False.
- **aggregation_level** (*str*) - Aggregation level for predictions. Either 'slide'
or 'patient'. Defaults to None.
- **attention_pooling** (*str*) - Attention pooling strategy. Either 'avg'
or 'max'. Defaults to None.

Returns:

Dataframe of predictions.

list(np.ndarray): Attention scores (if `attention=True`)

Return type:

pd.DataFrame

histox.mil.predict_multimodal_mil(*model: str | Callable*, *dataset: [Dataset](dataset.html#histox.Dataset)*, *outcomes: str | List[str]*, *bags: ndarray | List[List[str]]*, ***, *config: TrainerConfig | None = None*, *attention: bool = False*, *aggregation_level: str | None = None*, ***kwargs*) → DataFrame | Tuple[DataFrame, List[ndarray]][[source]](_modules/histox/mil/eval.html#predict_multimodal_mil)

Generate predictions for a dataset from a saved multimodal MIL model.

Parameters:

- **model** (*torch.nn.Module*) - Model from which to generate predictions.
- **dataset** (*hx.Dataset*) - Dataset from which to generation predictions.
- **outcomes** (*str**,**list**(**str**)*) - Outcomes.
- **bags** (*str**,**list**(**str**)*) - Path to bags, or list of bag file paths.
Each bag should contain PyTorch array of features from all tiles in
a slide, with the shape `(n_tiles, n_features)`.

Keyword Arguments:

- **config** (`histox.mil.TrainerConfig`) - Configuration for the MIL model. Required if model is a loaded `torch.nn.Module`.
Defaults to None.
- **attention** (*bool*) - Whether to calculate attention scores. Defaults to False.
- **uq** (*bool*) - Whether to generate uncertainty estimates. Defaults to False.
- **aggregation_level** (*str*) - Aggregation level for predictions. Either 'slide'
or 'patient'. Defaults to None.
- **attention_pooling** (*str*) - Attention pooling strategy. Either 'avg'
or 'max'. Defaults to None.

Returns:

Dataframe of predictions.

list(np.ndarray): Attention scores (if `attention=True`)

Return type:

pd.DataFrame

histox.mil.predict_from_bags(*model: torch.nn.Module*, *bags: ndarray | List[str]*, ***, *attention: bool = False*, *attention_pooling: str | None = None*, *use_lens: bool = False*, *device: Any | None = None*, *apply_softmax: bool | None = None*, *uq: bool = False*) → Tuple[ndarray, List[ndarray]][[source]](_modules/histox/mil/eval.html#predict_from_bags)

Generate MIL predictions for a list of bags.

Predictions are generated for each bag in the list one at a time, and not batched.

Parameters:

- **model** (*torch.nn.Module*) - Loaded PyTorch MIL model.
- **bags** (*np.ndarray**,**list**(**str**)*) - Bags to generate predictions for. Each bag
should contain PyTorch array of features from all tiles in a slide,
with the shape `(n_tiles, n_features)`.

Keyword Arguments:

- **attention** (*bool*) - Whether to calculate attention scores. Defaults to False.
- **attention_pooling** (*str**,**optional*) - Pooling strategy for attention scores.
Can be 'avg', 'max', or None. Defaults to None.
- **use_lens** (*bool*) - Whether to use the length of each bag as an additional
input to the model. Defaults to False.
- **device** (*str**,**optional*) - Device on which to run inference. Defaults to None.
- **apply_softmax** (*bool*) - Whether to apply softmax to the model output. Defaults
to True for categorical outcomes, False for continuous outcomes.
- **uq** (*bool*) - Whether to generate uncertainty estimates. Defaults to False.

Returns:

Predictions and attention scores.

Return type:

Tuple[np.ndarray, List[np.ndarray]]

histox.mil.predict_from_multimodal_bags(*model: torch.nn.Module*, *bags: List[ndarray] | List[List[str]]*, ***, *attention: bool = True*, *attention_pooling: str | None = None*, *use_lens: bool = True*, *device: Any | None = None*, *apply_softmax: bool | None = None*) → Tuple[ndarray, List[List[ndarray]]][[source]](_modules/histox/mil/eval.html#predict_from_multimodal_bags)

Generate multi-mag MIL predictions for a nested list of bags.

Parameters:

- **model** (*torch.nn.Module*) - Loaded PyTorch MIL model.
- **bags** (*list**(**list**(**str**)**)*) - Nested list of bags to generate predictions for.
Each bag should contain PyTorch array of features from all tiles in a slide,
with the shape `(n_tiles, n_features)`.

Keyword Arguments:

- **attention** (*bool*) - Whether to calculate attention scores. Defaults to False.
- **attention_pooling** (*str**,**optional*) - Pooling strategy for attention scores.
Can be 'avg', 'max', or None. Defaults to None.
- **use_lens** (*bool*) - Whether to use the length of each bag as an additional
input to the model. Defaults to False.
- **device** (*str**,**optional*) - Device on which to run inference. Defaults to None.
- **apply_softmax** (*bool*) - Whether to apply softmax to the model output. Defaults
to True for categorical outcomes, False for continuous

Returns:

Predictions and attention scores.

Return type:

Tuple[np.ndarray, List[List[np.ndarray]]]

histox.mil.predict_slide(*model: str*, *slide: str | [WSI](slide.html#histox.slide.WSI)*, *extractor: BaseFeatureExtractor | None = None*, ***, *normalizer: [StainNormalizer](norm.html#histox.norm.StainNormalizer) | None = None*, *config: TrainerConfig | None = None*, *attention: bool = False*, *native_normalizer: bool | None = True*, *extractor_kwargs: dict | None = None*, ***kwargs*) → Tuple[ndarray, ndarray | None][[source]](_modules/histox/mil/eval.html#predict_slide)

Generate predictions (and attention) for a single slide.

Parameters:

- **model** (*str*) - Path to MIL model.
- **slide** (*str*) - Path to slide.
- **extractor** (`histox.mil.BaseFeatureExtractor`, optional) - 

Feature extractor. If not provided, will attempt to auto-detect
extractor from model.

Note

If the extractor has a stain normalizer, this will be used to
normalize the slide before extracting features.

Keyword Arguments:

- **normalizer** (`histox.stain.StainNormalizer`, optional) - Stain normalizer. If not provided, will attempt to use stain
normalizer from extractor.
- **config** (`histox.mil.TrainerConfig`) - Configuration for building model. If None, will attempt to read
`mil_params.json` from the model directory and load saved
configuration. Defaults to None.
- **attention** (*bool*) - Whether to return attention scores. Defaults to
False.
- **attention_pooling** (*str*) - Attention pooling strategy. Either 'avg'
or 'max'. Defaults to None.
- **native_normalizer** (*bool**,**optional*) - Whether to use PyTorch/Tensorflow-native
stain normalization, if applicable. If False, will use the OpenCV/Numpy
implementations. Defaults to None, which auto-detects based on the
slide backend (False if libvips, True if cucim). This behavior is due
to performance issued when using native stain normalization with
libvips-compatible multiprocessing.

Returns:

Predictions and attention scores.
Attention scores are None if `attention` is False. For single-channel attention,
this is a masked 2D array with the same shape as the slide grid (arranged as a
heatmap, with unused tiles masked). For multi-channel attention, this is a
masked 3D array with shape `(n_channels, X, Y)`.

Return type:

Tuple[np.ndarray, Optional[np.ndarray]]

histox.mil.get_mil_tile_predictions(*weights: str*, *dataset: [Dataset](dataset.html#histox.Dataset)*, *bags: str | ndarray | List[str]*, ***, *config: TrainerConfig | None = None*, *outcomes: str | List[str] | None = None*, *dest: str | None = None*, *uq: bool = False*, *device: Any | None = None*, *tile_batch_size: int = 512*, ***kwargs*) → DataFrame[[source]](_modules/histox/mil/eval.html#get_mil_tile_predictions)

Generate tile-level predictions for a MIL model.

Parameters:

- **weights** (*str*) - Path to model weights to load.
- **dataset** ([`histox.Dataset`](dataset.html#histox.Dataset)) - Dataset.
- **bags** (*str**,**list**(**str**)*) - Path to bags, or list of bag file paths.
Each bag should contain PyTorch array of features from all tiles in
a slide, with the shape `(n_tiles, n_features)`.

Keyword Arguments:

- **config** (`histox.mil.TrainerConfig`) - Configuration for building model. If `weights` is a path to a
model directory, will attempt to read `mil_params.json` from this
location and load saved configuration. Defaults to None.
- **outcomes** (*str**,**list**(**str**)*) - Outcomes.
- **dest** (*str*) - Path at which to save tile predictions.
- **uq** (*bool*) - Whether to generate uncertainty estimates. Defaults to False.
- **device** (*str**,**optional*) - Device on which to run inference. Defaults to None.
- **tile_batch_size** (*int*) - Batch size for tile-level predictions. Defaults
to 512.
- **attention_pooling** (*str*) - Attention pooling strategy. Either 'avg'
or 'max'. Defaults to None.

Returns:

Dataframe of tile predictions.

Return type:

pd.DataFrame

histox.mil.generate_attention_heatmaps(*outdir: str*, *dataset: [Dataset](dataset.html#histox.Dataset)*, *bags: List[str] | ndarray*, *attention: ndarray | List[ndarray]*, ***kwargs*) → None[[source]](_modules/histox/mil/eval.html#generate_attention_heatmaps)

Generate and save attention heatmaps for a dataset.

Parameters:

- **outdir** (*str*) - Path at which to save heatmap images.
- **dataset** (*hx.Dataset*) - Dataset.
- **bags** (*str**,**list**(**str**)*) - List of bag file paths.
Each bag should contain PyTorch array of features from all tiles in
a slide, with the shape `(n_tiles, n_features)`.
- **attention** (*list**(**np.ndarray**)*) - Attention scores for each slide.
Length of `attention` should equal the length of `bags`.

Keyword Arguments:

- **interpolation** (*str**,**optional*) - Interpolation strategy for smoothing
heatmap. Defaults to 'bicubic'.
- **cmap** (*str**,**optional*) - Matplotlib colormap for heatmap. Can be any
valid matplotlib colormap. Defaults to 'inferno'.
- **norm** (*str**,**optional*) - Normalization strategy for assigning heatmap
values to colors. Either 'two_slope', or any other valid value
for the `norm` argument of `matplotlib.pyplot.imshow`.
If 'two_slope', normalizes values less than 0 and greater than 0
separately. Defaults to None.

histox.mil.generate_mil_features(*weights: str*, *dataset: hx.Dataset*, *bags: str | ndarray | List[str]*, ***, *config: TrainerConfig | None = None*) → MILFeatures[[source]](_modules/histox/mil/eval.html#generate_mil_features)

Generate activations weights from the last layer of an MIL model.

Returns MILFeatures object.

Parameters:

- **weights** (*str*) - Path to model weights to load.
- **config** (`histox.mil.TrainerConfig`) - Configuration for building model. If `weights` is a path to a
model directory, will attempt to read `mil_params.json` from this
location and load saved configuration. Defaults to None.
- **dataset** ([`histox.Dataset`](dataset.html#histox.Dataset)) - Dataset.
- **outcomes** (*str**,**list**(**str**)*) - Outcomes.
- **bags** (*str**,**list**(**str**)*) - Path to bags, or list of bag file paths.
Each bag should contain PyTorch array of features from all tiles in
a slide, with the shape `(n_tiles, n_features)`.

## TrainerConfig

mil.TrainerConfig

alias of <MagicMock name='mock.TrainerConfig' id='140677591211120'>

| `TrainerConfig.model_fn` | MIL model architecture (class/module). |
| --- | --- |
| `TrainerConfig.loss_fn` | MIL loss function. |
| `TrainerConfig.is_multimodal` | Whether the model is multimodal. |
| `TrainerConfig.model_type` | Type of model (classification or regression). |

mil.TrainerConfig.to_dict(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

mil.TrainerConfig.json_dump(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

mil.TrainerConfig.is_classification(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

mil.TrainerConfig.get_metrics(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

mil.TrainerConfig.prepare_training(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

mil.TrainerConfig.build_model(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

mil.TrainerConfig.predict(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

mil.TrainerConfig.batched_predict(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

mil.TrainerConfig.train(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

mil.TrainerConfig.eval(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

mil.TrainerConfig.build_train_dataloader(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

mil.TrainerConfig.build_val_dataloader(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

mil.TrainerConfig.inspect_batch(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

mil.TrainerConfig.run_metrics(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

## MILModelConfig

*class*histox.mil.MILModelConfig(*model: str | Callable = 'attention_mil'*, ***, *use_lens: bool | None = None*, *apply_softmax: bool = True*, *model_kwargs: dict | None = None*, *validate: bool = True*, *loss: str | Callable = 'cross_entropy'*, ***kwargs*)[[source]](_modules/histox/mil/_params.html#MILModelConfig)

| `MILModelConfig.apply_softmax` | Whether softmax will be applied to model outputs. |
| --- | --- |
| `MILModelConfig.loss_fn` | MIL loss function. |
| `MILModelConfig.model_fn` | MIL model architecture (class/module). |
| `MILModelConfig.model_type` | Type of model (classification or regression). |
| `MILModelConfig.is_multimodal` | Whether the model is multimodal. |

mil.MILModelConfig.is_classification(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

mil.MILModelConfig.to_dict(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

mil.MILModelConfig.inspect_batch(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

mil.MILModelConfig.build_model(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

mil.MILModelConfig.predict(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

mil.MILModelConfig.batched_predict(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

mil.MILModelConfig.run_metrics(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.

## CLAMModelConfig

The CLAM model configuration class requires `slideflow-gpl`, which can be installed with:

```
pip install slideflow-gpl
```

Once installed, the class is available at `histox.clam.CLAMModelConfig`.

clam.CLAMModelConfig

alias of <MagicMock name='mock.CLAMModelConfig' id='140677588969504'>