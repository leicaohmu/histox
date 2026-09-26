# histox.util

This module contains a variety of utility functions used throughout the package.

*class*histox.util.EasyDict[[source]](_modules/histox/util.html#EasyDict)

Convenience class that behaves like a dict but allows access
with the attribute syntax.

*class*histox.util.FeatureExtractionProgress(**columns: str | ProgressColumn*, *console: Console | None = None*, *auto_refresh: bool = True*, *refresh_per_second: float = 10*, *speed_estimate_period: float = 30.0*, *transient: bool = False*, *redirect_stdout: bool = True*, *redirect_stderr: bool = True*, *get_time: Callable[[], float] | None = None*, *disable: bool = False*, *expand: bool = False*)[[source]](_modules/histox/util.html#FeatureExtractionProgress)

get_renderables()[[source]](_modules/histox/util.html#FeatureExtractionProgress.get_renderables)

Get a number of renderables for the progress display.

*class*histox.util.ImgBatchSpeedColumn(*batch_size=1*, **args*, ***kwargs*)[[source]](_modules/histox/util.html#ImgBatchSpeedColumn)

Renders human readable transfer speed.

render(*task: Task*) → Text[[source]](_modules/histox/util.html#ImgBatchSpeedColumn.render)

Show data transfer speed.

*class*histox.util.LabeledMofNCompleteColumn(*unit: str*, **args*, ***kwargs*)[[source]](_modules/histox/util.html#LabeledMofNCompleteColumn)

Renders a completion column with labels.

render(*task: Task*) → Text[[source]](_modules/histox/util.html#LabeledMofNCompleteColumn.render)

Show completion status with labels.

*class*histox.util.MultiprocessProgress(*pb*)[[source]](_modules/histox/util.html#MultiprocessProgress)

Wrapper for a rich.progress bar that can be shared across processes.

*class*histox.util.MultiprocessProgressTracker(*tasks*)[[source]](_modules/histox/util.html#MultiprocessProgressTracker)

Wrapper for a rich.progress tracker that can be shared across processes.

*class*histox.util.TileExtractionProgress(**columns: str | ProgressColumn*, *console: Console | None = None*, *auto_refresh: bool = True*, *refresh_per_second: float = 10*, *speed_estimate_period: float = 30.0*, *transient: bool = False*, *redirect_stdout: bool = True*, *redirect_stderr: bool = True*, *get_time: Callable[[], float] | None = None*, *disable: bool = False*, *expand: bool = False*)[[source]](_modules/histox/util.html#TileExtractionProgress)

get_renderables()[[source]](_modules/histox/util.html#TileExtractionProgress.get_renderables)

Get a number of renderables for the progress display.

*class*histox.util.TileExtractionSpeedColumn(*table_column: Column | None = None*)[[source]](_modules/histox/util.html#TileExtractionSpeedColumn)

Renders human readable transfer speed.

render(*task: Task*) → Text[[source]](_modules/histox/util.html#TileExtractionSpeedColumn.render)

Show data transfer speed.

*class*histox.util.ValidJSONEncoder(***, *skipkeys=False*, *ensure_ascii=True*, *check_circular=True*, *allow_nan=True*, *sort_keys=False*, *indent=None*, *separators=None*, *default=None*)[[source]](_modules/histox/util.html#ValidJSONEncoder)

default(*obj*)[[source]](_modules/histox/util.html#ValidJSONEncoder.default)

Implement this method in a subclass such that it returns
a serializable object for `o`, or calls the base implementation
(to raise a `TypeError`).

For example, to support arbitrary iterators, you could
implement default like this:

```
def default(self, o):
 try:
 iterable = iter(o)
 except TypeError:
 pass
 else:
 return list(iterable)
 # Let the base class default method raise the TypeError
 return JSONEncoder.default(self, o)
```

histox.util.about(*console=None*) → None[[source]](_modules/histox/util.html#about)

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

histox.util.batch(*iterable: List*, *n: int = 1*) → Iterable[[source]](_modules/histox/util.html#batch)

Separates an interable into batches of maximum size n.

histox.util.batch_generator(*iterable: Iterable*, *n: int = 1*) → Iterable[[source]](_modules/histox/util.html#batch_generator)

Separates an interable into batches of maximum size n.

histox.util.bin_values_to_slide_grid(*locations: ndarray*, *values: ndarray*, *wsi: [WSI](slide.html#histox.slide.WSI)*, *background: str = 'min'*) → ndarray[[source]](_modules/histox/util.html#bin_values_to_slide_grid)

Bin heatmap values to a slide grid, using tile location information.

Parameters:

- **locations** (*np.ndarray*) - Array of shape `(n_tiles, 2)` containing x, y
coordinates for all image tiles. Coordinates represent the center
for an associated tile, and must be in a grid.
- **values** (*np.ndarray*) - Array of shape `(n_tiles,)` containing heatmap
values for each tile.
- **wsi** (*histox.wsi.WSI*) - WSI object.

Keyword Arguments:

**background** (*str**,**optional*) - Background strategy for heatmap. Can be
'min', 'mean', 'median', 'max', or 'mask'. Defaults to 'min'.

histox.util.choice_input(*prompt*, *valid_choices*, *default=None*, *multi_choice=False*, *input_type=<class 'str'>*)[[source]](_modules/histox/util.html#choice_input)

Prompts user for multi-choice input.

histox.util.create_triangles(*vertices*, *hole_vertices=None*, *hole_points=None*)[[source]](_modules/histox/util.html#create_triangles)

Tessellate a complex polygon, possibly with holes.

Parameters:

- **vertices** - A list of vertices [(x1, y1), (x2, y2), ...] defining the polygon boundary.
- **holes** - An optional list of points [(hx1, hy1), (hx2, hy2), ...] inside each hole in the polygon.

Returns:

A numpy array of vertices for the tessellated triangles.

histox.util.download_from_tcga(*uuid: str*, *dest: str*, *message: str = 'Downloading...'*) → None[[source]](_modules/histox/util.html#download_from_tcga)

Download a file from TCGA (GDC) by UUID.

histox.util.getLoggingLevel()[[source]](_modules/histox/util.html#getLoggingLevel)

Return the current logging level.

histox.util.get_ensemble_model_config(*model_path: str*) → Dict[[source]](_modules/histox/util.html#get_ensemble_model_config)

Loads ensemble model configuration JSON file.

histox.util.get_gan_config(*model_path: str*) → Dict[[source]](_modules/histox/util.html#get_gan_config)

Loads a GAN training_options.json for an associated network PKL.

histox.util.get_model_config(*model_path: str*) → Dict[[source]](_modules/histox/util.html#get_model_config)

Loads model configuration JSON file.

histox.util.get_model_normalizer(*model_path: str*) → [StainNormalizer](norm.html#histox.norm.StainNormalizer) | None[[source]](_modules/histox/util.html#get_model_normalizer)

Loads and fits normalizer using configuration at a model path.

histox.util.get_preprocess_fn(*model_path: str*)[[source]](_modules/histox/util.html#get_preprocess_fn)

Returns a function which preprocesses a uint8 image for a model.

Parameters:

**model_path** (*str*) - Path to a saved HistoX model.

Returns:

A function which accepts a single image or batch of uint8 images,
and returns preprocessed (and stain normalized) float32 images.

histox.util.get_relative_tfrecord_paths(*root: str*, *directory: str = ''*) → List[str][[source]](_modules/histox/util.html#get_relative_tfrecord_paths)

Returns relative tfrecord paths with respect to the given directory.

histox.util.get_slide_paths(*slides_dir: str*) → List[str][[source]](_modules/histox/util.html#get_slide_paths)

Get all slide paths from a given directory containing slides.

histox.util.get_slides_from_model_manifest(*model_path: str*, *dataset: str | None = None*) → List[str][[source]](_modules/histox/util.html#get_slides_from_model_manifest)

Get list of slides from a model manifest.

Parameters:

- **model_path** (*str*) - Path to model from which to load the model manifest.
- **dataset** (*str*) - 'training' or 'validation'. Will return only slides
from this dataset. Defaults to None (all).

Returns:

List of slide names.

Return type:

list(str)

histox.util.get_valid_model_dir(*root: str*) → List[[source]](_modules/histox/util.html#get_valid_model_dir)

This function returns the path of the first indented directory from root.
This only works when the indented folder name starts with a 5 digit number,
like "00000%".

Examples

If the root has 3 files:
root/00000-foldername/
root/00001-foldername/
root/00002-foldername/

The function returns "root/00000-foldername/"

histox.util.global_path(*root: str*, *path_string: str*)[[source]](_modules/histox/util.html#global_path)

Returns global path from a local path.

histox.util.infer_stride(*locations*, *wsi*)[[source]](_modules/histox/util.html#infer_stride)

Infer the stride of a grid of locations from a set of locations.

Parameters:

- **locations** (*np.ndarray*) - Nx2 array of locations
- **wsi** (*histox.wsi.WSI*) - WSI object

Returns:

inferred stride divisor in pixels

Return type:

float

histox.util.is_model(*path: str*) → bool[[source]](_modules/histox/util.html#is_model)

Checks if the given path is a valid HistoX model.

histox.util.is_project(*path: str*) → bool[[source]](_modules/histox/util.html#is_project)

Checks if the given path is a valid HistoX project.

histox.util.is_simclr_model_path(*path: Any*) → bool[[source]](_modules/histox/util.html#is_simclr_model_path)

Checks if the given path is a valid SimCLR model or checkpoint.

histox.util.is_slide(*path: str*) → bool[[source]](_modules/histox/util.html#is_slide)

Checks if the given path is a supported slide.

histox.util.is_tensorflow_model_path(*path: str*) → bool[[source]](_modules/histox/util.html#is_tensorflow_model_path)

Check if the given path points to a valid Histox/TensorFlow model.

A valid Histox TensorFlow model is identified by the following criteria:

1. The path points to an existing directory (TensorFlow saves models
in the SavedModel directory format via `tf.keras.Model.save()`).
2. A `params.json` file exists either inside the directory itself
or in its parent directory, which stores the training
hyperparameters required to reconstruct the model architecture
before loading weights.

Parameters:

**path** (*str*) - Path to the directory to be checked.

Returns:

`True` if the path points to a valid Histox/TensorFlow model,
`False` otherwise.

Return type:

bool

Notes

Histox loads TensorFlow models in two steps, as implemented in
[histox.model.load][]:

1. **Reconstruct architecture**: `params.json` is read via
[histox.util.get_model_config][] to retrieve the saved
hyperparameters, which are then passed to
`ModelParams.from_dict()` and `ModelParams.build_model()`
to rebuild the model structure.
2. **Load weights**: weights are loaded from
`<path>/variables/variables` via
`tf.keras.Model.load_weights()`.

Both files must therefore be present for a model to be considered
valid. This check is performed before calling
[histox.model.Features][] or [histox.model.load][].

Unlike [histox.util.is_torch_model_path][], `params.json` is
accepted in either of two locations, accommodating both the flat
layout (before training completes) and the nested layout (after
`params.json` is copied into the model directory at the end of
each epoch by the training callback):

```
outdir/ # params.json written here at
│ # training start (flat layout)
├── params.json
└── <model_name>_epoch<N>/ # SavedModel directory
 ├── variables/
 │ └── variables # Model weights
 └── params.json # Copied here at epoch end
 # (nested layout)
```

Examples

```
>>> is_tensorflow_model_path('/models/lung_model_epoch10')
True
```

```
>>> is_tensorflow_model_path('/models/lung_model_epoch10.zip')
False
```

```
>>> is_tensorflow_model_path('/models/not_a_model')
False
```

histox.util.is_tile_size_compatible(*tile_px1: int*, *tile_um1: str | int*, *tile_px2: int*, *tile_um2: str | int*) → bool[[source]](_modules/histox/util.html#is_tile_size_compatible)

Check whether tile sizes are compatible.

Compatibility is defined as:

- Equal size in pixels
- If tile width (tile_um) is defined in microns (int) for both, these must be equal
- If tile width (tile_um) is defined as a magnification (str) for both, these must be equal
- If one is defined in microns and the other as a magnification, the calculated magnification must be +/- 2.

Example 1:
- tile_px1=299, tile_um1=302
- tile_px2=299, tile_um2=304
- Incompatible (unequal micron width)

Example 2:
- tile_px1=299, tile_um1=10x
- tile_px2=299, tile_um2=9x
- Incompatible (unequal magnification)

Example 3:
- tile_px1=299, tile_um1=302
- tile_px2=299, tile_um2=10x
- Compatible (first has an equivalent magnification of 9.9x, which is +/- 2 compared to 10x)

Parameters:

- **tile_px1** (*int*) - Tile size (in pixels) of first slide.
- **tile_um1** (*int**or**str*) - Tile size (in microns) of first slide.
Can also be expressed as a magnification level, e.g. `'10x'`
- **tile_px2** (*int*) - Tile size (in pixels) of second slide.
- **tile_um2** (*int**or**str*) - Tile size (in microns) of second slide.
Can also be expressed as a magnification level, e.g. `'10x'`

Returns:

Whether the tile sizes are compatible.

Return type:

bool

histox.util.is_torch_model_path(*path: str*) → bool[[source]](_modules/histox/util.html#is_torch_model_path)

Check if the given path points to a valid Histox/PyTorch model.

A valid Histox PyTorch model is identified by the following criteria:

1. The path points to an existing file.
2. The file has a `.zip` extension (PyTorch saves model weights in
ZIP format via `torch.save()`).
3. A `params.json` file exists in the same directory as the model
file, which stores the training hyperparameters required to
reconstruct the model architecture before loading weights.

Parameters:

**path** (*str*) - Path to the file to be checked.

Returns:

`True` if the path points to a valid Histox/PyTorch model,
`False` otherwise.

Return type:

bool

Notes

Histox loads PyTorch models in two steps:

1. **Reconstruct architecture**: `params.json` is read via
[histox.util.get_model_config][] to retrieve the saved
hyperparameters, which are then used to rebuild the model
structure via [histox.ModelParams.build_model][].
2. **Load weights**: the state dictionary is loaded from the
`.zip` file via `torch.nn.Module.load_state_dict()`.

Both files must therefore be present for a model to be considered
valid. This check is performed before calling
[histox.model.Features][] or [histox.model.load][].

The minimum required directory structure of a saved Histox model is:

```
outdir/
├── <model_name>_epoch<N>.zip # Model weights saved by torch.save()
└── params.json # Training config and hyperparameters
```

Examples

```
>>> is_torch_model_path('/models/lung_model_epoch10.zip')
True
```

```
>>> is_torch_model_path('/models/lung_model_epoch10.pt')
False
```

```
>>> is_torch_model_path('/models/not_a_model.zip')
False
```

histox.util.is_uq_model(*model_path: str*) → bool[[source]](_modules/histox/util.html#is_uq_model)

Checks if the given model path points to a UQ-enabled model.

histox.util.isnumeric(*val: Any*) → bool[[source]](_modules/histox/util.html#isnumeric)

Check if the given value is numeric (numpy or python).

Tensors will return False.

Specifically checks if the value is a python int or float,
or if the value is a numpy array with a numeric dtype (int or float).

histox.util.load_json(*filename: str*) → Any[[source]](_modules/histox/util.html#load_json)

Reads JSON data from file.

histox.util.load_predictions(*path: str*, ***kwargs*) → DataFrame[[source]](_modules/histox/util.html#load_predictions)

Loads a 'csv', 'parquet' or 'feather' file to a pandas dataframe.

Parameters:

**path** (*str*) - Path to the file to be read.

Returns:

The dataframe read from the path.

Return type:

df (pd.DataFrame)

histox.util.location_heatmap(*locations: ndarray*, *values: ndarray*, *slide: str*, *tile_px: int*, *tile_um: int | str*, *filename: str*, ***, *interpolation: str | None = 'bicubic'*, *cmap: str = 'inferno'*, *norm: str | None = None*, *background: str = 'min'*) → None[[source]](_modules/histox/util.html#location_heatmap)

Generate a heatmap for a slide.

Parameters:

- **locations** (*np.ndarray*) - Array of shape `(n_tiles, 2)` containing x, y
coordinates for all image tiles. Coordinates represent the center
for an associated tile, and must be in a grid.
- **values** (*np.ndarray*) - Array of shape `(n_tiles,)` containing heatmap
values for each tile.
- **slide** (*str*) - Path to corresponding slide.
- **tile_px** (*int*) - Tile pixel size.
- **tile_um** (*int**,**str*) - Tile micron or magnification size.
- **filename** (*str*) - Destination filename for heatmap.

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

histox.util.log_manifest(*train_tfrecords: List[str] | None = None*, *val_tfrecords: List[str] | None = None*, ***, *labels: Dict[str, Any] | None = None*, *filename: str | None = None*, *remove_extension: bool = True*) → str[[source]](_modules/histox/util.html#log_manifest)

Saves the training manifest in CSV format and returns as a string.

Parameters:

- **train_tfrecords** (*list**(**str**)**]**,**optional*) - List of training TFRecords.
Defaults to None.
- **val_tfrecords** (*list**(**str**)**]**,**optional*) - List of validation TFRecords.
Defaults to None.

Keyword Arguments:

- **labels** (*dict**,**optional*) - TFRecord outcome labels. Defaults to None.
- **filename** (*str**,**optional*) - Path to CSV file to save. Defaults to None.
- **remove_extension** (*bool**,**optional*) - Remove file extension from slide
names. Defaults to True.

Returns:

Saved manifest in str format.

Return type:

str

histox.util.make_dir(*_dir: str*) → None[[source]](_modules/histox/util.html#make_dir)

Makes a directory if one does not already exist,
in a manner compatible with multithreading.

histox.util.map_values_to_slide_grid(*locations: ndarray*, *values: ndarray*, *wsi: [WSI](slide.html#histox.slide.WSI)*, *background: str = 'min'*, ***, *interpolation: str | None = 'bicubic'*) → ndarray[[source]](_modules/histox/util.html#map_values_to_slide_grid)

Map heatmap values to a slide grid, using tile location information.

Parameters:

- **locations** (*np.ndarray*) - Array of shape `(n_tiles, 2)` containing x, y
coordinates for all image tiles. Coordinates represent the center
for an associated tile, and must be in a grid.
- **values** (*np.ndarray*) - Array of shape `(n_tiles,)` containing heatmap
values for each tile.
- **wsi** (*histox.wsi.WSI*) - WSI object.

Keyword Arguments:

- **background** (*str**,**optional*) - Background strategy for heatmap. Can be
'min', 'mean', 'median', 'max', or 'mask'. Defaults to 'min'.
- **interpolation** (*str**,**optional*) - Interpolation strategy for smoothing
heatmap. Defaults to 'bicubic'.

histox.util.md5(*path: str*) → str[[source]](_modules/histox/util.html#md5)

Calculate and return MD5 checksum for a file.

histox.util.multi_warn(*arr: List*, *compare: Callable*, *msg: Callable | str*) → int[[source]](_modules/histox/util.html#multi_warn)

Logs multiple warning

Parameters:

- **arr** (*List*) - Array to compare.
- **compare** (*Callable*) - Comparison to perform on array. If True, will warn.
- **msg** (*str*) - Warning message.

Returns:

Number of warnings.

Return type:

int

histox.util.path_input(*prompt: str*, *root: str*, *default: str | None = None*, *create_on_invalid: bool = False*, *filetype: str | None = None*, *verify: bool = True*) → str[[source]](_modules/histox/util.html#path_input)

Prompts user for directory input.

histox.util.path_to_ext(*path: str*) → str[[source]](_modules/histox/util.html#path_to_ext)

Returns extension of a file path string.

histox.util.path_to_name(*path: str*) → str[[source]](_modules/histox/util.html#path_to_name)

Returns name of a file, without extension,
from a given full path string.

histox.util.read_annotations(*path: str*) → Tuple[List[str], List[Dict]][[source]](_modules/histox/util.html#read_annotations)

Read an annotations file.

histox.util.relative_path(*path: str*, *root: str*)[[source]](_modules/histox/util.html#relative_path)

Returns a relative path, from a given root directory.

histox.util.setLoggingLevel(*level*)[[source]](_modules/histox/util.html#setLoggingLevel)

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

histox.util.set_ignore_sigint()[[source]](_modules/histox/util.html#set_ignore_sigint)

Ignore keyboard interrupts.

histox.util.split_list(*a: List*, *n: int*) → List[List][[source]](_modules/histox/util.html#split_list)

Function to split a list into n components

histox.util.tfrecord_heatmap(*tfrecord: str*, *slide: str*, *tile_px: int*, *tile_um: int | str*, *tile_dict: Dict[int, float]*, *filename: str*, ***kwargs*) → None[[source]](_modules/histox/util.html#tfrecord_heatmap)

Creates a tfrecord-based WSI heatmap using a dictionary of tile values
for heatmap display.

Parameters:

- **tfrecord** (*str*) - Path to tfrecord.
- **slide** (*str*) - Path to whole-slide image.
- **tile_dict** (*dict*) - Dictionary mapping tfrecord indices to a
tile-level value for display in heatmap format.
- **tile_px** (*int*) - Tile width in pixels.
- **tile_um** (*int**or**str*) - Tile width in microns (int) or magnification
(str, e.g. "20x").
- **filename** (*str*) - Destination filename for heatmap.

histox.util.tile_size_label(*tile_px: int*, *tile_um: str | int*) → str[[source]](_modules/histox/util.html#tile_size_label)

Return the string label of the given tile size.

histox.util.to_onehot(*val: int*, *max: int*) → ndarray[[source]](_modules/histox/util.html#to_onehot)

Converts value to one-hot encoding

Parameters:

- **val** (*int*) - Value to encode
- **max** (*int*) - Maximum value (length of onehot encoding)

histox.util.update_results_log(*results_log_path: str*, *model_name: str*, *results_dict: Dict*) → None[[source]](_modules/histox/util.html#update_results_log)

Dynamically update results_log when recording training metrics.

histox.util.write_json(*data: Any*, *filename: str*) → None[[source]](_modules/histox/util.html#write_json)

Write data to JSON file.

Parameters:

- **data** (*Any*) - Data to write.
- **filename** (*str*) - Path to JSON file.

histox.util.yes_no_input(*prompt: str*, *default: str = 'no'*) → bool[[source]](_modules/histox/util.html#yes_no_input)

Prompts user for yes/no input.