# histox.stats

This module contains internal utility functions for generating and evaluating model predictions and metrics.

histox.stats.df_from_pred(*y_true: List[Any] | None*, *y_pred: List[Any]*, *y_std: List[Any] | None*, *tile_to_slides: List | ndarray*, *locations: List | ndarray | None = None*) → DataFrame[[source]](_modules/histox/stats/metrics.html#df_from_pred)

Converts arrays of model predictions to a pandas dataframe.

Parameters:

- **y_true** (*list**(**np.ndarray**)*) - List of y_true numpy arrays, one array for
each outcome. For continuous outcomes, the length of the outer
list should be one, and the second shape dimension of the numpy
array should be the number of continuous outcomes.
- **y_pred** (*list**(**np.ndarray**)*) - List of y_pred numpy arrays, one array for
each outcome. For continuous outcomes, the length of the outer
list should be one, and the second shape dimension of the numpy
array should be the number of continuous outcomes.
- **y_std** (*list**(**np.ndarray**)*) - List of uncertainty numpy arrays, formatted
in the same way as y_pred.
- **tile_to_slides** (*np.ndarray*) - Array of slide names for each tile. Length
should match the numpy arrays in y_true, y_pred, and y_std.

Returns:

DataFrame of predictions.

Return type:

DataFrame

histox.stats.eval_dataset(*model: tf.keras.Model | torch.nn.Module*, *dataset: tf.data.Dataset | torch.utils.data.DataLoader*, *model_type: str*, *num_tiles: int = 0*, *uq: bool = False*, *uq_n: int = 30*, *reduce_method: str | Callable = 'average'*, *patients: Dict[str, str] | None = None*, *outcome_names: List[str] | None = None*, *loss: Callable | None = None*, *torch_args: SimpleNamespace | None = None*) → Tuple[DataFrame, float, float][[source]](_modules/histox/stats/metrics.html#eval_dataset)

Generates predictions and accuracy/loss from a given model and dataset.

Parameters:

- **model** (*str*) - Path to PyTorch model.
- **dataset** (*tf.data.Dataset*) - PyTorch dataloader.
- **model_type** (*str**,**optional*) - 'classification', 'regression', or 'survival'.
If multiple continuous outcomes are present, y_true is stacked into a
single vector for each image. Defaults to 'classification'.
- **num_tiles** (*int**,**optional*) - Used for progress bar with Tensorflow.
Defaults to 0.
- **uq_n** (*int**,**optional*) - Number of forward passes to perform
when calculating MC Dropout uncertainty. Defaults to 30.
- **reduce_method** (*str**,**optional*) - Reduction method for calculating
slide-level and patient-level predictions for categorical
outcomes. Options include 'average', 'mean', 'proportion',
'median', 'sum', 'min', 'max', or a callable function.
'average' and 'mean' are synonymous, with both options kept
for backwards compatibility. If 'average' or 'mean', will
reduce with average of each logit across tiles. If
'proportion', will convert tile predictions into onehot encoding
then reduce by averaging these onehot values. For all other
values, will reduce with the specified function, applied via
the pandas `DataFrame.agg()` function. Defaults to 'average'.
- **patients** (*dict**,**optional*) - Dictionary mapping slide names to patient
names. Required for generating patient-level metrics.
- **outcome_names** (*list**,**optional*) - List of str, names for outcomes.
Defaults to None (outcomes will not be named).
- **torch_args** (*namespace*) - Used for PyTorch models. Namespace containing
num_slide_features, slide_input, update_corrects, and
update_loss functions.

Returns:

pd.DataFrame, accuracy, loss

histox.stats.group_reduce(*df: DataFrame*, *method: str | Callable = 'average'*, *patients: Dict[str, str] | None = None*) → Dict[str, DataFrame][[source]](_modules/histox/stats/metrics.html#group_reduce)

Reduces tile-level predictions to group-level predictions.

Parameters:

- **df** (*DataFrame*) - Tile-level predictions.
- **method** (*str**,**optional*) - Reduction method for calculating
slide-level and patient-level predictions for categorical outcomes.
Options include 'average', 'mean', 'proportion', 'median', 'sum',
'min', 'max', or a callable function. 'average' and 'mean' are
synonymous, with both options kept for backwards compatibility. If
'average' or 'mean', will reduce with average of each logit across
tiles. If 'proportion', will convert tile predictions into onehot
encoding then reduce by averaging these onehot values. For all other
values, will reduce with the specified function, applied via
the pandas `DataFrame.agg()` function. Defaults to 'average'.
- **patients** (*dict**,**optional*) - Dictionary mapping slide names to patient
names. Required for generating patient-level metrics.

histox.stats.metrics_from_dataset(*model: tf.keras.Model | torch.nn.Module*, *model_type: str*, *patients: Dict[str, str]*, *dataset: tf.data.Dataset | torch.utils.data.DataLoader*, *num_tiles: int = 0*, *outcome_names: List[str] | None = None*, *reduce_method: str | Callable = 'average'*, *label: str = ''*, *save_predictions: str | bool = False*, *data_dir: str = ''*, *uq: bool = False*, *loss: Callable | None = None*, *torch_args: SimpleNamespace | None = None*, ***kwargs*) → Tuple[Dict, float, float][[source]](_modules/histox/stats/metrics.html#metrics_from_dataset)

Evaluate performance of a given model on a given TFRecord dataset,
generating a variety of statistical outcomes and graphs.

Parameters:

- **model** (*tf.keras.Model**or**torch.nn.Module*) - Keras/Torch model to eval.
- **model_type** (*str*) - 'classification', 'regression', or 'survival'.
- **patients** (*dict*) - Dictionary mapping slidenames to patients.
- **dataset** (*tf.data.Dataset**or**torch.utils.data.DataLoader*) - Dataset.
- **num_tiles** (*int**,**optional*) - Number of total tiles expected in dataset.
Used for progress bar. Defaults to 0.

Keyword Arguments:

- **outcome_names** (*list**,**optional*) - List of str, names for outcomes.
Defaults to None.
- **reduce_method** (*str**,**optional*) - Reduction method for calculating
slide-level and patient-level predictions for categorical
outcomes. Options include 'average', 'mean', 'proportion',
'median', 'sum', 'min', 'max', or a callable function.
'average' and 'mean' are synonymous, with both options kept
for backwards compatibility. If 'average' or 'mean', will
reduce with average of each logit across tiles. If
'proportion', will convert tile predictions into onehot encoding
then reduce by averaging these onehot values. For all other
values, will reduce with the specified function, applied via
the pandas `DataFrame.agg()` function. Defaults to 'average'.
- **label** (*str**,**optional*) - Label prefix/suffix for saving.
Defaults to None.
- **save_predictions** (*bool**,**optional*) - Save tile, slide, and patient-level
predictions to CSV. Defaults to True.
- **data_dir** (*str*) - Path to data directory for saving.
Defaults to empty string (current directory).
- **neptune_run** (`neptune.Run`, optional) - Neptune run in which to
log results. Defaults to None.

Returns:

metrics [dict], accuracy [float], loss [float]

histox.stats.name_columns(*df: DataFrame*, *model_type: str*, *outcome_names: List[str] | None = None*)[[source]](_modules/histox/stats/metrics.html#name_columns)

Renames columns in a DataFrame to correspond to the given outcome names.

Assumes the DataFrame supplied was generated by hx.stats.df_from_pred().

Parameters:

- **df** (*DataFrame*) - DataFrame from hx.stats.df_from_pred(), containing
predictions and labels.
- **model_type** (*str*) - Type of model ('classification', 'regression', or 'survival').
- **outcome_names** (*list**(**str**)**)**,**optional*) - Outcome names to apply to the
DataFrame. If this is from a survival model, the standard names "time"
and "event" will be used.

Raises:

- **ValueError** - If outcome_names are not supplied and it is not a survival model.
- **errors.StatsError** - If the length of outcome_names is incompatible
 with the DataFrame.

Returns:

DataFrame with renamed columns.

Return type:

DataFrame

histox.stats.predict_dataset(*model: tf.keras.Model | torch.nn.Module*, *dataset: tf.data.Dataset | torch.utils.data.DataLoader*, *model_type: str*, *num_tiles: int = 0*, *uq: bool = False*, *uq_n: int = 30*, *reduce_method: str | Callable = 'average'*, *patients: Dict[str, str] | None = None*, *outcome_names: List[str] | None = None*, *torch_args: SimpleNamespace | None = None*) → Dict[str, DataFrame][[source]](_modules/histox/stats/metrics.html#predict_dataset)

Generates predictions from model and dataset.

Parameters:

- **model** (*str*) - Path to PyTorch model.
- **dataset** (*tf.data.Dataset*) - PyTorch dataloader.
- **model_type** (*str**,**optional*) - 'classification', 'regression', or 'survival'.
If multiple continuous outcomes are present, y_true is stacked into a
single vector for each image. Defaults to 'classification'.
- **num_tiles** (*int**,**optional*) - Used for progress bar with Tensorflow.
Defaults to 0.
- **uq_n** (*int**,**optional*) - Number of forward passes to perform
when calculating MC Dropout uncertainty. Defaults to 30.
- **reduce_method** (*str**,**optional*) - Reduction method for calculating
slide-level and patient-level predictions for categorical
outcomes. Options include 'average', 'mean', 'proportion',
'median', 'sum', 'min', 'max', or a callable function.
'average' and 'mean' are synonymous, with both options kept
for backwards compatibility. If 'average' or 'mean', will
reduce with average of each logit across tiles. If
'proportion', will convert tile predictions into onehot encoding
then reduce by averaging these onehot values. For all other
values, will reduce with the specified function, applied via
the pandas `DataFrame.agg()` function. Defaults to 'average'.
- **patients** (*dict**,**optional*) - Dictionary mapping slide names to patient
names. Required for generating patient-level metrics.
- **outcome_names** (*list**,**optional*) - List of str, names for outcomes.
Defaults to None (outcomes will not be named).
- **torch_args** (*namespace*) - Used for PyTorch backend. Namespace containing
num_slide_features and slide_input.

Returns:

Dictionary with keys 'tile', 'slide', and
'patient', and values containing DataFrames with tile-, slide-,
and patient-level predictions.

Return type:

Dict[str, pd.DataFrame]

histox.stats.calculate_centroid(*act: Dict[str, ndarray]*) → Tuple[Dict[str, int], Dict[str, ndarray]][[source]](_modules/histox/stats/stats_utils.html#calculate_centroid)

Calcultes slide-level centroid indices for a provided activations dict.

Parameters:

**activations** (*dict*) - Dict mapping slide names to ndarray of activations
across tiles, of shape (n_tiles, n_features)

Returns:

A tuple containing

> dict: Dict mapping slides to index of tile nearest to centroid
> 
> 
> 
> 
> dict: Dict mapping slides to activations of tile nearest to centroid

histox.stats.get_centroid_index(*arr: ndarray*) → int[[source]](_modules/histox/stats/stats_utils.html#get_centroid_index)

Calculate index nearest to centroid from a given 2D input array.