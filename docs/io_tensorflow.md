# histox.io.tensorflow

TFRecord interleaving in the Tensorflow backend is accomplished with `histox.io.tensorflow.interleave()`, which interleaves a set of tfrecords together into a `tf.data.Datasets` object that can be used for training. This interleaving can include patient or category-level balancing for returned batches (see [Oversampling with balancing](dataloaders.html#balancing)).

Note

The TFRecord reading and interleaving implemented in this module is only compatible with Tensorflow models. The [`histox.io.torch`](io_torch.html#module-histox.io.torch) module includes a PyTorch-specific TFRecord reader.

histox.io.tensorflow.checkpoint_to_tf_model(*models_dir: str*, *model_name: str*) → None[[source]](_modules/histox/io/tensorflow.html#checkpoint_to_tf_model)

Convert a checkpoint file into a saved model.

Parameters:

- **models_dir** - Directory containing the model.
- **model_name** - Name of the model to convert.

histox.io.tensorflow.decode_image(*img_string: bytes*, *img_type: str*, *crop_left: int | None = None*, *crop_width: int | None = None*, *resize_target: int | None = None*, *resize_method: str = 'lanczos3'*, *resize_aa: bool = True*, *size: int | None = None*) → tensorflow.Tensor

Decodes an image.

Parameters:

- **img_string** (*bytes*) - Image bytes (JPG/PNG).
- **img_type** (*str*) - Type of image data; 'jpg', 'jpeg', or 'png'.
- **crop_left** (*int**,**optional*) - Crop image starting at this top-left
coordinate. Defaults to None.
- **crop_width** (*int**,**optional*) - Crop image to this width.
Defaults to None.
- **resize_target** (*int**,**optional*) - Resize image, post-crop, to this target
size in pixels. Defaults to None.
- **resize_method** (*str**,**optional*) - Resizing method, if applicable.
Defaults to 'lanczos3'.
- **resize_aa** (*bool**,**optional*) - If resizing, use antialiasing.
Defaults to True.
- **size** (*int**,**optional*) - Set the image size/width (pixels).
Defaults to None.

Returns:

Processed image (uint8).

Return type:

tf.Tensor

histox.io.tensorflow.get_tfrecord_parser(*tfrecord_path: str*, *features_to_return: Iterable[str] | None = None*, *to_numpy: bool = False*, *decode_images: bool = True*, *img_size: int | None = None*, *error_if_invalid: bool = True*, ***decode_kwargs: Any*) → Callable | None[[source]](_modules/histox/io/tensorflow.html#get_tfrecord_parser)

Returns a tfrecord parsing function based on the specified parameters.

Parameters:

- **tfrecord_path** (*str*) - Path to tfrecord to parse.
- **features_to_return** (*list**or**dict**,**optional*) - Designates format for how
features should be returned from parser. If a list of feature names
is provided, the parsing function will return tfrecord features as
a list in the order provided. If a dictionary of labels (keys)
mapping to feature names (values) is provided, features will be
returned from the parser as a dictionary matching the same format.
If None, will return all features as a list.
- **to_numpy** (*bool**,**optional*) - Convert records from tensors->numpy arrays.
Defaults to False.
- **decode_images** (*bool**,**optional*) - Decode image strings into arrays.
Defaults to True.
- **standardize** (*bool**,**optional*) - Standardize images into the range (0,1).
Defaults to False.
- **img_size** (*int*) - Width of images in pixels. Will call tf.set_shape(...)
if provided. Defaults to False.
- **normalizer** ([`histox.norm.StainNormalizer`](norm.html#histox.norm.StainNormalizer)) - Stain normalizer
to use on images. Defaults to None.
- **augment** (*str**or**bool*) - 

Image augmentations to perform. Augmentations include:

- `'x'`: Random horizontal flip
- `'y'`: Random vertical flip
- `'r'`: Random 90-degree rotation
- `'j'`: Random JPEG compression (50% chance to compress with quality between 50-100)
- `'b'`: Random Gaussian blur (10% chance to blur with sigma between 0.5-2.0)
- `'n'`: Random [Stain Augmentation](norm.html#stain-augmentation) (requires stain normalizer)

Combine letters to define augmentations, such as `'xyrjn'`.
A value of True will use `'xyrjb'`.
- **error_if_invalid** (*bool**,**optional*) - Raise an error if a tfrecord cannot
be read. Defaults to True.

histox.io.tensorflow.interleave(*paths: List[str]*, ***, *augment: bool = False*, *batch_size: int | None*, *clip: Dict[str, int] | None = None*, *deterministic: bool = False*, *drop_last: bool = False*, *from_wsi: bool = False*, *incl_loc: str | None = None*, *incl_slidenames: bool = False*, *infinite: bool = True*, *img_size: int*, *labels: Dict[str, str] | Dict[str, int] | Dict[str, List[float]] | None = None*, *normalizer: [StainNormalizer](norm.html#histox.norm.StainNormalizer) | None = None*, *num_parallel_reads: int = 4*, *num_shards: int | None = None*, *pool: mp.pool.Pool | None = None*, *prob_weights: Dict[str, float] | None = None*, *rois: List[str] | None = None*, *roi_method: str = 'auto'*, *shard_idx: int | None = None*, *standardize: bool = True*, *tile_um: int | None = None*, *tfrecord_parser: Callable | None = None*, *transform: Callable | None = None*, ***decode_kwargs: Any*) → Iterable[[source]](_modules/histox/io/tensorflow.html#interleave)

Generate an interleaved dataset from a collection of tfrecord files.

The interleaved dataset samples from tfrecord files randomly according to
balancing, if provided. Requires manifest for balancing. Assumes TFRecord
files are named by slide.

Parameters:

**paths** (*list**(**str**)*) - List of paths to TFRecord files or whole-slide
images.

Keyword Arguments:

- **augment** (*str**or**bool*) - 

Image augmentations to perform. Augmentations include:

- `'x'`: Random horizontal flip
- `'y'`: Random vertical flip
- `'r'`: Random 90-degree rotation
- `'j'`: Random JPEG compression (50% chance to compress with quality between 50-100)
- `'b'`: Random Gaussian blur (10% chance to blur with sigma between 0.5-2.0)
- `'n'`: Random [Stain Augmentation](norm.html#stain-augmentation) (requires stain normalizer)

Combine letters to define augmentations, such as `'xyrjn'`.
A value of True will use `'xyrjb'`.
- **batch_size** (*int*) - Batch size.
- **clip** (*dict**,**optional*) - Dict mapping tfrecords to number of tiles to
take per tfrecord. Defaults to None.
- **deterministic** (*bool**,**optional*) - When num_parallel_calls is specified,
if this boolean is specified, it controls the order in which the
transformation produces elements. If set to False, the
transformation is allowed to yield elements out of order to trade
determinism for performance. Defaults to False.
- **drop_last** (*bool**,**optional*) - Drop the last non-full batch.
Defaults to False.
- **from_wsi** (*bool*) - Generate predictions from tiles dynamically
extracted from whole-slide images, rather than TFRecords.
Defaults to False (use TFRecords).
- **incl_loc** (*str**,**optional*) - 'coord', 'grid', or None. Return (x,y)
coordinates ('coord') for each tile center along with tile
images, or the (x,y) grid coordinates for each tile.
Defaults to 'coord'.
- **incl_slidenames** (*bool**,**optional*) - Include slidenames as third returned
variable. Defaults to False.
- **infinite** (*bool**,**optional*) - Create an finite dataset. WARNING: If
infinite is False && balancing is used, some tiles will be skipped.
Defaults to True.
- **img_size** (*int*) - Image width in pixels.
- **labels** (*dict**or**str**,**optional*) - Dict or function. If dict, must map
slide names to outcome labels. If function, function must accept an
image (tensor) and slide name (str), and return a dict
{'image_raw': image (tensor)} and label (int or float). If not
provided, all labels will be None.
- **normalizer** ([`histox.norm.StainNormalizer`](norm.html#histox.norm.StainNormalizer), optional) - Normalizer to use on images. Defaults to None.
- **num_parallel_reads** (*int**,**optional*) - Number of parallel reads for each
TFRecordDataset. Defaults to 4.
- **num_shards** (*int**,**optional*) - Shard the tfrecord datasets, used for
multiprocessing datasets. Defaults to None.
- **pool** (*multiprocessing.Pool*) - Shared multiprocessing pool. Useful
if `from_wsi=True`, for sharing a unified processing pool between
dataloaders. Defaults to None.
- **prob_weights** (*dict**,**optional*) - Dict mapping tfrecords to probability of
including in batch. Defaults to None.
- **rois** (*list**(**str**)**,**optional*) - List of ROI paths. Only used if
from_wsi=True. Defaults to None.
- **roi_method** (*str**,**optional*) - Method for extracting ROIs. Only used if
from_wsi=True. Defaults to 'auto'.
- **shard_idx** (*int**,**optional*) - Index of the tfrecord shard to use.
Defaults to None.
- **standardize** (*bool**,**optional*) - Standardize images to (0,1).
Defaults to True.
- **tile_um** (*int**,**optional*) - Size of tiles to extract from WSI, in
microns. Only used if from_wsi=True. Defaults to None.
- **tfrecord_parser** (*Callable**,**optional*) - Custom parser for TFRecords.
Defaults to None.
- **transform** (*Callable**,**optional*) - Arbitrary transform function.
Performs transformation after augmentations but before
standardization. Defaults to None.
- ****decode_kwargs** (*dict*) - Keyword arguments to pass to
`histox.io.tensorflow.decode_image()`.

histox.io.tensorflow.join_tfrecord(*input_folder: str*, *output_file: str*, *assign_slide: str | None = None*) → None[[source]](_modules/histox/io/tensorflow.html#join_tfrecord)

Randomly sample from tfrecords in the input folder with shuffling,
and combine into a single tfrecord file.

Parameters:

- **input_folder** (*str*) - Folder containing tfrecord files.
- **output_file** (*str*) - Output tfrecord file.
- **assign_slide** (*str**,**optional*) - Assign a slide name to all images.
Defaults to None.

histox.io.tensorflow.multi_image_example(*slide: bytes*, *image_dict: Dict*) → Example[[source]](_modules/histox/io/tensorflow.html#multi_image_example)

Returns a Tensorflow Data example for storage with multiple images.

Parameters:

- **slide** (*bytes*) - Slide name.
- **image_dict** (*Dict*) - Dictionary of image names and image bytes.

Returns:

Tensorflow Data example.

Return type:

Example

histox.io.tensorflow.parser_from_labels(*labels: Dict[str, str] | Dict[str, int] | Dict[str, List[float]]*) → Callable[[source]](_modules/histox/io/tensorflow.html#parser_from_labels)

Create a label parsing function used for parsing slides into single
or multi-outcome labels.

Parameters:

**labels** (*dict*) - Dictionary mapping slide names to outcome labels.

Returns:

Label parsing function.

Return type:

Callable

histox.io.tensorflow.preprocess_uint8(*img: tensorflow.Tensor*, *normalizer: [StainNormalizer](norm.html#histox.norm.StainNormalizer) | None = None*, *standardize: bool = True*, *resize_px: int | None = None*, *resize_method: str = 'lanczos3'*, *resize_aa: bool = True*, *as_dict: bool = True*) → Dict[str, tensorflow.Tensor]

Process batch of tensorflow images, resizing, normalizing,
and standardizing.

Parameters:

- **img** (*tf.Tensor*) - Batch of tensorflow images (uint8).
- **normalizer** (*hx.norm.StainNormalizer**,**optional*) - Normalizer.
Defaults to None.
- **standardize** (*bool**,**optional*) - Standardize images. Defaults to True.
- **resize_px** (*Optional**[**int**]**,**optional*) - Resize images. Defaults to None.
- **resize_method** (*str**,**optional*) - Resize method. Defaults to 'lanczos3'.
- **resize_aa** (*bool**,**optional*) - Apply antialiasing during resizing.
Defaults to True.

Returns:

Processed image.

Return type:

Dict[str, tf.Tensor]

histox.io.tensorflow.print_tfrecord(*target: str*) → None[[source]](_modules/histox/io/tensorflow.html#print_tfrecord)

Print the slide names (and locations, if present) for records
in the given tfrecord file.

Parameters:

**target** - Path to the tfrecord file or folder containing tfrecord files.

histox.io.tensorflow.process_image(*record: tensorflow.Tensor | Dict[str, tensorflow.Tensor]*, **args: Any*, *standardize: bool = False*, *augment: bool | str = False*, *transform: Callable | None = None*, *size: int | None = None*) → Tuple[Dict | tensorflow.Tensor, ...]

Applies augmentations and/or standardization to an image Tensor.

Parameters:

**record** (*Union**[**tf.Tensor**,**Dict**[**str**,**tf.Tensor**]**]*) - Image Tensor.

Keyword Arguments:

- **standardize** (*bool**,**optional*) - Standardize images. Defaults to False.
- **augment** (*str**or**bool*) - 

Image augmentations to perform. Augmentations include:

- `'x'`: Random horizontal flip
- `'y'`: Random vertical flip
- `'r'`: Random 90-degree rotation
- `'j'`: Random JPEG compression (50% chance to compress with quality between 50-100)
- `'b'`: Random Gaussian blur (10% chance to blur with sigma between 0.5-2.0)

Combine letters to define augmentations, such as `'xyrj'`.
A value of True will use `'xyrjb'`.
Note: this function does not support stain augmentation.
- **transform** (*Callable**,**optional*) - Arbitrary transform function.
Performs transformation after augmentations but before standardization.
Defaults to None.
- **size** (*int**,**optional*) - Set the image shape. Defaults to None.

histox.io.tensorflow.read_and_return_record(*record: bytes*, *parser: Callable*, *assign_slide: bytes | None = None*) → Example[[source]](_modules/histox/io/tensorflow.html#read_and_return_record)

Process raw TFRecord bytes into a format that can be written with
`tf.io.TFRecordWriter`.

Parameters:

- **record** (*bytes*) - Raw TFRecord bytes (unparsed)
- **parser** (*Callable*) - TFRecord parser, as returned by
`hx.io.get_tfrecord_parser()`
- **assign_slide** (*str**,**optional*) - Slide name to override the record with.
Defaults to None.

Returns:

Dictionary mapping record key to a tuple containing (bytes, dtype).

histox.io.tensorflow.serialized_record(*slide: bytes*, *image_raw: bytes*, *loc_x: int = 0*, *loc_y: int = 0*) → bytes[[source]](_modules/histox/io/tensorflow.html#serialized_record)

Serialize a record for TFRecord storage.

The serialized record will be in a data format ready to be written
by a TFRecordWriter.

Parameters:

- **slide** (*bytes*) - Slide name.
- **image_raw** (*bytes*) - Image bytes.
- **loc_x** (*int**,**optional*) - X coordinate of image. Defaults to 0.
- **loc_y** (*int**,**optional*) - Y coordinate of image. Defaults to 0.

Returns:

Serialized record.

Return type:

bytes

histox.io.tensorflow.shuffle_tfrecord(*target: str*) → None[[source]](_modules/histox/io/tensorflow.html#shuffle_tfrecord)

Shuffle records in a TFRecord, saving the original to a .old file.

Parameters:

**target** - Path to the tfrecord file.

histox.io.tensorflow.shuffle_tfrecords_by_dir(*directory: str*) → None[[source]](_modules/histox/io/tensorflow.html#shuffle_tfrecords_by_dir)

For each TFRecord in a directory, shuffle records in the TFRecord,
saving the original to a .old file.

Parameters:

**directory** - Path to the directory containing tfrecord files.

histox.io.tensorflow.split_tfrecord(*tfrecord_file: str*, *output_folder: str*) → None[[source]](_modules/histox/io/tensorflow.html#split_tfrecord)

Split records from a single tfrecord into individual tfrecord
files, stratified by slide.

Parameters:

- **tfrecord_file** (*str*) - Path to tfrecord file.
- **output_folder** (*str*) - Path to output folder.

histox.io.tensorflow.tfrecord_example(*slide: bytes*, *image_raw: bytes*, *loc_x: int | None = 0*, *loc_y: int | None = 0*) → Example[[source]](_modules/histox/io/tensorflow.html#tfrecord_example)

Return a Tensorflow Data example for TFRecord storage.

Parameters:

- **slide** (*bytes*) - Slide name.
- **image_raw** (*bytes*) - Image bytes.
- **loc_x** (*Optional**[**int**]**,**optional*) - X coordinate of image. Defaults to 0.
- **loc_y** (*Optional**[**int**]**,**optional*) - Y coordinate of image. Defaults to 0.

Returns:

Tensorflow Data example.

Return type:

Example

histox.io.tensorflow.transform_tfrecord(*origin: str*, *target: str*, *assign_slide: str | None = None*, *hue_shift: float | None = None*, *resize: float | None = None*) → None[[source]](_modules/histox/io/tensorflow.html#transform_tfrecord)

Transform images in a single tfrecord.

Can perform hue shifting, resizing, or re-assigning slide label.

Parameters:

- **origin** - Path to the original tfrecord file.
- **target** - Path to the new tfrecord file.
- **assign_slide** - If provided, will assign this slide name to all
records in the new tfrecord.
- **hue_shift** - If provided, will shift the hue of all images by
this amount.
- **resize** - If provided, will resize all images to this size.