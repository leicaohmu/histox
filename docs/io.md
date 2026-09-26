# histox.io

This module contains utility functions for working with TFRecords, cross-compatible
with both Tensorflow and PyTorch.

Functions included in this module assist with processing TFRecords, detecting image and data format,
extracting tiles, splitting and merging TFrecords, and a variety of other manipulations.

Additional Tensorflow-specific TFRecord reading/writing utility functions are
available in [`histox.io.tensorflow`](io_tensorflow.html#module-histox.io.tensorflow), and additional PyTorch-specific
functions are in [`histox.io.torch`](io_torch.html#module-histox.io.torch).

histox.io.convert_dtype(*img: Any*, *dtype: dtype | tf.dtypes.DType | torch.dtype*) → Any[[source]](_modules/histox/io/io_utils.html#convert_dtype)

Converts an image from one type to another.

Images can be converted to and from numpy arrays, Torch Tensors and
Tensorflow Tensors. Images can also be converted from standardized
float images to RGB uint8 images, and vice versa.

Supported formats for starting and ending dtype include:

| `np.uint8` | Image in RGB (WHC) uint8 format. |
| --- | --- |
| `np.float32` | RGB (WHC) image. If the source image is a numpy uint8 or torch uint8, it will be standardized with `(img / 127.5) - 1`. If the source image is a tensorflow image, standardization uses `tf.image.per_image_standardization()`. |
| `torch.uint8` | Image in RGB (CWH) uint8 format. |
| `torch.float32` | Image converted with `(img / 127.5) - 1` and WHC -> CWH. |
| `tf.uint8` | Image in RGB (WHC) uint8 format. |
| `tf.float32` | Image converted with `tf.image.per_image_standardization()` |

Parameters:

- **img** (*Any*) - Input image or batch of images.
- **start_dtype** (*type*) - Starting dtype.
- **end_dtype** (*type*) - Target dtype for conversion.

Returns:

Converted image or batch of images.

histox.io.detect_tfrecord_format(*tfr: str*) → Tuple[List[str] | None, str | None][[source]](_modules/histox/io/io_utils.html#detect_tfrecord_format)

Detects tfrecord format.

Parameters:

**tfr** (*str*) - Path to tfrecord.

Returns:

A tuple containing

> list(str): List of detected features.
> 
> 
> 
> 
> str: Image file type (png/jpeg)

histox.io.extract_tiles(*tfrecord: str*, *destination: str*) → None[[source]](_modules/histox/io.html#extract_tiles)

Extracts images within a TFRecord to a destination folder.

Parameters:

- **tfrecord** (*str*) - Path to tfrecord.
- **destination** (*str*) - Destination path to write loose images.

histox.io.get_locations_from_tfrecord(*filename: str*) → List[Tuple[int, int]][[source]](_modules/histox/io.html#get_locations_from_tfrecord)

Return list of tile locations (X, Y) for all items in the TFRecord.

histox.io.get_tfrecord_by_index(*tfrecord: str*, *index: int*, ***, *compression_type: str | None = None*, *index_array: ndarray | None = None*) → Dict[[source]](_modules/histox/util/tfrecord2idx.html#get_tfrecord_by_index)

Read a specific record in a TFRecord file.

Parameters:

- **tfrecord** (*str*) - TFRecord file to read.
- **index** (*int*) - Index of record to read from the file.
- **compression_type** (*str*) - Type of compression in the TFRecord file.
Either 'gzip' or None. Defaults to None.

Returns:

A dictionary mapping record names (e.g., `'slide'`, `'image_raw'`,
`'loc_x'`, and `'loc_y'`) to their values. `'slide'` will be a
string, `image_raw` will be bytes, and `'loc_x'` and `'loc_y'`
will be int.

Raises:

- **histox.error.EmptyTFRecordsError** - If the file is empty.
- **histox.error.InvalidTFRecordIndex** - If the given index cannot be found.

histox.io.get_tfrecord_by_location(*tfrecord: str*, *location: Tuple[int, int]*, *decode: bool = True*, ***, *locations_array: List[Tuple[int, int]] | None = None*, *index_array: ndarray | None = None*) → Any[[source]](_modules/histox/io.html#get_tfrecord_by_location)

Reads and returns an individual record from a tfrecord by index,
including slide name and processed image data.

Parameters:

- **tfrecord** (*str*) - Path to TFRecord file.
- **location** (*tuple**(**int**,**int**)*) - `(x, y)` tile location.
Searches the TFRecord for the tile that corresponds to this
location.
- **decode** (*bool*) - Decode the associated record, returning Tensors.
Defaults to True.

Returns:

Unprocessed raw TFRecord bytes if `decode=False`, otherwise a
tuple containing `(slide, image)`, where `image` is a
uint8 Tensor.

histox.io.get_tfrecord_parser(*tfrecord_path: str*, *features_to_return: Iterable[str] = None*, *decode_images: bool = True*, *standardize: bool = False*, *normalizer: [StainNormalizer](norm.html#histox.norm.StainNormalizer) | None = None*, *augment: bool = False*, ***kwargs*) → Callable[[source]](_modules/histox/io/torch/data_utils.html#get_tfrecord_parser)

Gets tfrecord parser using dareblopy reader. Torch implementation;
different than hx.io.tensorflow

Parameters:

- **tfrecord_path** (*str*) - Path to tfrecord to parse.
- **features_to_return** (*list**or**dict**,**optional*) - Designates format for how
features should be returned from parser. If a list of feature names
is provided, the parsing function will return tfrecord features as
a list in the order provided. If a dictionary of labels (keys)
mapping to feature names (values) is provided, features will be
returned from the parser as a dictionary matching the same format.
If None, will return all features as a list.
- **decode_images** (*bool**,**optional*) - Decode raw image strings into image
arrays. Defaults to True.
- **standardize** (*bool**,**optional*) - Standardize images into the range (0,1).
Defaults to False.
- **normalizer** ([`histox.norm.StainNormalizer`](norm.html#histox.norm.StainNormalizer)) - Stain normalizer
to use on images. Defaults to None.
- **augment** (*str**or**bool*) - 

Image augmentations to perform. Augmentations include:

- `'x'`: Random horizontal flip
- `'y'`: Random vertical flip
- `'r'`: Random 90-degree rotation
- `'j'`: Random JPEG compression (50% chance to compress with quality between 50-100)
- `'b'`: Random Gaussian blur (10% chance to blur with sigma between 0.5-2.0)

Combine letters to define augmentations, such as `'xyrjn'`.
A value of True will use `'xyrjb'`.
Note: this function does not support stain augmentation.

Returns:

A tuple containing

> func: Parsing function
> 
> 
> 
> 
> dict: Detected feature description for the tfrecord

histox.io.get_tfrecord_length(*tfrecord: str*) → int[[source]](_modules/histox/util/tfrecord2idx.html#get_tfrecord_length)

Return the number of records in a TFRecord file.

Uses an index file if available, otherwise iterates through
the file to find the total record length.

Parameters:

**tfrecord** (*str*) - Path to TFRecord.

Returns:

Number of records.

Return type:

int

histox.io.read_and_return_record(*record: bytes*, *parser: Callable*, *assign_slide: str | None = None*) → Dict[[source]](_modules/histox/io/torch/data_utils.html#read_and_return_record)

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

histox.io.serialized_record(*slide: bytes*, *image_raw: bytes*, *loc_x: int = 0*, *loc_y: int = 0*)[[source]](_modules/histox/io/torch/data_utils.html#serialized_record)

Returns a serialized example for TFRecord storage, ready to be written
by a TFRecordWriter.

histox.io.tfrecord_has_locations(*filename: str*, *check_x: int = True*, *check_y: bool = False*) → bool[[source]](_modules/histox/io.html#tfrecord_has_locations)

Check if a given TFRecord has location information stored.

histox.io.update_manifest_at_dir(*directory: str*, *force_update: bool = False*) → str | Dict | None[[source]](_modules/histox/io.html#update_manifest_at_dir)

Log number of tiles in each TFRecord file present in the given
directory and all subdirectories, saving manifest to file within
the parent directory.

histox.io.write_tfrecords_multi(*input_directory: str*, *output_directory: str*) → None[[source]](_modules/histox/io.html#write_tfrecords_multi)

Write multiple tfrecords, one for each slide, from a directory of images.

Scans a folder for subfolders, assumes subfolders are slide names.
Assembles all image tiles within subfolders, assuming the subfolder is the
slide name. Collects all image tiles and exports into multiple tfrecord
files, one for each slide.

Parameters:

- **input_directory** (*str*) - Directory of images.
- **output_directory** (*str*) - Directory in which to write TFRecord files.

histox.io.write_tfrecords_single(*input_directory: str*, *output_directory: str*, *filename: str*, *slide: str*) → int[[source]](_modules/histox/io.html#write_tfrecords_single)

Scans a folder for image tiles, annotates using the provided slide,
exports into a single tfrecord file.

Parameters:

- **input_directory** (*str*) - Directory of images.
- **output_directory** (*str*) - Directory in which to write TFRecord file.
- **filename** (*str*) - TFRecord filename (without path).
- **slide** (*str*) - Slide name to assign to records inside TFRecord.

Returns:

Number of records written.

Return type:

int

histox.io.write_tfrecords_merge(*input_directory: str*, *output_directory: str*, *filename: str*) → int[[source]](_modules/histox/io.html#write_tfrecords_merge)

Scans a folder for subfolders, assumes subfolders are slide names.
Assembles all image tiles within subfolders and labels using the provided
annotation_dict, assuming the subfolder is the slide name. Collects all
image tiles and exports into a single tfrecord file.

Parameters:

- **input_directory** (*str*) - Directory of images.
- **output_directory** (*str*) - Directory in which to write TFRecord file.
- **filename** (*str*) - TFRecord filename (without path).

Returns:

Number of records written.

Return type:

int

## histox.io.preservedsite

io.preservedsite.generate_crossfolds(**args*, ***kwargs*)

MagicMock is a subclass of Mock with default implementations
of most of the magic methods. You can use MagicMock without having to
configure the magic methods yourself.

If you use the spec or spec_set arguments then *only* magic
methods that exist in the spec will be created.

Attributes and the return value of a MagicMock will also be MagicMocks.