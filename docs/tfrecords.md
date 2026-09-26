# TFRecords: Reading and Writing

TFRecords are binary files designed for storing large amounts of data. In HistoX, TFRecords are used to store compressed image tiles extracted from whole-slide images. TFRecords are used instead of loose image files (such as `*.jpg` or `*.png`) because they are compact, more easily distributed, and significantly improve data reading efficiency during model training. TFRecords were originally designed for Tensorflow, but they can also be used with PyTorch.

The following sections describe the TFRecord data format and provide examples of how to create, read, and manipulate TFRecords using HistoX.

## TFRecord Format

TFRecords are binary files that contain a sequence of records, where each record represents an individual image tile. Each record contains a serialized [protocol buffer](https://protobuf.dev/overview/) with a list of named features. Each feature can be a list of bytes, floats, or integers. TFRecords are expected to have the following features:

- **"image_raw"**: Bytes containing the image data (either JPG or PNG).
- **"slide"**: Bytes containing the slide name (in UTF-8 format).
- **"loc_x"**: Integer containing the x-coordinate of the tile (optional).
- **"loc_y"**: Integer containing the y-coordinate of the tile (optional).

HistoX expects each TFRecord to contain images from only a single slide, with the TFRecord name matching the slide name. The `loc_x` and `loc_y` features are optional, but are required for some operations (such as generating TFRecord heatmaps).

Note

When reading TFRecords with Tensorflow, records are internally decoded using `tf.train.Example`. When Tensorflow is not being used (such as when using the PyTorch backend), tfrecords are decoded using `hx.util.example_pb2.Example`, providing an alternative decoder that does not require Tensorflow. Tensorflow's `tf.train.Example` and HistoX's `hx.util.example_pb2.Example` are identical, except that `hx.util.example_pb2.Example` does not require Tensorflow and supports `protobuf` version 4.

## TFRecord Indices

HistoX uses TFRecord index files to keep track of the internal structure of each TFRecord, improving efficiency of data reading. These index files are automatically built and stored in the same directory as the TFRecords upon first use. A TFRecord index is an `*.npz` file with the same name as the TFRecord, but with the `*.index.npz` extension. A TFRecord index contains the following fields:

- **"arr_0"**: An array of shape `(n_tiles, 2)` containing the starting bytes and length of each record.
- **"locations"**: An array of shape `(n_tiles, 2)` containing the x- and y-coordinates of each tile.

Index files for an entire dataset can be rebuilt using `histox.Dataset.rebuild_index()`. You can manually create an index file for a single TFRecord using `hx.util.tfrecord2idx.create_index()`.

## Creating TFRecords

### From a Dataset

The typical way to create TFRecords is to use the `histox.Dataset.extract_tiles()` function, as described in [Slide Processing](slide_processing.html#filtering). TFRecords will be exported to the destination configured in the [`histox.Dataset`](dataset.html#histox.Dataset) object (see: [Datasets](datasets_and_val.html#datasets-and-validation)).

### From a slide

A TFRecord file for a single slide can be manually created using `histox.WSI.extract_tiles()` function. The first argument of this function is the TFRecord destination folder.

### From a directory of images

A directory of loose image files can be assembled into a TFRecord using [`histox.io.write_tfrecords_single()`](io.html#histox.io.write_tfrecords_single):

```
hx.io.write_tfrecords_single(
 '/path/to/images',
 '/path/to/destination',
 filename='filename',
 slide='slide',
)
```

A nested directory of loose image tiles, organized into subdirectory by slide name, can be simultaneously assembled into multiple TFRecords (one for each slide) using [`histox.io.write_tfrecords_multi()`](io.html#histox.io.write_tfrecords_multi). Slide names are determined from the subdirectory names:

```
hx.io.write_tfrecords_multi(
 '/path/to/nested_images',
 '/path/to/destination'
)
```

## Inspecting TFRecords

### Individual TFRecords

The quickest way to inspect a TFRecord is to use `histox.TFRecord`:

```
>>> import histox as hx
>>> tfr = hx.TFRecord('/path/to/tfrecord')
```

An index file will be automatically created if one is not found. To disable automatic index creation, set `create_index=False`.

The TFRecord object has several useful attributes:

```
>>> tfr.fields
['image_raw', 'slide', 'loc_x', 'loc_y']
>>> tfr.img_format
'jpeg'
>>> tfr.length
1000
>>> tfr.locations
[(768, 256), (768, 512), ...]
```

The `fields` attribute is a list of the fields in the TFRecord.

The `img_format` attribute is the image format of the TFRecord (either `"jpeg"` or `"png"`).

The `length` attribute is the number of tiles in the TFRecord.

The `locations` attribute is a list of the x- and y- center coordinates of each tile, if available, otherwise None.

### Inspecting Datasets

The [`histox.Dataset`](dataset.html#histox.Dataset) object provides several methods for inspecting the TFRecords in a dataset generated through `histox.Dataset.extract_tiles()`.

The `histox.Dataset.summary()` method provides a summary of the dataset, including the location TFRecords are stored and the number of total number of tiles across all TFRecords in the dataset.

```
# Prepare a dataset of image tiles.
dataset = project.dataset(
 tile_px=299, # Tile size, in pixels.
 tile_um='10x' # Tile size, in microns or magnification.
)
dataset.summary()
```

```
Overview:
╒===============================================╕
│ Configuration file: │ /mnt/data/datasets.json │
│ Tile size (px): │ 299 │
│ Tile size (um): │ 10x │
│ Slides: │ 941 │
│ Patients: │ 941 │
│ Slides with ROIs: │ 941 │
│ Patients with ROIs: │ 941 │
╘===============================================╛

Filters:
╒====================╕
│ Filters: │ {} │
├--------------------┤
│ Filter Blank: │ [] │
├--------------------┤
│ Min Tiles: │ 0 │
╘====================╛

Sources:

TCGA_LUNG
╒==============================================╕
│ slides │ /mnt/raid/SLIDES/TCGA_LUNG │
│ roi │ /mnt/raid/SLIDES/TCGA_LUNG │
│ tiles │ /mnt/rocket/tiles/TCGA_LUNG │
│ tfrecords │ /mnt/rocket/tfrecords/TCGA_LUNG/ │
│ label │ 299px_10x │
╘==============================================╛

Number of tiles in TFRecords: 284114
Annotation columns:
Index(['patient', 'subtype', 'site', 'slide'],
 dtype='object')
```

The `histox.Dataset.tfrecords()` method returns a list of paths to tfrecords.

```
>>> tfrecords = dataset.tfrecords()
>>> len(tfrecords)
941
>>> tfrecords[0]
'/path/to/tfrecords1'
```

The `histox.Dataset.num_tiles` attribute returns the total number of tiles across all TFRecords in the dataset.

```
>>> dataset.num_tiles
284114
```

Finally, the `histox.Dataset.manifest()` method returns a dictionary mapping TFRecord paths to the number tiles in each TFRecord. Each value returned by the dictionary is a nested dictionary with two keys: `"total"`, which is the total number of tiles in the TFRecords, and `"clipped"`, which is the number of tiles that will be taken from the TFRecord as a result of [clipping/undersampling](dataloaders.html#sampling).

```
>>> dataset.manifest()
{'/path/to/tfrecords1': {'total': 1000, 'clipped': 512},
 '/path/to/tfrecords2': {'total': 2000, 'clipped': 512},
 ...}
```

## Reading TFRecords

HistoX provides several tools for reading and parsing TFRecords. These tools are intended for debugging and development, and are not recommended for model training. Higher-level dataloaders, which supervise sampling, shuffling, sharding, batching, labeling, and augmenting, are discussed in [Dataloaders: Sampling and Augmentation](dataloaders.html#dataloaders).

### Reading a single image tile

To get a single parsed record according to its index, use `histox.TFRecord.__getitem__()`, which returns a dictionary of the parsed record:

```
>>> import histox as hx
>>> tfr = hx.TFRecord('/path/to/tfrecord')
>>> tfr[0]
{'image_raw': b'...', 'slide': 'SLIDE_NAME', 'loc_x': 0, 'loc_y': 0}
```

The `'image_raw'` field contains raw image bytes, in either JPG or PNG format.

To get a single parsed record according to its location, use `histox.TFRecord.get_record_by_xy()`, which returns the slide name and image bytes:

```
>>> tfr.get_record_by_xy(768, 256)
('SLIDE_NAME', b'...')
```

Image bytes can be decoded into Tensors (according to the active backend) using `histox.io.decode_image()`:

```
>>> import histox as hx
>>> slide, image = tfr.get_record_by_xy(768, 256)
>>> print(type(image))
<class 'bytes'>
>>> hx.io.decode_image(image)
<torch.Tensor shape=(256, 256, 3) dtype=torch.uint8
```

### Reading from a single TFRecord

The function `histox.tfrecord_loader()` provides an interface for reading images from a single TFRecord in sequence. Start by loading the TFRecord index, creating one if it does not already exist:

```
>>> import histox as hx
>>> tfr = '/path/to/tfrecords'
>>> hx.io.tfrecord2idx.create_index(tfr)
>>> index = hx.io.tfrecord2idx.load_index(tfr)
```

Then, use `histox.tfrecord_loader()` to create a generator that yields parsed records from the TFRecord:

```
>>> loader = hx.tfrecord.tfrecord_loader(tfr, index)
>>> record = next(iter(loader))
{'image_raw': <np.ndarray>, 'slide': <np.ndarray>, 'loc_x': [0], 'loc_y': [0]}
```

Both `"image_raw"` and `"slide"` fields are returned as bytes in numpy arrays. The `"loc_x"` and `"loc_y"` fields are returned as integers. The image and slide name can be decoded using `histox.io.decode_image()` and `.decode('utf-8')`, respectively:

```
>>> image = hx.io.decode_image(bytes(record['image_raw']))
>>> slide = bytes(record['slide']).decode('utf-8')
```

This iterator can be used to read all images from a TFRecord in sequence:

```
>>> for record in loader:
... image = hx.io.decode_image(bytes(record['image_raw']))
... slide = bytes(record['slide']).decode('utf-8')
```

The iterator can be split into separate shards (data partitions) with the `shard` argument, a tuple of `(shard_id, n_shards)`. This is useful for parallelizing data reading across multiple processes, threads, or compute nodes:

```
>>> loader = hx.tfrecord.tfrecord_loader(tfr, index, shard=(0, 2))
```

Data sharding ensures that each shard reads a unique subset of the data, and that each record is read exactly once.

An index file is recommended for improving efficiency of data reading, and required if using data sharding.

### Interleaving multiple TFRecords

You can also interleave multiple TFRecords using `histox.multi_tfrecord_loader()`. This function takes a list of TFRecord paths and a list of corresponding TFRecord indices, and returns a generator that randomly samples from TFRecords and parses the records:

```
>>> import histox as hx
>>> tfrs = ['/path/to/tfrecord1', '/path/to/tfrecord2']
>>> indices = [hx.io.tfrecord2idx.load_index(tfr) for tfr in tfrs]
>>> loader = hx.tfrecord.multi_tfrecord_loader(tfrs, indices)
>>> record = next(iter(loader))
{'image_raw': <np.ndarray>, 'slide': <np.ndarray>, 'loc_x': [0], 'loc_y': [0]}
```

By default, records are sampled from TFRecords with equal probability (i.e. uniform sampling). You can also specify a list of weights to sample from TFRecords with different probabilities (i.e. weighted sampling) via the `weights` argument. The weights should be a list of floats, one for each TFRecord, that sum to 1.0:

```
>>> loader = hx.tfrecord.multi_tfrecord_loader(tfrs, indices, weights=[0.5, 0.5])
```

Records will be sampled infinitely by default. To disable infinite sampling, set `infinite=False`.

TFRecord sharding is also supported for `multi_tfrecord_loader()` via the `shard` argument.