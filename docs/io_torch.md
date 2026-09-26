# histox.io.torch

The purpose of this module is to provide a performant, backend-agnostic TFRecord reader and interleaver to use as
input for PyTorch models. Its TFRecord reader is a modified and optimized version of
[vahidk/tfrecord](https://github.com/vahidk/tfrecord), included as the module `histox.tfrecord`. TFRecord file reading and
interleaving is supervised by `histox.io.torch.interleave()`, while the
`histox.io.torch.interleave_dataloader()` function provides a PyTorch DataLoader object which can be directly used.

PyTorch-specific I/O utilities.

*class*histox.io.torch.InterleaveIterator(**args: Any*, ***kwargs: Any*)[[source]](_modules/histox/io/torch/iterable.html#InterleaveIterator)

Bases: `IterableDataset`

Pytorch Iterable Dataset that interleaves tfrecords with the
interleave() function below. Serves as a bridge between the python
generator returned by interleave() and the pytorch DataLoader class.

__init__(*tfrecords: List[str]*, ***, *img_size: int | None = None*, *labels: Dict[str, str] | Dict[str, int] | Dict[str, List[float]] | None = None*, *incl_slidenames: bool = False*, *incl_loc: bool = False*, *rank: int = 0*, *num_replicas: int = 1*, *augment: str | bool = False*, *standardize: bool = True*, *num_tiles: int | None = None*, *infinite: bool = True*, *prob_weights: Dict[str, float] | None = None*, *normalizer: [StainNormalizer](norm.html#histox.norm.StainNormalizer) | None = None*, *clip: List[int] | None = None*, *chunk_size: int = 1*, *use_labels: bool = True*, *model_type: str = 'classification'*, *onehot: bool = False*, *indices: ndarray | None = None*, *from_wsi: bool = False*, *tile_um: int | None = None*, *rois: List[str] | None = None*, *roi_method: str = 'auto'*, *pool: Any | None = None*, *transform: Any | None = None*, ***interleave_kwargs*) → None[[source]](_modules/histox/io/torch/iterable.html#InterleaveIterator.__init__)

Pytorch IterableDataset that interleaves tfrecords with
`histox.io.torch.interleave()`.

Parameters:

**tfrecords** (*list**(**str**)*) - Path to tfrecord files to interleave.

Keyword Arguments:

- **img_size** (*int*) - Image width in pixels.
- **labels** (*dict**,**optional*) - Dict mapping slide names to labels.
Defaults to None.
- **incl_slidenames** (*bool**,**optional*) - Include slide names when iterated
(returns image, label, slide). Defaults to False.
- **incl_loc** (*bool**,**optional*) - Include location info (tile center
coordinates). Returns samples in the form `(returns ..., loc_x,
loc_y)`. Defaults to False.
- **rank** (*int**,**optional*) - Which GPU replica this dataset is used for.
Assists with synchronization across GPUs. Defaults to 0.
- **num_replicas** (*int**,**optional*) - Total number of GPU replicas.
Defaults to 1.
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
- **standardize** (*bool**,**optional*) - Standardize images to mean 0 and
variance of 1. Defaults to True.
- **num_tiles** (*int**,**optional*) - Dict mapping tfrecord names to number
of total tiles. Defaults to None.
- **infinite** (*bool**,**optional*) - Inifitely loop through dataset.
Defaults to True.
- **prob_weights** (*list**(**float**)**,**optional*) - Probability weights for
interleaving tfrecords. Defaults to None.
- **normalizer** ([`histox.norm.StainNormalizer`](norm.html#histox.norm.StainNormalizer), optional) - Normalizer. Defaults to None.
- **clip** (*list**(**int**)**,**optional*) - Array of maximum tiles to take for each
tfrecord. Defaults to None.
- **chunk_size** (*int**,**optional*) - Chunk size for image decoding.
Defaults to 1.
- **use_labels** (*bool**,**optional*) - Enable use of labels (disabled for
non-conditional GANs). Defaults to True.
- **model_type** (*str**,**optional*) - Used to generate random labels
(for StyleGAN2). Not required. Defaults to 'classification'.
- **onehot** (*bool**,**optional*) - Onehot encode outcomes. Defaults to False.
- **indices** (*numpy.ndarray**,**optional*) - Indices in form of array,
with np.loadtxt(index_path, dtype=np.int64) for each tfrecord.
Defaults to None.
- **max_size** (*bool**,**optional*) - Unused argument present for legacy
compatibility; will be removed.
- **from_wsi** (*bool*) - Generate predictions from tiles dynamically
extracted from whole-slide images, rather than TFRecords.
Defaults to False (use TFRecords).
- **tile_um** (*int**,**optional*) - Size of tiles to extract from WSI, in
microns. Only used if from_wsi=True. Defaults to None.
- **rois** (*list**(**str**)**,**optional*) - List of ROI paths. Only used if
from_wsi=True. Defaults to None.
- **roi_method** (*str**,**optional*) - Method for extracting ROIs. Only used if
from_wsi=True. Defaults to 'auto'.
- **pool** (*multiprocessing.Pool*) - Shared multiprocessing pool. Useful
if `from_wsi=True`, for sharing a unified processing pool between
dataloaders. Defaults to None.
- **transform** (*Callable**,**optional*) - Arbitrary torchvision transform
function. Performs transformation after augmentations but
before standardization. Defaults to None.
- **tfrecord_parser** (*Callable**,**optional*) - Custom parser for TFRecords.
Defaults to None.

*class*histox.io.torch.IndexedInterleaver(**args: Any*, ***kwargs: Any*)[[source]](_modules/histox/io/torch/indexed.html#IndexedInterleaver)

Bases: `IndexedMultiTFRecordDataset`

__init__(*tfrecords: List[str]*, ***, *labels: Dict[str, str] | Dict[str, int] | Dict[str, List[float]] | None = None*, *incl_slidenames: bool = False*, *incl_loc: bool = False*, *rank: int = 0*, *num_replicas: int = 1*, *augment: bool | str = False*, *standardize: bool = True*, *normalizer: [StainNormalizer](norm.html#histox.norm.StainNormalizer) | None = None*, *clip: Dict[str, int] | None = None*, *use_labels: bool = True*, *onehot: bool = False*, *indices: List[ndarray] | None = None*, *transform: Any | None = None*, *tfrecord_parser: Callable | None = None*, ***kwargs*)[[source]](_modules/histox/io/torch/indexed.html#IndexedInterleaver.__init__)

Interleave TFRecords with an indexable `torch.utils.data.Dataset`.

Provides an alternative TFRecord IO pipeline to `InterleaveIterator`,
which only supports Iterable-style datasets. This class supports
both Iterable and Indexable datasets.

Differences from `InterleaveIterator`:

- Supports direct indexing.
- No "infinite" argument. Looping is handled by the dataloader.
- No "prob_weights" argument. Sampling is handled by the dataloader.
- Does not support dynamic reading from WSI ("from_wsi", "tile_um", "rois", "roi_method", and "pool" arguments).

Parameters:

**tfrecords** (*list**(**str**)*) - Path to tfrecord files to interleave.

Keyword Arguments:

- **labels** (*dict**,**optional*) - Dict mapping slide names to labels.
Defaults to None.
- **incl_slidenames** (*bool**,**optional*) - Include slide names when iterated
(returns image, label, slide). Defaults to False.
- **incl_loc** (*bool**,**optional*) - Include location info (tile center
coordinates). Returns samples in the form `(returns ..., loc_x,
loc_y)`. Defaults to False.
- **rank** (*int**,**optional*) - Which GPU replica this dataset is used for.
Assists with synchronization across GPUs. Defaults to 0.
- **num_replicas** (*int**,**optional*) - Total number of GPU replicas.
Defaults to 1.
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
- **standardize** (*bool**,**optional*) - Standardize images to mean 0 and
variance of 1. Defaults to True.
- **normalizer** ([`histox.norm.StainNormalizer`](norm.html#histox.norm.StainNormalizer), optional) - Normalizer. Defaults to None.
- **clip** (*list**(**int**)**,**optional*) - Array of maximum tiles to take for each
tfrecord. Defaults to None.
- **use_labels** (*bool**,**optional*) - Enable use of labels (disabled for
non-conditional GANs). Defaults to True.
- **onehot** (*bool**,**optional*) - Onehot encode outcomes. Defaults to False.
- **indices** (*numpy.ndarray**,**optional*) - Indices in form of array,
with np.loadtxt(index_path, dtype=np.int64) for each tfrecord.
Defaults to None.
- **transform** (*Callable**,**optional*) - Arbitrary torchvision transform
function. Performs transformation after augmentations but
before standardization. Defaults to None.
- **tfrecord_parser** (*Callable**,**optional*) - Custom parser for TFRecords.
Defaults to None.
- **compression_type** (*str**,**optional*) - Compression type for TFRecords.
Either 'gzip' or None. Defaults to None.
- **shuffle** (*bool*) - Shuffle records within TFRecord files during
reading. Defaults to False.
- **seed** (*int**,**optional*) - Seed for random TFRecord interleaving and
intra-tfrecord shuffling. Defaults to None.

build_parser(*tfrecord_parser: Callable | None = None*) → Callable[[source]](_modules/histox/io/torch/indexed.html#IndexedInterleaver.build_parser)

Build a parser function for TFRecords.

The parser will be responsible for processing images and labels.