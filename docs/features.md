# Generating Features

Converting images into feature vectors is a common step for many machine learning tasks, including [feature space analysis](activations) and [multiple-instance learning (MIL)](mil). HistoX provides a simple API for generating features from image tiles and includes several pretrained feature extractors. You can see a list of all available feature extractors with `histox.list_extractors()`.

## Generating Features

The first step in generating features from a dataset of images is creating a feature extractor. Many types of feature extractors can be used, including imagenet-pretrained models, models finetuned in HistoX, histology-specific pretrained feature extractors (ie. "foundation models"), or fine-tuned SSL models. In all cases, feature extractors are built with [`histox.build_feature_extractor()`](histox.html#histox.build_feature_extractor), and features are generated for a [Dataset](datasets_and_val) using `histox.Dataset.generate_feature_bags()`, as described below.

```
# Build a feature extractor
ctranspath = hx.build_feature_extractor('ctranspath')

# Generate features for a dataset
dataset.generate_feature_bags(ctranspath, outdir='/path/to/features')
```

## Pretrained Extractors

HistoX includes several pathology-specific feature extractors, also referred to as foundation models, pretrained on large-scale histology datasets.

| Model | Type | WSIs | Input size | Dim | Source | Package | Link |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Virchow** | DINOv2 | 1.5M | 224 | 2560 | Paige | `histox` | [Paper](http://arxiv.org/pdf/2309.07778v5) |
| **CTransPath** | SRCL | 32K | 224 | 768 | Tencent AI Lab | `slideflow-gpl` | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S1361841522002043) |
| **RetCCL** | CCL | 32K | 256 | 2048 | Tencent AI Lab | `slideflow-gpl` | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S1361841522002730) |
| **Phikon** | iBOT | 6.1K | 224 | 768 | Owkin | `slideflow-noncommercial` | [Paper](https://www.medrxiv.org/content/10.1101/2023.07.21.23292757v2.full.pdf) |
| **PLIP** | CLIP | N/A | 224 | 512 | Zhao Lab | `slideflow-noncommercial` | [Paper](https://www.nature.com/articles/s41591-023-02504-3) |
| **UNI** | DINOv2 | 100K | 224 | 1024 | Mahmood Lab | `slideflow-noncommercial` | [Paper](https://www.nature.com/articles/s41591-024-02857-3) |
| **GigaPath** | DINOv2 | 170K | 256 | 1536 | Microsoft | `slideflow-noncommercial` | [Paper](https://aka.ms/gigapath) |

In order to respect the original licensing agreements, pretrained models are distributed in separate packages. The core `histox` package provides access to models under the **Apache-2.0** license, while models under **GPL-3.0** are available in the `slideflow-gpl` package. Models restricted to non-commercial use are available under the **CC BY-NC 4.0** license through the `slideflow-noncommercial` package.

### Loading weights

Pretrained feature extractors will automatically download their weights from Hugging Face upon creation. Some models, such as PLIP, GigaPath, UNI, and Phikon, require approval for access. Request approval on Hugging Face and ensure your local machine has been [authenticated](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication).

All pretrained models can also be loaded using local weights. Use the `weights` argument when creating a feature extractor.

```
# Load UNI with local weights
uni = hx.build_feature_extractor('uni', weights='../pytorch_model.bin')
```

### Image preprocessing

Each feature extractor includes a default image preprocessing pipeline that matches the original implementation. However, preprocessing can also be manually adjusted using various keyword arguments when creating a feature extractor.

- **resize**: `int` or `bool`. If an `int`, resizes images to this size. If `True`, resizes images to the input size of the feature extractor. Default is `False`.
- **center_crop**: `int` or `bool`. If an `int`, crops images to this size. If `True`, crops images to the input size of the feature extractor. Center-cropping happens after resizing, if both are used. Default is `False`.
- **interpolation**: `str`. Interpolation method for resizing images. Default is `bilinear` for most models, but is `bicubic` for GigaPath and Virchow.
- **antialias**: `bool`. Whether to apply antialiasing to resized images. Default is `False` (matching the default behavior of torchvision < 0.17).
- **norm_mean**: `list`. Mean values for image normalization. Default is `[0.485, 0.456, 0.406]` for all models except PLIP.
- **norm_std**: `list`. Standard deviation values for image normalization. Default is `[0.229, 0.224, 0.225]` for all models except PLIP.

Example:

```
# Load a feature extractor with custom preprocessing
extractor = hx.build_feature_extractor(
 'ctranspath',
 resize=224,
 interpolation='bicubic',
 antialias=True
)
```

Default values for these processing arguments are determined by the feature extractor. One notable exception to the standard preprocessing algorithm is GigaPath, for which images are resized first (default to 256x256) and then center cropped (default to 224x224), which mirrors the official implementation.

For transparency, you can see the current preprocessing pipeline with `extractor.transform`:

```
>>> import histox as hx
>>> ctranspath = hx.build_feature_extractor(
... 'ctranspath',
... resize=256,
... interpolation='bicubic',
... center_crop=224
... )
>>> ctranspath.transform
Compose(
 CenterCrop(size=(224, 224))
 Resize(size=256, interpolation=bicubic, max_size=None, antialias=False)
 Lambda()
 Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
)
```

### GigaPath

GigaPath is a DINOv2-based model from Microsoft/Providence trained on 170k whole-slide images and is bundled with `slideflow-noncommercial`. The GigaPath model includes additional dependencies which are not broadly compatible with all OS distributions, and are thus not installed by default. To install the GigaPath dependencies:

```
pip install slideflow-noncommercial[gigapath] git+ssh://git@github.com/prov-gigapath/prov-gigapath
```

GigaPath has two stages: a tile encoder and slide-level encoder. The tile encoder (`"gigapath.tile"`) works the same as all other feature extractors in HistoX. You can build this encoder directly:

```
# Build the tile encoder
gigapath_tile = hx.build_feature_extractor("gigapath.tile")

# Use the tile encoder
project.generate_feature_bags(gigapath_tile, ...)
```

or you can build the combined tile+slide model, and then use `gigapath.tile`:

```
# Build the tile encoder
gigapath = hx.build_feature_extractor("gigapath")

# Use the tile encoder
project.generate_feature_bags(gigapath.tile, ...)
```

As there are two stages to GigaPath, there are also separate model weights. As with other pretrained feature extractors, the weights will be auto-downloaded from Hugging Face upon first use if you are logged into Hugging Face and have been granted access to the repository. If you have manually downloaded the weights, these can be used with the following:

```
# Example of how to supply tile + slide weights
# For the full GigaPath model
gigapath = hx.build_feature_extractor(
 'gigapath',
 tile_encoder_weights='../pytorch_model.bin',
 slide_encoder_weights='../slide_encoder.pth'
)

# Or, just supply the tile weights
gigapath_tile = hx.build_feature_extractor(
 'gigapath.tile',
 weights='pytorch_model.bin'
)
```

Once feature bags have been generated and saved with the GigaPath tile encoder, you can then generate slide-level embeddings with `gigapath.slide`:

```
# Load GigaPath
gigapath = hx.build_feature_extractor('gigapath')

# Generate tile-level features
project.generate_feature_bags(gigapath.tile, ..., outdir='/gigapath_bags')

# Generate slide-level embeddings
gigapath.slide.generate_and_save('/gigapath_bags', outdir='/gigapath_embeddings')
```

In addition to running the tile and slide encoder steps separately, you can also run the combined pipeline all at once on a whole-slide image, generating a final slide-level embedding.

```
# Load GigaPath
gigapath = hx.build_feature_extractor('gigapath')

# Load slide
wsi = hx.WSI('slide.svs', tile_px=256, tile_um=128)

# Generate slide embedding
embedding = gigapath(wsi)
```

## ImageNet Features

To calculate features from an ImageNet-pretrained network, first build an imagenet feature extractor with [`histox.build_feature_extractor()`](histox.html#histox.build_feature_extractor). The first argument should be the name of an architecture followed by `_imagenet`, and the expected tile size should be passed to the keyword argument `tile_px`. You can optionally specify the layer from which to generate features with the `layers` argument; if not provided, it will default to calculating features from post-convolutional layer activations. For example, to build a ResNet50 feature extractor for images at 299 x 299 pixels:

```
resnet50 = hx.build_feature_extractor(
 'resnet50_imagenet',
 tile_px=299
)
```

This will calculate features using activations from the post-convolutional layer. You can also concatenate activations from multiple neural network layers and apply pooling for layers with 2D output shapes.

```
resnet50 = hx.build_feature_extractor(
 'resnet50_imagenet',
 layers=['conv1_relu', 'conv3_block1_2_relu'],
 pooling='avg',
 tile_px=299
)
```

If a model architecture is available in both the Tensorflow and PyTorch backends, HistoX will default to using the active backend. You can manually set the feature extractor backend using `backend`.

```
# Create a PyTorch feature extractor
extractor = hx.build_feature_extractor(
 'resnet50_imagenet',
 layers=['layer2.0.conv1', 'layer3.1.conv2'],
 pooling='avg',
 tile_px=299,
 backend='torch'
)
```

You can view all available feature extractors with [`histox.model.list_extractors()`](model.html#histox.model.list_extractors).

## Layer Activations

You can also calculate features from any model trained in HistoX. The first argument to `build_feature_extractor()` should be the path of the trained model. You can optionally specify the layer at which to calculate activations using the `layers` keyword argument. If not specified, activations are calculated at the post-convolutional layer.

```
# Calculate features from trained model.
features = build_feature_extractor(
 '/path/to/model',
 layers='sepconv3_bn'
)
```

## Self-Supervised Learning

Finally, you can also generate features from a trained [self-supervised learning](ssl.html#simclr-ssl) model (either [SimCLR](https://github.com/jamesdolezal/simclr) or [DinoV2](https://github.com/jamesdolezal/dinov2)).

For SimCLR models, use `'simclr'` as the first argument to `build_feature_extractor()`, and pass the path to a saved model (or saved checkpoint file) via the keyword argument `ckpt`.

```
simclr = hx.build_feature_extractor(
 'simclr',
 ckpt='/path/to/simclr.ckpt'
)
```

For DinoV2 models, use `'dinov2'` as the first argument, and pass the model configuration YAML file to `cfg` and the teacher checkpoint weights to `weights`.

```
dinov2 = hx.build_feature_extractor(
 'dinov2',
 weights='/path/to/teacher_checkpoint.pth',
 cfg='/path/to/config.yaml'
)
```

## Custom Extractors

HistoX also provides an API for integrating your own custom, pretrained feature extractor. See [Custom Feature Extractors](custom_extractors.html#custom-extractors) for additional information.

## Exporting Features

### Feature bags

Once you have prepared a feature extractor, features can be generated for a dataset and exported to disk for later use. Pass a feature extractor to the first argument of `histox.Project.generate_feature_bags()`, with a [`histox.Dataset`](dataset.html#histox.Dataset) as the second argument.

```
# Load a project and dataset.
P = hx.Project(...)
dataset = P.dataset(tile_px=299, tile_um=302)

# Create a feature extractor.
ctranspath = hx.build_feature_extractor('ctranspath', resize=True)

# Calculate & export feature bags.
P.generate_feature_bags(ctranspath, dataset)
```

Note

If you are generating features from a SimCLR model trained with stain normalization,
you should specify the stain normalizer using the `normalizer` argument to `histox.Project.generate_feature_bags()` or [`histox.DatasetFeatures`](dataset_features.html#histox.DatasetFeatures).

Features are calculated for slides in batches, keeping memory usage low. By default, features are saved to disk in a directory named `pt_files` within the project directory, but you can override the destination directory using the `outdir` argument.

Alternatively, you can calculate features for a dataset using [`histox.DatasetFeatures`](dataset_features.html#histox.DatasetFeatures) and the `.to_torch()` method. This will calculate features for your entire dataset at once, which may require a large amount of memory. The first argument should be the feature extractor, and the second argument should be a [`histox.Dataset`](dataset.html#histox.Dataset).

```
# Calculate features for the entire dataset.
features = hx.DatasetFeatures(ctranspath, dataset)

# Export feature bags.
features.to_torch('/path/to/bag_directory/')
```

Warning

Using [`histox.DatasetFeatures`](dataset_features.html#histox.DatasetFeatures) directly may result in a large amount of memory usage, particularly for sizable datasets. When generating feature bags for training MIL models, it is recommended to use `histox.Project.generate_feature_bags()` instead.

Feature "bags" are PyTorch tensors of features for all images in a slide, saved to disk as `.pt` files. These bags are used to train MIL models. Bags can be manually loaded and inspected using `torch.load()`.

```
>>> import torch
>>> bag = torch.load('/path/to/bag.pt')
>>> bag.shape
torch.Size([2310, 768])
>>> bag.dtype
torch.float32
```

When image features are exported for a dataset, the feature extractor configuration is saved to `bags_config.json` in the same directory as the exported features. This configuration file can be used to rebuild the feature extractor. An example file is shown below.

```
{
 "extractor": {
 "class": "histox.model.extractors.ctranspath.CTransPathFeatures",
 "kwargs": {
 "center_crop": true
 }
 },
 "normalizer": {
 "method": "macenko",
 "fit": {
 "stain_matrix_target": [
 [
 0.5062568187713623,
 0.22186939418315887
 ],
 [
 0.7532230615615845,
 0.8652154803276062
 ],
 [
 0.4069173336029053,
 0.42241501808166504
 ]
 ],
 "target_concentrations": [
 1.7656903266906738,
 1.2797492742538452
 ]
 }
 },
 "num_features": 2048,
 "tile_px": 299,
 "tile_um": 302
}
```

The feature extractor can be manually rebuilt using [`histox.model.rebuild_extractor()`](model.html#histox.model.rebuild_extractor):

```
from histox.model import rebuild_extractor

# Recreate the feature extractor
# and stain normalizer, if applicable
extractor, normalizer = rebuild_extractor('/path/to/bags_config.json')
```

### From a TFRecord

In addition to generating and exporting feature bags for a dataset, features can also be generated from a single TFRecord file. This may be useful for debugging or testing purposes.

```
import histox as hx

# Create a feature extractor
ctranspath = hx.build_feature_extractor('ctranspath')

# Bags is a tensor of shape (n_tiles, n_features)
# Coords is a tensor of shape (n_tiles, 2), containing x/y tile coordinates.
bags, coords = ctranspath('file.tfrecords')
```

### From a whole-slide image

Feature extractors can also create features from a whole-slide image. This is useful for single-slide analysis, MIL inference, and other tasks where features are needed for the entire slide. Features are returned as a 3D tensor, with shape `(width, height, n_features)`, reflecting the spatial arrangement of features for tiles across the image.

```
# Load a feature extractor.
ctranspath = hx.build_feature_extractor('ctranspath')

# Load a whole-slide image.
wsi = hx.WSI('slide.svs', tile_px=256, tile_um=128)

# Generate features for the whole slide.
# Shape: (width, height, n_features)
features = ctranspath(wsi)
```

### Mixed precision

All feature extractors will use mixed precision by default. This can be disabled by setting the `mixed_precision` argument to `False` when creating the feature extractor.

```
# Load a feature extractor without mixed precision
extractor = hx.build_feature_extractor('ctranspath', mixed_precision=False)
```

### License & Citation

Licensing and citation information for the pretrained feature extractors is accessible with the `.license` and `.citation` attributes.

```
>>> ctranspath.license
'GNU General Public License v3.0'
>>> print(ctranspath.citation)

@{wang2022,
 title={Transformer-based Unsupervised Contrastive Learning for Histopathological Image Classification},
 author={Wang, Xiyue and Yang, Sen and Zhang, Jun and Wang, Minghui and Zhang, Jing and Yang, Wei and Huang, Junzhou and Han, Xiao},
 journal={Medical Image Analysis},
 year={2022},
 publisher={Elsevier}
}
```