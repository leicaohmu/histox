# Self-Supervised Learning (SSL)

HistoX provides easy access to training the self-supervised, contrastive learning framework [SimCLR](https://arxiv.org/abs/2002.05709). Self-supervised learning provides an avenue for learning useful visual representations in your dataset without requiring ground-truth labels. These visual representations can be exported as feature vectors and used for downstream analyses such as [dimensionality reduction](posthoc.html#slidemap) or [multi-instance learning](mil.html#mil).

The `histox.simclr` module contains a [forked Tensorflow implementation](https://github.com/jamesdolezal/simclr/) minimally modified to interface with HistoX. SimCLR models can be trained with `histox.Project.train_simclr()`, and SimCLR features can be calculated as with other models using `histox.Project.generate_features()`.

## Training SimCLR

First, determine the SimCLR training parameters with `histox.simclr.get_args()`. This function accepts parameters via keyword arguments, such as `learning_rate` and `temperature`, and returns a configured `histox.simclr.SimCLR_Args`.

```
from histox import simclr

args = simclr.get_args(
 temperature=0.1,
 learning_rate=0.3,
 train_epochs=100,
 image_size=299
)
```

Next, assemble a training and (optionally) a validation dataset. The validation dataset is used to assess contrastive loss during training, but is not required.

```
import histox as hx

# Load a project and dataset
P = hx.load_project('path')
dataset = P.dataset(tile_px=299, tile_um=302)

# Split dataset into training/validation
train_dts, val_dts = dataset.split(
 val_fraction=0.3,
 model_type='classification',
 labels='subtype')
```

Finally, SimCLR can be trained with `histox.Project.train_simclr()`. You can train with a single dataset:

```
P.train_simclr(args, dataset)
```

You can train with an optional validation dataset:

```
P.train_simclr(
 args,
 train_dataset=train_dts,
 val_dataset=val_dts
)
```

And you can also optionally provide labels for training the supervised head. To train a supervised head, you'll also need to set the SimCLR argument `lineareval_while_pretraining=True`.

```
# SimCLR args
args = simclr.get_args(
 ...,
 lineareval_while_pretraining=True
)

# Train with validation & supervised head
P.train_simclr(
 args,
 train_dataset=train_dts,
 val_dataset=val_dts,
 outcomes='subtype'
)
```

The SimCLR model checkpoints and final saved model will be saved in the `simclr/` folder within the project root directory.

## Training DINOv2

A lightly modified version of [DINOv2](https://arxiv.org/abs/2304.07193) with HistoX integration is available on [GitHub](https://github.com/jamesdolezal/dinov2). This version facilitates training DINOv2 with HistoX datasets and adds stain augmentation to the training pipeline.

To train DINOv2, first install the package:

```
pip install git+https://github.com/jamesdolezal/dinov2.git
```

Next, configure the training parameters and datsets by providing a configuration YAML file. This configuration file should contain a `histox` section, which specifies the HistoX project and dataset to use for training. An example YAML file is shown below:

```
train:
 dataset_path: histox
 batch_size_per_gpu: 32
 histox:
 project: "/mnt/data/projects/TCGA_THCA_BRAF"
 dataset:
 tile_px: 299
 tile_um: 302
 filters:
 brs_class:
 - "Braf-like"
 - "Ras-like"
 seed: 42
 outcome_labels: "brs_class"
 normalizer: "reinhard_mask"
 interleave_kwargs: null
```

See the [DINOv2 README](https://github.com/jamesdolezal/dinov2) for more details on the configuration file format.

Finally, train DINOv2 using the same command-line interface as the original DINOv2 implementation. For example, to train DINOv2 on 4 GPUs on a single node:

```
torchrun --nproc_per_node=4 -m "dinov2.train.train" \
 --config-file /path/to/config.yaml \
 --output-dir /path/to/output_dir
```

The teacher weights will be saved in `outdir/eval/.../teacher_checkpoint.pth`, and the final configuration YAML will be saved in `outdir/config.yaml`.

## Generating features

Generating features from a trained SSL is straightforward - use the same `histox.Project.generate_features()` and [`histox.DatasetFeatures`](dataset_features.html#histox.DatasetFeatures) interfaces as [previously described](posthoc.html#dataset-features), providing a path to a saved SimCLR model or checkpoint.

```
import histox as hx

# Create the SimCLR feature extractor
simclr = hx.build_feature_extractor(
 'simclr',
 ckpt='/path/to/simclr.ckpt'
)

# Calculate SimCLR features for a dataset
features = P.generate_features(simclr, ...)
```

For DINOv2 models, use `'dinov2'` as the first argument, and pass the model configuration YAML file to `cfg` and the teacher checkpoint weights to `weights`.

```
dinov2 = build_feature_extractor(
 'dinov2',
 weights='/path/to/teacher_checkpoint.pth',
 cfg='/path/to/config.yaml'
)
```