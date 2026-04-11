# histox

**histox** is a deep learning library for digital pathology, built on top of [Slideflow](https://github.com/slideflow/slideflow) with customized enhancements.

Designed for medical researchers and AI practitioners, histox provides an accessible, easy-to-use interface for developing state-of-the-art computational pathology models.

## ✨ Features

- Easy-to-use, highly customizable training pipelines
- Robust slide processing and stain normalization toolkit
- Support for weakly-supervised or strongly-supervised labels
- Built-in foundation model support
- Multiple-instance learning (MIL)
- Self-supervised learning (SSL)
- Generative adversarial networks (GANs)
- Explainability tools: Heatmaps, mosaic maps, saliency maps
- Layer activation analysis tools
- Uncertainty quantification

## 📦 Installation

```bash
pip install histox
```

Or install from source:

```bash
git clone https://github.com/leicaohmu/histox
cd histox
pip install -e .
```

## ⚙️ Requirements

- Python >= 3.7
- PyTorch >= 1.9 or Tensorflow 2.5-2.11

## 🚀 Quick Start

### Create a Project

```python
import histox as hx

P = hx.create_project(
    root='/project/path',
    annotations="/patient/annotations.csv",
    slides="/slides/directory",
    tfrecords="/tfrecords/directory"
)
```

### Extract Tiles

```python
P.extract_tiles(
    tile_px=299,  # Tile size, in pixels
    tile_um=302   # Tile size, in microns
)
```

### Train a Model

```python
params = hx.ModelParams(
    tile_px=299,
    tile_um=302,
    batch_size=32,
    model='xception',
    learning_rate=0.0001,
)

P.train(
    'category1',
    params=params,
    save_predictions=True,
    multi_gpu=True
)
```

## 📄 License

This project is based on [Slideflow](https://github.com/slideflow/slideflow).  
Please refer to the original license for usage terms.

## 🙏 Acknowledgements

- [Slideflow](https://github.com/slideflow/slideflow) — the upstream library this project is based on.