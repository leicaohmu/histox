---

# histox

![Python Version](https://img.shields.io/badge/python->=3.7-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-brightgreen)
![Language](https://img.shields.io/badge/language-Python-blue)

**histox** is a powerful, open-source deep learning library for digital pathology. Built on top of [Slideflow](https://github.com/slideflow/slideflow), it provides researchers and AI practitioners with a comprehensive toolkit for analyzing whole slide images (WSI) and building state-of-the-art computational pathology models.

🌐 **Website**: [slideflow.dev](https://slideflow.dev)

## ✨ Key Features

- **End-to-end WSI analysis pipeline** - From raw slides to trained models
- **Robust slide processing** - Automatic tile extraction with stain normalization
- **Flexible learning paradigms**
  - Strongly-supervised and weakly-supervised learning
  - Multiple Instance Learning (MIL) for slide-level labels
  - Self-Supervised Learning (SSL) for representation learning
  - Generative Adversarial Networks (GANs) for data augmentation
- **Pre-trained foundation models** - Integrated support for modern architectures
- **Interpretability tools** - Generate heatmaps, saliency maps, and mosaic visualizations
- **Layer activation analysis** - Understand model decision-making
- **Uncertainty quantification** - Measure model confidence

## 📋 Requirements

- **Python**: >= 3.7
- **Deep Learning Framework** (at least one):
  - PyTorch >= 1.9
  - TensorFlow >= 2.5, < 2.12

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install histox
```

### From Source

```bash
git clone https://github.com/leicaohmu/histox
cd histox
pip install -e .
```

### Optional GPU Support

Install with PyTorch GPU support:

```bash
pip install -e ".[torch]"
```

Or TensorFlow GPU support:

```bash
pip install -e ".[tf]"
```

NVIDIA RAPIDS acceleration (optional):

```bash
pip install -e ".[cucim]"
```

## 🚀 Quick Start Example

This example demonstrates how to train a lung cancer adenocarcinoma classifier using the included TCGA dataset.

### 1. Prepare Your Data Structure

```
project_root/
├── slides/                    # WSI files (.svs, .ndpi, etc.)
│   ├── TCGA-83-5908-01Z-00-DX1.svs
│   ├── TCGA-62-A46V-01Z-00-DX1.svs
│   └── ...
├── annotations.csv            # Slide labels
└── tfrecords/                # Output directory for processed data (auto-created)
```

### 2. Create Annotation File

Create `annotations.csv` with slide-level labels:

```csv
patient,subtype,site,slide
TCGA-83-5908,adenocarcinoma,Site-28,TCGA-83-5908-01Z-00-DX1
TCGA-62-A46V,adenocarcinoma,Site-124,TCGA-62-A46V-01Z-00-DX1
TCGA-44-2655,squamous,Site-29,TCGA-44-2655-01Z-00-DX1
```

(A complete dataset example is available in `lung_labels.csv` in this repository)

### 3. Initialize and Train

```python
import histox as hx
import os

# Define paths
project_root = '/path/to/project'
annotations_file = '/path/to/annotations.csv'
slides_directory = '/path/to/slides'
tfrecords_directory = os.path.join(project_root, 'tfrecords')

# Create project
print("[1/5] Creating project...")
project = hx.create_project(
    root=project_root,
    annotations=annotations_file,
    slides=slides_directory,
    tfrecords=tfrecords_directory
)

# Extract tiles from slides
print("[2/5] Extracting tiles...")
project.extract_tiles(
    tile_px=299,        # Tile size in pixels
    tile_um=302,        # Tile size in micrometers
    workers=8           # Number of parallel workers
)

# Define model parameters
print("[3/5] Configuring model...")
params = hx.ModelParams(
    tile_px=299,
    tile_um=302,
    batch_size=32,
    model='xception',            # Base architecture
    learning_rate=0.0001,
    epochs=50,
    validation_fraction=0.2
)

# Train the model
print("[4/5] Training model...")
project.train(
    'subtype',                   # Column name for classification target
    params=params,
    save_predictions=True,
    multi_gpu=True               # Enable multi-GPU training if available
)

# Generate explanations
print("[5/5] Generating interpretations...")
project.generate_heatmaps()      # Attention/saliency visualizations
project.generate_mosaic_maps()   # Tile-level predictions

print("✓ Training complete!")
```

### 4. Evaluate and Interpret Results

```python
# Access trained models and predictions
results = project.get_results('subtype')

# Generate patient-level predictions
slide_predictions = project.predict_slides('subtype')

# Create visualizations
heatmap = hx.Heatmap.from_project(
    project=project,
    outcome_name='subtype',
    model_idx=0
)
heatmap.save('/path/to/output/heatmap.png')

# Access detailed metrics
print(f"Validation Accuracy: {results['val_accuracy']:.4f}")
print(f"Validation AUC: {results['val_auc']:.4f}")
```

## 📊 Complete Workflow Example (Lung Cancer Dataset)

Here's a ready-to-run example using the included TCGA lung cancer data:

```python
import histox as hx
import pandas as pd
import os

# Configuration
data_root = './datasets/lung_adeno_squam'
project_dir = './lung_project'

# The lung_labels.csv is already in the repository with TCGA samples
annotations_file = os.path.join(data_root, 'lung_labels.csv')
slides_dir = os.path.join(data_root, 'slides')  # Add actual .svs files here
tfrecords_dir = os.path.join(project_dir, 'tfrecords')

# Check available data
labels_df = pd.read_csv(annotations_file)
print(f"Total slides: {len(labels_df)}")
print(f"Subtypes: {labels_df['subtype'].unique()}")
print(labels_df.head())

# Step 1: Create project
try:
    project = hx.create_project(
        root=project_dir,
        annotations=annotations_file,
        slides=slides_dir,
        tfrecords=tfrecords_dir
    )
    print("✓ Project created successfully")
except Exception as e:
    print(f"Project already exists or error: {e}")

# Step 2: Extract tiles (if slides are available)
try:
    project.extract_tiles(
        tile_px=299,
        tile_um=302,
        workers=4,
        verbose=True
    )
    print("✓ Tiles extracted")
except Exception as e:
    print(f"Could not extract tiles: {e}")

# Step 3: Configure and train
params = hx.ModelParams(
    tile_px=299,
    tile_um=302,
    batch_size=32,
    model='xception',
    learning_rate=0.0001,
    epochs=30,
    validation_fraction=0.2
)

# Train on adenocarcinoma vs squamous classification
project.train(
    'subtype',
    params=params,
    save_predictions=True,
    multi_gpu=False  # Set to True if using multi-GPU
)

print("✓ Model training complete!")
```

## 🔧 Advanced Features

### Multiple Instance Learning (MIL)

Use when you only have slide-level labels, not individual tile annotations:

```python
params = hx.ModelParams(
    tile_px=299,
    model='xception',
    mil=True,                 # Enable MIL
    mil_method='attention',   # 'attention' or 'max-pooling'
    learning_rate=0.0001
)

project.train('diagnosis', params=params)
```

### Self-Supervised Pre-training

Pre-train feature extractors without labeled data:

```python
# Perform self-supervised learning on unlabeled slides
project.train_ssl(
    method='simclr',          # Self-supervised learning method
    epochs=100,
    batch_size=64,
    model='xception'
)

# Fine-tune on labeled data
project.train('diagnosis', params=params)
```

### Stain Normalization

Automatic handling of staining variations across labs:

```python
# Extract tiles with stain normalization
project.extract_tiles(
    tile_px=299,
    tile_um=302,
    stain_norm=True,          # Enable normalization
    norm_method='macenko'     # 'macenko' or 'reinhard'
)
```

### Generate Heatmaps

Visualize model predictions across slides:

```python
heatmap = hx.Heatmap.from_project(
    project=project,
    outcome_name='diagnosis',
    model_idx=0,
    cmap='RdYlBu_r'
)

# Save with metadata
heatmap.save(
    '/output/heatmap.png',
    high_res=True,
    stride=32
)
```

## 📚 Key Modules

| Module | Purpose |
|--------|---------|
| `histox.project` | Main Project class for pipeline management |
| `histox.dataset` | Dataset handling and tile management |
| `histox.model` | Model architecture and training |
| `histox.slide` | WSI (Whole Slide Image) handling |
| `histox.norm` | Stain normalization algorithms |
| `histox.heatmap` | Heatmap generation and visualization |
| `histox.stats` | Statistical analysis and reporting |

## 🎯 Supported Models

- **CNN Architectures**: Xception, ResNet-50, EfficientNet, InceptionV3, DenseNet
- **Vision Transformers**: ViT, DeiT
- **Specialized Architectures**: Cellpose (instance segmentation)

## 📖 Documentation

For detailed documentation and API reference:
- [Slideflow Documentation](https://slideflow.dev)
- [GitHub Issues](https://github.com/leicaohmu/histox/issues)

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the **Apache License 2.0**. See `LICENSE` file for details.

**Note**: This project is built on [Slideflow](https://github.com/slideflow/slideflow). Please refer to the original project's license for additional terms.

## 🙏 Acknowledgments

- **Slideflow** - The powerful foundation this project is built upon
- **TCGA** - The Cancer Genome Atlas for dataset resources
- **PyTorch & TensorFlow** - Deep learning frameworks

## 📞 Support & Contact

- **Email**: james@histox.ai
- **Website**: [slideflow.dev](https://slideflow.dev)
- **Repository**: [github.com/leicaohmu/histox](https://github.com/leicaohmu/histox)

## 🔗 Related Resources

- [Slideflow Documentation](https://slideflow.dev)
- [TCGA Data Portal](https://portal.gdc.cancer.gov/)
- [Digital Pathology Resources](https://www.digipathonet.org/)

---

**Status**: Active development  
**Last Updated**: 2026-04-11

---