import setuptools
from runpy import run_path

with open("README.md", "r") as fh:
    long_description = fh.read()

VERSION = run_path("histox/_release.py")["__version__"]

setuptools.setup(
    name="histox",
    version=VERSION,
    author="Lei Cao",
    author_email="caolei@hrbmu.edu.cn",
    description="Deep learning tools for digital histology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leicaohmu/histox",
    license_files=[
        'LICENSE',
        'THIRD_PARTY_NOTICES.md',
        'licenses/*.txt',
    ],
    packages=setuptools.find_packages(),
    package_data={
        'histox': [
            'assets/branding/*.svg',
            'assets/branding/README.md',
            'data/registry.json',
            'slide/histox-logo-name-small.png',
            'studio/gui/icons/logo.png',
            'studio/gui/logo_dark_outline.png',
            'studio/gui/splash.png',
        ],
    },
    scripts=['scripts/histox-studio'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
    install_requires=[
        'gast>=0.3.3',
        'scipy',
        'scikit-learn',
        'matplotlib>=3.2',
        'imageio',
        'opencv-python-headless',
        'shapely',
        'umap-learn',
        'seaborn<0.14',
        'pandas',
        'pyvips',
        'fpdf2',
        'scikit-image',
        'tqdm',
        'click',
        'protobuf<3.21',
        'tensorboard',
        'crc32c',
        'numpy',
        'tabulate',
        'rasterio',
        'rich',
        'pillow>=6.0.0',
        'saliency',
        'requests',
        'parameterized',
        # histox.cellseg relies on zarr 2 internals removed in zarr 3.
        'zarr>=2.16,<3',
        'gdown',
        'triangle',
        'pyarrow',
    ],
    extras_require={
        'torch': [
            'torch',
            'torchvision',
            'pretrainedmodels',
            'six',  # needed by pretrainedmodels, which does not declare it
            'cellpose<2.2',
            'fastai',
            'pytorch-lightning',
            'timm',
            'segmentation-models-pytorch',
        ],
        # Bayesian hyperparameter search (Project.smac_search). SMAC 1.x
        # builds pyrfr from source on Python 3.11+, which requires SWIG.
        'hpo': [
            'smac==1.4.0',
            'ConfigSpace',
        ],
        # HistoX Studio, the interactive slide viewer (histox-studio).
        # pyimgui has no Python 3.13 build yet.
        'studio': [
            'imgui>=2.0.0',
            'pyopengl',
            'glfw',
            'pyperclip',
        ],
        'huggingface': [
            'transformers>=4.32,<5',
            'safetensors>=0.4',
        ],
        'tf': [
            # Deprecated backend. TensorFlow 2.12-2.15 supports Python 3.11
            # and still uses Keras 2, which the legacy code is written for.
            'tensorflow>=2.12,<2.16',
            'tensorflow_probability>=0.20,<0.24',
            'tensorflow_datasets>=4.9,<4.10',
        ],
        'cucim': [
            'cucim',
        ],
        'cucim-cuda12': [
            'cucim',
            'cupy-cuda12x',
        ],
        'cucim-cuda11': [
            'cucim',
            'cupy-cuda11x',
        ],
        'dicom': [
            'wsidicom>=0.36.1,<0.37',
        ],
        'noncommercial': [
            'slideflow-noncommercial',
        ],
        'gpl': [
            'slideflow-gpl',
        ],
        'all': [
            'slideflow-noncommercial',
            'slideflow-gpl',
        ],
    },
)
