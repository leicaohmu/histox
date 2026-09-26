__author__ = 'Lei Cao'
__license__ = 'Apache-2.0'
from ._release import __version__
__gitcommit__ = ""
__github__ = 'https://github.com/leicaohmu/histox'

# Configure deep learning and slide backends
from ._backend import backend, slide_backend

# Import logging functions required for other submodules
from histox.util import getLoggingLevel, log, setLoggingLevel, about

...
from histox import data, io, model, norm, stats, gan
from histox.dataset import Dataset
from histox.heatmap import Heatmap
from histox.model import DatasetFeatures, ModelParams
from histox.model import (
    list_extractors, list_torch_extractors, list_tensorflow_extractors,
    is_extractor, is_torch_extractor, is_tensorflow_extractor,
    build_feature_extractor, build_torch_feature_extractor,
    build_tensorflow_feature_extractor, rebuild_extractor
)
from histox.mosaic import Mosaic
from histox.project import Project
from histox.project import create as create_project
from histox.project import load as load_project
from histox.slide import WSI
from histox.stats import SlideMap
from histox.tfrecord import TFRecord, tfrecord_loader, multi_tfrecord_loader
from histox.plugin import load_plugins

# Optional extension packages still use their upstream distribution and
# import names. Keep these identifiers until HistoX-owned replacements are
# published; they are compatibility boundaries, not public HistoX namespaces.
_NONCOMMERCIAL_MODULES = {'biscuit', 'stylegan2', 'stylegan3', 'extractors'}
_GPL_MODULES = {'clam'}

def __getattr__(name):
    if name in _NONCOMMERCIAL_MODULES:
        try:
            import importlib
            mod = importlib.import_module(f'slideflow_noncommercial.{name}')
            globals()[name] = mod
            return mod
        except ImportError:
            raise AttributeError(
                f"histox.{name} requires the 'slideflow-noncommercial' package.\n"
                f"Install it with:  pip install slideflow-noncommercial"
            )
    if name in _GPL_MODULES:
        try:
            import importlib
            mod = importlib.import_module(f'slideflow_gpl.{name}')
            globals()[name] = mod
            return mod
        except ImportError:
            raise AttributeError(
                f"histox.{name} requires the 'slideflow-gpl' package.\n"
                f"Install it with:  pip install slideflow-gpl"
            )
    raise AttributeError(f"module 'histox' has no attribute {name!r}")
