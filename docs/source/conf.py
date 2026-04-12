import os, sys
import importlib
import importlib.machinery
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath('../..'))

MOCK_MODULES = [
    'torch', 'torch.nn', 'torch.nn.functional',
    'torch.utils', 'torch.utils.data',
    'torch.optim', 'torch.cuda', 'torch.distributed',
    'torchvision', 'torchvision.transforms', 'torchvision.models',
    'tensorflow', 'tensorflow.keras',
    'tensorflow.keras.layers', 'tensorflow.keras.models',
    'tensorflow.keras.optimizers',
    'cv2', 'openslide', 'pyvips',
    'cucim', 'cucim.clara',
    'glfw', 'imgui', 'OpenGL', 'OpenGL.GL',
    'numba', 'llvmlite',
    'umap', 'umap.umap_',
    'cellpose', 'cellpose.models',
    'pytorch_lightning',
    'pytorch_lightning.callbacks',
    'pytorch_lightning.trainer',
    'pytorch_lightning.core',
    'pytorch_lightning.core.lightning',
]
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = MagicMock()
for mod_name in MOCK_MODULES:
    sys.modules[mod_name].__spec__ = importlib.machinery.ModuleSpec(mod_name, None)

# ⬇️ 把 histox 子模块注册为顶层别名
# ⚠️ 必须排除与 Python 标准库同名的模块！
STDLIB_NAMES = {
    'io', 'os', 'sys', 'test', 'stat', 'util',
    'model', 'segment', 'stats', 'norm',
}
HISTOX_SUBMODULES = [
    'biscuit', 'cellseg', 'dataset', 'experimental',
    'gan', 'grad', 'heatmap', 'mil',
    'mosaic', 'project', 'simclr',
    'slide', 'studio',
]
for sub in HISTOX_SUBMODULES:
    if sub in STDLIB_NAMES:
        continue                         # 跳过与标准库冲突的名字
    full = f'histox.{sub}'
    try:
        mod = importlib.import_module(full)
        sys.modules[sub] = mod
    except Exception:
        pass

project = 'histox'
copyright = '2026, histox team'
author = 'histox team'
release = '0.1.3'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'myst_parser',
]

html_theme = 'sphinx_rtd_theme'
html_logo = './_static/logo.png'
html_static_path = ['_static']
html_css_files = ['custom.css']
