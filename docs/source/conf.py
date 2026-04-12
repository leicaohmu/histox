import os, sys
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

# ⬇️ 关键修复：把 histox 子模块也注册为顶层别名
# 这样 RST 里写 `from module 'biscuit'` 也能找到 histox.biscuit
import histox
HISTOX_SUBMODULES = [
    'biscuit', 'cellseg', 'dataset', 'experimental',
    'gan', 'grad', 'heatmap', 'io', 'mil', 'model',
    'mosaic', 'norm', 'project', 'segment', 'simclr',
    'slide', 'stats', 'studio', 'util',
]
for sub in HISTOX_SUBMODULES:
    full = f'histox.{sub}'
    try:
        import importlib
        mod = importlib.import_module(full)
        sys.modules[sub] = mod          # 注册短名别名
    except Exception:
        pass                             # import 失败则跳过，不影响构建

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
