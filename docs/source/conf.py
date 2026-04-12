import os, sys
import importlib.machinery
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath('../..'))

# ── Mock 所有重量级依赖，让 RTD 能正常 import histox ──
MOCK_MODULES = [
    # PyTorch
    'torch', 'torch.nn', 'torch.nn.functional',
    'torch.utils', 'torch.utils.data',
    'torch.optim', 'torch.cuda', 'torch.distributed',
    'torchvision', 'torchvision.transforms', 'torchvision.models',
    # TensorFlow
    'tensorflow', 'tensorflow.keras',
    'tensorflow.keras.layers', 'tensorflow.keras.models',
    'tensorflow.keras.optimizers',
    # 图像/WSI
    'cv2', 'openslide', 'pyvips',
    'cucim', 'cucim.clara',
    # GUI
    'glfw', 'imgui', 'OpenGL', 'OpenGL.GL',
    # 地理/图形
    'shapely', 'shapely.geometry',
    'rasterio', 'rasterio.transform',
    # 数值计算
    'numba', 'llvmlite',
    'umap', 'umap.umap_',
    # 细胞分割
    'cellpose', 'cellpose.models',
]
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = MagicMock()

# ⬇️ 关键修复：给所有 mock 模块补上合法的 __spec__
for mod_name in MOCK_MODULES:
    mock = sys.modules[mod_name]
    mock.__spec__ = importlib.machinery.ModuleSpec(mod_name, None)
# ─────────────────────────────────────────────────────

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
