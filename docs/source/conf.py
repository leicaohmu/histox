import io as _builtin_io   # ← 提前保护内置 io 模块，防止与 histox.io 冲突
import os, sys
import importlib.machinery
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath('../..'))

# ── autodoc_mock_imports：Sphinx 官方机制，专为 autodoc 设计 ──────────
# 所有有 C 扩展 / 重依赖的库都放这里，autodoc import 时自动走 mock
autodoc_mock_imports = [
    # TensorFlow
    'tensorflow',
    'tensorflow.keras',
    'tensorflow.keras.layers',
    'tensorflow.keras.models',
    'tensorflow.keras.optimizers',
    # PyTorch
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torch.utils',
    'torch.utils.data',
    'torch.optim',
    'torch.cuda',
    'torch.distributed',
    'torchvision',
    'torchvision.transforms',
    'torchvision.models',
    # 图像/WSI
    'cv2',
    'openslide',
    'pyvips',
    'cucim',
    'cucim.clara',
    # GUI
    'glfw',
    'imgui',
    'OpenGL',
    'OpenGL.GL',
    # 数值计算
    'numba',
    'llvmlite',
    'umap',
    'umap.umap_',
    # 细胞分割
    'cellpose',
    'cellpose.models',
    # Lightning
    'pytorch_lightning',
    'pytorch_lightning.callbacks',
    'pytorch_lightning.trainer',
    'pytorch_lightning.core',
    'pytorch_lightning.core.lightning',
]

# ── 手动 mock：RST 文件里用到的短名别名模块 ──────────────────────────
# 注意：histox 自身的子模块不在这里 mock，全部交给 autodoc 按需处理
MOCK_MODULES = [
    # biscuit（slideflow-noncommercial）
    'biscuit', 'biscuit.hp', 'biscuit.threshold',
    'biscuit.utils', 'biscuit.delong',
    # clam（slideflow-gpl）
    'clam', 'clam.models', 'clam.utils',
    # slideflow 扩展包
    'slideflow_noncommercial',
    'slideflow_noncommercial.biscuit',
    'slideflow_noncommercial.biscuit.hp',
    'slideflow_noncommercial.biscuit.threshold',
    'slideflow_noncommercial.biscuit.utils',
    'slideflow_noncommercial.biscuit.delong',
    'slideflow_gpl',
    'slideflow_gpl.clam',
    'slideflow_gpl.clam.models',
    'slideflow_gpl.clam.utils',
    # cellseg 别名
    'cellseg', 'cellseg.models',
    # RST 文件用类名/短名作为模块名
    'Dataset', 'DatasetFeatures',
    'Heatmap', 'Project', 'Mosaic',
    'WSI', 'SlideMap',
    'ModelParams', 'TrainerConfig',
    'MILModelConfig', 'CLAMModelConfig',
    'mil', 'model',
    'norm', 'util', 'studio',
    'simclr', 'slide',
    'gan', 'grad', 'heatmap', 'mosaic', 'project', 'dataset',
    # io 子模块短名别名
    'io.torch', 'io.tensorflow', 'io.preservedsite', 'io.io_utils',
]
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = MagicMock()
for mod_name in MOCK_MODULES:
    sys.modules[mod_name].__spec__ = importlib.machinery.ModuleSpec(mod_name, None)

# ── 尝试用真实扩展包覆盖 biscuit/clam 的 mock ────────────────────────
import importlib
EXT_SUBMODULES = {
    'biscuit':           'slideflow_noncommercial.biscuit',
    'biscuit.hp':        'slideflow_noncommercial.biscuit.hp',
    'biscuit.threshold': 'slideflow_noncommercial.biscuit.threshold',
    'biscuit.utils':     'slideflow_noncommercial.biscuit.utils',
    'biscuit.delong':    'slideflow_noncommercial.biscuit.delong',
    'clam':              'slideflow_gpl.clam',
    'clam.models':       'slideflow_gpl.clam.models',
    'clam.utils':        'slideflow_gpl.clam.utils',
}
for alias, full in EXT_SUBMODULES.items():
    try:
        mod = importlib.import_module(full)
        sys.modules[alias] = mod
    except Exception:
        pass  # 真实包 import 失败，保留 mock，不影响构建

project = 'histox'
copyright = '2026, histox team'
author = 'histox team'
release = '0.1.4'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'myst_parser',
    'sphinxcontrib.video',
]

html_theme = 'sphinx_rtd_theme'
html_logo = './_static/logo.png'
html_static_path = ['_static']
html_css_files = ['custom.css']