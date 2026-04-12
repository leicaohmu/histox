import io as _builtin_io   # ← 提前保护内置 io 模块，防止与 histox.io 冲突
import os, sys
import importlib
import importlib.machinery
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath('../..'))

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
    # 数值计算
    'numba', 'llvmlite',
    'umap', 'umap.umap_',
    # 细胞分割
    'cellpose', 'cellpose.models',
    # Lightning
    'pytorch_lightning',
    'pytorch_lightning.callbacks',
    'pytorch_lightning.trainer',
    'pytorch_lightning.core',
    'pytorch_lightning.core.lightning',
    # biscuit（slideflow-noncommercial，先 mock，后面用真实模块覆盖）
    'biscuit', 'biscuit.hp', 'biscuit.threshold',
    'biscuit.utils', 'biscuit.delong',
    # clam（slideflow-gpl，先 mock，后面用真实模块覆盖）
    'clam', 'clam.models', 'clam.utils',
    # slideflow 扩展包本身（未安装时先 mock）
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
    # RST 文件用类名/短名作为模块名，全部 mock
    'Dataset', 'DatasetFeatures',
    'Heatmap', 'Project', 'Mosaic',
    'WSI', 'SlideMap',
    'ModelParams', 'TrainerConfig',
    'MILModelConfig', 'CLAMModelConfig',
    'mil', 'model',
    'norm', 'util', 'studio',
    'simclr', 'slide',
    'io.torch', 'io.tensorflow', 'io.preservedsite', 'io.io_utils',
]
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = MagicMock()
for mod_name in MOCK_MODULES:
    sys.modules[mod_name].__spec__ = importlib.machinery.ModuleSpec(mod_name, None)

# ── 注册真实 histox 子模块为顶层别名（覆盖上面的 mock）──────────────
REAL_SUBMODULES = {
    'gan':     'histox.gan',
    'grad':    'histox.grad',
    'heatmap': 'histox.heatmap',
    'mosaic':  'histox.mosaic',
    'project': 'histox.project',
    'cellseg': 'histox.cellseg',
    'dataset': 'histox.dataset',
    'mil':     'histox.mil',
    'model':   'histox.model',
    'norm':    'histox.norm',
    'simclr':  'histox.simclr',
    'slide':   'histox.slide',
    'studio':  'histox.studio',
    'util':    'histox.util',
    # ── 新增 io 系列 ──────────────────────────
    'io.torch':                   'histox.io.torch',
    'io.tensorflow':              'histox.io.tensorflow',
    'io.preservedsite':           'histox.io.preservedsite',
    'io.io_utils':                'histox.io.io_utils',
}
for alias, full in REAL_SUBMODULES.items():
    try:
        mod = importlib.import_module(full)
        sys.modules[alias] = mod
    except Exception:
        pass   # import 失败保留 mock，不影响构建

# ── 尝试用真实扩展包覆盖 mock（RTD 安装了 slideflow-noncommercial/gpl 时生效）──
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
        pass   # 未安装扩展包时保留 mock，不影响构建

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
    'sphinxcontrib.video',    # ← 新增，修复 video 指令 ERROR
]

html_theme = 'sphinx_rtd_theme'
html_logo = './_static/logo.png'
html_static_path = ['_static']
html_css_files = ['custom.css']