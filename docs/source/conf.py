import io as _builtin_io   # ← 提前保护内置 io 模块，防止与 histox.io 冲突
import os, sys
import importlib.machinery
from runpy import run_path
from unittest.mock import MagicMock

import pytorch_sphinx_theme2

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
    'segmentation_models_pytorch',
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
    # biscuit (legacy optional distribution)
    'biscuit', 'biscuit.hp', 'biscuit.threshold',
    'biscuit.utils', 'biscuit.delong',
    # clam (legacy optional distribution)
    'clam', 'clam.models', 'clam.utils',
    # Legacy extension import namespaces retained for compatibility.
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
release = run_path(
    os.path.join(os.path.dirname(__file__), '..', '..', 'histox', '_release.py')
)["__version__"]

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'myst_parser',
    'sphinx_copybutton',
    'sphinx_design',
    'sphinxcontrib.video',
]

# -- HTML theme -----------------------------------------------------------
#
# HistoX uses the same maintained theme family as the current PyTorch docs,
# with a small, project-owned token layer in ``_static/custom.css``.
html_theme = 'pytorch_sphinx_theme2'
html_theme_path = [pytorch_sphinx_theme2.get_html_theme_path()]
templates_path = [
    '_templates',
    os.path.join(os.path.dirname(pytorch_sphinx_theme2.__file__), 'templates'),
]
html_title = 'HistoX documentation'
html_logo = None
# The previous O/X mark is intentionally not used as a favicon while the
# HistoX identity is being redesigned.
html_favicon = None
html_static_path = ['_static']
html_css_files = ['custom.css']
html_js_files = ['tutorials.js']

html_theme_options = {
    'show_toc_level': 2,
    'navigation_with_keys': True,
    'navbar_align': 'left',
    'navbar_start': ['navbar-logo'],
    'navbar_center': ['histox_navbar'],
    'navbar_end': [
        'search-field-custom',
        'theme-switcher',
        'navbar-icon-links',
    ],
    'navbar_persistent': [],
    'header_links_before_dropdown': 5,
    'use_edit_page_button': True,
    'show_version_warning_banner': False,
    'show_lf_header': False,
    'show_lf_footer': False,
    'show_pytorch_org_link': False,
    'announcement_banner': {
        'text': 'HistoX is under active development.',
        'url': 'https://github.com/leicaohmu/histox',
        'link_text': 'Follow the project on GitHub',
        'dismissible': False,
    },
    'icon_links': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/leicaohmu/histox',
            'icon': 'fa-brands fa-github',
        },
        {
            'name': 'PyPI',
            'url': 'https://pypi.org/project/histox/',
            'icon': 'fa-brands fa-python',
        },
    ],
    'pytorch_project': 'docs',
}

html_context = {
    'github_user': 'leicaohmu',
    'github_repo': 'histox',
    'github_version': 'develop',
    'doc_path': 'docs/source',
}

# The PyTorch theme's generated global toctree is intentionally minimal.
# Keep HistoX's high-level task taxonomy stable across reference pages.
html_sidebars = {
    '**': ['histox_sidebar.html'],
}

html_show_sphinx = False
html_last_updated_fmt = '%b %d, %Y'

# Keep interactive prompts out of copied snippets while leaving output intact.
copybutton_prompt_text = r'>>> |\.\.\. '
copybutton_prompt_is_regexp = True
