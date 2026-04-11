# conf.py

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # 指向项目根目录

project = 'histox'
copyright = '2026, histox team'
author = 'histox team'
release = '0.1.3'

extensions = [
    'sphinx.ext.autodoc',       # 自动从 docstring 生成 API 文档
    'sphinx.ext.napoleon',      # 支持 Google/NumPy 风格 docstring
    'sphinx.ext.viewcode',      # 在文档中显示源码链接
    'sphinx.ext.autosummary',   # 自动生成模块摘要
    'myst_parser',              # 支持 Markdown 写文档
]

# 使用 Read the Docs 主题（和 slideflow 一样）
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'logo_only': False,
}

# 静态文件（logo 等）
html_static_path = ['_static']
html_logo = '_static/logo.png'  # 如果有 logo 取消注释
