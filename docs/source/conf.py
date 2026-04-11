import os, sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'histox'
copyright = '2026, histox team'
author = 'histox team'
release = '0.1.3'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',
]

html_theme = 'sphinx_rtd_theme'
html_logo = './_static/logo.png'  # 可选：侧边栏logo
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]
