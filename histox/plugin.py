"""
This module is responsible for loading all the plugins that are installed in the system.
"""

from importlib.metadata import entry_points

def load_plugins():
    eps = entry_points()
    # 兼容 Python 3.9 和 3.10+
    if hasattr(eps, 'select'):
        plugins = eps.select(group='histox.plugins')
    else:
        plugins = eps.get('histox.plugins', [])
    for entry_point in plugins:
        register = entry_point.load()
        register()

load_plugins()
