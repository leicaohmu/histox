"""Registries for HistoX workflow tasks and methods.

Both registries are :class:`histox.registry.Registry` instances. Names are
case-insensitive and duplicate registrations are rejected unless
``replace=True`` is passed.
"""

from typing import List

from histox.registry import Registry

from .exceptions import (
    DuplicateRegistrationError,
    UnknownMethodError,
    UnknownTaskError,
)


def _normalise_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Workflow names must be non-empty strings")
    return name.strip().lower()


_TASK_REGISTRY = Registry(
    'task',
    normalize=_normalise_name,
    duplicates='error',
    unknown_error=UnknownTaskError,
    duplicate_error=DuplicateRegistrationError,
)
_METHOD_REGISTRY = Registry(
    'method',
    normalize=_normalise_name,
    duplicates='error',
    unknown_error=UnknownMethodError,
    duplicate_error=DuplicateRegistrationError,
)


def register_task(name=None, replace: bool = False):
    """Register a task factory or class.

    Use as ``@register_task``, ``@register_task("survival")``, or
    ``register_task("survival")(SurvivalTask)``. Existing registrations are
    protected unless ``replace=True``.
    """
    return _TASK_REGISTRY.register(name, replace=replace)


def register_method(name=None, replace: bool = False):
    """Register a workflow method factory or class.

    Use as ``@register_method``, ``@register_method("ps3")``, or
    ``register_method("ps3")(PS3Method)``. Existing registrations are
    protected unless ``replace=True``.
    """
    return _METHOD_REGISTRY.register(name, replace=replace)


def list_tasks() -> List[str]:
    """Return registered task names in stable order."""
    return _TASK_REGISTRY.names()


def list_methods() -> List[str]:
    """Return registered workflow method names in stable order."""
    return _METHOD_REGISTRY.names()


def build_task(name: str, **config):
    """Construct a registered task from keyword configuration."""
    return _TASK_REGISTRY.get(name)(**config)


def build_method(name: str, **config):
    """Construct a registered method from keyword configuration."""
    return _METHOD_REGISTRY.get(name)(**config)
