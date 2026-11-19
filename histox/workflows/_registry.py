"""Registries for HistoX workflow tasks and methods."""

from typing import Any, Callable, Dict, List, Optional

from .exceptions import (
    DuplicateRegistrationError,
    UnknownMethodError,
    UnknownTaskError,
)


_TASK_REGISTRY: Dict[str, Callable[..., Any]] = {}
_METHOD_REGISTRY: Dict[str, Callable[..., Any]] = {}


def _normalise_name(name: str, kind: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("{} name must be a non-empty string".format(kind))
    return name.strip().lower()


def _register(
    registry: Dict[str, Callable[..., Any]],
    factory: Callable[..., Any],
    name: Optional[str],
    kind: str,
    replace: bool,
) -> Callable[..., Any]:
    key = _normalise_name(name or factory.__name__, kind)
    if key in registry and not replace:
        raise DuplicateRegistrationError(
            "{} {!r} is already registered".format(kind.capitalize(), key)
        )
    registry[key] = factory
    return factory


def _decorator(
    registry: Dict[str, Callable[..., Any]],
    name: Optional[str],
    kind: str,
    replace: bool,
):
    def decorate(factory: Callable[..., Any]) -> Callable[..., Any]:
        return _register(registry, factory, name, kind, replace)

    return decorate


def register_task(name=None, replace: bool = False):
    """Register a task class or factory.

    The decorator accepts either ``@register_task`` or
    ``@register_task("survival")``. Existing registrations are protected by
    default so importing an extension cannot silently replace another task.
    """
    if callable(name):
        factory = name
        return _register(_TASK_REGISTRY, factory, None, "task", replace)
    return _decorator(_TASK_REGISTRY, name, "task", replace)


def register_method(name=None, replace: bool = False):
    """Register a workflow method class or factory."""
    if callable(name):
        factory = name
        return _register(_METHOD_REGISTRY, factory, None, "method", replace)
    return _decorator(_METHOD_REGISTRY, name, "method", replace)


def list_tasks() -> List[str]:
    """Return registered task names in stable order."""
    return sorted(_TASK_REGISTRY)


def list_methods() -> List[str]:
    """Return registered workflow method names in stable order."""
    return sorted(_METHOD_REGISTRY)


def build_task(name: str, **config):
    """Construct a registered task from keyword configuration."""
    key = _normalise_name(name, "task")
    try:
        factory = _TASK_REGISTRY[key]
    except KeyError:
        raise UnknownTaskError(
            "Unknown task {!r}. Available tasks: {}".format(key, list_tasks())
        )
    return factory(**config)


def build_method(name: str, **config):
    """Construct a registered method from keyword configuration."""
    key = _normalise_name(name, "method")
    try:
        factory = _METHOD_REGISTRY[key]
    except KeyError:
        raise UnknownMethodError(
            "Unknown method {!r}. Available methods: {}".format(
                key, list_methods()
            )
        )
    return factory(**config)
