"""A single registry type for every pluggable HistoX component.

HistoX looks up feature extractors, MIL models and trainers, and workflow
tasks and methods by name. Each of these kinds of component is held in a
:class:`Registry`. All registries share the same lookup rules, error messages,
optional metadata (for example a license or citation), and hints for
components that live in optional extras packages.

.. code-block:: python

    import histox as hx
    import histox.mil   # MIL registries are created when histox.mil is imported

    hx.registry.list_registries()
    # ['extractor.tensorflow', 'extractor.torch', 'method', 'mil_model', ...]

    extractors = hx.registry.get_registry('extractor.torch')
    extractors.names()
    extractors.metadata('resnet50_imagenet')

Third-party packages add components with the ``register`` decorator of the
relevant registry, usually from a ``histox.plugins`` entry point.

See RFC 0001 (``rfcs/0001-core-abstractions.md``), proposal P1.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import (Any, Callable, Dict, Iterator, List, Optional, Type,
                    TypeVar)

__all__ = [
    'Registry',
    'RegistryError',
    'DuplicateEntryError',
    'UnknownEntryError',
    'get_registry',
    'list_registries',
]

T = TypeVar('T')

_DUPLICATE_POLICIES = ('error', 'replace')


class RegistryError(Exception):
    """Base class for registry errors."""


class DuplicateEntryError(RegistryError):
    """Raised when a name is registered twice and duplicates are not allowed."""


class UnknownEntryError(RegistryError, KeyError):
    """Raised when a name is not registered."""

    def __str__(self) -> str:
        # KeyError quotes its message; show it as plain text instead.
        return str(self.args[0]) if self.args else ''


_REGISTRIES: Dict[str, "Registry"] = {}


class Registry(Mapping):
    """A named collection of components, looked up by string key.

    A registry behaves like a read-only mapping from names to registered
    objects (``name in registry``, ``registry[name]``, ``registry.keys()``).
    Objects are added with :meth:`register`.

    Args:
        kind (str): Name of this registry, e.g. ``'extractor.torch'``.
            Must be unique across HistoX.
        normalize (callable, optional): Function applied to every name before
            it is stored or looked up, e.g. ``str.lower``. It may raise
            ``ValueError`` to reject a name. Defaults to stripping surrounding
            whitespace and rejecting empty names.
        duplicates (str): What to do when a name is registered again.
            ``'error'`` raises ``duplicate_error``; ``'replace'`` keeps the
            newer object. Either can be overridden per call with
            ``register(..., replace=True)``. Defaults to ``'error'``.
        extras (dict, optional): Names that are provided by optional extras
            packages, mapped to the package name. Looking up such a name when
            it is not registered raises an error that names the package.
        unknown_error (type): Exception raised for unknown names. Must be a
            subclass of :class:`UnknownEntryError` or accept a single message
            argument. Defaults to :class:`UnknownEntryError`.
        duplicate_error (type): Exception raised for duplicate names.
            Defaults to :class:`DuplicateEntryError`.
        label (str, optional): Human-readable name of one entry, used in
            error messages, e.g. ``'feature extractor'``. Defaults to
            ``kind``.
    """

    def __init__(
        self,
        kind: str,
        *,
        normalize: Optional[Callable[[str], str]] = None,
        duplicates: str = 'error',
        extras: Optional[Dict[str, str]] = None,
        unknown_error: Type[Exception] = UnknownEntryError,
        duplicate_error: Type[Exception] = DuplicateEntryError,
        label: Optional[str] = None,
    ) -> None:
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("Registry kind must be a non-empty string")
        if duplicates not in _DUPLICATE_POLICIES:
            raise ValueError(
                f"duplicates must be one of {_DUPLICATE_POLICIES}, "
                f"got {duplicates!r}")
        kind = kind.strip()
        if kind in _REGISTRIES:
            raise DuplicateEntryError(f"Registry {kind!r} already exists")
        self.kind = kind
        self.label = label or kind
        self._normalize = normalize or _default_normalize
        self._duplicates = duplicates
        self._extras = dict(extras or {})
        self._unknown_error = unknown_error
        self._duplicate_error = duplicate_error
        self._entries: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        _REGISTRIES[kind] = self

    # --- Registration -----------------------------------------------------

    def register(
        self,
        name: Any = None,
        *,
        replace: Optional[bool] = None,
        **metadata: Any
    ) -> Callable[[T], T]:
        """Register an object, usually as a decorator.

        Supports three forms:

        .. code-block:: python

            @registry.register
            def resnet(): ...                 # registered as 'resnet'

            @registry.register('uni', license='CC-BY-NC-ND-4.0')
            def build_uni(): ...              # registered as 'uni'

            registry.register('uni')(build_uni)

        Args:
            name (str, optional): Name to register under. Defaults to the
                object's ``__name__``.
            replace (bool, optional): Allow replacing an existing entry.
                Defaults to this registry's duplicate policy.
            **metadata: Stored alongside the entry; see :meth:`metadata`.

        Returns:
            The decorator, which returns the registered object unchanged.
        """
        if callable(name) and not isinstance(name, str):
            obj, name = name, None
            return self._decorator(None, replace, metadata)(obj)
        return self._decorator(name, replace, metadata)

    def add(self, name: str, obj: Any, *, replace: Optional[bool] = None,
            **metadata: Any) -> Any:
        """Register ``obj`` under ``name`` without using a decorator."""
        return self._decorator(name, replace, metadata)(obj)

    def _decorator(self, name, replace, metadata):
        def decorate(obj):
            key = name if name is not None else getattr(obj, '__name__', None)
            if key is None:
                raise ValueError(
                    f"Cannot infer a name for {obj!r}; pass one explicitly")
            key = self._normalize(key)
            allow = (self._duplicates == 'replace') if replace is None else replace
            if key in self._entries and not allow:
                raise self._duplicate_error(
                    f"{self.label.capitalize()} {key!r} is already registered")
            self._entries[key] = obj
            self._metadata[key] = dict(metadata)
            return obj
        return decorate

    def unregister(self, name: str) -> None:
        """Remove an entry. Mainly useful in tests."""
        key = self._normalize(name)
        if key not in self._entries:
            raise self._unknown(key)
        del self._entries[key]
        del self._metadata[key]

    # --- Lookup -------------------------------------------------------------

    def get(self, name: str, default: Any = ...) -> Any:  # type: ignore[override]
        """Return the object registered under ``name``.

        Args:
            name (str): Registered name.
            default: Returned instead of raising when ``name`` is unknown.

        Raises:
            UnknownEntryError: If ``name`` is not registered and no default is
                given. The message names the extras package if the entry is
                provided by one, and otherwise lists the available names.
        """
        try:
            key = self._normalize(name)
        except ValueError:
            if default is not ...:
                return default
            raise
        if key in self._entries:
            return self._entries[key]
        if default is not ...:
            return default
        raise self._unknown(key)

    def metadata(self, name: str) -> Mapping:
        """Return the read-only metadata stored with ``name``."""
        key = self._normalize(name)
        if key not in self._metadata:
            raise self._unknown(key)
        return MappingProxyType(self._metadata[key])

    def names(self) -> List[str]:
        """Return the registered names, sorted."""
        return sorted(self._entries)

    def extras_package(self, name: str) -> Optional[str]:
        """Return the extras package that provides ``name``, if any."""
        return self._extras.get(self._normalize(name))

    @property
    def extras(self) -> Mapping:
        """Names provided by optional extras packages, mapped to the package."""
        return MappingProxyType(self._extras)

    def _unknown(self, key: str) -> Exception:
        package = self._extras.get(key)
        if package is not None:
            msg = (f"{self.label.capitalize()} {key!r} is provided by the "
                   f"'{package}' package. Install it with: pip install {package}")
        else:
            msg = (f"Unknown {self.label} {key!r}. "
                   f"Available: {self.names()}")
        return self._unknown_error(msg)

    # --- Mapping interface --------------------------------------------------

    def __getitem__(self, name: str) -> Any:
        return self.get(name)

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        try:
            return self._normalize(name) in self._entries
        except ValueError:
            return False

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"<Registry {self.kind!r} ({len(self)} entries)>"


def _default_normalize(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Registry names must be non-empty strings")
    return name.strip()


def get_registry(kind: str) -> Registry:
    """Return the registry named ``kind``."""
    try:
        return _REGISTRIES[kind]
    except KeyError:
        raise UnknownEntryError(
            f"Unknown registry {kind!r}. Available: {list_registries()}"
        ) from None


def list_registries() -> List[str]:
    """Return the names of all registries created so far, sorted.

    Registries are created when the module that owns them is imported, so
    this lists only the registries of modules already imported.
    """
    return sorted(_REGISTRIES)
