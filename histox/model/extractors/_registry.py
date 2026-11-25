"""Feature extractor registry."""

from histox.registry import Registry

_known_extras_packages = {
     'histox-contrib': ['retccl', 'ctranspath'],
     'slideflow-noncommercial': ['gigapath', 'gigapath.tile', 'gigapath.slide', 'histossl']
}
_extras_extractors = {
    extractor: package 
    for package, extractors in _known_extras_packages.items() 
    for extractor in extractors
}

# Both registries replace silently on re-registration, as before: optional
# packages and plugins may override a built-in extractor of the same name.
_tf_extractors = Registry(
    'extractor.tensorflow', duplicates='replace', extras=_extras_extractors,
    label='TensorFlow feature extractor')
_torch_extractors = Registry(
    'extractor.torch', duplicates='replace', extras=_extras_extractors,
    label='PyTorch feature extractor')

__all__ = ['list_extractors', 'list_tensorflow_extractors', 'list_torch_extractors',
           'is_extractor', 'is_tensorflow_extractor', 'is_torch_extractor']

# -----------------------------------------------------------------------------

def list_extractors():
    """Return a list of all available feature extractors.

    Scans both the Tensorflow and PyTorch extractor registries and returns
    a deduplicated list of all registered extractor names.

    Extractors are registered automatically via the ``@register_tf`` and
    ``@register_torch`` decorators when the corresponding backend module
    (``histox.model.tensorflow`` or ``histox.model.torch``) is imported.
    Therefore, the returned list reflects only the extractors available
    under the **currently active backend**.

    Returns:
        extractor_lists: list[str], a deduplicated list of extractor names,
            sorted by name.

    See Also:
        - `list_tensorflow_extractors`: List only Tensorflow extractors.
        - `list_torch_extractors`: List only PyTorch extractors.
        - `build_feature_extractor`: Build an extractor by name.

    Examples:
        .. code-block:: python

            import histox as hx

            # List all available extractors
            extractors = hx.model.list_extractors()
            print(extractors)
            # ['vgg19_imagenet', 'resnet101_v2_imagenet', 'virchow', ...]
    """
    return sorted(set(_tf_extractors) | set(_torch_extractors))

def list_tensorflow_extractors():
    """Return a list of all Tensorflow feature extractors."""
    return _tf_extractors.names()

def list_torch_extractors():
    """Return a list of all PyTorch feature extractors."""
    return _torch_extractors.names()

def is_extractor(name):
    """Checks if a given name is a valid feature extractor."""
    _valid_extractors = list_extractors()
    return (name in _valid_extractors or name+'_imagenet' in _valid_extractors)

def is_tensorflow_extractor(name):
    """Checks if a given name is a valid Tensorflow feature extractor."""
    return name in _tf_extractors or name+'_imagenet' in _tf_extractors

def is_torch_extractor(name):
    """Checks if a given name is a valid PyTorch feature extractor."""
    return name in _torch_extractors or name+'_imagenet' in _torch_extractors

# -----------------------------------------------------------------------------

def register_torch(key_name=None, **metadata):
    """Decorator to register a PyTorch feature extractor.

    Use as ``@register_torch`` or ``@register_torch('name')``. Keyword
    arguments (e.g. ``license=...``, ``citation=...``) are stored as registry
    metadata.
    """
    return _torch_extractors.register(key_name, **metadata)


def register_tf(key_name=None, **metadata):
    """Decorator to register a Tensorflow feature extractor.

    Use as ``@register_tf`` or ``@register_tf('name')``.
    """
    return _tf_extractors.register(key_name, **metadata)
