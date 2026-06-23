"""Feature extractor registry."""

_tf_extractors = dict()
_torch_extractors = dict()
_known_extras_packages = {
     'histox-contrib': ['retccl', 'ctranspath'],
     'histox-noncommercial': ['gigapath', 'gigapath.tile', 'gigapath.slide', 'histossl', 'plip']
}
_extras_extractors = {
    extractor: package 
    for package, extractors in _known_extras_packages.items() 
    for extractor in extractors
}

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
        extractor_lists: list[str], a deduplicated list of extractor names. Note that the
            order of the returned list is not guaranteed.

    See Also:
        - `list_tensorflow_extractors`: List only Tensorflow extractors.
        - `list_torch_extractors`: List only PyTorch extractors.
        - `build_feature_extractor`: Build an extractor by name.

    Examples:
        ```python
        import histox as hx

        # List all available extractors
        extractors = hx.model.list_extractors()
        print(extractors)
        # ['vgg19_imagenet', 'resnet101_v2_imagenet', 'virchow', ...]
        ```
    """
    return list(set(list(_tf_extractors.keys()) + list(_torch_extractors.keys())))

def list_tensorflow_extractors():
    """Return a list of all Tensorflow feature extractors."""
    return list(_tf_extractors.keys())

def list_torch_extractors():
    """Return a list of all PyTorch feature extractors."""
    return list(_torch_extractors.keys())

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

def register_torch(key_name=None):
    """Decorator to register a PyTorch feature extractor."""

    def decorator(fn):
        # Use the custom key name if provided, otherwise use the function's name
        name = key_name if isinstance(key_name, str) else fn.__name__
        _torch_extractors[name] = fn
        return fn

    # If the decorator is used without arguments, the key_name will be the function itself
    if callable(key_name):
        return decorator(key_name)

    return decorator

def register_tf(key_name=None):
    """Decorator to register a Tensorflow feature extractor."""

    def decorator(fn):
        # Use the custom key name if provided, otherwise use the function's name
        name = key_name if isinstance(key_name, str) else fn.__name__
        _tf_extractors[name] = fn
        return fn

    # If the decorator is used without arguments, the key_name will be the function itself
    if callable(key_name):
        return decorator(key_name)

    return decorator
