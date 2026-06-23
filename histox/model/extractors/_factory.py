"""Factory for building feature extractors."""

import importlib
import histox as hx
from os.path import join, exists
from typing import Optional, Tuple, Dict, Any, TYPE_CHECKING
from histox import errors
from histox.model import BaseFeatureExtractor

from ._registry import (is_tensorflow_extractor, is_torch_extractor,
                        _tf_extractors, _torch_extractors, _extras_extractors)
from ._factory_tensorflow import build_tensorflow_feature_extractor
from ._factory_torch import build_torch_feature_extractor

if TYPE_CHECKING:
    from histox.norm import StainNormalizer

# -----------------------------------------------------------------------------

def build_feature_extractor(
    name: str,
    backend: Optional[str] = None,
    **kwargs
) -> BaseFeatureExtractor:
    """Build a feature extractor.

    The returned feature extractor is a callable object, which returns
    features (often layer activations) for either a batch of images or a
    ``histox.WSI`` object.

    If generating features for a batch of images, images are expected to be in
    ``(B, W, H, C)`` format, non-standardized (scaled 0-255), with dtype
    ``uint8``. The feature extractor performs all needed preprocessing
    (normalization, resizing, channel reordering) on the fly.

    If generating features for a slide, the slide is expected to be a
    ``histox.WSI`` object. The feature extractor will generate features
    for each tile in the slide, returning a numpy array of shape ``(W, H, F)``,
    where ``F`` is the number of features.

    Parameters
    ----------
    name : str
        Name of the feature extractor to build, or a path to a saved
        histox model (Tensorflow or PyTorch). When a model path is
        provided, a ``Features`` (or ``UncertaintyInterface`` if UQ is
        enabled) object is returned directly from the saved model.
        Available named extractors can be listed with [`histox.model.list_extractors`][].
    backend : str, optional
        Deep learning backend to use, one of ``{'tensorflow', 'torch'}``.
        If ``None``, the backend is selected automatically based on the
        ``name`` argument (see Notes). Defaults to ``None``.
    **kwargs : Any
        Additional keyword arguments passed to the feature extractor factory
        function. Common options include:

        - **tile_px** (*int*) - Tile size (input image size), in pixels.
          Required for ImageNet-pretrained models (e.g. ``resnet50_imagenet``).
        - **layers** (*str or list of str*) - Layer name(s) at which to
          intercept activations during the forward pass. A ``forward hook``
          is registered on each specified layer; activations are captured
          automatically at inference time and returned as the feature output.
          The special value ``'postconv'`` registers a hook on the
          post-convolutional layer, which is predefined for each supported
          architecture (e.g. ``avgpool`` for ResNet, outputting a feature
          vector of shape ``(B, 2048)`` for ResNet50). Any other string is
          resolved by name against the model's module tree via
          [`histox.model.torch_utils.get_module_by_name`][]; use
          ``print(extractor.ftrs._model)`` to inspect available layer names.
          When multiple layers are specified as a list, their outputs are
          concatenated along the feature dimension, and ``num_features``
          equals the sum of all individual layer output sizes.
          Defaults to ``'postconv'``.

        - **include_preds** (*bool*) - Whether to append model predictions
          (final logits) to the output features. Defaults to ``False``.
        - **mixed_precision** (*bool*) - Use FP16 mixed precision.
          Defaults to ``True``.
        - **pooling** (*str or Callable*) - Pooling applied to intermediate
          feature maps before flattening. May be ``'avg'``
          (adaptive average pooling), ``'max'`` (adaptive max pooling),
          or a custom callable that accepts and returns a ``Tensor``.
          Only applied to layers whose output has 4 dimensions ``(B, C, H, W)``.
          Has no effect on ``'postconv'``, which uses its own pooling logic.
          Defaults to ``None`` (no pooling).

        The following preprocessing transform arguments are also accepted
        for PyTorch ImageNet-pretrained extractors:

        - **center_crop** (*int or bool*) - If an integer, center-crop
          images to this size before inference. If ``True``, crop to
          ``tile_px``. Defaults to ``None`` (no cropping).
        - **resize** (*int or bool*) - If an integer, resize images to
          this size before inference. If ``True``, resize to ``tile_px``.
          Defaults to ``None`` (no resizing).
        - **interpolation** (*str*) - Interpolation mode used when resizing.
          One of ``'bilinear'``, ``'bicubic'``, ``'nearest'``,
          ``'nearest_exact'``. Defaults to ``'bilinear'``.
        - **antialias** (*bool*) - Apply antialiasing filter when resizing.
          Defaults to ``False``.
        - **norm_mean** (*tuple of float*) - Per-channel normalization mean
          applied after scaling to ``[0, 1]``.
          Defaults to ``(0.485, 0.456, 0.406)`` (ImageNet mean).
        - **norm_std** (*tuple of float*) - Per-channel normalization std
          applied after scaling to ``[0, 1]``.
          Defaults to ``(0.229, 0.224, 0.225)`` (ImageNet std).

    Returns
    -------
    BaseFeatureExtractor
        A callable object which accepts either:

        - A batch of images of shape ``(B, W, H, C)``, dtype ``uint8``,
          and returns features of shape ``(B, F)``, dtype ``float32``.
        - A ``histox.WSI`` object, and returns a spatially-mapped feature
          array of shape ``(W, H, F)``, dtype ``float32``.

    Raises
    ------
    ValueError
        If ``backend`` is not one of ``{'tensorflow', 'torch'}``.
    InvalidFeatureExtractor
        If ``name`` is not a recognized feature extractor for the
        specified or active backend. If the extractor requires an optional
        package that is not installed, the error message will indicate
        the package name and the install command.

    Notes
    -----
    The ``name`` parameter supports two modes:

    **1. Registered extractor name** (e.g. ``'resnet50_imagenet'``,
    ``'ctranspath'``):
    histox maintains an internal registry mapping extractor names to
    their implementations, which may exist in one or both backends
    (``'torch'`` and ``'tensorflow'``). When ``name`` is a registered
    extractor name, the backend is resolved in the following order:

    - Step 1: Use the manually specified ``backend`` argument, if provided.
    - Step 2: If ``backend=None`` and ``name`` is only available in one
      backend, that backend is used automatically.
    - Step 3: If ``backend=None`` and ``name`` is available in both backends,
      the currently active backend (``histox.backend()``) is used,
      and a notice is logged.

    For ImageNet-pretrained models, the ``_imagenet`` suffix in ``name``
    may be omitted; e.g. ``'resnet50'`` is automatically resolved to
    ``'resnet50_imagenet'``.

    The models ``'xception_imagenet'`` and ``'nasnet_large_imagenet'``
    use a different normalization strategy (mean and std of
    ``[0.5, 0.5, 0.5]``) compared to the standard ImageNet normalization
    used by other models. Custom ``norm_mean`` / ``norm_std`` kwargs are
    ignored for these two models.

    **2. Model path** (e.g. ``'/path/to/saved/model'``):
    When ``name`` is a path to a saved histox model, a ``Features``
    object (or ``UncertaintyInterface`` if UQ is enabled) is returned
    directly, bypassing the extractor registry and backend resolution
    entirely. The backend is inferred automatically from the saved
    model format.

    For PyTorch ImageNet-pretrained models, the underlying ``nn.Module``
    is accessible via ``extractor.ftrs._model.model`` after construction.
    To inspect available layer names for use with the ``layers`` argument,
    print the model wrapper::

        print(extractor.ftrs._model)


    Examples
    --------
    Create an extractor using an ImageNet-pretrained ResNet50, extracting
    from the default post-convolutional layer (``avgpool``, 2048-dim):

    >>> import histox as hx
    >>> extractor = hx.build_feature_extractor(
    ...     'resnet50_imagenet',
    ...     tile_px=224
    ... )
    >>> extractor.num_features  # 2048

    Equivalently, specify ``'postconv'`` explicitly:

    >>> extractor = hx.build_feature_extractor(
    ...     'resnet50_imagenet',
    ...     tile_px=224,
    ...     layers='postconv'
    ... )

    Extract activations from a specific intermediate layer by name
    (use ``print(extractor.ftrs._model)`` to inspect available layer names):

    >>> extractor = hx.build_feature_extractor(
    ...     'resnet50_imagenet',
    ...     tile_px=224,
    ...     layers='model.layer3'
    ... )
    >>> extractor.num_features  # 1024

    Concatenate activations from multiple layers:

    >>> extractor = hx.build_feature_extractor(
    ...     'resnet50_imagenet',
    ...     tile_px=224,
    ...     layers=['model.layer3', 'model.layer4']
    ... )
    >>> extractor.num_features  # 1024 + 2048 = 3072

    Create a pretrained CTransPath extractor:

    >>> extractor = hx.build_feature_extractor('ctranspath')

    Load a feature extractor from a saved finetuned model:

    >>> extractor = hx.build_feature_extractor('/path/to/saved/model')

    Calculate features for an entire dataset:

    >>> P = hx.load_project('/path/to/project')
    >>> dataset = P.dataset(tile_px=224, tile_um=302)
    >>> resnet = hx.build_feature_extractor('resnet50_imagenet', tile_px=224)
    >>> features = hx.DatasetFeatures(resnet, dataset=dataset)

    Generate a map of features across a whole-slide image:

    >>> wsi = hx.WSI('/path/to/slide.svs', tile_px=224, tile_um=302)
    >>> retccl = hx.build_feature_extractor('retccl', resize=True)
    >>> features = retccl(wsi)  # shape: (W, H, F)
    """
    # Build feature extractor according to manually specified backend
    if backend is not None and backend not in ('tensorflow', 'torch'):
        raise ValueError(f"Invalid backend: {backend}")

    # Build a feature extractor from a finetuned model
    if hx.util.is_tensorflow_model_path(name):
        model_config = hx.util.get_model_config(name)
        if model_config['hp']['uq']:
            from histox.model.tensorflow import UncertaintyInterface
            return UncertaintyInterface(name, **kwargs)
        else:
            from histox.model.tensorflow import Features
            return Features(name, **kwargs)
    elif hx.util.is_torch_model_path(name):
        model_config = hx.util.get_model_config(name)
        if model_config['hp']['uq']:
            from histox.model.torch import UncertaintyInterface
            return UncertaintyInterface(name, **kwargs)
        else:
            from histox.model.torch import Features  # noqa: F401
            return Features(name, **kwargs)

    # Build feature extractor with a specific backend
    if backend == 'tensorflow':
        if not is_tensorflow_extractor(name):
            raise errors.InvalidFeatureExtractor(
                f"Feature extractor {name} not available in Tensorflow backend")
        return build_tensorflow_feature_extractor(name, **kwargs)
    elif backend == 'torch':
        if not is_torch_extractor(name):
            raise errors.InvalidFeatureExtractor(
                f"Feature extractor {name} not available in PyTorch backend")
        return build_torch_feature_extractor(name, **kwargs)

    # Auto-build feature extractor according to available backends
    if is_tensorflow_extractor(name) and is_torch_extractor(name):
        hx.log.info(
            f"Feature extractor {name} available in both Tensorflow and "
            f"PyTorch backends; using active backend {hx.backend()}")
        if hx.backend() == 'tensorflow':
            return build_tensorflow_feature_extractor(name, **kwargs)
        else:
            return build_torch_feature_extractor(name, **kwargs)
    if is_tensorflow_extractor(name):
        return build_tensorflow_feature_extractor(name, **kwargs)
    elif is_torch_extractor(name):
        return build_torch_feature_extractor(name, **kwargs)
    elif name in _extras_extractors:
        raise errors.InvalidFeatureExtractor(
            "{} requires the package {}, please install with 'pip install {}'".format(
                name, _extras_extractors[name], _extras_extractors[name]
        ))
    else:
        raise errors.InvalidFeatureExtractor(f"Unrecognized feature extractor: {name}")


def rebuild_extractor(
    bags_or_model: str,
    allow_errors: bool = False,
    native_normalizer: bool = True
) -> Tuple[Optional["BaseFeatureExtractor"], Optional["StainNormalizer"]]:
    """Recreate the extractor used to generate features stored in bags.

    Args:
        bags_or_model (str): Either a path to directory containing feature bags,
            or a path to a trained MIL model. If a path to a trained MIL model,
            the extractor used to generate features will be recreated.
        allow_errors (bool): If True, return None if the extractor
            cannot be rebuilt. If False, raise an error. Defaults to False.
        native_normalizer (bool, optional): Whether to use PyTorch/Tensorflow-native
            stain normalization, if applicable. If False, will use the OpenCV/Numpy
            implementations. Defaults to True.

    Returns:
        Optional[BaseFeatureExtractor]: Extractor function, or None if ``allow_errors`` is
            True and the extractor cannot be rebuilt.

        Optional[StainNormalizer]: Stain normalizer used when generating
            feature bags, or None if no stain normalization was used.

    """
    # Load bags configuration
    is_bag_config = bags_or_model.endswith('bags_config.json')
    is_bag_dir = exists(join(bags_or_model, 'bags_config.json'))
    is_model_dir = exists(join(bags_or_model, 'mil_params.json'))
    if not (is_bag_dir or is_model_dir or is_bag_config):
        if allow_errors:
            return None, None
        else:
            raise ValueError(
                'Could not find bags or MIL model configuration at '
                f'{bags_or_model}.'
            )
    if is_bag_config:
        bags_config = hx.util.load_json(bags_or_model)
    elif is_model_dir:
        mil_config = hx.util.load_json(join(bags_or_model, 'mil_params.json'))
        if 'bags_extractor' not in mil_config:
            if allow_errors:
                return None, None
            else:
                raise ValueError(
                    'Could not rebuild extractor from configuration at '
                    f'{bags_or_model}; missing "bags_extractor" key in '
                    'mil_params.json.'
                )
        bags_config = mil_config['bags_extractor']
    else:
        bags_config = hx.util.load_json(join(bags_or_model, 'bags_config.json'))
    if ('extractor' not in bags_config
       or any(n not in bags_config['extractor'] for n in ['class', 'kwargs'])):
        if allow_errors:
            return None, None
        else:
            raise ValueError(
                'Could not rebuild extractor from configuration at '
                f'{bags_or_model}; missing "extractor" class or kwargs.'
            )

    # Rebuild extractor
    extractor_name = bags_config['extractor']['class'].split('.')
    extractor_class = extractor_name[-1]
    extractor_kwargs = bags_config['extractor']['kwargs']
    try:
        module = importlib.import_module('.'.join(extractor_name[:-1]))
        extractor = getattr(module, extractor_class)(**extractor_kwargs)
    except Exception:
        submodule_name = extractor_name[-2]
        if submodule_name in _extras_extractors:
            raise errors.InvalidFeatureExtractor(
                "{} requires the package {}, please install with 'pip install {}'".format(
                    submodule_name, 
                    _extras_extractors[submodule_name], 
                    _extras_extractors[submodule_name]
            ))
        if allow_errors:
            return None
        else:
            raise ValueError(
                f'Could not rebuild extractor from configuration at {bags_or_model}.'
            )

    # Rebuild stain normalizer
    if bags_config['normalizer'] is not None:
        normalizer = hx.norm.autoselect(
            bags_config['normalizer']['method'],
            backend=(extractor.backend if native_normalizer else 'opencv')
        )
        normalizer.set_fit(**bags_config['normalizer']['fit'])
    else:
        normalizer = None
    if (hasattr(extractor, 'normalizer')
       and extractor.normalizer is not None
       and normalizer is not None):
        hx.log.warning(
            'Extractor already has a stain normalizer. Overwriting with '
            'normalizer from bags configuration.'
        )
        extractor.normalizer = normalizer
    elif hasattr(extractor, 'normalizer') and extractor.normalizer is not None:
        normalizer = extractor.normalizer

    return extractor, normalizer

# -----------------------------------------------------------------------------

def extractor_to_config(extractor: BaseFeatureExtractor) -> Dict[str, Any]:
    """Return a dictionary of configuration parameters for the extractor.

    These configuration parameters can be used to reconstruct the
    feature extractor, using ``build_extractor_from_cfg()``.

    Args:
        extractor (BaseFeatureExtractor): Feature extractor.

    Returns:
        Dict[str, Any]: Configuration dictionary.

    """
    cfg = extractor.dump_config()
    if extractor.backend == 'torch':
        cfg['mixed_precision'] = extractor.mixed_precision
        cfg['channels_last'] = extractor.channels_last
    return cfg


def build_extractor_from_cfg(
    cfg: Dict[str, Any],
    **kwargs: Any
) -> BaseFeatureExtractor:
    """Rebuild an extractor from a configuration dictionary.

    Args:
        cfg (Dict[str, Any]): Configuration dictionary.
        **kwargs (Any): All remaining keyword arguments are passed
            to the feature extractor factory function, and may be different
            for each extractor.

    Returns:
        BaseFeatureExtractor: The rebuilt feature extractor.

    """
    extractor_name = cfg['class'].split('.')
    extractor_class = extractor_name[-1]
    extractor_kwargs = cfg['kwargs']
    module = importlib.import_module('.'.join(extractor_name[:-1]))
    extractor = getattr(module, extractor_class)(**extractor_kwargs, **kwargs)
    for k, v in cfg.items():
        if k not in ['class', 'kwargs']:
            setattr(extractor, k, v)
    return extractor
