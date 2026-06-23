"""Configure the deep learning and slide reading backends."""

import os
import importlib.util

# Deep learning backend - use Tensorflow if available.
_valid_backends = ('torch', 'tensorflow')
if 'HX_BACKEND' not in os.environ:
    if importlib.util.find_spec('torch'):
        os.environ['HX_BACKEND'] = 'torch'
    elif importlib.util.find_spec('tensorflow'):
        os.environ['HX_BACKEND'] = 'tensorflow'
    else:
        os.environ['HX_BACKEND'] = 'torch'
elif os.environ['HX_BACKEND'] not in _valid_backends:
    raise ValueError("Unrecognized backend set via environmental variable "
                    "HX_BACKEND: {}. Expected one of: {}".format(
                        os.environ['HX_BACKEND'],
                        ', '.join(_valid_backends)
                    ))

# Slide backend - use cuCIM if available.
_valid_slide_backends = ('cucim', 'libvips')
if 'HX_SLIDE_BACKEND' not in os.environ:
    os.environ['HX_SLIDE_BACKEND'] = 'libvips'
    if importlib.util.find_spec('cucim'):
        import cucim
        if cucim.is_available():
            os.environ['HX_SLIDE_BACKEND'] = 'cucim'
elif os.environ['HX_SLIDE_BACKEND'] not in _valid_slide_backends:
    raise ValueError("Unrecognized slide backend set via environmental variable"
                    " HX_SLIDE_BACKEND: {}. Expected one of: {}".format(
                        os.environ['HX_SLIDE_BACKEND'],
                        ', '.join(_valid_slide_backends)
                    ))

# -----------------------------------------------------------------------------

def backend():
    """
    Return the current deep learning backend.
    
    The backend is determined by the environment variable ``HX_BACKEND``.
    If not explicitly set, the backend is auto-detected in the following order:

    1. ``torch``        - used if PyTorch is installed.
    2. ``tensorflow``   - used if TensorFlow is installed.
    3. ``torch``        - default fallback if neither is found.

    Returns
    -------
    str
        Name of the activate backend, one of ``{'torch', 'tensorflow'}``.

    Examples
    --------
    >>> import histox as hx
    >>> hx.backend()
    'torch'
    """
    return os.environ['HX_BACKEND']


def slide_backend():
    """
    Return the current slide reading backend.
    
    The backend is determined by the environment variable ``HX_SLIDE_BACKEND``.
    If not explicitly set, the backend is auto-detected in the following order:
    
    1. ``cucim``    - used if cuCIM is installed and available (GPU-accelerated).
    2. ``libvips``  - default fallback.

    Returns
    -------
    str
        Name of the active slide backend, one of ``{'cucim', 'libvips'}``.

    Examples
    --------
    >>> import histox as hx
    >>> hx.slide_backend()
    'cucim'
    """
    return os.environ['HX_SLIDE_BACKEND']
