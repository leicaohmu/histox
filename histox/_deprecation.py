"""Deprecation warnings for HistoX.

HistoX follows the policy in RFC 0001, section 6: a deprecated API emits
:class:`HistoXDeprecationWarning` naming its replacement, and stays for at
least two minor releases before it is removed.
"""

import warnings
from typing import Optional

__all__ = ['HistoXDeprecationWarning', 'warn_deprecated', 'warn_tensorflow']


class HistoXDeprecationWarning(FutureWarning):
    """Warning for HistoX features that are deprecated and will be removed.

    It subclasses :class:`FutureWarning` rather than
    :class:`DeprecationWarning` so that Python shows it to end users by
    default, not only to developers running tests.

    To silence it::

        import warnings
        import histox as hx
        warnings.filterwarnings('ignore', category=hx.HistoXDeprecationWarning)
    """


def warn_deprecated(
    what: str,
    *,
    since: str,
    removal: str,
    replacement: Optional[str] = None,
    stacklevel: int = 3,
) -> None:
    """Emit a :class:`HistoXDeprecationWarning`.

    Args:
        what (str): The deprecated feature, e.g. ``"Dataset.foo()"``.
        since (str): Version in which the feature was deprecated.
        removal (str): Earliest version in which it may be removed.
        replacement (str, optional): What to use instead.
        stacklevel (int): Passed to :func:`warnings.warn`. The default points
            at the caller of the function that calls ``warn_deprecated``.
    """
    msg = '{} is deprecated since HistoX {} and will be removed no earlier than {}.'.format(
        what, since, removal)
    if replacement:
        msg += ' Use {} instead.'.format(replacement)
    warnings.warn(msg, HistoXDeprecationWarning, stacklevel=stacklevel)


_TENSORFLOW_WARNED = False


def warn_tensorflow(stacklevel: int = 3) -> None:
    """Warn, once per process, that the TensorFlow backend is deprecated."""
    global _TENSORFLOW_WARNED
    if _TENSORFLOW_WARNED:
        return
    _TENSORFLOW_WARNED = True
    warn_deprecated(
        'The TensorFlow backend',
        since='0.3',
        removal='0.5',
        replacement='the PyTorch backend (HX_BACKEND=torch)',
        stacklevel=stacklevel + 1,
    )
