import os
import subprocess
import sys
import unittest
import warnings

import histox as hx
from histox import _deprecation


def _caught(fn):
    """Run ``fn`` and return the HistoX deprecation warnings it emitted.

    ``assertWarns`` is avoided because it inspects every loaded module, which
    fails in environments with lazily imported packages.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fn()
    return [w for w in caught if issubclass(w.category, hx.HistoXDeprecationWarning)]


class DeprecationTest(unittest.TestCase):

    def test_warning_class(self):
        self.assertIs(hx.HistoXDeprecationWarning, _deprecation.HistoXDeprecationWarning)
        self.assertTrue(issubclass(hx.HistoXDeprecationWarning, FutureWarning))

    def test_warn_deprecated_message(self):
        found = _caught(lambda: _deprecation.warn_deprecated(
            'old()', since='0.3', removal='0.5', replacement='new()'))
        self.assertEqual(len(found), 1)
        msg = str(found[0].message)
        self.assertIn('old() is deprecated since HistoX 0.3', msg)
        self.assertIn('no earlier than 0.5', msg)
        self.assertIn('Use new() instead.', msg)

    def test_warn_tensorflow_once(self):
        _deprecation._TENSORFLOW_WARNED = False
        try:
            first = _caught(_deprecation.warn_tensorflow)
            self.assertEqual(len(first), 1)
            self.assertIn('TensorFlow backend', str(first[0].message))
            self.assertEqual(_caught(_deprecation.warn_tensorflow), [])
        finally:
            _deprecation._TENSORFLOW_WARNED = True

    def test_tensorflow_backend_warns_on_import(self):
        # The warning is emitted when the backend is chosen, before any
        # TensorFlow import, so it appears even if TensorFlow is missing.
        env = dict(os.environ, HX_BACKEND='tensorflow')
        proc = subprocess.run(
            [sys.executable, '-c', 'import histox'],
            env=env, capture_output=True, text=True, timeout=300)
        self.assertIn('The TensorFlow backend is deprecated', proc.stderr)
        self.assertIn('HistoXDeprecationWarning', proc.stderr)

    def test_torch_backend_is_silent(self):
        env = dict(os.environ, HX_BACKEND='torch')
        proc = subprocess.run(
            [sys.executable, '-c', 'import histox'],
            env=env, capture_output=True, text=True, timeout=300)
        self.assertNotIn('HistoXDeprecationWarning', proc.stderr)


if __name__ == '__main__':
    unittest.main()
