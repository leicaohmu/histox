import subprocess
import sys
import unittest


def _run(code):
    return subprocess.run([sys.executable, '-c', code], capture_output=True,
                          text=True, timeout=300)


class LazySubmoduleTest(unittest.TestCase):
    """Subpackages are reachable as attributes right after ``import histox``.

    Each check runs in a fresh interpreter, so no earlier import in the test
    session can make it pass by accident.
    """

    def test_mil_without_explicit_import(self):
        proc = _run("import histox as hx; print(callable(hx.mil.mil_config))")
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn('True', proc.stdout)

    def test_transforms_without_explicit_import(self):
        proc = _run("import histox as hx; print(callable(hx.transforms.normalizer))")
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn('True', proc.stdout)

    def test_unknown_attribute(self):
        proc = _run("import histox as hx; print(hasattr(hx, 'no_such_module'))")
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn('False', proc.stdout)


if __name__ == '__main__':
    unittest.main()
