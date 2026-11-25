import importlib.util
import unittest

from histox.util import smac_utils

HAS_HPO = all(importlib.util.find_spec(m) for m in ('smac', 'ConfigSpace'))


class HPOOptionalTest(unittest.TestCase):

    @unittest.skipIf(HAS_HPO, 'smac and ConfigSpace are installed')
    def test_missing_hpo_gives_install_hint(self):
        with self.assertRaises(ImportError) as cm:
            smac_utils.create_search_space(dropout=(0.1, 0.5))
        self.assertIn("histox[hpo]", str(cm.exception))

    @unittest.skipUnless(HAS_HPO, 'smac and ConfigSpace are not installed')
    def test_search_space_with_hpo(self):
        cs = smac_utils.create_search_space(dropout=(0.1, 0.5))
        self.assertIn('dropout', [h.name for h in cs.values()] if hasattr(cs, 'values')
                      else cs.get_hyperparameter_names())


if __name__ == '__main__':
    unittest.main()
