import importlib.util
import unittest


class TestVirchowMetadata(unittest.TestCase):

    @unittest.skipUnless(
        importlib.util.find_spec('timm') is not None,
        'timm is required to import the Virchow extractor',
    )
    def test_license_matches_upstream_model_card(self):
        from histox.model.extractors.virchow import VirchowFeatures

        self.assertIn('Apache-2.0', VirchowFeatures.license)
        self.assertIn(
            'https://huggingface.co/paige-ai/Virchow',
            VirchowFeatures.license,
        )
        self.assertNotIn('CC-BY-NC-ND', VirchowFeatures.license)
        self.assertNotIn('non-commercial', VirchowFeatures.license.lower())


if __name__ == '__main__':
    unittest.main()
