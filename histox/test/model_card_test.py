"""Tests for histox.model.ModelCard and extractor cards."""

import unittest

import histox as hx
from histox import errors
from histox.model import ModelCard, extractor_card


class ModelCardTest(unittest.TestCase):

    def test_dict_and_summary(self):
        card = ModelCard('enc', backend='torch', license='MIT', num_features=8,
                         input_px=224, norm_mean=(0.5, 0.5, 0.5), norm_std=(0.2, 0.2, 0.2),
                         weights={'model_id': 'org/enc', 'revision': None})
        d = card.to_dict()
        self.assertEqual(d['norm_mean'], [0.5, 0.5, 0.5])
        self.assertEqual(d['weights'], {'model_id': 'org/enc', 'revision': None})
        text = card.summary()
        for piece in ('enc (tile_encoder, torch)', 'features: 8', 'input: 224 px',
                      'model_id=org/enc', 'license: MIT', 'citation: not recorded'):
            self.assertIn(piece, text)
        self.assertNotIn('revision', text)    # None values are omitted
        missing = ModelCard('big', extras='histox-contrib', installed=False)
        self.assertIn('pip install histox-contrib', missing.summary())

    def test_registered_card_without_building(self):
        card = extractor_card('dinov2')
        self.assertEqual((card.name, card.backend, card.license), ('dinov2', 'torch', 'Apache-2.0'))
        self.assertIn('DINOv2', card.citation)
        self.assertEqual(card.details['class'], 'histox.model.extractors.dinov2.DinoV2Features')
        self.assertIsNone(card.num_features)          # only known once built

    def test_imagenet_alias(self):
        card = extractor_card('resnet50')
        self.assertEqual(card.name, 'resnet50_imagenet')
        self.assertIsNone(card.license)               # not recorded in the code

    def test_extras_and_unknown(self):
        from histox.model.extractors import _registry as reg
        if 'gigapath' not in reg._torch_extractors:
            card = extractor_card('gigapath')
            self.assertFalse(card.installed)
            self.assertEqual(card.extras, 'slideflow-noncommercial')
        with self.assertRaises(errors.InvalidFeatureExtractor):
            extractor_card('no_such_extractor')


class ExtractorCardTest(unittest.TestCase):

    def _extractor(self):
        import torch
        from histox.model.extractors._factory_torch import TorchFeatureExtractor

        class Tiny(TorchFeatureExtractor):
            tag = 'tiny'
            license = 'MIT'
            citation = '@misc{tiny}'

            def __init__(self):
                super().__init__()
                self.model = torch.nn.Linear(3, 5)
                self.num_features = 5
                self.transform = self.build_transform(img_size=224, resize=True)

            def dump_config(self):
                return self._dump_config(class_name='tests.Tiny', model_id='org/tiny')

        return Tiny()

    def test_card_from_built_extractor(self):
        ex = self._extractor()
        card = ex.card
        self.assertEqual((card.name, card.backend, card.num_features), ('tiny', 'torch', 5))
        self.assertEqual(card.input_px, 224)
        self.assertEqual(card.norm_mean, (0.485, 0.456, 0.406))
        self.assertEqual(card.norm_std, (0.229, 0.224, 0.225))
        self.assertEqual(card.weights, {'model_id': 'org/tiny'})
        self.assertEqual((card.license, card.citation), ('MIT', '@misc{tiny}'))
        self.assertEqual(card.details['class'], 'tests.Tiny')

    def test_module(self):
        import torch
        ex = self._extractor()
        self.assertIsInstance(ex.module, torch.nn.Linear)
        self.assertIs(ex.module, ex.model)
        ex.model = None
        self.assertIsNone(ex.module)


if __name__ == '__main__':
    unittest.main()
