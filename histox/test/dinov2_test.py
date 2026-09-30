import sys
import types
import unittest
from unittest import mock

import torch


class _FakeDinoV2(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=384)
        self.last_pixel_values = None

    def forward(self, pixel_values):
        self.last_pixel_values = pixel_values.detach().clone()
        return types.SimpleNamespace(
            pooler_output=torch.ones(
                pixel_values.shape[0],
                self.config.hidden_size,
                device=pixel_values.device,
            ),
            last_hidden_state=None,
        )


class _FakeAutoModel:
    calls = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.calls.append((args, kwargs))
        return _FakeDinoV2()


class TestDinoV2Features(unittest.TestCase):

    def setUp(self):
        _FakeAutoModel.calls = []

    def _build_extractor(self, **kwargs):
        fake_transformers = types.SimpleNamespace(AutoModel=_FakeAutoModel)
        with mock.patch.dict(sys.modules, {'transformers': fake_transformers}):
            from histox.model.extractors.dinov2 import DinoV2Features
            with mock.patch(
                'histox.model.extractors.dinov2.torch_utils.get_device',
                return_value=torch.device('cpu'),
            ):
                return DinoV2Features(**kwargs)

    def _rebuild_extractor(self, config):
        fake_transformers = types.SimpleNamespace(AutoModel=_FakeAutoModel)
        with mock.patch.dict(sys.modules, {'transformers': fake_transformers}):
            from histox.model.extractors._factory import build_extractor_from_cfg
            with mock.patch(
                'histox.model.extractors.dinov2.torch_utils.get_device',
                return_value=torch.device('cpu'),
            ):
                return build_extractor_from_cfg(config)

    def test_huggingface_model_forward_and_config(self):
        extractor = self._build_extractor(
            model_id='facebook/dinov2-small',
            revision='fixed-revision',
            local_files_only=True,
            mixed_precision=False,
        )

        output = extractor(
            torch.zeros(2, 240, 320, 3, dtype=torch.uint8)
        )

        self.assertEqual(tuple(output.shape), (2, 384))
        self.assertEqual(extractor.num_features, 384)
        processed = extractor.model.model.last_pixel_values
        self.assertEqual(tuple(processed.shape), (2, 3, 224, 224))
        self.assertEqual(processed.dtype, torch.float32)
        expected_zero = torch.tensor([
            -0.485 / 0.229,
            -0.456 / 0.224,
            -0.406 / 0.225,
        ])
        torch.testing.assert_close(processed[0, :, 0, 0], expected_zero)
        self.assertEqual(
            _FakeAutoModel.calls,
            [(('facebook/dinov2-small',), {
                'revision': 'fixed-revision',
                'local_files_only': True,
            })],
        )
        self.assertEqual(extractor.dump_config()['kwargs'], {
            'model_id': 'facebook/dinov2-small',
            'revision': 'fixed-revision',
            'local_files_only': True,
        })

    def test_huggingface_config_rebuilds_extractor(self):
        from histox.model.extractors._factory import extractor_to_config

        extractor = self._build_extractor(
            model_id='facebook/dinov2-small',
            revision='fixed-revision',
            local_files_only=True,
            resize=280,
            mixed_precision=False,
        )
        config = extractor_to_config(extractor)

        rebuilt = self._rebuild_extractor(config)

        self.assertEqual(rebuilt.dump_config(), extractor.dump_config())
        self.assertFalse(rebuilt.mixed_precision)
        self.assertFalse(rebuilt.channels_last)

    def test_original_checkpoint_path_is_preserved(self):
        fake_model = _FakeDinoV2()
        fake_model.embed_dim = 768
        fake_omegaconf = types.ModuleType('omegaconf')
        fake_omegaconf.OmegaConf = types.SimpleNamespace(
            load=mock.Mock(return_value={'model': 'config'})
        )
        fake_dinov2 = types.ModuleType('dinov2')
        fake_dinov2.__path__ = []
        fake_eval = types.ModuleType('dinov2.eval')
        fake_eval.__path__ = []
        fake_setup = types.ModuleType('dinov2.eval.setup')
        fake_setup.build_model_for_eval = mock.Mock(return_value=fake_model)
        modules = {
            'omegaconf': fake_omegaconf,
            'dinov2': fake_dinov2,
            'dinov2.eval': fake_eval,
            'dinov2.eval.setup': fake_setup,
        }

        with mock.patch.dict(sys.modules, modules):
            from histox.model.extractors.dinov2 import DinoV2Features
            with mock.patch(
                'histox.model.extractors.dinov2.torch_utils.get_device',
                return_value=torch.device('cpu'),
            ):
                extractor = DinoV2Features(
                    cfg='config.yaml',
                    weights='model.pth',
                    mixed_precision=False,
                )

        fake_omegaconf.OmegaConf.load.assert_called_once_with('config.yaml')
        fake_setup.build_model_for_eval.assert_called_once_with(
            {'model': 'config'},
            'model.pth',
        )
        self.assertEqual(extractor.num_features, 768)
        self.assertEqual(extractor.dump_config()['kwargs'], {
            'cfg': 'config.yaml',
            'weights': 'model.pth',
        })

    def test_wrapper_falls_back_to_cls_token(self):
        from histox.model.extractors.dinov2 import _HuggingFaceDinoV2Model

        class _LastHiddenStateOnly(torch.nn.Module):
            def forward(self, pixel_values):
                batch_size = pixel_values.shape[0]
                hidden = torch.arange(
                    batch_size * 3 * 4,
                    dtype=torch.float32,
                ).reshape(batch_size, 3, 4)
                return types.SimpleNamespace(last_hidden_state=hidden)

        wrapper = _HuggingFaceDinoV2Model(_LastHiddenStateOnly())
        output = wrapper(torch.zeros(2, 3, 224, 224))

        expected = torch.tensor([
            [0., 1., 2., 3.],
            [12., 13., 14., 15.],
        ])
        torch.testing.assert_close(output, expected)

    def test_wrapper_rejects_output_without_embedding(self):
        from histox.model.extractors.dinov2 import _HuggingFaceDinoV2Model

        class _EmptyOutputModel(torch.nn.Module):
            def forward(self, pixel_values):
                return types.SimpleNamespace()

        wrapper = _HuggingFaceDinoV2Model(_EmptyOutputModel())

        with self.assertRaisesRegex(RuntimeError, 'did not return'):
            wrapper(torch.zeros(1, 3, 224, 224))

    def test_model_source_is_required(self):
        with self.assertRaisesRegex(ValueError, 'Provide model_id'):
            self._build_extractor()

    def test_cfg_and_weights_must_be_paired(self):
        with self.assertRaisesRegex(ValueError, 'provided together'):
            self._build_extractor(cfg='config.yaml')

    def test_sources_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, 'either model_id'):
            self._build_extractor(
                cfg='config.yaml',
                weights='model.pth',
                model_id='facebook/dinov2-small',
            )

    def test_huggingface_options_require_model_id(self):
        with self.assertRaisesRegex(ValueError, 'only valid with model_id'):
            self._build_extractor(
                cfg='config.yaml',
                weights='model.pth',
                revision='fixed-revision',
            )


if __name__ == '__main__':
    unittest.main()
