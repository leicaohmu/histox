import sys
import types
import unittest
from unittest import mock

import torch


class _FakePLIP(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(projection_dim=512)
        self.last_pixel_values = None

    def forward(self, pixel_values):
        self.last_pixel_values = pixel_values.detach().clone()
        return types.SimpleNamespace(
            image_embeds=torch.ones(pixel_values.shape[0], 512)
        )


class _FakePLIPLoader:
    calls = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.calls.append((args, kwargs))
        return _FakePLIP()


class TestPLIPFeatures(unittest.TestCase):

    def setUp(self):
        _FakePLIPLoader.calls = []

    def _build(self, **kwargs):
        fake_transformers = types.SimpleNamespace(
            CLIPVisionModelWithProjection=_FakePLIPLoader
        )
        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            from histox.model.extractors.plip import PLIPFeatures
            with mock.patch(
                "histox.model.extractors.plip.torch_utils.get_device",
                return_value=torch.device("cpu"),
            ):
                return PLIPFeatures(**kwargs)

    def test_forward_preprocess_and_config(self):
        extractor = self._build(
            revision="fixed-revision",
            local_files_only=True,
            mixed_precision=False,
        )
        output = extractor(torch.zeros(2, 256, 256, 3, dtype=torch.uint8))
        self.assertEqual(tuple(output.shape), (2, 512))
        self.assertEqual(extractor.num_features, 512)
        processed = extractor.model.model.last_pixel_values
        self.assertEqual(tuple(processed.shape), (2, 3, 224, 224))
        expected = torch.tensor([
            -0.48145466 / 0.26862954,
            -0.4578275 / 0.26130258,
            -0.40821073 / 0.27577711,
        ])
        torch.testing.assert_close(processed[0, :, 0, 0], expected)
        self.assertEqual(_FakePLIPLoader.calls, [
            (("vinid/plip",), {
                "revision": "fixed-revision",
                "local_files_only": True,
            })
        ])
        self.assertEqual(extractor.dump_config()["kwargs"], {
            "model_id": "vinid/plip",
            "revision": "fixed-revision",
            "local_files_only": True,
        })

    def test_wrapper_requires_projected_embedding(self):
        from histox.model.extractors.plip import _HuggingFacePLIPVisionModel

        class _BadModel(torch.nn.Module):
            def forward(self, pixel_values):
                return types.SimpleNamespace()

        wrapper = _HuggingFacePLIPVisionModel(_BadModel())
        with self.assertRaisesRegex(RuntimeError, "image_embeds"):
            wrapper(torch.zeros(1, 3, 224, 224))


if __name__ == "__main__":
    unittest.main()
