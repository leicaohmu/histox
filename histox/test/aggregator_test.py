import importlib.util
import unittest

import torch
from torch import nn

import histox as hx
from histox.mil import SlideAggregator
from histox.mil.eval import run_inference
from histox.mil.models import Attention_MIL
from histox.models import ABMIL
from histox.nn import Aggregator, AggregatorOutput, lens_to_mask, masked_softmax

HAS_NYSTROM = importlib.util.find_spec('nystrom_attention') is not None


class MeanPool(Aggregator):
    provides_attention = False

    def __init__(self, in_features, num_classes):
        super().__init__()
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, features, mask=None, *, coords=None):
        mask = self.full_mask(features) if mask is None else mask
        w = mask.unsqueeze(-1).to(features.dtype)
        pooled = (features * w).sum(1) / w.sum(1).clamp(min=1)
        return AggregatorOutput(self.head(pooled), embedding=pooled)


def _padded(lens, f=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(len(lens), max(lens), f, generator=g)
    mask = lens_to_mask(torch.tensor(lens))
    return x * mask.unsqueeze(-1), mask


class HelpersTest(unittest.TestCase):

    def test_lens_to_mask(self):
        m = lens_to_mask(torch.tensor([3, 1]), 4)
        self.assertEqual(m.tolist(), [[True, True, True, False], [True, False, False, False]])

    def test_masked_softmax(self):
        s = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        m = torch.tensor([[True, True, False], [False, False, False]])
        w = masked_softmax(s, m)
        self.assertEqual(w[0, 2].item(), 0.0)
        self.assertAlmostEqual(w[0].sum().item(), 1.0, places=6)
        self.assertEqual(w[1].tolist(), [0.0, 0.0, 0.0])

    def test_base_forward(self):
        with self.assertRaises(NotImplementedError):
            Aggregator()(torch.zeros(1, 2, 3))
        self.assertIs(hx.nn.Aggregator, Aggregator)
        self.assertIs(hx.models.ABMIL, ABMIL)


class CustomAggregatorTest(unittest.TestCase):

    def test_slide_aggregator_with_custom_model(self):
        torch.manual_seed(0)
        model = MeanPool(8, 3).eval()
        x, mask = _padded([5, 2])
        agg = SlideAggregator(model)
        self.assertTrue(agg.is_aggregator)
        self.assertFalse(agg.supports_attention)
        self.assertTrue(agg.supports_activations)
        logits = agg.predict(x, mask.sum(1))
        single = model(x[1:, :2]).logits
        torch.testing.assert_close(logits[1:], single)
        h, att = agg.activations(x, mask.sum(1))
        self.assertEqual(tuple(h.shape), (2, 8))
        self.assertIsNone(att)
        with self.assertRaises(RuntimeError):
            agg.predict(x, attention=True)
        y, _, _ = run_inference(model, x[:1], apply_softmax=True)
        self.assertAlmostEqual(y.sum().item(), 1.0, places=5)


class ABMILTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.legacy = Attention_MIL(8, 3, z_dim=16).eval()
        self.model = ABMIL.from_legacy(self.legacy).eval()

    def test_same_outputs_as_legacy(self):
        x, mask = _padded([6, 3, 1])
        lens = mask.sum(1)
        out = self.model(x, mask)
        with torch.no_grad():
            torch.testing.assert_close(out.logits, self.legacy(x, lens))
            torch.testing.assert_close(out.attention,
                                       self.legacy.calculate_attention(x, lens).squeeze(-1))
            torch.testing.assert_close(out.embedding, self.legacy.get_last_layer_activations(x, lens))
        self.assertTrue(torch.all(out.attention[~mask] == 0))

    def test_padded_batch_equals_single_bags(self):
        x, mask = _padded([6, 3, 1])
        out = self.model(x, mask)
        for i, n in enumerate([6, 3, 1]):
            torch.testing.assert_close(out.logits[i:i + 1], self.model(x[i:i + 1, :n]).logits)

    def test_gate_and_temperature_match_legacy(self):
        legacy = Attention_MIL(8, 3, z_dim=16, attention_gate=0.5, temperature=2.0).eval()
        model = ABMIL.from_legacy(legacy).eval()
        x, mask = _padded([6, 4])
        with torch.no_grad():
            torch.testing.assert_close(model(x, mask).logits, legacy(x, mask.sum(1)))

    def test_new_model_loads_legacy_state_dict(self):
        fresh = ABMIL(8, 3, z_dim=16).eval()
        fresh.load_state_dict(self.legacy.state_dict())
        x, mask = _padded([4])
        torch.testing.assert_close(fresh(x, mask).logits, self.model(x, mask).logits)

    def test_slide_aggregator_and_run_inference(self):
        x, mask = _padded([5])
        agg = SlideAggregator(self.model)
        logits, att = agg.predict(x, attention=True)
        self.assertEqual(tuple(att.shape), (1, 5, 1))
        new = run_inference(self.model, x, attention=True)
        old = run_inference(self.legacy, x, attention=True)
        torch.testing.assert_close(new[0], old[0])
        torch.testing.assert_close(new[1], old[1])
        mean, std = self.model.predict_uq(x)
        self.assertEqual(tuple(mean.shape), (1, 3))
        self.assertEqual(tuple(std.shape), (1, 3))


@unittest.skipUnless(HAS_NYSTROM, 'nystrom_attention is not installed')
class TransMILTest(unittest.TestCase):

    def test_same_outputs_as_legacy_and_masking(self):
        from histox.mil.models import TransMIL as LegacyTransMIL
        from histox.models import TransMIL
        torch.manual_seed(0)
        legacy = LegacyTransMIL(8, 2).eval()
        model = TransMIL.from_legacy(legacy).eval()
        x, mask = _padded([7, 3])
        with torch.no_grad():
            torch.testing.assert_close(model(x[:1, :7]).logits, legacy(x[:1, :7]))
            out = model(x, mask)
            torch.testing.assert_close(out.logits[1:], legacy(x[1:, :3]))
        self.assertIsNone(out.attention)
        self.assertEqual(tuple(out.embedding.shape), (2, 512))


if __name__ == '__main__':
    unittest.main()
