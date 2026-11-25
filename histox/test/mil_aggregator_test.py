"""Tests for histox.mil.SlideAggregator and MIL inference built on it."""

import importlib.util
import unittest

import torch

from histox.mil import SlideAggregator, uses_lens
from histox.mil.eval import predict_from_bags, run_inference
from histox.mil.models import Attention_MIL, MultiModal_Attention_MIL


class _CLAMLike(torch.nn.Module):
    """Mimics CLAM: forward(h) -> (logits, extra); activations -> (h, A)."""

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 2)

    def forward(self, h):
        return self.fc(h).mean(1), 'extra'

    def get_last_layer_activations(self, h):
        return h.mean(1), torch.ones(1, h.shape[1])


class _NoLens(torch.nn.Module):
    """TransMIL's convention: forward(h), calculate_attention(h), activations(h)."""

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 3)
        self.att = torch.nn.Linear(8, 1)

    def forward(self, h):
        return self.fc(self.get_last_layer_activations(h))

    def calculate_attention(self, h):
        return torch.softmax(self.att(h), dim=1)

    def get_last_layer_activations(self, h):
        return (self.calculate_attention(h) * h).sum(1)


class _LensBySignature(torch.nn.Module):
    def forward(self, bags, lens):
        return bags.sum(1) * 0 + lens.float().unsqueeze(-1)


class AggregatorTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.bags = torch.randn(2, 7, 8)
        self.att_mil = Attention_MIL(n_feats=8, n_out=3).eval()
        self.transmil = _NoLens().eval()

    def test_uses_lens(self):
        self.assertTrue(uses_lens(self.att_mil))          # use_lens attribute
        self.assertFalse(uses_lens(self.transmil))        # forward(h)
        self.assertTrue(uses_lens(_LensBySignature()))    # parameter named 'lens'
        self.assertFalse(uses_lens(_CLAMLike()))

    def test_args(self):
        args = SlideAggregator(self.att_mil).args(self.bags)
        self.assertEqual(len(args), 2)
        self.assertEqual(args[1].tolist(), [7, 7])
        self.assertEqual(len(SlideAggregator(self.transmil).args(self.bags)), 1)
        custom = torch.tensor([3, 5])
        self.assertIs(SlideAggregator(self.att_mil).args(self.bags, custom)[1], custom)

    def test_matches_direct_calls(self):
        lens = torch.tensor([7, 7])
        with torch.no_grad():
            agg = SlideAggregator(self.att_mil)
            torch.testing.assert_close(agg.predict(self.bags), self.att_mil(self.bags, lens))
            y, a = agg.predict(self.bags, attention=True)
            y0, a0 = self.att_mil(self.bags, lens, return_attention=True)
            torch.testing.assert_close((y, a), (y0, a0))
            torch.testing.assert_close(agg.attention(self.bags),
                                       self.att_mil.calculate_attention(self.bags, lens))
            torch.testing.assert_close(agg.activations(self.bags)[0],
                                       self.att_mil.get_last_layer_activations(self.bags, lens))
            tagg = SlideAggregator(self.transmil)
            torch.testing.assert_close(tagg.predict(self.bags), self.transmil(self.bags))
            y, a = tagg.predict(self.bags, attention=True)
            torch.testing.assert_close(a, self.transmil.calculate_attention(self.bags))

    def test_uq(self):
        agg = SlideAggregator(self.att_mil)
        self.assertTrue(agg.supports_uq)
        torch.manual_seed(1)
        y, unc = agg.predict(self.bags, uq=True)
        self.assertEqual(tuple(unc.shape), (2, 3))
        y, att, unc = agg.predict(self.bags, attention=True, uq=True)
        self.assertEqual(att.shape[:2], (2, 7))
        with self.assertRaises(RuntimeError):
            SlideAggregator(self.transmil).predict(self.bags, uq=True)

    def test_clam_like_outputs(self):
        agg = SlideAggregator(_CLAMLike())
        self.assertEqual(tuple(agg.predict(self.bags).shape), (2, 2))   # tuple -> logits
        h, A = agg.activations(self.bags)
        self.assertEqual(tuple(A.shape), (1, 7))
        self.assertFalse(agg.supports_attention)
        with self.assertRaises(RuntimeError):
            agg.attention(self.bags)

    def test_multimodal(self):
        mm = MultiModal_Attention_MIL(n_feats=[8, 4], n_out=2).eval()
        agg = SlideAggregator(mm)
        self.assertTrue(agg.is_multimodal and agg.use_lens)
        bags = [torch.randn(1, 5, 8), torch.randn(1, 3, 4)]
        args = agg.args(bags)
        self.assertEqual([a[1].tolist() for a in args], [[5], [3]])
        with torch.no_grad():
            torch.testing.assert_close(agg.predict(bags), mm(*args))

    def test_run_inference_infers_lens(self):
        # Before, the default use_lens=False made Attention_MIL fail here.
        with torch.no_grad():
            y, att, unc = run_inference(self.att_mil, self.bags, attention=True)
            y0 = self.att_mil(self.bags, torch.tensor([7, 7]))
            torch.testing.assert_close(y, y0)
            # run_inference handles one bag at a time; attention is squeezed.
            _, att1, _ = run_inference(self.att_mil, self.bags[:1], attention=True)
            self.assertEqual(tuple(att1.shape), (7,))
            y, _, _ = run_inference(self.transmil, self.bags)
            torch.testing.assert_close(y, self.transmil(self.bags))
            # Explicit lens tensor
            y, _, _ = run_inference(self.att_mil, self.bags, use_lens=torch.tensor([7, 7]))
            torch.testing.assert_close(y, y0)

    def test_predict_from_bags_default(self):
        bags = [torch.randn(5, 8), torch.randn(9, 8)]
        yp, _ = predict_from_bags(self.att_mil, bags, device='cpu')
        yp_explicit, _ = predict_from_bags(self.att_mil, bags, use_lens=True, device='cpu')
        torch.testing.assert_close(torch.from_numpy(yp), torch.from_numpy(yp_explicit))
        self.assertEqual(yp.shape, (2, 3))


@unittest.skipUnless(importlib.util.find_spec('nystrom_attention'), 'nystrom_attention not installed')
class TransMILTest(unittest.TestCase):

    def test_transmil(self):
        from histox.mil.models import TransMIL
        torch.manual_seed(0)
        model = TransMIL(n_feats=8, n_out=3).eval()
        bags = torch.randn(1, 10, 8)
        agg = SlideAggregator(model)
        self.assertFalse(agg.use_lens)
        with torch.no_grad():
            torch.testing.assert_close(agg.predict(bags), model(bags))
            y, att = agg.predict(bags, attention=True)
            torch.testing.assert_close(att, model.calculate_attention(bags))


if __name__ == '__main__':
    unittest.main()
