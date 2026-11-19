"""Tests for the independent PS3 prototype-fusion implementation."""

import importlib.util
from pathlib import Path
import unittest

import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "mil" / "models" / "ps3.py"
SPEC = importlib.util.spec_from_file_location("histox_ps3_under_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Unable to load {}".format(MODULE_PATH))
PS3_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PS3_MODULE)

CoxPHLoss = PS3_MODULE.CoxPHLoss
DiagnosticPrototypeSelector = PS3_MODULE.DiagnosticPrototypeSelector
PS3PrototypeFusion = PS3_MODULE.PS3PrototypeFusion
PS3SurvivalModel = PS3_MODULE.PS3SurvivalModel
PathwayPrototypeEncoder = PS3_MODULE.PathwayPrototypeEncoder


class DiagnosticPrototypeSelectorTest(unittest.TestCase):

    def test_selector_respects_padding_and_returns_top_k(self):
        torch.manual_seed(11)
        selector = DiagnosticPrototypeSelector(8, n_prototypes=3, output_dim=6)
        reports = torch.randn(2, 7, 8)
        mask = torch.tensor(
            [[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0, 0]],
            dtype=torch.bool,
        )
        prototypes, details = selector(reports, mask, return_attention=True)
        self.assertEqual(tuple(prototypes.shape), (2, 3, 6))
        self.assertEqual(tuple(details["attention"].shape), (2, 7, 7))
        self.assertTrue(torch.all(details["selected_indices"][0] < 5))
        self.assertTrue(torch.all(details["selected_indices"][1] < 4))
        self.assertTrue(torch.isneginf(details["importance"][0, 5:]).all())

    def test_selector_interpolates_short_reports_without_selecting_padding(self):
        torch.manual_seed(12)
        selector = DiagnosticPrototypeSelector(4, n_prototypes=3)
        reports = torch.randn(2, 4, 4)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
        prototypes, details = selector(reports, mask, return_attention=True)
        self.assertEqual(tuple(prototypes.shape), (2, 3, 4))
        self.assertTrue(details["interpolated"].tolist() == [True, False])
        self.assertTrue(torch.all(details["selected_indices"][0] == -1))
        self.assertTrue(torch.all(details["selected_indices"][1] < 3))
        self.assertTrue(torch.isfinite(prototypes).all())

    def test_selector_rejects_report_without_valid_segments(self):
        selector = DiagnosticPrototypeSelector(4, n_prototypes=3)
        with self.assertRaisesRegex(ValueError, "at least one valid segment"):
            selector(torch.randn(1, 4, 4), torch.zeros(1, 4, dtype=torch.bool))

    def test_crc_reproduction_mode_can_select_padded_query_outputs(self):
        torch.manual_seed(13)
        selector = DiagnosticPrototypeSelector(
            4,
            n_prototypes=3,
            mode="crc_reproduction",
        )
        reports = torch.randn(1, 5, 4)
        mask = torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.bool)
        prototypes, details = selector(reports, mask, return_attention=True)
        self.assertEqual(tuple(prototypes.shape), (1, 3, 4))
        self.assertTrue(torch.any(details["selected_indices"] >= 2))
        self.assertFalse(details["interpolated"].any())
        self.assertTrue(torch.isfinite(prototypes).all())

    def test_unknown_selector_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "mode must be"):
            DiagnosticPrototypeSelector(4, n_prototypes=3, mode="unknown")

    def test_traceable_residual_keeps_real_source_segments(self):
        torch.manual_seed(131)
        selector = DiagnosticPrototypeSelector(
            4,
            n_prototypes=3,
            mode="traceable",
            representation_mode="residual",
        )
        with torch.no_grad():
            selector.query.weight.zero_()
            selector.query.bias.zero_()
            selector.key.weight.zero_()
            selector.key.bias.zero_()
            selector.value.weight.zero_()
            selector.value.bias.zero_()
            selector.output.weight.copy_(torch.eye(4))
            selector.output.bias.zero_()
        reports = torch.randn(1, 5, 4)
        mask = torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.bool)
        prototypes, details = selector(reports, mask, return_attention=True)
        indices = details["selected_indices"]
        self.assertTrue(torch.all(indices >= 0))
        self.assertTrue(torch.all(indices < 2))
        expected = torch.gather(
            reports,
            1,
            indices.unsqueeze(-1).expand(-1, -1, reports.shape[-1]),
        )
        torch.testing.assert_close(prototypes, expected)

    def test_residual_scale_controls_source_segment_contribution(self):
        selector = DiagnosticPrototypeSelector(
            4,
            n_prototypes=2,
            mode="traceable",
            representation_mode="residual",
            residual_scale=0.25,
        )
        with torch.no_grad():
            selector.query.weight.zero_()
            selector.query.bias.zero_()
            selector.key.weight.zero_()
            selector.key.bias.zero_()
            selector.value.weight.zero_()
            selector.value.bias.zero_()
            selector.output.weight.copy_(torch.eye(4))
            selector.output.bias.zero_()
        reports = torch.randn(1, 3, 4)
        mask = torch.ones(1, 3, dtype=torch.bool)
        prototypes, details = selector(reports, mask, return_attention=True)
        expected = torch.gather(
            reports,
            1,
            details["selected_indices"].unsqueeze(-1).expand(-1, -1, 4),
        )
        torch.testing.assert_close(prototypes, 0.25 * expected)

    def test_nonpositive_residual_scale_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "residual_scale"):
            DiagnosticPrototypeSelector(4, n_prototypes=3, residual_scale=0)

    def test_unknown_representation_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "representation_mode"):
            DiagnosticPrototypeSelector(
                4,
                n_prototypes=3,
                representation_mode="unknown",
            )


class PathwayPrototypeEncoderTest(unittest.TestCase):

    def test_variable_gene_sets_produce_fixed_width_tokens(self):
        torch.manual_seed(12)
        encoder = PathwayPrototypeEncoder(
            n_genes=7,
            pathway_gene_indices=[[0, 2], [1, 3, 4], [5, 6]],
            output_dim=5,
            hidden_dim=9,
        )
        output = encoder(torch.randn(4, 7))
        self.assertEqual(tuple(output.shape), (4, 3, 5))
        self.assertEqual(encoder.n_pathways, 3)

    def test_duplicate_gene_index_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            PathwayPrototypeEncoder(4, [[0, 0, 1]])


class PS3PrototypeFusionTest(unittest.TestCase):

    def _model(self, **kwargs):
        return PS3PrototypeFusion(
            pathway_dim=5,
            histology_dim=7,
            report_dim=6,
            n_pathway_prototypes=3,
            n_histology_prototypes=4,
            n_report_prototypes=2,
            embedding_dim=8,
            modality_embedding_dim=4,
            feedforward_dim=16,
            dropout=0.0,
            **kwargs,
        )

    def test_default_configuration_preserves_legacy_parameter_shapes(self):
        model = self._model()
        self.assertEqual(model.prototype_embedding_mode, "modality")
        self.assertTrue(model.fusion_residual)
        self.assertEqual(
            [tuple(embedding.shape) for embedding in model.modality_embeddings],
            [(1, 1, 4), (1, 1, 4), (1, 1, 4)],
        )
        clone = self._model()
        clone.load_state_dict(model.state_dict(), strict=True)

    def test_prototype_embedding_mode_assigns_each_position_an_embedding(self):
        torch.manual_seed(1301)
        model = self._model(prototype_embedding_mode="prototype")
        self.assertEqual(
            [tuple(embedding.shape) for embedding in model.modality_embeddings],
            [(1, 3, 4), (1, 4, 4), (1, 2, 4)],
        )
        prepared, valid = model._prepare_modality(
            torch.zeros(1, 3, 5), 0, None, "pathway"
        )
        self.assertTrue(valid.all())
        self.assertFalse(
            torch.allclose(prepared[0, 0, -4:], prepared[0, 1, -4:])
        )

    def test_unknown_prototype_embedding_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "prototype_embedding_mode"):
            self._model(prototype_embedding_mode="unknown")

    def test_attention_residual_can_be_disabled(self):
        model = self._model(fusion_residual=False).eval()
        self.assertFalse(model.fusion_residual)
        self.assertTrue(
            all(not layer.attention_residual for layer in model.fusion_layers)
        )
        risk = model(
            torch.randn(2, 3, 5),
            torch.randn(2, 4, 7),
            torch.randn(2, 2, 6),
        )
        self.assertTrue(torch.isfinite(risk).all())

    def test_forward_returns_risk_and_joint_attention(self):
        torch.manual_seed(13)
        model = self._model().eval()
        risk, attention = model(
            torch.randn(2, 3, 5),
            torch.randn(2, 4, 7),
            torch.randn(2, 2, 6),
            return_attention=True,
        )
        self.assertEqual(tuple(risk.shape), (2, 1))
        self.assertEqual(tuple(attention.shape), (2, 9, 9))
        torch.testing.assert_close(
            attention.sum(dim=-1),
            torch.ones(2, 9),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_forward_can_return_pre_softmax_attention_logits(self):
        torch.manual_seed(130)
        model = self._model().eval()
        risk, attention, logits = model(
            torch.randn(2, 3, 5),
            torch.randn(2, 4, 7),
            torch.randn(2, 2, 6),
            return_attention=True,
            return_attention_logits=True,
        )
        self.assertEqual(tuple(risk.shape), (2, 1))
        self.assertEqual(tuple(logits.shape), (2, 9, 9))
        torch.testing.assert_close(
            torch.softmax(logits, dim=-1), attention
        )

    def test_masked_histology_prototype_receives_no_attention(self):
        torch.manual_seed(14)
        model = self._model().eval()
        _, attention = model(
            torch.randn(1, 3, 5),
            torch.randn(1, 4, 7),
            torch.randn(1, 2, 6),
            histology_mask=torch.tensor([[1, 1, 1, 0]], dtype=torch.bool),
            return_attention=True,
        )
        masked_key = 3 + 3
        torch.testing.assert_close(
            attention[..., masked_key],
            torch.zeros_like(attention[..., masked_key]),
        )

    def test_wrong_prototype_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Expected 4 histology"):
            self._model()(
                torch.randn(1, 3, 5),
                torch.randn(1, 3, 7),
                torch.randn(1, 2, 6),
            )


class PS3SurvivalModelTest(unittest.TestCase):

    def test_architecture_options_are_forwarded_to_fusion(self):
        model = PS3SurvivalModel(
            n_genes=4,
            pathway_gene_indices=[[0, 1], [2, 3]],
            histology_dim=5,
            report_dim=6,
            n_histology_prototypes=2,
            n_report_prototypes=3,
            embedding_dim=4,
            modality_embedding_dim=2,
            feedforward_dim=8,
            dropout=0.0,
            prototype_embedding_mode="prototype",
            fusion_residual=False,
            report_representation_mode="residual",
            report_residual_scale=0.25,
        )
        self.assertEqual(model.fusion.prototype_embedding_mode, "prototype")
        self.assertFalse(model.fusion.fusion_residual)
        self.assertEqual(model.report_selector.representation_mode, "residual")
        self.assertEqual(model.report_selector.residual_scale, 0.25)

    def test_end_to_end_tensor_contract_and_gradient(self):
        torch.manual_seed(15)
        model = PS3SurvivalModel(
            n_genes=7,
            pathway_gene_indices=[[0, 1], [2, 3, 4], [5, 6]],
            histology_dim=9,
            report_dim=8,
            n_histology_prototypes=4,
            n_report_prototypes=3,
            embedding_dim=6,
            modality_embedding_dim=2,
            feedforward_dim=12,
            dropout=0.0,
        )
        risk, attention = model(
            histology_prototypes=torch.randn(3, 4, 9),
            gene_expression=torch.randn(3, 7),
            report_embeddings=torch.randn(3, 6, 8),
            report_mask=torch.tensor(
                [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]],
                dtype=torch.bool,
            ),
            return_attention=True,
        )
        self.assertEqual(tuple(risk.shape), (3, 1))
        self.assertEqual(tuple(attention.shape), (3, 10, 10))
        loss = CoxPHLoss()(
            risk,
            duration=torch.tensor([8.0, 5.0, 5.0]),
            event=torch.tensor([1, 1, 0]),
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.fusion.risk_head.weight.grad)

    def test_report_evidence_maps_attention_to_real_source_segments(self):
        torch.manual_seed(151)
        model = PS3SurvivalModel(
            n_genes=7,
            pathway_gene_indices=[[0, 1], [2, 3, 4], [5, 6]],
            histology_dim=9,
            report_dim=8,
            n_histology_prototypes=4,
            n_report_prototypes=3,
            embedding_dim=6,
            modality_embedding_dim=2,
            feedforward_dim=12,
            dropout=0.0,
            report_selection_mode="crc_reproduction",
        ).eval()
        mask = torch.tensor(
            [[1, 1, 0, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool
        )
        evidence = model.report_segment_evidence(
            histology_prototypes=torch.randn(2, 4, 9),
            gene_expression=torch.randn(2, 7),
            report_embeddings=torch.randn(2, 5, 8),
            report_mask=mask,
        )
        self.assertEqual(tuple(evidence["risk"].shape), (2, 1))
        self.assertEqual(
            tuple(evidence["report_prototype_scores"].shape), (2, 3)
        )
        self.assertEqual(
            tuple(evidence["histology_to_report_attention"].shape),
            (2, 4, 3),
        )
        selected = evidence["selected_source_segment"]
        self.assertTrue(torch.all(selected >= 0))
        self.assertTrue(torch.all(selected < mask.sum(dim=1)))
        torch.testing.assert_close(
            evidence["report_prototype_scores"].sum(dim=1),
            torch.ones(2),
        )

    def test_report_evidence_rejects_untraceable_interpolation(self):
        torch.manual_seed(152)
        model = PS3SurvivalModel(
            n_genes=4,
            pathway_gene_indices=[[0, 1], [2, 3]],
            histology_dim=5,
            report_dim=6,
            n_histology_prototypes=2,
            n_report_prototypes=3,
            embedding_dim=4,
            modality_embedding_dim=2,
            feedforward_dim=8,
            dropout=0.0,
            report_selection_mode="paper",
        ).eval()
        with self.assertRaisesRegex(ValueError, "No report prototype maps"):
            model.report_segment_evidence(
                histology_prototypes=torch.randn(1, 2, 5),
                gene_expression=torch.randn(1, 4),
                report_embeddings=torch.randn(1, 4, 6),
                report_mask=torch.tensor([[1, 1, 0, 0]], dtype=torch.bool),
            )


class CoxPHLossTest(unittest.TestCase):

    def test_breslow_ties_match_manual_result(self):
        risk = torch.tensor([0.2, -0.1, 0.4], requires_grad=True)
        duration = torch.tensor([5.0, 5.0, 8.0])
        event = torch.tensor([1, 1, 1])
        loss = CoxPHLoss()(risk, duration, event)
        expected = -(
            risk[2] - torch.logsumexp(risk[2:], dim=0)
            + risk[:2].sum() - 2 * torch.logsumexp(risk, dim=0)
        ) / 3
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(risk.grad).all())

    def test_batch_without_event_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one observed event"):
            CoxPHLoss()(torch.zeros(2), torch.ones(2), torch.zeros(2))


if __name__ == "__main__":
    unittest.main()
