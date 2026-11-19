"""Tests for the lightweight HistoX workflow contract."""

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import torch
import histox as hx

from histox.workflows import (
    DataSpec,
    DuplicateRegistrationError,
    Experiment,
    InvalidDataSpecError,
    Method,
    RunResult,
    Task,
    UnsupportedStageError,
    list_methods,
    list_tasks,
    register_method,
    register_task,
)


class RegistryTest(unittest.TestCase):

    def test_builtins_are_registered(self):
        self.assertIn("survival", list_tasks())
        self.assertIn("ps3", list_methods())

    def test_core_contract_is_available_from_top_level_package(self):
        self.assertIs(hx.Experiment, Experiment)
        self.assertIs(hx.DataSpec, DataSpec)
        self.assertIs(hx.UnsupportedStageError, UnsupportedStageError)

    def test_duplicate_registration_is_rejected(self):
        @register_task("workflow_test_duplicate")
        class FirstTask(Task):
            pass

        with self.assertRaises(DuplicateRegistrationError):
            @register_task("workflow_test_duplicate")
            class SecondTask(Task):
                pass


class ExperimentTest(unittest.TestCase):

    def _data_spec(self):
        return DataSpec(
            modalities=("histology", "report", "rna"),
            fields={
                "patient_id": "case_id",
                "duration": "dss_survival_days",
                "event": "dss_event",
            },
        )

    def test_ps3_builds_existing_model(self):
        experiment = Experiment(
            task="survival",
            method="ps3",
            data_spec=self._data_spec(),
            method_config={
                "n_genes": 7,
                "pathway_gene_indices": [[0, 1], [2, 3, 4], [5, 6]],
                "histology_dim": 9,
                "report_dim": 8,
                "n_histology_prototypes": 4,
                "n_report_prototypes": 3,
                "embedding_dim": 6,
                "modality_embedding_dim": 2,
                "feedforward_dim": 12,
                "dropout": 0.0,
            },
        )
        model = experiment.build_model()
        risk = experiment.method.forward_batch(
            model,
            {
                "histology": torch.randn(2, 4, 9),
                "rna": torch.randn(2, 7),
                "report": torch.randn(2, 4, 8),
                "report_mask": torch.ones(2, 4, dtype=torch.bool),
            },
        )
        self.assertEqual(tuple(risk.shape), (2, 1))

    def test_ps3_uses_cached_model_for_full_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = Experiment(
                task="survival",
                method="ps3",
                data_spec=self._data_spec(),
                output=directory,
                method_config={
                    "n_genes": 7,
                    "pathway_gene_indices": [[0, 1], [2, 3, 4], [5, 6]],
                    "histology_dim": 9,
                    "report_dim": 8,
                    "n_histology_prototypes": 4,
                    "n_report_prototypes": 3,
                    "embedding_dim": 6,
                    "modality_embedding_dim": 2,
                    "feedforward_dim": 12,
                    "dropout": 0.0,
                },
            )
            self.assertIs(experiment.build_model(), experiment.build_model())
            batch = {
                "patient_id": ["p0", "p1", "p2", "p3"],
                "histology": torch.randn(4, 4, 9),
                "rna": torch.randn(4, 7),
                "report": torch.randn(4, 4, 8),
                "report_mask": torch.ones(4, 4, dtype=torch.bool),
                "duration": torch.tensor([4.0, 3.0, 2.0, 1.0]),
                "event": torch.ones(4),
            }
            loader = [batch]
            fitted = experiment.fit(
                train_loader=loader,
                epochs=2,
                evaluation_loaders={"train": loader},
                device="cpu",
                scheduler=False,
            )
            self.assertIn("train_c_index", fitted.metrics)
            self.assertTrue((Path(directory) / "model_state.pt").is_file())
            self.assertTrue((Path(directory) / "history.csv").is_file())
            self.assertTrue(
                (Path(directory) / "train_predictions.csv").is_file()
            )

            evaluated = experiment.evaluate(loader=loader, device="cpu")
            predicted = experiment.predict(loader=loader, device="cpu")
            explained = experiment.explain(loader=loader, device="cpu")
            self.assertIn("c_index", evaluated.metrics)
            self.assertEqual(len(predicted.predictions["risk"]), 4)
            self.assertEqual(
                explained.artifacts["joint_attention"].kind,
                "joint_attention",
            )
            self.assertTrue(
                (Path(directory) / "joint_attention.npz").is_file()
            )

    def test_missing_modality_is_rejected_before_model_loading(self):
        data_spec = DataSpec(
            modalities=("histology", "rna"),
            fields={"duration": "time", "event": "event"},
        )
        with self.assertRaisesRegex(InvalidDataSpecError, "report"):
            Experiment(
                task="survival",
                method="ps3",
                data_spec=data_spec,
                method_config={
                    "n_genes": 2,
                    "pathway_gene_indices": [[0, 1]],
                    "histology_dim": 3,
                },
            )

    def test_unimplemented_stage_is_explicit(self):
        class BuildOnlyMethod(Method):
            name = "workflow_test_build_only"
            supported_tasks = ("survival",)

        experiment = Experiment(
            task="survival",
            method=BuildOnlyMethod(),
            data_spec=self._data_spec(),
        )
        with self.assertRaisesRegex(UnsupportedStageError, "fit"):
            experiment.fit()

    def test_custom_task_method_and_heatmap_artifact(self):
        @register_task("workflow_test_retrieval")
        class RetrievalTask(Task):
            name = "workflow_test_retrieval"
            required_fields = ("query_id",)
            default_metrics = ("recall_at_5",)

        @register_method("workflow_test_retriever")
        class Retriever(Method):
            name = "workflow_test_retriever"
            supported_tasks = ("workflow_test_retrieval",)
            required_modalities = ("histology", "text")

            def stage_evaluate(self, experiment, **kwargs):
                result = RunResult(
                    stage="evaluate",
                    metrics={"recall_at_5": 0.75},
                )
                result.add_artifact(
                    "query_map",
                    kind="concept_heatmap",
                    path="outputs/query_map.png",
                )
                return result

        experiment = Experiment(
            task="workflow_test_retrieval",
            method="workflow_test_retriever",
            data_spec=DataSpec(
                modalities=("histology", "text"),
                fields={"query_id": "query_id"},
            ),
        )
        result = experiment.evaluate()
        self.assertEqual(result.metrics["recall_at_5"], 0.75)
        self.assertEqual(
            result.artifacts["query_map"].kind, "concept_heatmap"
        )

    def test_ps3_pretrained_bundle_round_trip(self):
        from histox.mil import (
            PS3HistologyPrototypeEncoder,
            PS3PathwayPreprocessor,
        )

        expression = pd.DataFrame(
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 4.0, 6.0, 8.0],
                [4.0, 5.0, 7.0, 9.0],
            ],
            index=["p0", "p1", "p2"],
            columns=["G0", "G1", "G2", "G3"],
        )
        signatures = pd.DataFrame({
            "PATH_A": ["G0", "G1"],
            "PATH_B": ["G2", "G3"],
        })
        pathway = PS3PathwayPreprocessor().fit(expression, signatures)
        experiment = Experiment(
            task="survival",
            method="ps3",
            data_spec=self._data_spec(),
            method_config={
                "n_genes": 4,
                "pathway_gene_indices": pathway.pathway_gene_indices,
                "histology_dim": 5,
                "report_dim": 6,
                "n_histology_prototypes": 2,
                "n_report_prototypes": 2,
                "embedding_dim": 4,
                "modality_embedding_dim": 2,
                "feedforward_dim": 8,
                "dropout": 0.0,
            },
        )
        rna = pathway.transform(expression)
        batch = {
            "patient_id": list(expression.index),
            "histology": torch.randn(3, 2, 5),
            "rna": rna,
            "report": torch.randn(3, 3, 6),
            "report_mask": torch.ones(3, 3, dtype=torch.bool),
            "duration": torch.tensor([3.0, 2.0, 1.0]),
            "event": torch.ones(3),
        }
        experiment.build_model().eval()
        expected = experiment.predict(loader=[batch], device="cpu")
        histology_encoder = PS3HistologyPrototypeEncoder(
            n_prototypes=2,
            em_iterations=1,
        ).load_centers(torch.tensor([[-1.0, -1.0], [1.0, 1.0]]))
        with tempfile.TemporaryDirectory() as directory:
            bundle = hx.PS3PretrainedBundle.export(
                experiment=experiment,
                destination=Path(directory) / "bundle",
                pathway_preprocessor=pathway,
                histology_encoder=histology_encoder,
                report_config={"pad_to": 3},
                metadata={"seed": 1, "fold": 0},
            )
            observed = bundle.predict(loader=[batch], device="cpu")
            torch.testing.assert_close(
                torch.as_tensor(observed.predictions["risk"]),
                torch.as_tensor(expected.predictions["risk"]),
                rtol=0,
                atol=0,
            )
            self.assertTrue(
                (Path(directory) / "bundle" / "model.safetensors").is_file()
            )
            self.assertEqual(
                bundle.config["metadata"],
                {"seed": 1, "fold": 0},
            )
            spatial = bundle.spatial_attention(
                patch_features=torch.tensor([
                    [-1.2, -0.8],
                    [1.1, 0.9],
                    [-0.9, -1.1],
                ]),
                coordinates=torch.tensor([
                    [128, 128, 0, 0],
                    [384, 128, 1, 0],
                    [128, 384, 0, 1],
                ]),
                gene_expression=expression.iloc[[0]],
                report_embeddings=torch.randn(3, 6),
                source="pathway",
                source_index="PATH_A",
            )
            self.assertEqual(tuple(spatial["prototype_scores"].shape), (2,))
            self.assertEqual(tuple(spatial["patch_scores"].shape), (3,))
            self.assertEqual(tuple(spatial["responsibilities"].shape), (3, 2))
            self.assertEqual(tuple(spatial["coordinates"].shape), (3, 4))
            self.assertEqual(tuple(spatial["attention_logits"].shape), (1, 6, 6))
            self.assertTrue(
                torch.all(spatial["normalized_patch_scores"] >= 0)
            )
            self.assertTrue(
                torch.all(spatial["normalized_patch_scores"] <= 1)
            )
            explanation = bundle.prototype_explanation(
                patch_features=torch.tensor([
                    [-1.2, -0.8],
                    [1.1, 0.9],
                    [-0.9, -1.1],
                ]),
                coordinates=torch.tensor([
                    [128, 128, 0, 0],
                    [384, 128, 1, 0],
                    [128, 384, 0, 1],
                ]),
                gene_expression=expression.iloc[[0]],
                report_embeddings=torch.randn(3, 6),
                representative_patches=2,
            )
            self.assertEqual(
                tuple(explanation["prototype_assignments"].shape), (3,)
            )
            self.assertEqual(
                tuple(explanation["prototype_proportions"].shape), (2,)
            )
            torch.testing.assert_close(
                explanation["prototype_proportions"].sum(),
                torch.tensor(1.0),
            )
            self.assertEqual(
                tuple(explanation["representative_patch_indices"].shape),
                (2, 2),
            )
            self.assertEqual(
                tuple(explanation["pathway_to_histology"].shape), (2, 2)
            )
            self.assertEqual(
                tuple(explanation["pathway_to_histology_logits"].shape),
                (2, 2),
            )
            self.assertEqual(
                tuple(explanation["histology_to_pathway"].shape), (2, 2)
            )
            self.assertEqual(
                tuple(explanation["report_to_histology"].shape), (2, 2)
            )
            self.assertEqual(
                tuple(explanation["report_to_pathway"].shape), (2, 2)
            )
            torch.testing.assert_close(
                explanation["pathway_to_histology_normalized"].sum(dim=1),
                torch.ones(2),
            )
            torch.testing.assert_close(
                explanation["histology_to_pathway_normalized"].sum(dim=1),
                torch.ones(2),
            )
            self.assertEqual(
                tuple(explanation["report_to_histology_std"].shape), (2,)
            )
            self.assertEqual(
                tuple(explanation["report_to_pathway_std"].shape), (2,)
            )


class RunResultTest(unittest.TestCase):

    def test_result_json_records_typed_artifacts(self):
        result = RunResult(stage="evaluate", metrics={"c_index": 0.81})
        result.add_artifact(
            "risk_map",
            kind="heatmap",
            path="outputs/risk_map.png",
            metadata={"slide": "patient.svs"},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = result.save_json(Path(directory) / "result.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["metrics"]["c_index"], 0.81)
        self.assertEqual(payload["artifacts"]["risk_map"]["kind"], "heatmap")


if __name__ == "__main__":
    unittest.main()
