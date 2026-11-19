import importlib.util
from pathlib import Path
import types
import unittest

import numpy as np
import pandas as pd
import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "mil" / "ps3.py"
SPEC = importlib.util.spec_from_file_location("histox_ps3_inputs_under_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Unable to load {}".format(MODULE_PATH))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

PS3HistologyPrototypeEncoder = MODULE.PS3HistologyPrototypeEncoder
PS3PathwayPreprocessor = MODULE.PS3PathwayPreprocessor
PS3ReportEncoder = MODULE.PS3ReportEncoder


class _Tokenizer:

    def __call__(self, text, **kwargs):
        del kwargs
        count = len(text.split())
        return {"input_ids": torch.arange(count).reshape(1, count)}

    def decode(self, tokens, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(int(value)) for value in tokens)


class _Processor:

    def __init__(self):
        self.tokenizer = _Tokenizer()

    def __call__(self, text, **kwargs):
        del kwargs
        return {"input_ids": torch.ones(len(text), 3, dtype=torch.long)}


class _TextModel(torch.nn.Module):

    def forward(self, input_ids):
        return types.SimpleNamespace(
            text_embeds=torch.ones(input_ids.shape[0], 512)
        )


class ReportEncoderTest(unittest.TestCase):

    def test_segment_text_does_not_require_loading_the_text_model(self):
        segments = PS3ReportEncoder.segment_text(
            "ten tokens",
            tokenizer=_Tokenizer(),
            max_tokens=1,
        )
        self.assertEqual(segments, ["0", "1"])

    def test_report_is_segmented_and_encoded(self):
        encoder = PS3ReportEncoder(
            processor=_Processor(),
            model=_TextModel(),
            device="cpu",
        )
        report = " ".join("token{}".format(index) for index in range(160))
        segments = encoder.segment(report)
        embeddings = encoder.encode(report)
        self.assertEqual(len(segments), 3)
        self.assertEqual(tuple(embeddings.shape), (3, 512))
        self.assertEqual(embeddings.dtype, torch.float32)

    def test_blank_report_is_rejected(self):
        encoder = PS3ReportEncoder(
            processor=_Processor(), model=_TextModel(), device="cpu"
        )
        with self.assertRaisesRegex(ValueError, "non-empty"):
            encoder.encode("  ")

    def test_collate_pads_reports_and_returns_mask(self):
        padded, mask = PS3ReportEncoder.collate(
            [torch.ones(2, 4), torch.full((3, 4), 2.0)],
            pad_to=5,
        )
        self.assertEqual(tuple(padded.shape), (2, 5, 4))
        self.assertEqual(tuple(mask.shape), (2, 5))
        self.assertEqual(mask.sum(dim=1).tolist(), [2, 3])
        self.assertTrue(torch.all(padded[0, 2:] == 0))

    def test_collate_rejects_truncation(self):
        with self.assertRaisesRegex(ValueError, "shorter than the longest"):
            PS3ReportEncoder.collate([torch.ones(3, 4)], pad_to=2)


class PathwayPreprocessorTest(unittest.TestCase):

    def setUp(self):
        self.train = pd.DataFrame(
            [[1.0, 10.0, 5.0], [3.0, 14.0, 9.0]],
            index=["case-a", "case-b"],
            columns=["G1", "G2", "G3"],
        )
        self.signatures = pd.DataFrame({
            "PATH_A": ["G2", "G1", None],
            "PATH_B": ["G3", "NOT_ME", None],
        })

    def test_train_only_scaling_and_membership(self):
        preprocessor = PS3PathwayPreprocessor().fit(
            self.train, self.signatures
        )
        train_tensor = preprocessor.transform(self.train)
        test_tensor = preprocessor.transform(pd.DataFrame(
            [[5.0, 18.0, 13.0]],
            index=["case-c"],
            columns=["G1", "G2", "G3"],
        ))
        torch.testing.assert_close(
            train_tensor.mean(dim=0), torch.zeros(3), atol=1e-6, rtol=0
        )
        torch.testing.assert_close(
            test_tensor, torch.tensor([[3.0, 3.0, 3.0]])
        )
        self.assertEqual(preprocessor.pathway_names, ["PATH_A", "PATH_B"])
        self.assertEqual(preprocessor.pathway_gene_indices, [[0, 1], [2]])

    def test_test_frame_must_contain_fitted_genes(self):
        preprocessor = PS3PathwayPreprocessor().fit(
            self.train, self.signatures
        )
        with self.assertRaisesRegex(ValueError, "missing 1 fitted genes"):
            preprocessor.transform(self.train.drop(columns=["G2"]))

    def test_round_trip_preserves_scaling_and_membership(self):
        fitted = PS3PathwayPreprocessor().fit(self.train, self.signatures)
        restored = PS3PathwayPreprocessor.from_dict(fitted.to_dict())
        test = pd.DataFrame(
            [[5.0, 18.0, 13.0]],
            index=["case-c"],
            columns=["G1", "G2", "G3"],
        )
        torch.testing.assert_close(
            restored.transform(test),
            fitted.transform(test),
            rtol=0,
            atol=0,
        )
        self.assertEqual(restored.gene_names, fitted.gene_names)
        self.assertEqual(restored.pathway_names, fitted.pathway_names)
        self.assertEqual(
            restored.pathway_gene_indices,
            fitted.pathway_gene_indices,
        )


class HistologyPrototypeEncoderTest(unittest.TestCase):

    def test_panther_allcat_is_tokenized_before_reshaping(self):
        # Two prototypes, each with a two-dimensional mean and covariance.
        flat = torch.tensor([
            0.25, 0.75,
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
        ])
        tokens = PS3HistologyPrototypeEncoder.unpack_panther_allcat(flat, 2)
        expected = torch.tensor([
            [0.25, 1.0, 2.0, 5.0, 6.0],
            [0.75, 3.0, 4.0, 7.0, 8.0],
        ])
        torch.testing.assert_close(tokens, expected)
        self.assertFalse(torch.equal(tokens, flat.reshape(2, 5)))

    def test_panther_crc_reproduction_layout_preserves_direct_reshape(self):
        flat = torch.arange(10, dtype=torch.float32)
        tokens = PS3HistologyPrototypeEncoder.unpack_panther_allcat(
            flat,
            2,
            layout="crc_reproduction",
        )
        torch.testing.assert_close(tokens, flat.reshape(2, 5))

    def test_panther_unknown_layout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "layout must be"):
            PS3HistologyPrototypeEncoder.unpack_panther_allcat(
                torch.arange(10, dtype=torch.float32),
                2,
                layout="unknown",
            )

    def test_fit_and_transform_return_panther_contract(self):
        torch.manual_seed(4)
        bags = [
            torch.randn(20, 4) - 2,
            torch.randn(25, 4) + 2,
            torch.randn(18, 4),
        ]
        encoder = PS3HistologyPrototypeEncoder(
            n_prototypes=3,
            patches_per_prototype=12,
            max_iter=20,
            n_init=2,
            random_state=7,
        ).fit(bags)
        output = encoder.transform(bags[0])
        self.assertEqual(tuple(encoder.centers.shape), (3, 4))
        self.assertEqual(tuple(output.shape), (3, 9))
        torch.testing.assert_close(output[:, 0].sum(), torch.tensor(1.0))
        self.assertTrue(torch.all(output[:, 5:] > 0))
        self.assertTrue(torch.isfinite(output).all())

    def test_transform_with_assignments_aligns_patches_and_prototypes(self):
        centers = torch.tensor([[-2.0, -2.0], [2.0, 2.0]])
        patches = torch.tensor([
            [-2.1, -1.9],
            [2.2, 1.8],
            [-1.8, -2.2],
        ])
        encoder = PS3HistologyPrototypeEncoder(
            n_prototypes=2,
            em_iterations=1,
        ).load_centers(centers)
        prototypes, responsibilities = encoder.transform_with_assignments(
            patches
        )
        self.assertEqual(tuple(prototypes.shape), (2, 5))
        self.assertEqual(tuple(responsibilities.shape), (3, 2))
        torch.testing.assert_close(
            responsibilities.sum(dim=1), torch.ones(3), atol=1e-6, rtol=0
        )
        self.assertEqual(responsibilities.argmax(dim=1).tolist(), [0, 1, 0])

    def test_masked_feature_grid_is_flattened(self):
        data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        mask = np.zeros_like(data, dtype=bool)
        mask[0, 1] = True
        grid = np.ma.masked_array(data, mask=mask)
        bag = PS3HistologyPrototypeEncoder.flatten_feature_grid(grid)
        self.assertEqual(tuple(bag.shape), (5, 4))
        self.assertFalse(torch.equal(bag[1], torch.from_numpy(data[0, 1])))

    def test_masked_feature_grid_coordinates_follow_flattened_rows(self):
        data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        mask = np.zeros_like(data, dtype=bool)
        mask[0, 1] = True
        grid = np.ma.masked_array(data, mask=mask)
        bag, coordinates = (
            PS3HistologyPrototypeEncoder.flatten_feature_grid_with_coordinates(
                grid
            )
        )
        self.assertEqual(coordinates.tolist(), [[0, 0], [2, 0], [0, 1], [1, 1], [2, 1]])
        torch.testing.assert_close(bag[1], torch.from_numpy(data[0, 2]))

    def test_unfitted_transform_is_rejected(self):
        encoder = PS3HistologyPrototypeEncoder(n_prototypes=2)
        with self.assertRaisesRegex(RuntimeError, "fitted first"):
            encoder.transform(torch.randn(4, 3))


if __name__ == "__main__":
    unittest.main()
