"""Input preparation utilities for the HistoX PS3 implementation.

These utilities are independent implementations of the public PS3 data
contract. They do not contain source copied from the authors' code release.
"""

import math
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn


class PS3ReportEncoder:
    """Split machine-readable reports and encode every segment with PLIP.

    This class expects report text. PDF parsing, OCR, and clinical-document
    de-identification are intentionally outside its boundary.
    """

    def __init__(
        self,
        model_id: str = "vinid/plip",
        revision: Optional[str] = None,
        local_files_only: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        max_tokens: int = 77,
        processor: Optional[Any] = None,
        model: Optional[nn.Module] = None,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if (processor is None) != (model is None):
            raise ValueError("processor and model must be provided together")
        self.model_id = model_id
        self.revision = revision
        self.local_files_only = local_files_only
        self.max_tokens = max_tokens
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if processor is None:
            try:
                from transformers import CLIPProcessor, CLIPTextModelWithProjection
            except ImportError as exc:
                raise ImportError(
                    "PLIP report encoding requires Transformers. Install it "
                    "with `pip install 'histox[huggingface]'`."
                ) from exc
            try:
                processor = CLIPProcessor.from_pretrained(
                    model_id,
                    revision=revision,
                    local_files_only=local_files_only,
                )
                model = CLIPTextModelWithProjection.from_pretrained(
                    model_id,
                    revision=revision,
                    local_files_only=local_files_only,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                raise RuntimeError(
                    "Unable to load PLIP text weights from {!r}. The current "
                    "upstream repository may provide PyTorch .bin weights, "
                    "which recent Transformers releases refuse to load with "
                    "torch<2.6. Upgrade PyTorch to 2.6 or newer, or pass a "
                    "trusted local safetensors snapshot with model_id.".format(
                        model_id
                    )
                ) from exc
        self.processor = processor
        self.model = model.to(self.device)
        self.model.eval()

    @staticmethod
    def segment_text(
        report_text: str,
        tokenizer: Any,
        max_tokens: int = 77,
    ) -> List[str]:
        """Split one report with a supplied PLIP-compatible tokenizer.

        This lightweight entry point does not load the PLIP text model. It is
        useful when source text must be aligned to previously saved segment
        embeddings, for example when tracing a selected PS3 report prototype
        back to the original pathology report.
        """
        if not isinstance(report_text, str) or not report_text.strip():
            raise ValueError("report_text must be a non-empty string")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        tokens = tokenizer(
            report_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=False,
        )["input_ids"][0]
        if tokens.numel() == 0:
            raise ValueError("report_text produced no tokens")
        chunks = [
            tokens[start:start + max_tokens]
            for start in range(0, len(tokens), max_tokens)
        ]
        return [
            tokenizer.decode(chunk, skip_special_tokens=True)
            for chunk in chunks
        ]

    def segment(self, report_text: str) -> List[str]:
        """Return contiguous PLIP-token segments for one report."""
        return self.segment_text(
            report_text,
            tokenizer=self.processor.tokenizer,
            max_tokens=self.max_tokens,
        )

    def encode(self, report_text: str) -> Tensor:
        """Return a CPU float32 tensor shaped ``[segments, 512]``."""
        segments = self.segment(report_text)
        encoded = self.processor(
            text=segments,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
        )
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        with torch.inference_mode():
            output = self.model(**encoded)
        embeddings = getattr(output, "text_embeds", None)
        if embeddings is None:
            raise RuntimeError("PLIP did not return text_embeds")
        embeddings = embeddings.detach().cpu().float()
        if embeddings.ndim != 2 or embeddings.shape[0] != len(segments):
            raise RuntimeError(
                "Unexpected PLIP report embedding shape: {}".format(
                    tuple(embeddings.shape)
                )
            )
        if not torch.isfinite(embeddings).all():
            raise RuntimeError("PLIP produced non-finite report embeddings")
        return embeddings

    def encode_many(self, reports: Sequence[str]) -> List[Tensor]:
        """Encode reports without padding them to one global length."""
        return [self.encode(report) for report in reports]

    @staticmethod
    def collate(
        embeddings: Sequence[Tensor],
        pad_to: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Pad variable-length report embeddings and return their mask.

        Args:
            embeddings: Report tensors shaped ``[segments, features]``.
            pad_to: Padded segment length. If omitted, use the longest report
                in this call. For ``crc_reproduction`` selection, pass the
                training-fold maximum recorded by the reproduction protocol.

        Returns:
            A ``(padded, mask)`` pair shaped ``[batch, pad_to, features]`` and
            ``[batch, pad_to]``.
        """
        if not embeddings:
            raise ValueError("embeddings must contain at least one report")
        normalized = []
        for index, value in enumerate(embeddings):
            tensor = torch.as_tensor(value, dtype=torch.float32)
            if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] == 0:
                raise ValueError(
                    "Report {} must have shape [segments, features]".format(index)
                )
            if not torch.isfinite(tensor).all():
                raise ValueError("Report {} contains non-finite values".format(index))
            normalized.append(tensor)
        feature_dims = {int(tensor.shape[1]) for tensor in normalized}
        if len(feature_dims) != 1:
            raise ValueError("All reports must share one feature dimension")
        longest = max(int(tensor.shape[0]) for tensor in normalized)
        target = longest if pad_to is None else int(pad_to)
        if target < longest:
            raise ValueError(
                "pad_to={} is shorter than the longest report ({})".format(
                    target, longest
                )
            )
        if target <= 0:
            raise ValueError("pad_to must be positive")
        padded = torch.zeros(
            len(normalized), target, next(iter(feature_dims)), dtype=torch.float32
        )
        mask = torch.zeros(len(normalized), target, dtype=torch.bool)
        for index, tensor in enumerate(normalized):
            length = int(tensor.shape[0])
            padded[index, :length] = tensor
            mask[index, :length] = True
        return padded, mask


class PS3PathwayPreprocessor:
    """Fit train-only gene scaling and build pathway membership indices.

    Expression frames must use patients as rows and gene symbols as columns.
    Signature frames use one pathway per column and gene symbols in the
    non-null cells of that column.
    """

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.gene_names = []  # type: List[str]
        self.pathway_names = []  # type: List[str]
        self.pathway_gene_indices = []  # type: List[List[int]]
        self.fitted = False

    @staticmethod
    def _validated_expression(expression: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(expression, pd.DataFrame) or expression.empty:
            raise ValueError("expression must be a non-empty pandas DataFrame")
        frame = expression.copy()
        frame.columns = frame.columns.astype(str)
        if frame.columns.duplicated().any():
            raise ValueError("expression contains duplicate gene columns")
        if frame.index.duplicated().any():
            raise ValueError("expression contains duplicate patient rows")
        try:
            values = frame.to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("expression values must be numeric") from exc
        if not np.isfinite(values).all():
            raise ValueError("expression contains non-finite values")
        return frame

    def fit(
        self,
        train_expression: pd.DataFrame,
        signatures: pd.DataFrame,
    ) -> "PS3PathwayPreprocessor":
        """Fit scaling on training patients and resolve pathway membership."""
        frame = self._validated_expression(train_expression)
        if not isinstance(signatures, pd.DataFrame) or signatures.empty:
            raise ValueError("signatures must be a non-empty pandas DataFrame")
        if signatures.columns.duplicated().any():
            raise ValueError("signatures contains duplicate pathway columns")

        self.gene_names = list(frame.columns)
        gene_to_index = {gene: index for index, gene in enumerate(self.gene_names)}
        self.pathway_names = [str(name) for name in signatures.columns]
        self.pathway_gene_indices = []
        for pathway in signatures.columns:
            members = sorted(set(signatures[pathway].dropna().astype(str)))
            indices = [gene_to_index[gene] for gene in members if gene in gene_to_index]
            if not indices:
                raise ValueError(
                    "Pathway {!r} has no genes in the expression matrix".format(
                        pathway
                    )
                )
            self.pathway_gene_indices.append(indices)
        self.scaler.fit(frame.to_numpy(dtype=np.float64))
        self.fitted = True
        return self

    def transform(self, expression: pd.DataFrame) -> Tensor:
        """Apply the train-fitted scaler and return ``[patients, genes]``."""
        if not self.fitted:
            raise RuntimeError("PS3PathwayPreprocessor must be fitted first")
        frame = self._validated_expression(expression)
        missing = [gene for gene in self.gene_names if gene not in frame.columns]
        if missing:
            raise ValueError(
                "expression is missing {} fitted genes; first: {}".format(
                    len(missing), missing[:5]
                )
            )
        values = frame.loc[:, self.gene_names].to_numpy(dtype=np.float64)
        transformed = self.scaler.transform(values).astype(np.float32, copy=False)
        if not np.isfinite(transformed).all():
            raise RuntimeError("standardization produced non-finite values")
        return torch.from_numpy(transformed)

    def fit_transform(
        self,
        train_expression: pd.DataFrame,
        signatures: pd.DataFrame,
    ) -> Tensor:
        return self.fit(train_expression, signatures).transform(train_expression)

    def to_dict(self) -> dict:
        """Return the fitted pathway contract as JSON-compatible values.

        The payload records both feature semantics and the train-fitted
        standardization statistics. It is therefore sufficient to transform
        new expression tables without refitting on inference patients.
        """
        if not self.fitted:
            raise RuntimeError("PS3PathwayPreprocessor must be fitted first")
        return {
            "format": "histox.ps3-pathway-preprocessor",
            "format_version": 1,
            "gene_names": list(self.gene_names),
            "pathway_names": list(self.pathway_names),
            "pathway_gene_indices": [
                list(indices) for indices in self.pathway_gene_indices
            ],
            "scaler": {
                "mean": self.scaler.mean_.tolist(),
                "scale": self.scaler.scale_.tolist(),
                "var": self.scaler.var_.tolist(),
                "n_features_in": int(self.scaler.n_features_in_),
                "n_samples_seen": np.asarray(
                    self.scaler.n_samples_seen_
                ).tolist(),
            },
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "PS3PathwayPreprocessor":
        """Restore a pathway preprocessor produced by :meth:`to_dict`."""
        if not isinstance(payload, dict):
            raise TypeError("pathway payload must be a dictionary")
        if payload.get("format") != "histox.ps3-pathway-preprocessor":
            raise ValueError("Unsupported pathway preprocessor format")
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported pathway preprocessor format version")

        gene_names = [str(name) for name in payload.get("gene_names", ())]
        pathway_names = [
            str(name) for name in payload.get("pathway_names", ())
        ]
        pathway_gene_indices = [
            [int(index) for index in indices]
            for indices in payload.get("pathway_gene_indices", ())
        ]
        if not gene_names or len(set(gene_names)) != len(gene_names):
            raise ValueError("gene_names must be non-empty and unique")
        if (
            not pathway_names
            or len(set(pathway_names)) != len(pathway_names)
            or len(pathway_names) != len(pathway_gene_indices)
        ):
            raise ValueError(
                "pathway names and membership must be non-empty and aligned"
            )
        for pathway_name, indices in zip(
            pathway_names, pathway_gene_indices
        ):
            if not indices or len(set(indices)) != len(indices):
                raise ValueError(
                    "Pathway {!r} has empty or duplicate membership".format(
                        pathway_name
                    )
                )
            if min(indices) < 0 or max(indices) >= len(gene_names):
                raise ValueError(
                    "Pathway {!r} contains an out-of-range gene index".format(
                        pathway_name
                    )
                )

        scaler_payload = payload.get("scaler")
        if not isinstance(scaler_payload, dict):
            raise ValueError("pathway payload is missing scaler statistics")
        mean = np.asarray(scaler_payload.get("mean"), dtype=np.float64)
        scale = np.asarray(scaler_payload.get("scale"), dtype=np.float64)
        variance = np.asarray(scaler_payload.get("var"), dtype=np.float64)
        expected_shape = (len(gene_names),)
        if (
            mean.shape != expected_shape
            or scale.shape != expected_shape
            or variance.shape != expected_shape
        ):
            raise ValueError(
                "scaler statistics must match the fitted gene count"
            )
        if (
            not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or not np.isfinite(variance).all()
            or np.any(scale <= 0)
            or np.any(variance < 0)
        ):
            raise ValueError("scaler statistics contain invalid values")
        if int(scaler_payload.get("n_features_in", -1)) != len(gene_names):
            raise ValueError("scaler n_features_in does not match gene_names")

        restored = cls()
        restored.gene_names = gene_names
        restored.pathway_names = pathway_names
        restored.pathway_gene_indices = pathway_gene_indices
        restored.scaler.mean_ = mean
        restored.scaler.scale_ = scale
        restored.scaler.var_ = variance
        restored.scaler.n_features_in_ = len(gene_names)
        samples_seen = np.asarray(
            scaler_payload.get("n_samples_seen"), dtype=np.int64
        )
        restored.scaler.n_samples_seen_ = (
            int(samples_seen.item())
            if samples_seen.ndim == 0
            else samples_seen
        )
        restored.fitted = True
        return restored

    def build_encoder(self, **kwargs: Any) -> nn.Module:
        """Build a matching :class:`PathwayPrototypeEncoder`."""
        if not self.fitted:
            raise RuntimeError("PS3PathwayPreprocessor must be fitted first")
        from .models import PathwayPrototypeEncoder
        return PathwayPrototypeEncoder(
            n_genes=len(self.gene_names),
            pathway_gene_indices=self.pathway_gene_indices,
            **kwargs
        )


class PS3HistologyPrototypeEncoder(nn.Module):
    """Learn training-fold tissue centers and encode WSI patch bags.

    ``fit`` learns K-means centers from training slides only. ``forward`` then
    performs PANTHER-style MAP-EM for one slide and returns a tensor shaped
    ``[n_prototypes, 1 + 2 * feature_dim]`` containing mixture weight, mean,
    and diagonal covariance for every tissue prototype.
    """

    def __init__(
        self,
        n_prototypes: int = 16,
        patches_per_prototype: int = 100000,
        max_iter: int = 50,
        n_init: int = 3,
        em_iterations: int = 1,
        tau: float = 1.0,
        covariance_prior: float = 1.0,
        random_state: int = 1,
        min_variance: float = 1e-6,
    ) -> None:
        super().__init__()
        if n_prototypes <= 0 or patches_per_prototype <= 0:
            raise ValueError("prototype and patch counts must be positive")
        if max_iter <= 0 or n_init <= 0 or em_iterations <= 0:
            raise ValueError("iteration counts must be positive")
        if tau <= 0 or covariance_prior <= 0 or min_variance <= 0:
            raise ValueError("tau and variance values must be positive")
        self.n_prototypes = n_prototypes
        self.patches_per_prototype = patches_per_prototype
        self.max_iter = max_iter
        self.n_init = n_init
        self.em_iterations = em_iterations
        self.tau = tau
        self.covariance_prior = covariance_prior
        self.random_state = random_state
        self.min_variance = min_variance
        self.register_buffer("centers", torch.empty(0, dtype=torch.float32))

    @property
    def fitted(self) -> bool:
        return self.centers.ndim == 2 and self.centers.shape[0] == self.n_prototypes

    @property
    def feature_dim(self) -> int:
        if not self.fitted:
            raise RuntimeError("PS3HistologyPrototypeEncoder must be fitted first")
        return int(self.centers.shape[1])

    @property
    def output_dim(self) -> int:
        return 1 + 2 * self.feature_dim

    @staticmethod
    def flatten_feature_grid(features: Any) -> Tensor:
        """Convert a HistoX masked WSI feature grid or 2D bag to a tensor."""
        if np.ma.isMaskedArray(features):
            if features.ndim != 3:
                raise ValueError("masked WSI feature grid must be three-dimensional")
            mask = np.ma.getmaskarray(features)
            valid = ~mask[..., 0]
            features = np.asarray(features.data[valid])
        tensor = torch.as_tensor(features, dtype=torch.float32)
        if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] == 0:
            raise ValueError("patch features must have shape [patches, features]")
        if not torch.isfinite(tensor).all():
            raise ValueError("patch features contain non-finite values")
        return tensor

    @staticmethod
    def unpack_panther_allcat(
        values: Any,
        n_prototypes: int,
        layout: str = "prototype_aligned",
    ) -> Tensor:
        """Convert PANTHER ``allcat`` vectors to prototype-aligned tokens.

        PANTHER serializes all mixture weights first, followed by every mean
        and every diagonal covariance. A direct reshape would mix these three
        blocks across prototypes. This method returns ``[..., prototypes,
        1 + 2 * feature_dim]`` in ``(weight, mean, covariance)`` order.

        Args:
            values: One vector or a batch of serialized ``allcat`` vectors.
            n_prototypes: Number of PANTHER mixture components.
            layout: ``"prototype_aligned"`` reconstructs semantic prototype
                tokens and is the default for new studies.
                ``"crc_reproduction"`` preserves the direct reshape used to
                train the retained HistoX CRC checkpoints.
        """
        tensor = torch.as_tensor(values, dtype=torch.float32)
        squeeze = tensor.ndim == 1
        if squeeze:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[0] == 0:
            raise ValueError("PANTHER allcat values must have shape [samples, features]")
        if n_prototypes <= 0:
            raise ValueError("n_prototypes must be positive")
        if layout not in ("prototype_aligned", "crc_reproduction"):
            raise ValueError(
                "layout must be 'prototype_aligned' or 'crc_reproduction'"
            )
        remaining = int(tensor.shape[1]) - int(n_prototypes)
        denominator = 2 * int(n_prototypes)
        if remaining <= 0 or remaining % denominator:
            raise ValueError(
                "PANTHER allcat width is incompatible with n_prototypes"
            )
        feature_dim = remaining // denominator
        if layout == "crc_reproduction":
            tokens = tensor.reshape(
                -1, n_prototypes, 1 + 2 * feature_dim
            )
            return tokens[0] if squeeze else tokens
        weights = tensor[:, :n_prototypes].unsqueeze(-1)
        mean_end = n_prototypes * (1 + feature_dim)
        means = tensor[:, n_prototypes:mean_end].reshape(
            -1, n_prototypes, feature_dim
        )
        covariances = tensor[:, mean_end:].reshape(
            -1, n_prototypes, feature_dim
        )
        tokens = torch.cat((weights, means, covariances), dim=-1)
        if not torch.isfinite(tokens).all():
            raise ValueError("PANTHER allcat values contain non-finite values")
        return tokens[0] if squeeze else tokens

    @staticmethod
    def flatten_feature_grid_with_coordinates(
        features: Any,
    ) -> Tuple[Tensor, Tensor]:
        """Flatten a masked WSI feature grid and retain its grid coordinates.

        Returns:
            A ``(features, coordinates)`` pair. Features have shape
            ``[patches, channels]``. Coordinates have shape ``[patches, 2]``
            and use ``(grid_x, grid_y)`` order, aligned row-for-row with the
            flattened features.
        """
        if not np.ma.isMaskedArray(features) or features.ndim != 3:
            raise ValueError(
                "features must be a three-dimensional masked WSI feature grid"
            )
        mask = np.ma.getmaskarray(features)
        valid = ~mask[..., 0]
        grid_yx = np.argwhere(valid)
        if grid_yx.size == 0:
            raise ValueError("masked WSI feature grid contains no valid patches")
        bag = PS3HistologyPrototypeEncoder.flatten_feature_grid(features)
        grid_xy = np.stack((grid_yx[:, 1], grid_yx[:, 0]), axis=1)
        coordinates = torch.as_tensor(grid_xy, dtype=torch.int64)
        if len(bag) != len(coordinates):
            raise RuntimeError("feature and coordinate counts do not match")
        return bag, coordinates

    def load_centers(self, centers: Any) -> "PS3HistologyPrototypeEncoder":
        """Load externally fitted centers after validating their shape."""
        value = self.flatten_feature_grid(centers)
        if value.shape[0] != self.n_prototypes:
            raise ValueError(
                "Expected {} centers, got {}".format(
                    self.n_prototypes, value.shape[0]
                )
            )
        self.centers = value.detach().clone()
        return self

    def fit(self, training_bags: Sequence[Any]) -> "PS3HistologyPrototypeEncoder":
        """Fit K-means centers using only training-fold WSI feature bags."""
        if not training_bags:
            raise ValueError("training_bags must contain at least one slide")
        bags = [self.flatten_feature_grid(bag).cpu() for bag in training_bags]
        feature_dims = {int(bag.shape[1]) for bag in bags}
        if len(feature_dims) != 1:
            raise ValueError("all training bags must share one feature dimension")

        total_limit = self.n_prototypes * self.patches_per_prototype
        per_slide = int(math.ceil(float(total_limit) / len(bags)))
        sampled = []
        for index, bag in enumerate(bags):
            if len(bag) > per_slide:
                generator = torch.Generator().manual_seed(self.random_state + index)
                choice = torch.randperm(len(bag), generator=generator)[:per_slide]
                bag = bag.index_select(0, choice)
            sampled.append(bag)
        patches = torch.cat(sampled, dim=0)
        if len(patches) > total_limit:
            patches = patches[:total_limit]
        if len(patches) < self.n_prototypes:
            raise ValueError(
                "At least {} training patches are required".format(
                    self.n_prototypes
                )
            )

        kmeans = KMeans(
            n_clusters=self.n_prototypes,
            max_iter=self.max_iter,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        kmeans.fit(patches.numpy())
        self.centers = torch.as_tensor(
            kmeans.cluster_centers_, dtype=torch.float32
        )
        return self

    def _encode(self, patch_features: Any) -> Tuple[Tensor, Tensor]:
        """Return PANTHER prototypes and final patch responsibilities."""
        if not self.fitted:
            raise RuntimeError("PS3HistologyPrototypeEncoder must be fitted first")
        patches = self.flatten_feature_grid(patch_features).to(self.centers.device)
        if patches.shape[1] != self.feature_dim:
            raise ValueError(
                "Expected patch feature dimension {}, got {}".format(
                    self.feature_dim, patches.shape[1]
                )
            )
        centers = self.centers.to(dtype=patches.dtype)
        n_patches, feature_dim = patches.shape
        del n_patches
        mixture = torch.full(
            (self.n_prototypes,),
            1.0 / self.n_prototypes,
            dtype=patches.dtype,
            device=patches.device,
        )
        means = centers
        variances = torch.full_like(centers, self.covariance_prior)

        for _ in range(self.em_iterations):
            difference = patches.unsqueeze(1) - means.unsqueeze(0)
            log_likelihood = -0.5 * (
                feature_dim * math.log(2.0 * math.pi)
                + torch.log(variances).sum(dim=-1).unsqueeze(0)
                + (difference.square() / variances.unsqueeze(0)).sum(dim=-1)
            )
            log_likelihood = log_likelihood + torch.log(mixture).unsqueeze(0)
            responsibilities = torch.softmax(log_likelihood, dim=-1)
            responsibility_sum = responsibilities.sum(dim=0)
            regularized_sum = responsibility_sum + self.tau
            mixture = regularized_sum / regularized_sum.sum()
            weighted_sum = responsibilities.transpose(0, 1).matmul(patches)
            weighted_square_sum = responsibilities.transpose(0, 1).matmul(
                patches.square()
            )
            means = (
                weighted_sum + self.tau * centers
            ) / regularized_sum.unsqueeze(-1)
            second_moment = (
                weighted_square_sum
                + self.tau * (self.covariance_prior + centers.square())
            ) / regularized_sum.unsqueeze(-1)
            variances = (second_moment - means.square()).clamp_min(
                self.min_variance
            )

        # Recompute responsibilities with the final MAP-EM parameters so the
        # patch assignments used for visualization match the returned
        # prototype statistics.
        difference = patches.unsqueeze(1) - means.unsqueeze(0)
        log_likelihood = -0.5 * (
            feature_dim * math.log(2.0 * math.pi)
            + torch.log(variances).sum(dim=-1).unsqueeze(0)
            + (difference.square() / variances.unsqueeze(0)).sum(dim=-1)
        )
        log_likelihood = log_likelihood + torch.log(mixture).unsqueeze(0)
        responsibilities = torch.softmax(log_likelihood, dim=-1)
        prototypes = torch.cat(
            (mixture.unsqueeze(-1), means, variances), dim=-1
        ).float()
        return prototypes, responsibilities.float()

    def forward(self, patch_features: Any) -> Tensor:
        """Encode one WSI bag as mixture weights, means, and covariances."""
        prototypes, _ = self._encode(patch_features)
        return prototypes

    def transform_with_assignments(
        self,
        patch_features: Any,
    ) -> Tuple[Tensor, Tensor]:
        """Encode one WSI and return patch-to-prototype responsibilities.

        The first tensor follows the standard PANTHER contract with shape
        ``[n_prototypes, 1 + 2 * feature_dim]``. The second has shape
        ``[n_patches, n_prototypes]`` and every row sums to one.
        """
        return self._encode(patch_features)

    def transform(self, patch_features: Any) -> Tensor:
        return self.forward(patch_features)

    def transform_many(self, bags: Sequence[Any]) -> Tensor:
        """Encode equal-width variable-length bags into one batch tensor."""
        return torch.stack([self.forward(bag) for bag in bags], dim=0)
