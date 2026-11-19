"""Independent PS3 prototype-fusion components.

This module implements the equations described in the PS3 ICCV 2025 paper.
It does not contain source copied from the authors' unlicensed code release and
does not claim compatibility with upstream checkpoints.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _validate_mask(mask: Optional[Tensor], batch: int, tokens: int, name: str) -> Tensor:
    """Return a boolean prototype mask with shape ``[batch, tokens]``."""
    if mask is None:
        return torch.ones(batch, tokens, dtype=torch.bool)
    if mask.shape != (batch, tokens):
        raise ValueError(
            "{} mask must have shape ({}, {}), got {}".format(
                name, batch, tokens, tuple(mask.shape)
            )
        )
    return mask.to(dtype=torch.bool)


class DiagnosticPrototypeSelector(nn.Module):
    """Select fixed-length diagnostic prototypes from report segments.

    The selector applies masked self-attention, averages incoming attention for
    each valid report segment, and keeps the highest-scoring segments from the
    post-attention representation (PS3 equations 1-3). If a report contains
    fewer valid segments than ``n_prototypes``, its valid post-attention
    sequence is linearly resized to the requested length. This explicit HistoX
    extension avoids selecting padding for short reports.

    Args:
        input_dim: Dimension of each report-segment embedding.
        n_prototypes: Number of diagnostic prototypes retained per report.
        output_dim: Output prototype dimension. Defaults to ``input_dim``.
        mode: ``"paper"`` averages attention over valid query positions and
            interpolates short reports. ``"crc_reproduction"`` reproduces
            the released CRC run: attention is averaged over all padded query
            positions and top-k may select their post-attention outputs.
            ``"traceable"`` ranks only real segments and repeats real source
            indices when a report contains fewer segments than prototypes.
        representation_mode: ``"attended"`` uses the self-attention output.
            ``"residual"`` adds the original segment embedding so selected
            prototypes preserve patient-specific source information.
        residual_scale: Multiplier applied to the original segment embedding
            in ``"residual"`` mode. The default ``1.0`` preserves the
            unscaled residual formulation.
    """

    def __init__(
        self,
        input_dim: int,
        n_prototypes: int,
        output_dim: Optional[int] = None,
        mode: str = "paper",
        representation_mode: str = "attended",
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if n_prototypes <= 0:
            raise ValueError("n_prototypes must be positive")
        if mode not in ("paper", "crc_reproduction", "traceable"):
            raise ValueError(
                "mode must be 'paper', 'crc_reproduction', or 'traceable'"
            )
        if representation_mode not in ("attended", "residual"):
            raise ValueError(
                "representation_mode must be 'attended' or 'residual'"
            )
        if not math.isfinite(residual_scale) or residual_scale <= 0:
            raise ValueError("residual_scale must be a positive finite number")
        self.input_dim = input_dim
        self.n_prototypes = n_prototypes
        self.output_dim = output_dim or input_dim
        self.mode = mode
        self.representation_mode = representation_mode
        self.residual_scale = float(residual_scale)
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.output = nn.Linear(input_dim, self.output_dim)

    def forward(
        self,
        report_embeddings: Tensor,
        mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        """Build diagnostic prototypes.

        Args:
            report_embeddings: Tensor shaped ``[batch, segments, input_dim]``.
            mask: Boolean tensor shaped ``[batch, segments]``; ``True`` marks
                real segments and ``False`` marks padding.
            return_attention: Return attention, importance scores, selected
                segment indices, and a short-report interpolation flag
                alongside the prototypes. Interpolated prototypes have
                ``selected_indices == -1`` because they do not correspond to
                one original segment.
        """
        if report_embeddings.ndim != 3:
            raise ValueError("report_embeddings must have shape [batch, segments, features]")
        batch, segments, features = report_embeddings.shape
        if features != self.input_dim:
            raise ValueError(
                "Expected report feature dimension {}, got {}".format(
                    self.input_dim, features
                )
            )
        valid = _validate_mask(mask, batch, segments, "report").to(report_embeddings.device)
        valid_counts = valid.sum(dim=1)
        if torch.any(valid_counts == 0):
            raise ValueError("Each report must contain at least one valid segment")

        query = self.query(report_embeddings)
        key = self.key(report_embeddings)
        value = self.value(report_embeddings)
        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.input_dim)
        logits = logits.masked_fill(~valid.unsqueeze(1), -torch.inf)
        attention = torch.softmax(logits, dim=-1)

        post_attention = torch.matmul(attention, value)
        if self.representation_mode == "residual":
            post_attention = (
                post_attention + self.residual_scale * report_embeddings
            )
        if self.mode == "crc_reproduction":
            if segments < self.n_prototypes:
                raise ValueError(
                    "crc_reproduction mode requires the padded report length "
                    "to be at least n_prototypes"
                )
            importance = attention.mean(dim=1)
            selected_indices_tensor = torch.topk(
                importance,
                k=self.n_prototypes,
                dim=1,
                largest=True,
                sorted=True,
            ).indices
            gather_index = selected_indices_tensor.unsqueeze(-1).expand(
                -1, -1, post_attention.shape[-1]
            )
            prototypes = self.output(torch.gather(
                post_attention, 1, gather_index
            ))
            interpolated_tensor = torch.zeros(
                batch, dtype=torch.bool, device=report_embeddings.device
            )
            if return_attention:
                return prototypes, {
                    "attention": attention,
                    "importance": importance,
                    "selected_indices": selected_indices_tensor,
                    "interpolated": interpolated_tensor,
                }
            return prototypes

        query_mask = valid.unsqueeze(-1).to(attention.dtype)
        importance = (attention * query_mask).sum(dim=1)
        importance = importance / valid_counts.unsqueeze(-1).to(attention.dtype)
        importance = importance.masked_fill(~valid, -torch.inf)

        if self.mode == "traceable":
            selected = []
            selected_indices = []
            for batch_index in range(batch):
                count = int(valid_counts[batch_index].item())
                keep = min(count, self.n_prototypes)
                indices = torch.topk(
                    importance[batch_index],
                    k=keep,
                    largest=True,
                    sorted=True,
                ).indices
                if keep < self.n_prototypes:
                    repeats = math.ceil(self.n_prototypes / keep)
                    indices = indices.repeat(repeats)[:self.n_prototypes]
                selected.append(
                    post_attention[batch_index].index_select(0, indices)
                )
                selected_indices.append(indices)
            prototypes = self.output(torch.stack(selected, dim=0))
            selected_indices_tensor = torch.stack(selected_indices, dim=0)
            interpolated_tensor = torch.zeros(
                batch, dtype=torch.bool, device=report_embeddings.device
            )
            if return_attention:
                return prototypes, {
                    "attention": attention,
                    "importance": importance,
                    "selected_indices": selected_indices_tensor,
                    "interpolated": interpolated_tensor,
                }
            return prototypes

        selected = []
        selected_indices = []
        interpolated = []
        for batch_index in range(batch):
            count = int(valid_counts[batch_index].item())
            if count >= self.n_prototypes:
                indices = torch.topk(
                    importance[batch_index],
                    k=self.n_prototypes,
                    largest=True,
                    sorted=True,
                ).indices
                selected.append(post_attention[batch_index].index_select(0, indices))
                selected_indices.append(indices)
                interpolated.append(False)
            else:
                valid_sequence = post_attention[batch_index][valid[batch_index]]
                resized = F.interpolate(
                    valid_sequence.transpose(0, 1).unsqueeze(0),
                    size=self.n_prototypes,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0).transpose(0, 1)
                selected.append(resized)
                selected_indices.append(torch.full(
                    (self.n_prototypes,),
                    -1,
                    dtype=torch.long,
                    device=report_embeddings.device,
                ))
                interpolated.append(True)
        prototypes = self.output(torch.stack(selected, dim=0))
        selected_indices_tensor = torch.stack(selected_indices, dim=0)
        interpolated_tensor = torch.tensor(
            interpolated,
            dtype=torch.bool,
            device=report_embeddings.device,
        )
        if return_attention:
            return prototypes, {
                "attention": attention,
                "importance": importance,
                "selected_indices": selected_indices_tensor,
                "interpolated": interpolated_tensor,
            }
        return prototypes


class _PathwaySNN(nn.Module):
    """Self-normalizing MLP for one biological pathway."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.AlphaDropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.SELU(),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.layers(values)


class PathwayPrototypeEncoder(nn.Module):
    """Encode gene expression into pathway-specific prototypes.

    Each pathway owns an independent self-normalizing MLP, matching the
    pathway-specific construction in PS3 equation 9.

    Args:
        n_genes: Length of the input gene-expression vector.
        pathway_gene_indices: Gene-column indices for every pathway.
        output_dim: Shared output dimension for all pathway prototypes.
        hidden_dim: Hidden dimension of each pathway MLP.
        dropout: AlphaDropout probability.
    """

    def __init__(
        self,
        n_genes: int,
        pathway_gene_indices: Sequence[Sequence[int]],
        output_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_genes <= 0:
            raise ValueError("n_genes must be positive")
        if not pathway_gene_indices:
            raise ValueError("pathway_gene_indices must contain at least one pathway")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.n_genes = n_genes
        self.output_dim = output_dim
        self._index_names: List[str] = []
        networks = []
        for pathway_id, indices in enumerate(pathway_gene_indices):
            normalized = torch.as_tensor(list(indices), dtype=torch.long)
            if normalized.numel() == 0:
                raise ValueError("Pathway {} contains no genes".format(pathway_id))
            if torch.any(normalized < 0) or torch.any(normalized >= n_genes):
                raise ValueError("Pathway {} contains an out-of-range gene index".format(pathway_id))
            if normalized.unique().numel() != normalized.numel():
                raise ValueError("Pathway {} contains duplicate gene indices".format(pathway_id))
            name = "_pathway_indices_{}".format(pathway_id)
            self.register_buffer(name, normalized)
            self._index_names.append(name)
            networks.append(_PathwaySNN(len(normalized), hidden_dim, output_dim, dropout))
        self.networks = nn.ModuleList(networks)

    @property
    def n_pathways(self) -> int:
        return len(self.networks)

    def forward(self, gene_expression: Tensor) -> Tensor:
        """Return pathway prototypes shaped ``[batch, pathways, output_dim]``."""
        if gene_expression.ndim != 2:
            raise ValueError("gene_expression must have shape [batch, genes]")
        if gene_expression.shape[1] != self.n_genes:
            raise ValueError(
                "Expected {} genes, got {}".format(
                    self.n_genes, gene_expression.shape[1]
                )
            )
        encoded = []
        for name, network in zip(self._index_names, self.networks):
            indices = getattr(self, name)
            encoded.append(network(gene_expression.index_select(1, indices)))
        return torch.stack(encoded, dim=1)


class _FusionLayer(nn.Module):
    """Transformer layer exposing the full prototype attention matrix."""

    def __init__(
        self,
        dim: int,
        feedforward_dim: int,
        dropout: float,
        attention_residual: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.attention_residual = attention_residual
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(dim)
        self.feedforward = nn.Sequential(
            nn.Linear(dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, dim),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(dim)

    def forward(
        self, tokens: Tensor, mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        query = self.query(tokens)
        key = self.key(tokens)
        value = self.value(tokens)
        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.dim)
        logits = logits.masked_fill(~mask.unsqueeze(1), -torch.inf)
        attention = torch.softmax(logits, dim=-1)
        attended = torch.matmul(self.attention_dropout(attention), value)
        if self.attention_residual:
            attended = tokens + attended
        tokens = self.attention_norm(attended)
        tokens = self.output_norm(tokens + self.feedforward(tokens))
        tokens = tokens.masked_fill(~mask.unsqueeze(-1), 0)
        return tokens, attention, logits


class _PrototypeMLP(nn.Module):
    """Post-fusion MLP applied to one fixed prototype position."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, token: Tensor) -> Tensor:
        return self.layers(token)


class PS3PrototypeFusion(nn.Module):
    """Fuse pathway, histology, and report prototypes into survival risk.

    Inputs must already be compact prototype tensors. The model projects every
    modality to a common space, appends a learnable modality- or prototype-level
    embedding, applies joint self/cross-modal attention, transforms fixed
    prototype positions with independent MLPs, mean-pools each modality, and
    predicts one log-risk value.

    Attention token order is always pathway, histology, then report.
    """

    is_multimodal = True

    def __init__(
        self,
        pathway_dim: int,
        histology_dim: int,
        report_dim: int,
        n_pathway_prototypes: int,
        n_histology_prototypes: int = 16,
        n_report_prototypes: int = 10,
        embedding_dim: int = 256,
        modality_embedding_dim: int = 32,
        n_fusion_layers: int = 1,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
        prototype_embedding_mode: str = "modality",
        fusion_residual: bool = True,
    ) -> None:
        super().__init__()
        counts = (n_pathway_prototypes, n_histology_prototypes, n_report_prototypes)
        if any(count <= 0 for count in counts):
            raise ValueError("Prototype counts must be positive")
        if any(dim <= 0 for dim in (pathway_dim, histology_dim, report_dim, embedding_dim)):
            raise ValueError("Feature dimensions must be positive")
        if modality_embedding_dim <= 0:
            raise ValueError("modality_embedding_dim must be positive")
        if n_fusion_layers <= 0:
            raise ValueError("n_fusion_layers must be positive")
        if prototype_embedding_mode not in {"modality", "prototype"}:
            raise ValueError(
                "prototype_embedding_mode must be 'modality' or 'prototype'"
            )
        self.prototype_counts = counts
        self.input_dims = (pathway_dim, histology_dim, report_dim)
        self.embedding_dim = embedding_dim
        self.modality_embedding_dim = modality_embedding_dim
        self.fusion_dim = embedding_dim + modality_embedding_dim
        self.prototype_embedding_mode = prototype_embedding_mode
        self.fusion_residual = fusion_residual

        self.projections = nn.ModuleList(
            nn.Linear(input_dim, embedding_dim) for input_dim in self.input_dims
        )
        embedding_counts = (
            counts
            if prototype_embedding_mode == "prototype"
            else (1, 1, 1)
        )
        self.modality_embeddings = nn.ParameterList(
            nn.Parameter(torch.empty(1, count, modality_embedding_dim))
            for count in embedding_counts
        )
        embedding_std = 1.0 if prototype_embedding_mode == "prototype" else 0.02
        for embedding in self.modality_embeddings:
            nn.init.normal_(embedding, std=embedding_std)
        self.fusion_layers = nn.ModuleList(
            _FusionLayer(
                self.fusion_dim,
                feedforward_dim,
                dropout,
                attention_residual=fusion_residual,
            )
            for _ in range(n_fusion_layers)
        )
        self.post_fusion = nn.ModuleList(
            nn.ModuleList(
                _PrototypeMLP(self.fusion_dim, feedforward_dim, dropout)
                for _ in range(count)
            )
            for count in counts
        )
        self.modality_norms = nn.ModuleList(nn.LayerNorm(self.fusion_dim) for _ in range(3))
        self.risk_head = nn.Linear(3 * self.fusion_dim, 1)

    def _prepare_modality(
        self,
        tensor: Tensor,
        modality_id: int,
        mask: Optional[Tensor],
        name: str,
    ) -> Tuple[Tensor, Tensor]:
        if tensor.ndim != 3:
            raise ValueError("{} prototypes must have shape [batch, prototypes, features]".format(name))
        batch, count, features = tensor.shape
        expected_count = self.prototype_counts[modality_id]
        expected_features = self.input_dims[modality_id]
        if count != expected_count:
            raise ValueError("Expected {} {} prototypes, got {}".format(expected_count, name, count))
        if features != expected_features:
            raise ValueError("Expected {} {} features, got {}".format(expected_features, name, features))
        valid = _validate_mask(mask, batch, count, name).to(tensor.device)
        if torch.any(valid.sum(dim=1) == 0):
            raise ValueError("Each sample must contain at least one valid {} prototype".format(name))
        projected = self.projections[modality_id](tensor)
        learned = self.modality_embeddings[modality_id].expand(batch, count, -1)
        return torch.cat((projected, learned), dim=-1), valid

    @staticmethod
    def _masked_mean(tokens: Tensor, mask: Tensor) -> Tensor:
        weights = mask.unsqueeze(-1).to(tokens.dtype)
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)

    def forward(
        self,
        pathway_prototypes: Tensor,
        histology_prototypes: Tensor,
        report_prototypes: Tensor,
        pathway_mask: Optional[Tensor] = None,
        histology_mask: Optional[Tensor] = None,
        report_mask: Optional[Tensor] = None,
        return_attention: bool = False,
        return_attention_logits: bool = False,
    ) -> Union[
        Tensor,
        Tuple[Tensor, Tensor],
        Tuple[Tensor, Tensor, Tensor],
    ]:
        """Return one log-risk value per patient."""
        inputs = (pathway_prototypes, histology_prototypes, report_prototypes)
        masks = (pathway_mask, histology_mask, report_mask)
        names = ("pathway", "histology", "report")
        if len({tensor.shape[0] for tensor in inputs}) != 1:
            raise ValueError("All modalities must have the same batch size")

        prepared = [
            self._prepare_modality(tensor, index, mask, name)
            for index, (tensor, mask, name) in enumerate(zip(inputs, masks, names))
        ]
        modality_tokens = [item[0] for item in prepared]
        modality_masks = [item[1] for item in prepared]
        tokens = torch.cat(modality_tokens, dim=1)
        valid = torch.cat(modality_masks, dim=1)

        attention = None
        attention_logits = None
        for layer in self.fusion_layers:
            tokens, attention, attention_logits = layer(tokens, valid)

        split_tokens = torch.split(tokens, self.prototype_counts, dim=1)
        pooled = []
        for modality_id, (current, mask) in enumerate(zip(split_tokens, modality_masks)):
            transformed = torch.stack(
                [
                    network(current[:, prototype_id])
                    for prototype_id, network in enumerate(self.post_fusion[modality_id])
                ],
                dim=1,
            )
            transformed = self.modality_norms[modality_id](transformed)
            pooled.append(self._masked_mean(transformed, mask))
        risk = self.risk_head(torch.cat(pooled, dim=-1))
        if return_attention:
            if attention is None or attention_logits is None:
                raise RuntimeError("Fusion produced no attention matrix")
            if return_attention_logits:
                return risk, attention, attention_logits
            return risk, attention
        if return_attention_logits:
            raise ValueError(
                "return_attention_logits requires return_attention=True"
            )
        return risk


class PS3SurvivalModel(nn.Module):
    """Paper-derived PS3 survival model with explicit preprocessing inputs.

    This first HistoX implementation constructs diagnostic and pathway
    prototypes internally. Histological prototypes must be prepared separately
    (for example, using a PANTHER/GMM workflow) and passed to ``forward``.
    """

    def __init__(
        self,
        n_genes: int,
        pathway_gene_indices: Sequence[Sequence[int]],
        histology_dim: int,
        report_dim: int = 512,
        n_histology_prototypes: int = 16,
        n_report_prototypes: int = 10,
        embedding_dim: int = 256,
        modality_embedding_dim: int = 32,
        n_fusion_layers: int = 1,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
        report_selection_mode: str = "paper",
        prototype_embedding_mode: str = "modality",
        fusion_residual: bool = True,
        report_representation_mode: str = "attended",
        report_residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.pathway_encoder = PathwayPrototypeEncoder(
            n_genes=n_genes,
            pathway_gene_indices=pathway_gene_indices,
            output_dim=embedding_dim,
            hidden_dim=embedding_dim,
            dropout=dropout,
        )
        self.report_selector = DiagnosticPrototypeSelector(
            input_dim=report_dim,
            n_prototypes=n_report_prototypes,
            output_dim=embedding_dim,
            mode=report_selection_mode,
            representation_mode=report_representation_mode,
            residual_scale=report_residual_scale,
        )
        self.fusion = PS3PrototypeFusion(
            pathway_dim=embedding_dim,
            histology_dim=histology_dim,
            report_dim=embedding_dim,
            n_pathway_prototypes=self.pathway_encoder.n_pathways,
            n_histology_prototypes=n_histology_prototypes,
            n_report_prototypes=n_report_prototypes,
            embedding_dim=embedding_dim,
            modality_embedding_dim=modality_embedding_dim,
            n_fusion_layers=n_fusion_layers,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            prototype_embedding_mode=prototype_embedding_mode,
            fusion_residual=fusion_residual,
        )

    def forward(
        self,
        histology_prototypes: Tensor,
        gene_expression: Tensor,
        report_embeddings: Tensor,
        report_mask: Optional[Tensor] = None,
        histology_mask: Optional[Tensor] = None,
        return_attention: bool = False,
        return_attention_logits: bool = False,
    ) -> Union[
        Tensor,
        Tuple[Tensor, Tensor],
        Tuple[Tensor, Tensor, Tensor],
    ]:
        pathway_prototypes = self.pathway_encoder(gene_expression)
        report_prototypes = self.report_selector(report_embeddings, report_mask)
        return self.fusion(
            pathway_prototypes,
            histology_prototypes,
            report_prototypes,
            histology_mask=histology_mask,
            return_attention=return_attention,
            return_attention_logits=return_attention_logits,
        )

    def report_segment_evidence(
        self,
        histology_prototypes: Tensor,
        gene_expression: Tensor,
        report_embeddings: Tensor,
        report_mask: Optional[Tensor] = None,
        histology_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Map histology-to-report attention back to source report segments.

        This method follows the report interpretation used in the PS3 paper:
        it normalizes the final-layer attention logits over report prototypes,
        averages the resulting scores over histology-prototype queries, and
        selects the highest-scoring report prototype that still maps to a
        real source segment.

        Args:
            histology_prototypes: Histology prototypes shaped
                ``[batch, histology_prototypes, features]``.
            gene_expression: Fold-preprocessed expression shaped
                ``[batch, genes]``.
            report_embeddings: Padded report-segment embeddings shaped
                ``[batch, segments, features]``.
            report_mask: Boolean source-segment mask shaped
                ``[batch, segments]``.
            histology_mask: Optional histology-prototype mask.

        Returns:
            A dictionary containing ``risk``, the joint attention tensors,
            report-prototype scores, source-segment indices, source validity,
            and the selected report prototype/source segment for every
            patient. ``selected_source_segment`` indexes the second dimension
            of ``report_embeddings``.

        Notes:
            Attention measures a trained association, not a causal effect.
            Interpolated report prototypes cannot be traced to one source
            segment and therefore raise ``ValueError`` if no traceable
            prototype is available for a patient.
        """
        if report_embeddings.ndim != 3:
            raise ValueError(
                "report_embeddings must have shape [batch, segments, features]"
            )
        batch, segments, _ = report_embeddings.shape
        source_mask = _validate_mask(
            report_mask, batch, segments, "report"
        ).to(report_embeddings.device)

        pathway_prototypes = self.pathway_encoder(gene_expression)
        report_prototypes, selection = self.report_selector(
            report_embeddings,
            source_mask,
            return_attention=True,
        )
        risk, attention, logits = self.fusion(
            pathway_prototypes,
            histology_prototypes,
            report_prototypes,
            histology_mask=histology_mask,
            return_attention=True,
            return_attention_logits=True,
        )

        n_pathway, n_histology, n_report = self.fusion.prototype_counts
        histology_start = n_pathway
        report_start = n_pathway + n_histology
        histology_to_report_logits = logits[
            :,
            histology_start:report_start,
            report_start:report_start + n_report,
        ]
        histology_to_report_attention = torch.softmax(
            histology_to_report_logits, dim=-1
        )
        report_scores = histology_to_report_attention.mean(dim=1)

        source_indices = selection["selected_indices"]
        in_bounds = (source_indices >= 0) & (source_indices < segments)
        safe_indices = source_indices.clamp(min=0, max=segments - 1)
        source_valid = in_bounds & torch.gather(source_mask, 1, safe_indices)
        if torch.any(~source_valid.any(dim=1)):
            failed = torch.nonzero(
                ~source_valid.any(dim=1), as_tuple=False
            ).reshape(-1).tolist()
            raise ValueError(
                "No report prototype maps to a source segment for batch "
                "indices {}. Interpolated report prototypes cannot be "
                "used for source-text evidence.".format(failed)
            )

        traceable_scores = report_scores.masked_fill(
            ~source_valid, -torch.inf
        )
        selected_prototype = traceable_scores.argmax(dim=1)
        selected_source = torch.gather(
            source_indices, 1, selected_prototype.unsqueeze(1)
        ).squeeze(1)
        selected_score = torch.gather(
            report_scores, 1, selected_prototype.unsqueeze(1)
        ).squeeze(1)

        return {
            "risk": risk,
            "joint_attention": attention,
            "joint_attention_logits": logits,
            "histology_to_report_attention": histology_to_report_attention,
            "report_prototype_scores": report_scores,
            "report_prototype_source_segments": source_indices,
            "report_prototype_source_valid": source_valid,
            "selected_report_prototype": selected_prototype,
            "selected_source_segment": selected_source,
            "selected_score": selected_score,
        }


class CoxPHLoss(nn.Module):
    """Negative Cox partial log-likelihood with Breslow tie handling.

    The risk sets are formed only from samples supplied to one ``forward``
    call. Use full-risk-set batches or a separately validated sampling scheme;
    arbitrary minibatches change the Cox objective.

    Args to ``forward``:
        log_risk: Predicted log-risk shaped ``[batch]`` or ``[batch, 1]``.
        duration: Observed event or censoring times shaped ``[batch]``.
        event: Event indicators shaped ``[batch]`` (1=event, 0=censored).
    """

    def forward(self, log_risk: Tensor, duration: Tensor, event: Tensor) -> Tensor:
        risk = log_risk.reshape(-1)
        duration = duration.reshape(-1)
        event = event.reshape(-1).to(dtype=torch.bool)
        if not (risk.numel() == duration.numel() == event.numel()):
            raise ValueError("log_risk, duration, and event must have equal lengths")
        if risk.numel() == 0:
            raise ValueError("CoxPHLoss received an empty batch")
        if not torch.any(event):
            raise ValueError("CoxPHLoss requires at least one observed event")
        if not torch.all(torch.isfinite(risk)) or not torch.all(torch.isfinite(duration)):
            raise ValueError("log_risk and duration must be finite")

        partial_log_likelihood = risk.new_zeros(())
        event_times = torch.unique(duration[event])
        for event_time in event_times:
            tied_events = event & (duration == event_time)
            at_risk = duration >= event_time
            n_tied = tied_events.sum().to(risk.dtype)
            partial_log_likelihood = partial_log_likelihood + risk[tied_events].sum()
            partial_log_likelihood = partial_log_likelihood - n_tied * torch.logsumexp(
                risk[at_risk], dim=0
            )
        return -partial_log_likelihood / event.sum().to(risk.dtype)


__all__ = [
    "CoxPHLoss",
    "DiagnosticPrototypeSelector",
    "PS3PrototypeFusion",
    "PS3SurvivalModel",
    "PathwayPrototypeEncoder",
]
