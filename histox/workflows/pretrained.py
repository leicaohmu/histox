"""Local pretrained bundles for reproducible HistoX workflow inference."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union


PathLike = Union[str, Path]
_BUNDLE_FORMAT = "histox.ps3-pretrained"
_BUNDLE_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safetensors():
    try:
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise ImportError(
            "PS3 pretrained bundles require safetensors. Install HistoX "
            "with `pip install 'histox[huggingface]'`."
        ) from exc
    return load_file, save_file


class PS3PretrainedBundle:
    """A validated PS3 checkpoint plus its inference preprocessing contract.

    Bundle loading is local in A6a. A later Hub integration can download the
    same directory layout before calling :meth:`load`.
    """

    def __init__(
        self,
        root: Path,
        experiment,
        pathway_preprocessor,
        histology_encoder,
        report_config: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> None:
        self.root = root
        self.experiment = experiment
        self.pathway_preprocessor = pathway_preprocessor
        self.histology_encoder = histology_encoder
        self.report_config = dict(report_config)
        self.config = dict(config)

    @property
    def model(self):
        """Return the loaded evaluation-mode PS3 model."""
        return self.experiment.build_model()

    @classmethod
    def export(
        cls,
        experiment,
        destination: PathLike,
        pathway_preprocessor,
        histology_encoder=None,
        histology_config: Optional[Mapping[str, Any]] = None,
        report_config: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "PS3PretrainedBundle":
        """Write a self-describing local bundle from a loaded experiment.

        ``experiment.build_model()`` must already contain the intended
        checkpoint. Existing non-empty destinations are rejected to avoid
        silently mixing files from different model versions.
        """
        from histox.mil import PS3PathwayPreprocessor

        if getattr(experiment.method, "name", None) != "ps3":
            raise ValueError("PS3PretrainedBundle requires method='ps3'")
        if not isinstance(pathway_preprocessor, PS3PathwayPreprocessor):
            raise TypeError(
                "pathway_preprocessor must be a PS3PathwayPreprocessor"
            )
        if not pathway_preprocessor.fitted:
            raise RuntimeError("pathway_preprocessor must be fitted")

        method_config = experiment.method.public_config()
        if int(method_config.get("n_genes", -1)) != len(
            pathway_preprocessor.gene_names
        ):
            raise ValueError(
                "model n_genes does not match the pathway preprocessor"
            )
        configured_indices = [
            list(indices)
            for indices in method_config.get("pathway_gene_indices", ())
        ]
        if configured_indices != pathway_preprocessor.pathway_gene_indices:
            raise ValueError(
                "model pathway membership does not match the preprocessor"
            )

        root = Path(destination).expanduser()
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(
                "Bundle destination is not empty: {}".format(root)
            )
        root.mkdir(parents=True, exist_ok=True)
        preprocessing = root / "preprocessing"
        preprocessing.mkdir(parents=True, exist_ok=True)

        load_file, save_file = _safetensors()
        del load_file
        weights_path = root / "model.safetensors"
        state = {
            name: value.detach().cpu().contiguous()
            for name, value in experiment.build_model().state_dict().items()
        }
        save_file(state, str(weights_path))

        pathway_path = preprocessing / "pathway.json"
        _write_json(pathway_path, pathway_preprocessor.to_dict())

        histology = {
            "input_type": "precomputed_prototypes",
            "n_prototypes": int(
                method_config.get("n_histology_prototypes", 16)
            ),
            "prototype_dim": int(method_config["histology_dim"]),
            "patch_bags_supported": False,
            "raw_wsi_supported": False,
        }
        histology.update(dict(histology_config or {}))
        if histology_encoder is not None:
            if not getattr(histology_encoder, "fitted", False):
                raise RuntimeError("histology_encoder must be fitted")
            centers_path = preprocessing / "histology_centers.safetensors"
            save_file(
                {"centers": histology_encoder.centers.detach().cpu().contiguous()},
                str(centers_path),
            )
            histology.update({
                "input_type": "patch_features",
                "centers_file": str(centers_path.relative_to(root)),
                "n_prototypes": int(histology_encoder.n_prototypes),
                "feature_dim": int(histology_encoder.feature_dim),
                "prototype_dim": int(histology_encoder.output_dim),
                "patches_per_prototype": int(
                    histology_encoder.patches_per_prototype
                ),
                "max_iter": int(histology_encoder.max_iter),
                "n_init": int(histology_encoder.n_init),
                "em_iterations": int(histology_encoder.em_iterations),
                "tau": float(histology_encoder.tau),
                "covariance_prior": float(
                    histology_encoder.covariance_prior
                ),
                "random_state": int(histology_encoder.random_state),
                "min_variance": float(histology_encoder.min_variance),
                "patch_bags_supported": True,
                "raw_wsi_supported": False,
            })

        report = {
            "input_type": "precomputed_embeddings",
            "encoder": "vinid/plip",
            "revision": None,
            "max_tokens": 77,
            "embedding_dim": int(method_config.get("report_dim", 512)),
        }
        report.update(dict(report_config or {}))

        config = {
            "format": _BUNDLE_FORMAT,
            "format_version": _BUNDLE_VERSION,
            "task": {
                "name": experiment.task.name,
                "config": dict(experiment.task.config),
            },
            "method": {
                "name": experiment.method.name,
                "config": method_config,
            },
            "data_spec": {
                "modalities": list(experiment.data_spec.modalities),
                "fields": dict(experiment.data_spec.fields),
                "metadata": dict(experiment.data_spec.metadata),
            },
            "experiment_metadata": dict(experiment.metadata),
            "preprocessing": {
                "pathway_file": str(pathway_path.relative_to(root)),
                "histology": histology,
                "report": report,
            },
            "metadata": dict(metadata or {}),
        }
        config_path = root / "config.json"
        _write_json(config_path, config)

        files = [config_path, weights_path, pathway_path]
        centers_file = histology.get("centers_file")
        if centers_file is not None:
            files.append(root / centers_file)
        manifest = {
            "format": _BUNDLE_FORMAT,
            "format_version": _BUNDLE_VERSION,
            "files": {
                str(path.relative_to(root)): {
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in files
            },
        }
        _write_json(root / "manifest.json", manifest)
        return cls.load(root)

    @classmethod
    def load(
        cls,
        source: PathLike,
        device: Optional[Union[str, Any]] = None,
        verify: bool = True,
    ) -> "PS3PretrainedBundle":
        """Load and validate a local PS3 pretrained bundle."""
        from histox.mil import (
            PS3HistologyPrototypeEncoder,
            PS3PathwayPreprocessor,
        )
        from histox.workflows import DataSpec, Experiment

        root = Path(source).expanduser()
        config_path = root / "config.json"
        manifest_path = root / "manifest.json"
        weights_path = root / "model.safetensors"
        for required in (config_path, manifest_path, weights_path):
            if not required.is_file():
                raise FileNotFoundError(
                    "Bundle is missing {}".format(required.name)
                )

        manifest = _read_json(manifest_path)
        if (
            manifest.get("format") != _BUNDLE_FORMAT
            or manifest.get("format_version") != _BUNDLE_VERSION
        ):
            raise ValueError("Unsupported PS3 bundle manifest")
        if verify:
            for relative, expected in manifest.get("files", {}).items():
                path = root / relative
                if not path.is_file():
                    raise FileNotFoundError(
                        "Bundle manifest file is missing: {}".format(relative)
                    )
                if path.stat().st_size != int(expected["bytes"]):
                    raise ValueError(
                        "Bundle file size mismatch: {}".format(relative)
                    )
                if _sha256(path) != expected["sha256"]:
                    raise ValueError(
                        "Bundle checksum mismatch: {}".format(relative)
                    )

        config = _read_json(config_path)
        if (
            config.get("format") != _BUNDLE_FORMAT
            or config.get("format_version") != _BUNDLE_VERSION
        ):
            raise ValueError("Unsupported PS3 bundle configuration")
        if config.get("method", {}).get("name") != "ps3":
            raise ValueError("Bundle method must be 'ps3'")

        data_spec_payload = config["data_spec"]
        experiment = Experiment(
            task=config["task"]["name"],
            method="ps3",
            data_spec=DataSpec(
                modalities=data_spec_payload["modalities"],
                fields=data_spec_payload["fields"],
                metadata=data_spec_payload.get("metadata"),
            ),
            task_config=config["task"].get("config"),
            method_config=config["method"]["config"],
            metadata=config.get("experiment_metadata"),
        )
        model = experiment.build_model()
        load_file, save_file = _safetensors()
        del save_file
        state = load_file(str(weights_path), device="cpu")
        model.load_state_dict(state, strict=True)
        if device is not None:
            model.to(device)
        model.eval()

        preprocessing = config["preprocessing"]
        pathway = PS3PathwayPreprocessor.from_dict(
            _read_json(root / preprocessing["pathway_file"])
        )
        histology_config = dict(preprocessing["histology"])
        histology_encoder = None
        centers_file = histology_config.get("centers_file")
        if centers_file is not None:
            centers_state = load_file(
                str(root / centers_file), device="cpu"
            )
            histology_encoder = PS3HistologyPrototypeEncoder(
                n_prototypes=int(histology_config["n_prototypes"]),
                patches_per_prototype=int(
                    histology_config["patches_per_prototype"]
                ),
                max_iter=int(histology_config["max_iter"]),
                n_init=int(histology_config["n_init"]),
                em_iterations=int(histology_config["em_iterations"]),
                tau=float(histology_config["tau"]),
                covariance_prior=float(
                    histology_config["covariance_prior"]
                ),
                random_state=int(histology_config["random_state"]),
                min_variance=float(histology_config["min_variance"]),
            ).load_centers(centers_state["centers"])

        return cls(
            root=root,
            experiment=experiment,
            pathway_preprocessor=pathway,
            histology_encoder=histology_encoder,
            report_config=preprocessing["report"],
            config=config,
        )

    def transform_rna(self, expression):
        """Apply the training-fold gene order and scaler."""
        return self.pathway_preprocessor.transform(expression)

    def transform_histology(self, patch_features):
        """Convert patch features when fitted centers are bundled."""
        if self.histology_encoder is None:
            raise RuntimeError(
                "This bundle expects precomputed histology prototypes; "
                "fold-specific patch centers were not available."
            )
        return self.histology_encoder.transform(patch_features)

    def spatial_attention(
        self,
        patch_features,
        coordinates,
        gene_expression,
        report_embeddings,
        report_mask=None,
        source: str = "all",
        source_index=None,
    ) -> Dict[str, Any]:
        """Project PS3 histology-prototype attention onto WSI patches.

        Args:
            patch_features: One slide's PLIP patch features shaped
                ``[patches, feature_dim]``.
            coordinates: Patch-aligned coordinates shaped ``[patches, 2]`` or
                ``[patches, 4]``. For direct use with
                :func:`histox.util.location_heatmap`, the first two columns
                must be tile-center ``(x, y)`` slide coordinates.
            gene_expression: A one-row raw expression DataFrame or an already
                transformed tensor shaped ``[1, genes]``.
            report_embeddings: PLIP report-segment embeddings shaped
                ``[segments, features]`` or ``[1, segments, features]``.
            report_mask: Optional valid-segment mask.
            source: Attention query group: ``all``, ``pathway``, ``histology``,
                or ``report``.
            source_index: Optional prototype index within ``source``. A
                pathway name is also accepted when ``source='pathway'``.

        Returns:
            CPU tensors containing risk, joint attention, prototype scores,
            patch scores, normalized patch scores, responsibilities, and the
            unchanged coordinates.

        Notes:
            This is an attention-derived spatial concept map. Attention is not
            a causal estimate of each patch's contribution to predicted risk.
        """
        import pandas as pd
        import torch

        if self.histology_encoder is None:
            raise RuntimeError(
                "Spatial attention requires bundled histology centers."
            )
        patches = self.histology_encoder.flatten_feature_grid(patch_features)
        coordinate_tensor = torch.as_tensor(coordinates)
        if coordinate_tensor.ndim != 2 or coordinate_tensor.shape[1] not in (2, 4):
            raise ValueError("coordinates must have shape [patches, 2] or [patches, 4]")
        if len(coordinate_tensor) != len(patches):
            raise ValueError(
                "coordinates and patch_features must contain the same number of patches"
            )
        if not torch.isfinite(coordinate_tensor.float()).all():
            raise ValueError("coordinates contain non-finite values")

        histology, responsibilities = (
            self.histology_encoder.transform_with_assignments(patches)
        )
        if isinstance(gene_expression, pd.DataFrame):
            rna = self.transform_rna(gene_expression)
        else:
            rna = torch.as_tensor(gene_expression, dtype=torch.float32)
        if rna.ndim == 1:
            rna = rna.unsqueeze(0)
        if rna.ndim != 2 or rna.shape[0] != 1:
            raise ValueError("gene_expression must describe exactly one patient")

        reports = torch.as_tensor(report_embeddings, dtype=torch.float32)
        if reports.ndim == 2:
            reports = reports.unsqueeze(0)
        if reports.ndim != 3 or reports.shape[0] != 1:
            raise ValueError("report_embeddings must describe exactly one patient")
        if report_mask is None:
            mask = torch.ones(reports.shape[:2], dtype=torch.bool)
        else:
            mask = torch.as_tensor(report_mask, dtype=torch.bool)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            if tuple(mask.shape) != tuple(reports.shape[:2]):
                raise ValueError("report_mask must match report_embeddings")

        model = self.model
        device = next(model.parameters()).device
        model.eval()
        with torch.inference_mode():
            risk, attention, attention_logits = model(
                histology_prototypes=histology.unsqueeze(0).to(device),
                gene_expression=rna.to(device),
                report_embeddings=reports.to(device),
                report_mask=mask.to(device),
                return_attention=True,
                return_attention_logits=True,
            )

        method = self.config["method"]["config"]
        n_pathways = len(method["pathway_gene_indices"])
        n_histology = int(method.get("n_histology_prototypes", 16))
        n_reports = int(method.get("n_report_prototypes", 10))
        ranges = {
            "pathway": (0, n_pathways),
            "histology": (n_pathways, n_pathways + n_histology),
            "report": (
                n_pathways + n_histology,
                n_pathways + n_histology + n_reports,
            ),
        }
        if source not in ("all", "pathway", "histology", "report"):
            raise ValueError(
                "source must be 'all', 'pathway', 'histology', or 'report'"
            )
        matrix = attention[0]
        histology_slice = slice(n_pathways, n_pathways + n_histology)
        if source == "all":
            if source_index is not None:
                raise ValueError("source_index cannot be used with source='all'")
            query_rows = matrix
        else:
            start, stop = ranges[source]
            if source_index is None:
                query_rows = matrix[start:stop]
            else:
                if source == "pathway" and isinstance(source_index, str):
                    try:
                        source_index = self.pathway_preprocessor.pathway_names.index(
                            source_index
                        )
                    except ValueError as exc:
                        raise ValueError(
                            "Unknown pathway name: {}".format(source_index)
                        ) from exc
                index = int(source_index)
                if index < 0 or index >= stop - start:
                    raise IndexError(
                        "source_index is outside the {} prototype range".format(source)
                    )
                query_rows = matrix[start + index:start + index + 1]
        prototype_scores = query_rows[:, histology_slice].mean(dim=0)
        patch_scores = responsibilities.to(prototype_scores.device).matmul(
            prototype_scores
        )
        lower = patch_scores.min()
        upper = patch_scores.max()
        if torch.isclose(lower, upper):
            normalized = torch.zeros_like(patch_scores)
        else:
            normalized = (patch_scores - lower) / (upper - lower)

        return {
            "risk": risk.detach().cpu().reshape(-1),
            "attention": attention.detach().cpu(),
            "attention_logits": attention_logits.detach().cpu(),
            "prototype_scores": prototype_scores.detach().cpu(),
            "patch_scores": patch_scores.detach().cpu(),
            "normalized_patch_scores": normalized.detach().cpu(),
            "responsibilities": responsibilities.detach().cpu(),
            "coordinates": coordinate_tensor.detach().cpu(),
            "source": source,
            "source_index": source_index,
            "interpretation": "attention-derived spatial concept map",
        }

    def prototype_explanation(
        self,
        patch_features,
        coordinates,
        gene_expression,
        report_embeddings,
        report_mask=None,
        *,
        representative_patches: int = 3,
    ) -> Dict[str, Any]:
        """Build the patient-level prototype evidence used in PS3 Figure 2.

        The returned categorical assignments, prototype proportions, and
        representative patches all derive from the same PANTHER responsibility
        matrix. Cross-modal blocks use the final PS3 fusion attention matrix,
        whose query/key token order is pathway, histology, then report.

        Args:
            patch_features: One slide's patch features.
            coordinates: Patch-aligned center coordinates with either two or
                four columns. The first two columns must be base-level WSI
                ``(x, y)`` positions.
            gene_expression: One patient's raw or transformed RNA values.
            report_embeddings: One patient's report-segment embeddings.
            report_mask: Optional valid-segment mask.
            representative_patches: Maximum number of high-responsibility
                patches retained for each histology prototype.

        Returns:
            A dictionary containing the base spatial result plus hard prototype
            assignments, proportions, representative patch indices, raw and
            target-normalized cross-modal attention blocks, and the report-wise
            standard-deviation summaries described in the PS3 supplement.

        Notes:
            The bundle's report tokens are selected diagnostic prototypes. The
            report standard deviations are therefore computed over those valid
            diagnostic prototypes, not over padded text positions.
        """
        import torch

        if representative_patches <= 0:
            raise ValueError("representative_patches must be positive")
        result = self.spatial_attention(
            patch_features=patch_features,
            coordinates=coordinates,
            gene_expression=gene_expression,
            report_embeddings=report_embeddings,
            report_mask=report_mask,
            source="all",
        )
        responsibilities = torch.as_tensor(
            result["responsibilities"], dtype=torch.float32
        )
        assignments = responsibilities.argmax(dim=1)
        confidence = responsibilities.max(dim=1).values
        n_histology = responsibilities.shape[1]
        counts = torch.bincount(assignments, minlength=n_histology)
        proportions = counts.to(torch.float32) / max(int(len(assignments)), 1)

        representative_indices = torch.full(
            (n_histology, representative_patches), -1, dtype=torch.long
        )
        representative_scores = torch.full(
            (n_histology, representative_patches), float("nan")
        )
        for prototype in range(n_histology):
            assigned = torch.nonzero(
                assignments == prototype, as_tuple=False
            ).flatten()
            if assigned.numel() == 0:
                continue
            scores = responsibilities[assigned, prototype]
            keep = min(representative_patches, int(assigned.numel()))
            ranked = torch.topk(scores, k=keep, largest=True).indices
            representative_indices[prototype, :keep] = assigned[ranked]
            representative_scores[prototype, :keep] = scores[ranked]

        method = self.config["method"]["config"]
        n_pathways = len(method["pathway_gene_indices"])
        n_reports = int(method.get("n_report_prototypes", 10))
        matrix = torch.as_tensor(result["attention"])[0].to(torch.float32)
        logits = torch.as_tensor(result["attention_logits"])[0].to(
            torch.float32
        )
        pathway_slice = slice(0, n_pathways)
        histology_slice = slice(n_pathways, n_pathways + n_histology)
        report_slice = slice(
            n_pathways + n_histology,
            n_pathways + n_histology + n_reports,
        )

        pathway_to_histology = matrix[pathway_slice, histology_slice]
        histology_to_pathway = matrix[histology_slice, pathway_slice]
        report_to_histology = matrix[report_slice, histology_slice]
        report_to_pathway = matrix[report_slice, pathway_slice]
        pathway_to_histology_logits = logits[pathway_slice, histology_slice]
        histology_to_pathway_logits = logits[histology_slice, pathway_slice]
        report_to_histology_logits = logits[report_slice, histology_slice]
        report_to_pathway_logits = logits[report_slice, pathway_slice]
        pathway_to_histology_normalized = torch.softmax(
            pathway_to_histology_logits, dim=-1
        )
        histology_to_pathway_normalized = torch.softmax(
            histology_to_pathway_logits, dim=-1
        )
        report_to_histology_normalized = torch.softmax(
            report_to_histology_logits, dim=-1
        )
        report_to_pathway_normalized = torch.softmax(
            report_to_pathway_logits, dim=-1
        )

        result.update({
            "prototype_assignments": assignments,
            "prototype_assignment_confidence": confidence,
            "prototype_counts": counts,
            "prototype_proportions": proportions,
            "representative_patch_indices": representative_indices,
            "representative_patch_scores": representative_scores,
            "pathway_names": list(
                self.pathway_preprocessor.pathway_names
            ),
            "pathway_to_histology": pathway_to_histology,
            "pathway_to_histology_logits": pathway_to_histology_logits,
            "pathway_to_histology_normalized": (
                pathway_to_histology_normalized
            ),
            "histology_to_pathway": histology_to_pathway,
            "histology_to_pathway_logits": histology_to_pathway_logits,
            "histology_to_pathway_normalized": (
                histology_to_pathway_normalized
            ),
            "report_to_histology": report_to_histology,
            "report_to_histology_logits": report_to_histology_logits,
            "report_to_histology_normalized": (
                report_to_histology_normalized
            ),
            "report_to_histology_std": (
                report_to_histology_normalized.std(dim=0, unbiased=False)
            ),
            "report_to_pathway": report_to_pathway,
            "report_to_pathway_logits": report_to_pathway_logits,
            "report_to_pathway_normalized": report_to_pathway_normalized,
            "report_to_pathway_std": (
                report_to_pathway_normalized.std(dim=0, unbiased=False)
            ),
            "interpretation": "paper-style PS3 prototype evidence",
        })
        return result

    @staticmethod
    def save_prototype_map(
        result: Mapping[str, Any],
        slide: PathLike,
        tile_px: int,
        tile_um,
        destination: PathLike,
        *,
        alpha: float = 0.62,
        thumbnail_mpp: float = 5.0,
        colors=None,
    ) -> Path:
        """Render the closest-prototype category for every retained WSI patch."""
        import numpy as np
        import torch
        from PIL import Image, ImageDraw
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        import histox as hx

        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        coordinates = torch.as_tensor(result["coordinates"]).cpu().numpy()
        assignments = torch.as_tensor(
            result["prototype_assignments"], dtype=torch.long
        ).cpu().numpy()
        if coordinates.ndim != 2 or coordinates.shape[1] not in (2, 4):
            raise ValueError("result coordinates must have two or four columns")
        if len(coordinates) != len(assignments):
            raise ValueError("coordinates and prototype assignments are not aligned")
        n_prototypes = int(
            torch.as_tensor(result["prototype_proportions"]).numel()
        )
        if colors is None:
            palette = plt.get_cmap("tab20", n_prototypes)
            colors = [
                tuple(int(round(channel * 255)) for channel in palette(index)[:3])
                for index in range(n_prototypes)
            ]
        if len(colors) < n_prototypes:
            raise ValueError("colors must contain one color per prototype")

        wsi = hx.WSI(str(Path(slide).expanduser()), tile_px, tile_um, verbose=False)
        thumbnail = wsi.thumb(mpp=thumbnail_mpp).convert("RGBA")
        overlay = Image.new("RGBA", thumbnail.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        scale_x = thumbnail.width / float(wsi.dimensions[0])
        scale_y = thumbnail.height / float(wsi.dimensions[1])
        half = float(wsi.full_extract_px) / 2.0
        opacity = int(round(alpha * 255))
        for (center_x, center_y), prototype in zip(
            coordinates[:, :2], assignments
        ):
            left = int(round((float(center_x) - half) * scale_x))
            top = int(round((float(center_y) - half) * scale_y))
            right = int(round((float(center_x) + half) * scale_x))
            bottom = int(round((float(center_y) + half) * scale_y))
            color = tuple(colors[int(prototype)])
            draw.rectangle(
                (left, top, right, bottom), fill=(*color[:3], opacity)
            )
        composite = Image.alpha_composite(thumbnail, overlay)

        output = Path(destination).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure, axis = plt.subplots(figsize=(14, 8))
        axis.imshow(composite)
        axis.set_axis_off()
        handles = [
            Patch(
                facecolor=np.asarray(colors[index][:3]) / 255.0,
                label="C{}".format(index),
            )
            for index in range(n_prototypes)
        ]
        axis.legend(
            handles=handles,
            ncol=min(8, n_prototypes),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            frameon=False,
        )
        figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        return output

    @staticmethod
    def save_representative_patches(
        result: Mapping[str, Any],
        slide: PathLike,
        tile_px: int,
        tile_um,
        destination: PathLike,
        *,
        prototype_indices=None,
        top_prototypes: int = 8,
    ) -> Path:
        """Save high-responsibility WSI patches for important prototypes."""
        import numpy as np
        import torch
        import matplotlib.pyplot as plt
        import histox as hx

        coordinates = torch.as_tensor(result["coordinates"]).cpu()
        representatives = torch.as_tensor(
            result["representative_patch_indices"], dtype=torch.long
        ).cpu()
        proportions = torch.as_tensor(
            result["prototype_proportions"], dtype=torch.float32
        ).cpu()
        if coordinates.ndim != 2 or coordinates.shape[1] not in (2, 4):
            raise ValueError("result coordinates must have two or four columns")
        if prototype_indices is None:
            populated = torch.nonzero(proportions > 0, as_tuple=False).flatten()
            ranked = populated[
                torch.argsort(proportions[populated], descending=True)
            ]
            selected = ranked[: min(top_prototypes, int(ranked.numel()))].tolist()
        else:
            selected = [int(index) for index in prototype_indices]
        if not selected:
            raise ValueError("No populated prototypes are available")
        if any(index < 0 or index >= len(proportions) for index in selected):
            raise IndexError("prototype_indices contains an invalid prototype")

        wsi = hx.WSI(str(Path(slide).expanduser()), tile_px, tile_um, verbose=False)
        rows = representatives.shape[1]
        figure, axes = plt.subplots(
            rows,
            len(selected),
            figsize=(2.15 * len(selected), 2.15 * rows),
            squeeze=False,
        )
        palette = plt.get_cmap("tab20", len(proportions))
        for column, prototype in enumerate(selected):
            axes[0, column].set_title(
                "C{} ({:.1%})".format(prototype, float(proportions[prototype]))
            )
            for row in range(rows):
                axis = axes[row, column]
                patch_index = int(representatives[prototype, row])
                if patch_index >= 0:
                    center_x = float(coordinates[patch_index, 0])
                    center_y = float(coordinates[patch_index, 1])
                    half = float(wsi.full_extract_px) / 2.0
                    top_left = (
                        int(round(center_x - half)),
                        int(round(center_y - half)),
                    )
                    patch = wsi.slide.read_from_pyramid(
                        top_left=top_left,
                        window_size=(wsi.full_extract_px, wsi.full_extract_px),
                        target_size=(wsi.tile_px, wsi.tile_px),
                        convert="numpy",
                        flatten=True,
                        pad_missing=True,
                    )
                    axis.imshow(np.asarray(patch)[..., :3])
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_color(palette(prototype))
                    spine.set_linewidth(3)
        output = Path(destination).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        return output

    @staticmethod
    def save_prototype_proportions(
        result: Mapping[str, Any],
        destination: PathLike,
    ) -> Path:
        """Plot the per-slide fraction of patches assigned to each prototype."""
        import numpy as np
        import torch
        import matplotlib.pyplot as plt

        proportions = torch.as_tensor(
            result["prototype_proportions"], dtype=torch.float32
        ).cpu().numpy()
        output = Path(destination).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure, axis = plt.subplots(figsize=(9, 3.8))
        palette = plt.get_cmap("tab20", len(proportions))
        positions = np.arange(len(proportions))
        axis.bar(positions, proportions, color=palette(positions))
        axis.set_xticks(positions, ["C{}".format(i) for i in positions])
        axis.set_ylabel("Proportion of assigned patches")
        axis.set_xlabel("Histology prototype")
        axis.spines[["top", "right"]].set_visible(False)
        figure.tight_layout()
        figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        return output

    @staticmethod
    def save_cross_modal_attention(
        result: Mapping[str, Any],
        destination: PathLike,
        *,
        pathway,
        histology_prototype: Optional[int] = None,
        top_histology: int = 6,
        top_pathways: int = 10,
    ) -> Path:
        """Plot the four pathway/report/histology rankings from PS3 Figure 2."""
        import numpy as np
        import torch
        import matplotlib.pyplot as plt

        pathway_names = list(result["pathway_names"])
        if isinstance(pathway, str):
            try:
                pathway_index = pathway_names.index(pathway)
            except ValueError as exc:
                raise ValueError("Unknown pathway name: {}".format(pathway)) from exc
        else:
            pathway_index = int(pathway)
        if pathway_index < 0 or pathway_index >= len(pathway_names):
            raise IndexError("pathway is outside the available pathway range")
        proportions = torch.as_tensor(result["prototype_proportions"])
        if histology_prototype is None:
            histology_prototype = int(proportions.argmax())
        histology_prototype = int(histology_prototype)
        if histology_prototype < 0 or histology_prototype >= len(proportions):
            raise IndexError("histology_prototype is outside the available range")

        p_to_h = torch.as_tensor(
            result["pathway_to_histology_normalized"]
        )[pathway_index]
        report_to_h = torch.as_tensor(result["report_to_histology_std"])
        h_to_p = torch.as_tensor(
            result["histology_to_pathway_normalized"]
        )[histology_prototype]
        report_to_p = torch.as_tensor(result["report_to_pathway_std"])

        def pathway_label(name):
            label = str(name)
            if label.upper().startswith("HALLMARK_"):
                label = label[len("HALLMARK_"):]
            return label.replace("_", " ")

        def top(values, count):
            count = min(int(count), int(values.numel()))
            return torch.topk(values, k=count, largest=True).indices.cpu().numpy()

        figure, axes = plt.subplots(2, 2, figsize=(14, 9))
        panels = (
            (
                axes[0, 0], p_to_h, top(p_to_h, top_histology),
                ["C{}".format(i) for i in range(len(p_to_h))],
                "{} to histology".format(
                    pathway_label(pathway_names[pathway_index])
                ),
                "Normalized attention",
            ),
            (
                axes[0, 1], report_to_h,
                top(report_to_h, top_histology),
                ["C{}".format(i) for i in range(len(report_to_h))],
                "Pathology report to histology",
                "Std. dev. of normalized attention",
            ),
            (
                axes[1, 0], h_to_p, top(h_to_p, top_pathways),
                [pathway_label(name) for name in pathway_names],
                "Histology C{} to pathways".format(histology_prototype),
                "Normalized attention",
            ),
            (
                axes[1, 1], report_to_p,
                top(report_to_p, top_pathways),
                [pathway_label(name) for name in pathway_names],
                "Pathology report to pathways",
                "Std. dev. of normalized attention",
            ),
        )
        for axis, values, indices, labels, title, ylabel in panels:
            ordered = indices[::-1].copy()
            axis.barh(
                np.arange(len(ordered)),
                values[ordered].cpu().numpy(),
                color=plt.get_cmap("viridis")(np.linspace(0.25, 0.85, len(ordered))),
            )
            axis.set_yticks(
                np.arange(len(ordered)), [labels[index] for index in ordered]
            )
            axis.set_xlabel(ylabel)
            axis.set_title(title)
            axis.tick_params(axis="y", labelsize=8)
            axis.spines[["top", "right"]].set_visible(False)
        output = Path(destination).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        return output

    @staticmethod
    def save_spatial_heatmap(
        result: Mapping[str, Any],
        slide: PathLike,
        tile_px: int,
        tile_um,
        destination: PathLike,
        *,
        cmap: str = "magma",
        interpolation: Optional[str] = "bicubic",
        background: str = "min",
    ) -> Path:
        """Overlay a :meth:`spatial_attention` result on its source WSI."""
        import numpy as np
        import torch
        from histox.util import location_heatmap

        coordinates = torch.as_tensor(result["coordinates"]).cpu().numpy()
        if coordinates.ndim != 2 or coordinates.shape[1] not in (2, 4):
            raise ValueError("result coordinates must have two or four columns")
        values = torch.as_tensor(
            result["normalized_patch_scores"], dtype=torch.float32
        ).cpu().numpy()
        if len(coordinates) != len(values):
            raise ValueError("result coordinates and scores are not aligned")
        output = Path(destination).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        location_heatmap(
            locations=np.asarray(coordinates[:, :2]),
            values=np.asarray(values),
            slide=str(Path(slide).expanduser()),
            tile_px=int(tile_px),
            tile_um=tile_um,
            filename=str(output),
            interpolation=interpolation,
            cmap=cmap,
            background=background,
        )
        return output

    def predict(self, loader, device=None):
        """Run patient-level forward inference."""
        return self.experiment.predict(loader=loader, device=device)

    def explain(self, loader, device=None, output=None):
        """Return or save joint prototype attention."""
        return self.experiment.explain(
            loader=loader,
            device=device,
            output=output,
        )


__all__ = ["PS3PretrainedBundle"]
