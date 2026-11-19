"""Workflow adapter for the HistoX PS3 survival model."""

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .._base import Method
from .._registry import register_method
from .._types import DataSpec, RunResult


@register_method("ps3")
class PS3Method(Method):
    """Construct and invoke the existing HistoX PS3 implementation.

    Required model configuration keys are ``n_genes``,
    ``pathway_gene_indices``, and ``histology_dim``. Optional keys are passed
    to :class:`histox.mil.models.PS3SurvivalModel`.
    """

    name = "ps3"
    supported_tasks = ("survival",)
    required_modalities = ("histology", "report", "rna")
    supported_explainers = ("joint_attention",)

    _required_config = ("n_genes", "pathway_gene_indices", "histology_dim")

    def build_model(self, data_spec: DataSpec):
        """Build ``PS3SurvivalModel`` without importing PyTorch at package load."""
        missing = [
            name for name in self._required_config if name not in self.config
        ]
        if missing:
            raise ValueError(
                "PS3 model configuration is missing {}".format(missing)
            )
        from histox.mil.models import PS3SurvivalModel

        return PS3SurvivalModel(**self.config)

    def forward_batch(self, model, batch: Dict[str, Any], return_attention=False):
        """Map the standard multimodal batch names to the PS3 forward call."""
        aliases = {
            "histology": ("histology",),
            "rna": ("rna", "genes"),
            "report": ("report", "reports"),
        }
        resolved = {}
        for name, candidates in aliases.items():
            resolved[name] = next(
                (batch[key] for key in candidates if key in batch), None
            )
        missing = [name for name, value in resolved.items() if value is None]
        if missing:
            raise KeyError("PS3 batch is missing {}".format(missing))
        return model(
            histology_prototypes=resolved["histology"],
            gene_expression=resolved["rna"],
            report_embeddings=resolved["report"],
            report_mask=batch.get("report_mask"),
            histology_mask=batch.get("histology_mask"),
            return_attention=return_attention,
        )

    @staticmethod
    def build_optimizer(model, learning_rate: float, weight_decay: float):
        """Build the AdamW parameter groups used by the CRC reproduction."""
        import torch

        def exclude(name, parameter):
            return (
                parameter.ndim < 2
                or "bn" in name
                or "ln" in name
                or "bias" in name
                or "logit_scale" in name
            )

        named = list(model.named_parameters())
        no_decay = [
            parameter
            for name, parameter in named
            if parameter.requires_grad and exclude(name, parameter)
        ]
        decay = [
            parameter
            for name, parameter in named
            if parameter.requires_grad and not exclude(name, parameter)
        ]
        return torch.optim.AdamW(
            [
                {"params": no_decay, "weight_decay": 0.0},
                {"params": decay, "weight_decay": float(weight_decay)},
            ],
            lr=float(learning_rate),
        )

    def _engine(self, experiment, device=None):
        from histox.mil.models import CoxPHLoss
        from histox.workflows.engines import SurvivalEngine

        return SurvivalEngine(
            model=experiment.build_model(),
            forward_batch=self.forward_batch,
            loss_fn=CoxPHLoss(),
            device=device,
        )

    def stage_fit(
        self,
        experiment,
        train_loader,
        epochs: int = 50,
        evaluation_loaders: Optional[Mapping[str, Any]] = None,
        device=None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        optimizer=None,
        scheduler="cosine",
        warmup_steps: Optional[int] = None,
        output=None,
        progress=None,
        save_outputs: bool = True,
    ) -> RunResult:
        """Train PS3 with the shared survival engine."""
        engine = self._engine(experiment, device=device)
        if optimizer is None:
            optimizer = self.build_optimizer(
                engine.model,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )
        if scheduler == "cosine":
            from transformers import get_cosine_schedule_with_warmup

            steps_per_epoch = len(train_loader)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=(
                    steps_per_epoch
                    if warmup_steps is None
                    else int(warmup_steps)
                ),
                num_training_steps=steps_per_epoch * int(epochs),
            )
        elif scheduler is False:
            scheduler = None
        destination = output if output is not None else experiment.output
        result = engine.fit(
            train_loader=train_loader,
            epochs=epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            evaluation_loaders=evaluation_loaders,
            output=destination,
            progress=progress,
            save_outputs=save_outputs,
        )
        result.metadata["learning_rate"] = float(learning_rate)
        result.metadata["weight_decay"] = float(weight_decay)
        return result

    def stage_evaluate(
        self,
        experiment,
        loader,
        device=None,
    ) -> RunResult:
        """Evaluate the cached PS3 model on one patient loader."""
        return self._engine(experiment, device=device).evaluate(loader)

    def stage_predict(
        self,
        experiment,
        loader,
        device=None,
    ) -> RunResult:
        """Predict patient log-risk with the cached PS3 model."""
        return self._engine(experiment, device=device).predict(loader)

    def stage_explain(
        self,
        experiment,
        loader,
        device=None,
        output=None,
    ) -> RunResult:
        """Collect joint prototype attention for later visualization."""
        import numpy as np
        import torch

        engine = self._engine(experiment, device=device)
        engine.model.eval()
        patient_ids = []
        risks = []
        attentions = []
        with torch.inference_mode():
            for raw_batch in loader:
                batch = engine.move_batch(raw_batch)
                risk, attention = self.forward_batch(
                    engine.model,
                    batch,
                    return_attention=True,
                )
                risks.append(risk.reshape(-1).detach().cpu().numpy())
                attentions.append(attention.detach().cpu().numpy())
                raw_ids = raw_batch.get("patient_id")
                if raw_ids is None:
                    start = len(patient_ids)
                    patient_ids.extend(
                        str(index)
                        for index in range(start, start + len(risk))
                    )
                elif isinstance(raw_ids, (str, bytes)):
                    patient_ids.append(str(raw_ids))
                else:
                    patient_ids.extend(str(value) for value in raw_ids)
        if not attentions:
            raise ValueError("loader did not yield any batches")
        risk_array = np.concatenate(risks)
        attention_array = np.concatenate(attentions)
        result = RunResult(
            stage="explain",
            predictions={"patient_ids": patient_ids, "risk": risk_array},
        )
        destination = output if output is not None else experiment.output
        if destination is None:
            result.add_artifact(
                "joint_attention",
                "joint_attention",
                data=attention_array,
                metadata={"patient_ids": patient_ids},
            )
        else:
            path = Path(destination).expanduser() / "joint_attention.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                patient_ids=np.asarray(patient_ids),
                risk=risk_array,
                attention=attention_array,
            )
            result.add_artifact(
                "joint_attention",
                "joint_attention",
                path=path,
                metadata={
                    "patient_count": len(patient_ids),
                    "attention_shape": list(attention_array.shape),
                },
            )
        return result
