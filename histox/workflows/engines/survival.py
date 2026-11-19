"""Reusable PyTorch training and evaluation for time-to-event models."""

import csv
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Union

import numpy as np
import torch

from .._types import RunResult


PathLike = Union[str, Path]


def survival_concordance_index(event, duration, risk) -> float:
    """Return Harrell's C-index for predicted log-risk.

    HistoX's concordance implementation expects a predicted survival score,
    where a larger value means longer survival. A survival model's log-risk
    has the opposite direction, so this function passes ``-risk``.
    """
    from histox.stats.concordance import concordance_index

    event_array = np.asarray(event).reshape(-1)
    duration_array = np.asarray(duration, dtype=float).reshape(-1)
    risk_array = np.asarray(risk, dtype=float).reshape(-1)
    if not (
        len(event_array) == len(duration_array) == len(risk_array)
    ):
        raise ValueError("event, duration, and risk must have equal lengths")
    return float(
        concordance_index(duration_array, -risk_array, event_array)
    )


class SurvivalEngine:
    """Train and evaluate one PyTorch survival model.

    The engine owns the reusable optimization and result-writing mechanics.
    A method adapter supplies the model-specific ``forward_batch`` callable.

    Args:
        model: PyTorch module returning one log-risk value per patient.
        forward_batch: Callable receiving ``(model, batch)``.
        loss_fn: Callable receiving ``(risk, duration, event)``.
        device: Torch device. Defaults to CUDA when available, otherwise CPU.
        duration_key: Batch key containing observed times.
        event_key: Batch key containing event indicators.
        patient_id_key: Optional batch key containing patient identifiers.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        forward_batch: Callable[..., torch.Tensor],
        loss_fn: Callable[..., torch.Tensor],
        device=None,
        duration_key: str = "duration",
        event_key: str = "event",
        patient_id_key: str = "patient_id",
    ) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if not callable(forward_batch) or not callable(loss_fn):
            raise TypeError("forward_batch and loss_fn must be callable")
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.forward_batch = forward_batch
        self.loss_fn = loss_fn
        self.duration_key = str(duration_key)
        self.event_key = str(event_key)
        self.patient_id_key = str(patient_id_key)

    def move_batch(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        """Move tensor values in a batch to the engine device."""
        return {
            key: value.to(self.device)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in batch.items()
        }

    def _risk(self, batch: Mapping[str, Any]) -> torch.Tensor:
        risk = self.forward_batch(self.model, batch)
        if isinstance(risk, tuple):
            risk = risk[0]
        if not isinstance(risk, torch.Tensor):
            raise TypeError("forward_batch must return a tensor or tensor tuple")
        return risk.reshape(-1)

    @torch.inference_mode()
    def predict_arrays(self, loader) -> Dict[str, Any]:
        """Collect patient identifiers, log-risk, and optional labels."""
        self.model.eval()
        patient_ids = []
        risks = []
        durations = []
        events = []
        for raw_batch in loader:
            batch = self.move_batch(raw_batch)
            risk = self._risk(batch)
            risks.append(risk.detach().cpu().numpy())
            raw_ids = raw_batch.get(self.patient_id_key)
            if raw_ids is None:
                start = len(patient_ids)
                patient_ids.extend(
                    str(index) for index in range(start, start + len(risk))
                )
            elif isinstance(raw_ids, (str, bytes)):
                patient_ids.append(str(raw_ids))
            else:
                patient_ids.extend(str(value) for value in raw_ids)
            if self.duration_key in batch:
                durations.append(
                    batch[self.duration_key].detach().cpu().numpy()
                )
            if self.event_key in batch:
                events.append(batch[self.event_key].detach().cpu().numpy())
        if not risks:
            raise ValueError("loader did not yield any batches")
        result = {
            "patient_ids": patient_ids,
            "risk": np.concatenate(risks),
        }
        if durations:
            result["duration"] = np.concatenate(durations)
        if events:
            result["event"] = np.concatenate(events)
        return result

    def predict(self, loader) -> RunResult:
        """Return patient-level log-risk predictions."""
        return RunResult(stage="predict", predictions=self.predict_arrays(loader))

    def evaluate(self, loader) -> RunResult:
        """Return predictions and Harrell's C-index."""
        predictions = self.predict_arrays(loader)
        missing = [
            name
            for name in ("duration", "event")
            if name not in predictions
        ]
        if missing:
            raise KeyError(
                "Survival evaluation requires batch keys {}".format(missing)
            )
        score = survival_concordance_index(
            predictions["event"],
            predictions["duration"],
            predictions["risk"],
        )
        return RunResult(
            stage="evaluate",
            metrics={"c_index": score},
            predictions=predictions,
        )

    def fit(
        self,
        train_loader,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        evaluation_loaders: Optional[Mapping[str, Any]] = None,
        output: Optional[PathLike] = None,
        progress: Optional[Callable[[Mapping[str, Any]], None]] = None,
        save_outputs: bool = True,
    ) -> RunResult:
        """Fit the model and optionally evaluate named loaders.

        The scheduler is stepped once after every optimizer step, matching the
        validated PS3 CRC training program.
        """
        if int(epochs) <= 0:
            raise ValueError("epochs must be positive")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch optimizer")
        history = []
        for epoch in range(int(epochs)):
            self.model.train()
            losses = []
            for raw_batch in train_loader:
                batch = self.move_batch(raw_batch)
                if self.duration_key not in batch or self.event_key not in batch:
                    raise KeyError(
                        "Training batches require {!r} and {!r}".format(
                            self.duration_key, self.event_key
                        )
                    )
                if not torch.any(batch[self.event_key] > 0):
                    raise ValueError(
                        "Cox training requires an observed event in every batch"
                    )
                optimizer.zero_grad(set_to_none=True)
                risk = self._risk(batch)
                loss = self.loss_fn(
                    risk,
                    batch[self.duration_key],
                    batch[self.event_key],
                )
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "Epoch {} produced a non-finite loss".format(epoch + 1)
                    )
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                losses.append(float(loss.detach().cpu()))
            if not losses:
                raise ValueError("train_loader did not yield any batches")
            row = {
                "epoch": epoch + 1,
                "mean_batch_loss": float(np.mean(losses)),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            history.append(row)
            if progress is not None:
                progress(dict(row))

        named_predictions = {}
        metrics = {
            "epochs": int(epochs),
            "final_loss": history[-1]["mean_batch_loss"],
        }
        for name, loader in dict(evaluation_loaders or {}).items():
            evaluation = self.evaluate(loader)
            named_predictions[str(name)] = evaluation.predictions
            metrics["{}_c_index".format(name)] = evaluation.metrics["c_index"]

        result = RunResult(
            stage="fit",
            metrics=metrics,
            predictions=named_predictions,
            metadata={
                "device": str(self.device),
                "history": history,
                "optimizer": optimizer.__class__.__name__,
                "scheduler": (
                    scheduler.__class__.__name__
                    if scheduler is not None
                    else None
                ),
            },
        )
        if output is not None and save_outputs:
            self.write_outputs(result, output)
        return result

    def write_outputs(self, result: RunResult, output: PathLike) -> None:
        """Write a checkpoint, history, and named prediction tables."""
        destination = Path(output).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        checkpoint = destination / "model_state.pt"
        torch.save(self.model.state_dict(), checkpoint)
        result.add_artifact("checkpoint", "checkpoint", checkpoint)

        history = list(result.metadata.get("history", ()))
        if history:
            history_path = destination / "history.csv"
            with history_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
            result.add_artifact("history", "table", history_path)

        for name, prediction in dict(result.predictions or {}).items():
            path = destination / "{}_predictions.csv".format(name)
            self.write_predictions(path, prediction)
            result.add_artifact(
                "{}_predictions".format(name),
                "predictions",
                path,
            )

    @staticmethod
    def write_predictions(path: PathLike, prediction: Mapping[str, Any]) -> None:
        """Write patient-level survival predictions as CSV."""
        required = ("patient_ids", "risk")
        missing = [name for name in required if name not in prediction]
        if missing:
            raise KeyError("Prediction output is missing {}".format(missing))
        patient_ids = list(prediction["patient_ids"])
        risk = np.asarray(prediction["risk"]).reshape(-1)
        duration = prediction.get("duration")
        event = prediction.get("event")
        if len(patient_ids) != len(risk):
            raise ValueError("patient_ids and risk must have equal lengths")
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["patient_id"]
        if duration is not None:
            fieldnames.append("duration")
            duration = np.asarray(duration).reshape(-1)
            if len(duration) != len(risk):
                raise ValueError("duration and risk must have equal lengths")
        if event is not None:
            fieldnames.append("event")
            event = np.asarray(event).reshape(-1)
            if len(event) != len(risk):
                raise ValueError("event and risk must have equal lengths")
        fieldnames.append("log_risk")
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index, patient_id in enumerate(patient_ids):
                row = {
                    "patient_id": patient_id,
                    "log_risk": float(risk[index]),
                }
                if duration is not None:
                    row["duration"] = float(duration[index])
                if event is not None:
                    row["event"] = float(event[index])
                writer.writerow(row)
