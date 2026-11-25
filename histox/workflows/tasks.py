"""Built-in HistoX workflow tasks."""

from typing import Any, Dict, Optional, Sequence

import numpy as np

from ._base import Task
from ._registry import register_task


@register_task("survival")
class SurvivalTask(Task):
    """Time-to-event prediction task.

    Args:
        duration_field: Semantic field name mapped to observed time.
        event_field: Semantic field name mapped to the event indicator.
        metrics: Metric names recorded by default.
    """

    name = "survival"

    def __init__(
        self,
        duration_field: str = "duration",
        event_field: str = "event",
        metrics=None,
        **config
    ) -> None:
        duration_field = str(duration_field).strip()
        event_field = str(event_field).strip()
        if not duration_field or not event_field:
            raise ValueError("Survival field names must be non-empty")
        if duration_field == event_field:
            raise ValueError("Duration and event fields must be different")
        selected_metrics = tuple(metrics or ("c_index",))
        super().__init__(
            duration_field=duration_field,
            event_field=event_field,
            metrics=selected_metrics,
            **config
        )
        self.duration_field = duration_field
        self.event_field = event_field
        self.required_fields = (duration_field, event_field)
        self.default_metrics = selected_metrics


@register_task("classification")
class ClassificationTask(Task):
    """Predict a categorical label per slide (or patient).

    Args:
        label_field: Semantic field name mapped to the label column.
        metrics: Metric names computed by :meth:`score`. Defaults to
            ``("auroc", "accuracy")``.
    """

    name = "classification"
    available_metrics = ("auroc", "accuracy", "balanced_accuracy")

    def __init__(self, label_field: str = "label", metrics=None, **config) -> None:
        label_field = str(label_field).strip()
        if not label_field:
            raise ValueError("label_field must be non-empty")
        selected = tuple(metrics or ("auroc", "accuracy"))
        unknown = [m for m in selected if m not in self.available_metrics]
        if unknown:
            raise ValueError(f"Unknown metrics {unknown}; available: {self.available_metrics}")
        super().__init__(label_field=label_field, metrics=selected, **config)
        self.label_field = label_field
        self.required_fields = (label_field,)
        self.default_metrics = selected

    def score(self, y_true: Sequence[int], y_prob: np.ndarray) -> Dict[str, Any]:
        """Compute the task metrics.

        Args:
            y_true: Integer class labels, shape ``(N,)``.
            y_prob: Class probabilities, shape ``(N, C)``.
        """
        from sklearn import metrics as skm
        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob, dtype=float)
        if y_prob.ndim != 2 or len(y_prob) != len(y_true):
            raise ValueError("y_prob must have shape (N, C) matching y_true")
        y_pred = y_prob.argmax(axis=1)
        out: Dict[str, Any] = {"n": int(len(y_true))}
        for name in self.default_metrics:
            if name == "accuracy":
                out[name] = float(skm.accuracy_score(y_true, y_pred))
            elif name == "balanced_accuracy":
                out[name] = float(skm.balanced_accuracy_score(y_true, y_pred))
            elif name == "auroc":
                out[name] = _auroc(y_true, y_prob)
        return out


def _auroc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    """Binary AUROC, or one-vs-rest macro AUROC; None if only one class present."""
    from sklearn.metrics import roc_auc_score
    present = np.unique(y_true)
    if len(present) < 2:
        return None
    if y_prob.shape[1] == 2:
        return float(roc_auc_score(y_true, y_prob[:, 1]))
    return float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro",
                               labels=list(range(y_prob.shape[1]))))


@register_task("regression")
class RegressionTask(Task):
    """Predict a continuous value per slide (or patient).

    Args:
        target_field: Semantic field name mapped to the target column.
        metrics: Metric names computed by :meth:`score`. Defaults to
            ``("r2", "mae")``.
    """

    name = "regression"
    available_metrics = ("r2", "mae", "mse", "pearson")

    def __init__(self, target_field: str = "target", metrics=None, **config) -> None:
        target_field = str(target_field).strip()
        if not target_field:
            raise ValueError("target_field must be non-empty")
        selected = tuple(metrics or ("r2", "mae"))
        unknown = [m for m in selected if m not in self.available_metrics]
        if unknown:
            raise ValueError(f"Unknown metrics {unknown}; available: {self.available_metrics}")
        super().__init__(target_field=target_field, metrics=selected, **config)
        self.target_field = target_field
        self.required_fields = (target_field,)
        self.default_metrics = selected

    # Methods look up the outcome column through ``label_field``.
    @property
    def label_field(self) -> str:
        return self.target_field

    def score(self, y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, Any]:
        """Compute the task metrics for 1-D targets and predictions."""
        from sklearn import metrics as skm
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        out: Dict[str, Any] = {"n": int(len(y_true))}
        for name in self.default_metrics:
            if name == "r2":
                out[name] = float(skm.r2_score(y_true, y_pred))
            elif name == "mae":
                out[name] = float(skm.mean_absolute_error(y_true, y_pred))
            elif name == "mse":
                out[name] = float(skm.mean_squared_error(y_true, y_pred))
            elif name == "pearson":
                out[name] = float(np.corrcoef(y_true, y_pred)[0, 1])
        return out
