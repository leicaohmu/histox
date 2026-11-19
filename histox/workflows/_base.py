"""Base classes for extensible HistoX tasks and methods."""

from typing import Any, Dict, Sequence

from ._types import DataSpec, RunResult
from .exceptions import (
    IncompatibleWorkflowError,
    InvalidDataSpecError,
    UnsupportedStageError,
)


class Task:
    """Base class describing target fields and evaluation defaults."""

    name = "task"
    required_fields: Sequence[str] = ()
    default_metrics: Sequence[str] = ()

    def __init__(self, **config) -> None:
        self.config = dict(config)

    def validate_data_spec(self, data_spec: DataSpec) -> None:
        """Validate that the cohort declares all task fields."""
        missing = [
            name for name in self.required_fields if name not in data_spec.fields
        ]
        if missing:
            raise InvalidDataSpecError(
                "Task {!r} requires fields {}; declared fields are {}".format(
                    self.name, missing, sorted(data_spec.fields)
                )
            )


class Method:
    """Base class for one model or algorithm workflow adapter."""

    name = "method"
    supported_tasks: Sequence[str] = ()
    required_modalities: Sequence[str] = ()
    supported_explainers: Sequence[str] = ()

    def __init__(self, **config) -> None:
        self.config = dict(config)

    def validate(self, task: Task, data_spec: DataSpec) -> None:
        """Validate task compatibility and required modalities."""
        if self.supported_tasks and task.name not in self.supported_tasks:
            raise IncompatibleWorkflowError(
                "Method {!r} supports tasks {}, not {!r}".format(
                    self.name, list(self.supported_tasks), task.name
                )
            )
        missing = [
            name
            for name in self.required_modalities
            if name not in data_spec.modalities
        ]
        if missing:
            raise InvalidDataSpecError(
                "Method {!r} requires modalities {}; declared modalities are {}".format(
                    self.name, missing, list(data_spec.modalities)
                )
            )

    def build_model(self, data_spec: DataSpec):
        """Construct the model for this method."""
        raise NotImplementedError

    def run_stage(self, stage: str, experiment, **kwargs) -> RunResult:
        """Execute one lifecycle stage implemented as ``stage_<name>``."""
        handler = getattr(self, "stage_{}".format(stage), None)
        if handler is None:
            raise UnsupportedStageError(
                "Method {!r} does not implement stage {!r}".format(
                    self.name, stage
                )
            )
        result = handler(experiment=experiment, **kwargs)
        if isinstance(result, RunResult):
            return result
        return RunResult(stage=stage, predictions=result)

    def public_config(self) -> Dict[str, Any]:
        """Return configuration suitable for experiment metadata."""
        return dict(self.config)
