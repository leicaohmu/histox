"""Experiment orchestration for registered HistoX tasks and methods."""

from pathlib import Path
from typing import Any, Mapping, Optional, Union

from ._base import Method, Task
from ._registry import build_method, build_task
from ._types import DataSpec, RunResult


class Experiment:
    """Bind a task, method, declared cohort, and lifecycle results.

    Args:
        task: Registered task name or a :class:`Task` instance.
        method: Registered method name or a :class:`Method` instance.
        data_spec: Modalities and semantic fields available to the workflow.
        cohort: Optional user-owned cohort object consumed by method stages.
        task_config: Keyword configuration used when ``task`` is a name.
        method_config: Keyword configuration used when ``method`` is a name.
        output: Optional directory for checkpoints, tables, and artifacts.
        metadata: Free-form experiment metadata.
    """

    stages = ("prepare", "fit", "evaluate", "predict", "explain")

    def __init__(
        self,
        task: Union[str, Task],
        method: Union[str, Method],
        data_spec: DataSpec,
        cohort: Any = None,
        task_config: Optional[Mapping[str, Any]] = None,
        method_config: Optional[Mapping[str, Any]] = None,
        output=None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.task = (
            build_task(task, **dict(task_config or {}))
            if isinstance(task, str)
            else task
        )
        self.method = (
            build_method(method, **dict(method_config or {}))
            if isinstance(method, str)
            else method
        )
        if not isinstance(self.task, Task):
            raise TypeError("task must resolve to a Task instance")
        if not isinstance(self.method, Method):
            raise TypeError("method must resolve to a Method instance")
        if not isinstance(data_spec, DataSpec):
            raise TypeError("data_spec must be a DataSpec instance")
        self.data_spec = data_spec
        self.cohort = cohort
        self.output = Path(output).expanduser() if output is not None else None
        self.metadata = dict(metadata or {})
        self.results = {}
        self._model = None
        self.validate()

    def validate(self) -> None:
        """Validate task fields, method modalities, and task compatibility."""
        self.task.validate_data_spec(self.data_spec)
        self.method.validate(self.task, self.data_spec)

    def build_model(self, rebuild: bool = False):
        """Build and cache the method model after validating the cohort.

        Args:
            rebuild: Construct a fresh model even when one is already cached.
        """
        self.validate()
        if self._model is None or rebuild:
            self._model = self.method.build_model(self.data_spec)
        return self._model

    def run(self, stage: str, **kwargs) -> RunResult:
        """Run a supported lifecycle stage and retain its result."""
        if stage not in self.stages:
            raise ValueError(
                "Unknown stage {!r}; expected one of {}".format(
                    stage, self.stages
                )
            )
        self.validate()
        result = self.method.run_stage(stage, experiment=self, **kwargs)
        if result.stage != stage:
            raise ValueError(
                "Method {!r} returned stage {!r} while running {!r}".format(
                    self.method.name, result.stage, stage
                )
            )
        result.metadata.setdefault("task", self.task.name)
        result.metadata.setdefault("method", self.method.name)
        self.results[stage] = result
        return result

    def prepare(self, **kwargs) -> RunResult:
        return self.run("prepare", **kwargs)

    def fit(self, **kwargs) -> RunResult:
        return self.run("fit", **kwargs)

    def evaluate(self, **kwargs) -> RunResult:
        return self.run("evaluate", **kwargs)

    def predict(self, **kwargs) -> RunResult:
        return self.run("predict", **kwargs)

    def explain(self, **kwargs) -> RunResult:
        return self.run("explain", **kwargs)
