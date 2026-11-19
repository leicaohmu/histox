"""Composable experiment workflows for HistoX.

The base package is intentionally lightweight. Model frameworks and optional
dependencies are loaded by individual method adapters only when needed.
"""

from ._base import Method, Task
from ._registry import (
    build_method,
    build_task,
    list_methods,
    list_tasks,
    register_method,
    register_task,
)
from ._types import Artifact, DataSpec, RunResult
from .exceptions import (
    DuplicateRegistrationError,
    IncompatibleWorkflowError,
    InvalidDataSpecError,
    RegistryError,
    UnknownMethodError,
    UnknownTaskError,
    UnsupportedStageError,
    WorkflowError,
)
from .experiment import Experiment
from .pretrained import PS3PretrainedBundle
from .tasks import SurvivalTask
from .methods import PS3Method

__all__ = [
    "Artifact",
    "DataSpec",
    "DuplicateRegistrationError",
    "Experiment",
    "IncompatibleWorkflowError",
    "InvalidDataSpecError",
    "Method",
    "PS3PretrainedBundle",
    "PS3Method",
    "RegistryError",
    "RunResult",
    "SurvivalTask",
    "Task",
    "UnknownMethodError",
    "UnknownTaskError",
    "UnsupportedStageError",
    "WorkflowError",
    "build_method",
    "build_task",
    "list_methods",
    "list_tasks",
    "register_method",
    "register_task",
]
