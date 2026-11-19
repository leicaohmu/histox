"""Lightweight data and result types for HistoX workflows."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union


PathLike = Union[str, Path]


def _unique_names(values: Sequence[str], label: str) -> Tuple[str, ...]:
    names = tuple(str(value).strip() for value in values)
    if any(not name for name in names):
        raise ValueError("{} cannot contain empty names".format(label))
    if len(set(names)) != len(names):
        raise ValueError("{} cannot contain duplicate names".format(label))
    return names


@dataclass(frozen=True)
class DataSpec:
    """Describe the modalities and semantic fields available in a cohort.

    ``DataSpec`` deliberately does not own slide files or tensors. It allows
    HistoX to validate a workflow before loading a large cohort.
    """

    modalities: Sequence[str]
    fields: Mapping[str, str]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "modalities", _unique_names(self.modalities, "modalities")
        )
        clean_fields = {
            str(name).strip(): str(column).strip()
            for name, column in self.fields.items()
        }
        if any(not name or not column for name, column in clean_fields.items()):
            raise ValueError("fields must map non-empty semantic names to columns")
        object.__setattr__(self, "fields", clean_fields)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def has_modalities(self, names: Sequence[str]) -> bool:
        """Return whether every requested modality is declared."""
        return set(names).issubset(set(self.modalities))

    def has_fields(self, names: Sequence[str]) -> bool:
        """Return whether every requested semantic field is declared."""
        return set(names).issubset(set(self.fields))


@dataclass
class Artifact:
    """One named output produced by a workflow stage."""

    name: str
    kind: str
    path: Optional[PathLike] = None
    data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        self.kind = str(self.kind).strip()
        if not self.name or not self.kind:
            raise ValueError("Artifact name and kind must be non-empty")
        if self.path is not None:
            self.path = Path(self.path)

    def to_dict(self, include_data: bool = False) -> Dict[str, Any]:
        """Return a JSON-oriented representation."""
        payload = {
            "name": self.name,
            "kind": self.kind,
            "path": str(self.path) if self.path is not None else None,
            "metadata": dict(self.metadata),
        }
        if include_data:
            payload["data"] = self.data
        return payload


@dataclass
class RunResult:
    """Standard output container for training, evaluation, and explanation."""

    stage: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    predictions: Any = None
    artifacts: Dict[str, Artifact] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_artifact(
        self,
        name: str,
        kind: str,
        path: Optional[PathLike] = None,
        data: Any = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Artifact:
        """Attach a checkpoint, heatmap, table, figure, or custom artifact."""
        artifact = Artifact(
            name=name,
            kind=kind,
            path=path,
            data=data,
            metadata=dict(metadata or {}),
        )
        if artifact.name in self.artifacts:
            raise ValueError(
                "Artifact {!r} already exists".format(artifact.name)
            )
        self.artifacts[artifact.name] = artifact
        return artifact

    def to_dict(self, include_artifact_data: bool = False) -> Dict[str, Any]:
        """Return a serializable summary of the result."""
        return {
            "stage": self.stage,
            "metrics": dict(self.metrics),
            "predictions": self.predictions,
            "artifacts": {
                name: artifact.to_dict(include_data=include_artifact_data)
                for name, artifact in self.artifacts.items()
            },
            "metadata": dict(self.metadata),
        }

    def save_json(
        self,
        path: PathLike,
        include_artifact_data: bool = False,
    ) -> Path:
        """Write the result summary as UTF-8 JSON and return its path."""
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                self.to_dict(include_artifact_data=include_artifact_data),
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        return destination
