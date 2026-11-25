"""Cohort: patients, slides, and their labels.

A :class:`Cohort` answers "who is in the study, which slides do they have,
and what are their labels" -- and nothing else. It knows nothing about tile
sizes, TFRecords, or feature bags, so the same cohort can be tiled, split,
and modeled in different ways.

.. code-block:: python

    import histox as hx

    cohort = hx.Cohort.from_csv('annotations.csv', slides='/data/slides')
    cohort.patients[:3]
    cohort.labels('subtype')                 # {slide: label}
    cohort.patient_labels('subtype')         # {patient: label}; errors on conflicts
    cohort.missing_slides()                  # annotated slides with no file

    luad = cohort.filter(subtype='LUAD')
    folds = hx.splits.patient_kfold(cohort, k=5, stratify='subtype', seed=1)

    dataset.cohort                           # the cohort of an existing Dataset

See RFC 0001 (``rfcs/0001-core-abstractions.md``), proposal P8.
"""

import os
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import pandas as pd

if TYPE_CHECKING:
    from histox.workflows import DataSpec

__all__ = ['Cohort']

_EMPTY = ('', 'na', 'n/a', 'none', 'nan')


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in _EMPTY


class Cohort:
    """Patients, slides, and their labels.

    Args:
        table (pandas.DataFrame): One row per slide, with a slide column, an
            optional patient column, and any label columns.
        slides (str or list(str), optional): Where the slide files are: a
            directory, or a list of directories and/or slide file paths.
            Directories are searched the same way as ``Dataset`` slide
            sources (the directory and its immediate subdirectories, for
            supported slide formats).
        patient_col (str): Patient column. Defaults to ``'patient'``. If the
            table has no such column, each slide is its own patient.
        slide_col (str): Slide column. Defaults to ``'slide'``.

    Rows with an empty slide name are dropped.
    """

    def __init__(
        self,
        table: pd.DataFrame,
        *,
        slides: Optional[Union[str, Sequence[str]]] = None,
        patient_col: str = 'patient',
        slide_col: str = 'slide',
    ) -> None:
        if not isinstance(table, pd.DataFrame):
            raise TypeError("table must be a pandas DataFrame")
        if slide_col not in table.columns:
            raise ValueError(f"Column {slide_col!r} not found in table")
        table = table.loc[~table[slide_col].map(_is_empty)].copy()
        table[slide_col] = table[slide_col].astype(str)
        if table[slide_col].duplicated().any():
            dupes = table.loc[table[slide_col].duplicated(), slide_col].unique().tolist()
            raise ValueError(f"Slides listed more than once: {dupes[:5]}")
        if patient_col not in table.columns:
            table[patient_col] = table[slide_col]
        else:
            table[patient_col] = table[patient_col].astype(str)
        self._table = table.reset_index(drop=True)
        self.patient_col = patient_col
        self.slide_col = slide_col
        self._slide_sources = ([] if slides is None
                               else [slides] if isinstance(slides, str) else list(slides))
        self._paths: Optional[Dict[str, str]] = None

    @classmethod
    def from_csv(cls, path: str, *, slides: Optional[Union[str, Sequence[str]]] = None,
                 **kwargs) -> "Cohort":
        """Read a cohort from a CSV file (all columns as text, like ``Dataset``)."""
        table = pd.read_csv(path, dtype=str).fillna('')
        return cls(table, slides=slides, **kwargs)

    # --- Contents -------------------------------------------------------------

    @property
    def table(self) -> pd.DataFrame:
        """A copy of the annotation table (one row per slide)."""
        return self._table.copy()

    @property
    def slides(self) -> List[str]:
        """Slide names, in table order."""
        return self._table[self.slide_col].tolist()

    @property
    def patients(self) -> List[str]:
        """Patient IDs, sorted."""
        return sorted(self._table[self.patient_col].unique().tolist())

    @property
    def fields(self) -> List[str]:
        """Label columns (all columns except patient and slide)."""
        return [c for c in self._table.columns if c not in (self.patient_col, self.slide_col)]

    def __len__(self) -> int:
        return len(self._table)

    def __contains__(self, slide: object) -> bool:
        return slide in set(self.slides)

    def __repr__(self) -> str:
        return (f"<Cohort {len(self.patients)} patients, {len(self)} slides, "
                f"fields={self.fields}>")

    def patient_of(self, slide: str) -> str:
        """The patient a slide belongs to."""
        rows = self._table.loc[self._table[self.slide_col] == slide, self.patient_col]
        if not len(rows):
            raise KeyError(f"Slide {slide!r} not in cohort")
        return rows.iloc[0]

    def slides_of(self, patient: str) -> List[str]:
        """The slides of a patient."""
        return self._table.loc[self._table[self.patient_col] == patient, self.slide_col].tolist()

    def patient_map(self) -> Dict[str, str]:
        """``{slide: patient}``."""
        return dict(zip(self._table[self.slide_col], self._table[self.patient_col]))

    # --- Labels ---------------------------------------------------------------

    def _check_field(self, field: str) -> None:
        if field not in self._table.columns:
            raise KeyError(f"Field {field!r} not in cohort; available: {self.fields}")

    def labels(self, field: str, *, drop_empty: bool = True) -> Dict[str, Any]:
        """``{slide: value}`` for one field.

        Args:
            field (str): Column name.
            drop_empty (bool): Leave out slides with an empty value.
                Defaults to True.
        """
        self._check_field(field)
        pairs = zip(self._table[self.slide_col], self._table[field])
        return {s: v for s, v in pairs if not (drop_empty and _is_empty(v))}

    def patient_labels(self, field: str, *, drop_empty: bool = True) -> Dict[str, Any]:
        """``{patient: value}`` for one field.

        Raises:
            ValueError: If a patient's slides have different values.
        """
        self._check_field(field)
        out: Dict[str, Any] = {}
        for patient, value in zip(self._table[self.patient_col], self._table[field]):
            if drop_empty and _is_empty(value):
                continue
            if patient in out and out[patient] != value:
                raise ValueError(
                    f"Patient {patient!r} has conflicting values for {field!r}: "
                    f"{out[patient]!r} and {value!r}")
            out[patient] = value
        return out

    # --- Selection --------------------------------------------------------------

    def _new(self, table: pd.DataFrame) -> "Cohort":
        new = Cohort(table, slides=self._slide_sources, patient_col=self.patient_col,
                     slide_col=self.slide_col)
        new._paths = self._paths
        return new

    def subset(self, *, slides: Optional[Iterable[str]] = None,
               patients: Optional[Iterable[str]] = None) -> "Cohort":
        """A cohort with only the given slides and/or patients."""
        mask = pd.Series(True, index=self._table.index)
        if slides is not None:
            mask &= self._table[self.slide_col].isin(list(slides))
        if patients is not None:
            mask &= self._table[self.patient_col].isin(list(patients))
        return self._new(self._table.loc[mask])

    def filter(self, **conditions: Any) -> "Cohort":
        """A cohort with rows matching every ``field=value`` condition.

        A value may be a single value or a list of accepted values.
        """
        mask = pd.Series(True, index=self._table.index)
        for field, accepted in conditions.items():
            self._check_field(field)
            if isinstance(accepted, (list, tuple, set)):
                mask &= self._table[field].isin(list(accepted))
            else:
                mask &= self._table[field] == accepted
        return self._new(self._table.loc[mask])

    def drop_empty(self, *fields: str) -> "Cohort":
        """A cohort without the rows that have an empty value in any of ``fields``."""
        mask = pd.Series(True, index=self._table.index)
        for field in fields:
            self._check_field(field)
            mask &= ~self._table[field].map(_is_empty)
        return self._new(self._table.loc[mask])

    # --- Slide files ------------------------------------------------------------

    def _index_paths(self) -> Dict[str, str]:
        if self._paths is None:
            from histox.util import get_slide_paths, path_to_name
            paths: Dict[str, str] = {}
            for source in self._slide_sources:
                found = get_slide_paths(source) if os.path.isdir(source) else [source]
                for p in sorted(found):
                    paths.setdefault(path_to_name(p), p)
            self._paths = paths
        return self._paths

    def slide_path(self, slide: str) -> Optional[str]:
        """Path to a slide's file, or ``None`` if it was not found."""
        return self._index_paths().get(slide)

    def slide_paths(self) -> Dict[str, str]:
        """``{slide: path}`` for the cohort's slides that have a file."""
        index = self._index_paths()
        return {s: index[s] for s in self.slides if s in index}

    def missing_slides(self) -> List[str]:
        """Slides in the cohort with no file in the slide sources."""
        index = self._index_paths()
        return [s for s in self.slides if s not in index]

    # --- Workflows ----------------------------------------------------------------

    def data_spec(
        self,
        fields: Mapping[str, str],
        *,
        modalities: Sequence[str] = ('histology',),
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "DataSpec":
        """A :class:`histox.workflows.DataSpec` describing this cohort.

        Args:
            fields (dict): Semantic names mapped to columns, e.g.
                ``{'duration': 'os_days', 'event': 'os_event'}``.
            modalities (list(str)): Data modalities available. Defaults to
                ``('histology',)``.
        """
        from histox.workflows import DataSpec
        for column in fields.values():
            self._check_field(column)
        spec_fields = {'patient_id': self.patient_col, 'slide_id': self.slide_col}
        spec_fields.update(fields)
        meta = {'patients': len(self.patients), 'slides': len(self)}
        meta.update(metadata or {})
        return DataSpec(modalities=tuple(modalities), fields=spec_fields, metadata=meta)
