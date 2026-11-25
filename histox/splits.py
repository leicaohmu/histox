"""Patient-level data splits.

Every split here assigns whole **patients**, never single slides, so no
patient contributes slides to both sides of a split. Splits are
deterministic: the same cohort, arguments, and ``seed`` always give the same
result, on any machine.

.. code-block:: python

    import histox as hx

    folds = hx.splits.patient_kfold(cohort, k=5, stratify='subtype', seed=1)
    for fold in folds:
        train = cohort.subset(slides=fold.train_slides)
        val = cohort.subset(slides=fold.val_slides)

    holdout = hx.splits.patient_holdout(cohort, val_fraction=0.2, stratify='subtype')

:meth:`histox.Dataset.split` is unchanged and keeps its own split files.

See RFC 0001 (``rfcs/0001-core-abstractions.md``), proposal P8.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from histox.cohort import Cohort

__all__ = ['Split', 'patient_kfold', 'patient_holdout', 'site_preserved_kfold']


@dataclass(frozen=True)
class Split:
    """One train/validation split, by patient and by slide."""

    train_patients: Tuple[str, ...]
    val_patients: Tuple[str, ...]
    train_slides: Tuple[str, ...]
    val_slides: Tuple[str, ...]

    def __post_init__(self) -> None:
        overlap = set(self.train_patients) & set(self.val_patients)
        if overlap:
            raise ValueError(f"Patients on both sides of a split: {sorted(overlap)[:5]}")

    def to_dict(self) -> Dict[str, List[str]]:
        return {k: list(getattr(self, k)) for k in
                ('train_patients', 'val_patients', 'train_slides', 'val_slides')}


def _strata(cohort: "Cohort", stratify: Optional[str]) -> Dict[str, Any]:
    """``{patient: stratum}`` for the patients that take part in the split."""
    if stratify is None:
        return {p: None for p in cohort.patients}
    return cohort.patient_labels(stratify)          # drops empty, checks conflicts


def _make_split(cohort: "Cohort", val_patients: Sequence[str],
                patients: Sequence[str]) -> Split:
    val = sorted(val_patients)
    val_set = set(val)
    train = sorted(p for p in patients if p not in val_set)
    by_patient: Dict[str, List[str]] = {}
    for slide, patient in cohort.patient_map().items():
        by_patient.setdefault(patient, []).append(slide)
    return Split(
        train_patients=tuple(train),
        val_patients=tuple(val),
        train_slides=tuple(s for p in train for s in by_patient[p]),
        val_slides=tuple(s for p in val for s in by_patient[p]),
    )


def _shuffled_by_stratum(strata: Dict[str, Any], seed: int) -> List[List[str]]:
    rng = np.random.default_rng(seed)
    groups: Dict[str, List[str]] = {}
    for patient in sorted(strata):
        groups.setdefault(str(strata[patient]), []).append(patient)
    out = []
    for key in sorted(groups):
        members = groups[key]
        out.append([members[i] for i in rng.permutation(len(members))])
    return out


def patient_kfold(
    cohort: "Cohort",
    k: int,
    *,
    stratify: Optional[str] = None,
    seed: int = 0,
) -> List[Split]:
    """K-fold cross-validation by patient.

    Args:
        cohort (histox.Cohort): Patients and slides.
        k (int): Number of folds (>= 2).
        stratify (str, optional): Field whose values are balanced across
            folds (e.g. the outcome). Patients with an empty value are left
            out. A patient whose slides have different values is an error.
        seed (int): Random seed. Defaults to 0.

    Returns:
        list(Split): ``k`` splits. Fold ``i`` validates on the ``i``-th group
        of patients and trains on all the others. Every patient is validated
        exactly once.
    """
    if int(k) != k or k < 2:
        raise ValueError(f"k must be an integer >= 2, got {k}")
    strata = _strata(cohort, stratify)
    if len(strata) < k:
        raise ValueError(f"{len(strata)} patients cannot be split into {k} folds")
    ordered = [p for group in _shuffled_by_stratum(strata, seed) for p in group]
    folds: List[List[str]] = [[] for _ in range(k)]
    for i, patient in enumerate(ordered):       # deal stratum by stratum, round-robin
        folds[i % k].append(patient)
    patients = sorted(strata)
    return [_make_split(cohort, fold, patients) for fold in folds]


def patient_holdout(
    cohort: "Cohort",
    val_fraction: float,
    *,
    stratify: Optional[str] = None,
    seed: int = 0,
) -> Split:
    """One train/validation split by patient.

    Args:
        cohort (histox.Cohort): Patients and slides.
        val_fraction (float): Fraction of patients for validation (0 to 1),
            applied within each stratum and rounded to the nearest patient.
        stratify (str, optional): Field whose values are balanced between the
            two sides. Patients with an empty value are left out.
        seed (int): Random seed. Defaults to 0.
    """
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")
    strata = _strata(cohort, stratify)
    val: List[str] = []
    for group in _shuffled_by_stratum(strata, seed):
        val += group[:int(round(len(group) * val_fraction))]
    split = _make_split(cohort, val, sorted(strata))
    if not split.val_patients or not split.train_patients:
        raise ValueError("Split leaves no patients on one side; adjust val_fraction")
    return split


def site_preserved_kfold(
    cohort: "Cohort",
    k: int,
    *,
    site: str = 'site',
    stratify: Optional[str] = None,
    method: str = 'auto',
) -> List[Split]:
    """K-fold cross-validation that keeps each site in a single fold.

    Uses the preserved-site cross-validation solver of HistoX
    (``histox.io.preservedsite``), which balances ``stratify`` across folds
    while placing all patients of a site in the same fold. It requires the
    CPLEX or Pyomo/Bonmin solver.

    Args:
        cohort (histox.Cohort): Patients and slides.
        k (int): Number of folds.
        site (str): Field with the site of each patient. Defaults to ``'site'``.
        stratify (str, optional): Field to balance across folds.
        method (str): Solver: ``'auto'``, ``'cplex'``, or ``'bonmin'``.
    """
    import pandas as pd
    import histox.io.preservedsite.crossfolds as cv

    sites = cohort.patient_labels(site)
    strata = _strata(cohort, stratify)
    patients = sorted(set(sites) & set(strata))
    df = pd.DataFrame({
        'patient': patients,
        'outcome_label': [strata[p] if stratify else 1 for p in patients],
        'site': [sites[p] for p in patients],
    })
    df = cv.generate(df, 'outcome_label', k=k, target_column='CV', method=method)
    folds = [df.loc[df.CV == str(i + 1), 'patient'].tolist() for i in range(k)]
    return [_make_split(cohort, fold, patients) for fold in folds]
