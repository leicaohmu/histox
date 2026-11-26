"""PyTorch datasets of feature bags.

Every HistoX dataset returns one ``dict`` per item, with standard keys (RFC
0002, P2). For feature bags:

============  ==========================  ===========================================
Key           Type                        Meaning
============  ==========================  ===========================================
``features``  float32 tensor ``(N, F)``   One row per tile.
``coords``    int64 tensor ``(N, 2)``     Tile centers, ``(x, y)`` in level-0 pixels
                                          (see :mod:`histox.geometry`); ``None`` if
                                          the bag has no tile index.
``slide``     ``str``                     Slide name. For patient-level items, the
                                          slides joined in order, as a list.
``patient``   ``str``                     Patient identifier.
``target``    tensor                      ``int64`` class index for categorical
                                          targets, ``float32`` for numeric ones;
                                          absent when no target is requested.
============  ==========================  ===========================================

Bags in one batch have different lengths, so batch them with
:func:`collate_bags`, which pads them and adds a boolean ``mask``.
"""

import os
from os.path import exists, join
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional,
                    Sequence, Union)

import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from histox.cohort import Cohort

__all__ = ['BagDataset', 'collate_bags']


class BagDataset(Dataset):
    """Feature bags of a cohort, one item per slide or per patient.

    Args:
        cohort (histox.Cohort or histox.Dataset): Patients, slides, and labels.
            A :class:`histox.Dataset` is accepted and its ``cohort`` is used.
            To train on one side of a split, pass
            ``cohort.subset(slides=split.train_slides)``.
        bags (str or list(str)): Directory, or directories, of feature bags
            written by :meth:`histox.Dataset.generate_feature_bags` or
            :meth:`histox.Bag.save`. The first directory with a bag for a slide
            is used.
        target (str, optional): Cohort field to return as ``target``. Slides
            with an empty value are dropped. Defaults to None (no target).
        target_type (str): ``'categorical'``, ``'numeric'``, or ``'auto'``.
            ``'auto'`` treats the target as numeric if every value converts to
            a number, and as categorical otherwise. Defaults to ``'auto'``.
        classes (list, optional): Class order for categorical targets, which
            defines the class indices. Defaults to the sorted unique values.
        level (str): ``'slide'`` for one item per slide, or ``'patient'`` for
            one item per patient, with the bags of the patient's slides joined.
            Defaults to ``'slide'``.
        max_tiles (int, optional): If a bag has more tiles, return a random
            subset of this size, drawn again on every access. Uses the PyTorch
            random generator, so ``torch.manual_seed`` and DataLoader worker
            seeding apply. Defaults to None (all tiles).
        transform (callable, optional): Applied to each item ``dict`` last.
        missing (str): What to do with cohort slides that have no bag:
            ``'drop'`` (log and skip them) or ``'error'``. Defaults to
            ``'drop'``.
        preload (bool): Keep loaded bags in memory. Defaults to False.

    Example:
        .. code-block:: python

            from torch.utils.data import DataLoader
            import histox as hx

            cohort = hx.Cohort.from_csv('annotations.csv')
            fold = hx.splits.patient_holdout(cohort, 0.2, stratify='subtype')
            train = hx.data.BagDataset(cohort.subset(slides=fold.train_slides),
                                       'bags/', target='subtype', max_tiles=2048)
            loader = DataLoader(train, batch_size=8, shuffle=True,
                                collate_fn=hx.data.collate_bags)
            batch = next(iter(loader))
            batch['features'].shape     # (8, N_max, F)
            batch['mask'].shape         # (8, N_max), True for real tiles
    """

    def __init__(
        self,
        cohort: Any,
        bags: Union[str, Sequence[str]],
        *,
        target: Optional[str] = None,
        target_type: str = 'auto',
        classes: Optional[Sequence[Any]] = None,
        level: str = 'slide',
        max_tiles: Optional[int] = None,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        missing: str = 'drop',
        preload: bool = False,
    ) -> None:
        super().__init__()
        from histox import log
        cohort = getattr(cohort, 'cohort', cohort)
        if level not in ('slide', 'patient'):
            raise ValueError(f"level must be 'slide' or 'patient', got {level!r}")
        if target_type not in ('auto', 'categorical', 'numeric'):
            raise ValueError(
                f"target_type must be 'auto', 'categorical', or 'numeric', got {target_type!r}")
        if missing not in ('drop', 'error'):
            raise ValueError(f"missing must be 'drop' or 'error', got {missing!r}")
        if max_tiles is not None and max_tiles < 1:
            raise ValueError("max_tiles must be a positive integer")

        self.cohort = cohort
        self.directories = [bags] if isinstance(bags, (str, os.PathLike)) else list(bags)
        self.target = target
        self.level = level
        self.max_tiles = max_tiles
        self.transform = transform
        self.preload = preload
        self._cache: Dict[str, Any] = {}

        # Slides with a bag, and optionally a non-empty target.
        slides = list(cohort.slides)
        if target is not None:
            labeled = cohort.labels(target)  # drops empty values
            slides = [s for s in slides if s in labeled]
        self._paths: Dict[str, str] = {}
        absent = []
        for slide in slides:
            path = self._find(slide)
            if path is None:
                absent.append(slide)
            else:
                self._paths[slide] = path
        if absent:
            msg = (f"{len(absent)} of {len(slides)} slides have no feature bag in "
                   f"{self.directories}: {absent[:5]}{' ...' if len(absent) > 5 else ''}")
            if missing == 'error':
                raise FileNotFoundError(msg)
            log.warning(msg + "; they are skipped.")
        if not self._paths:
            raise ValueError("No slides with feature bags were found.")

        # Items: one slide, or the slides of one patient (in cohort order).
        if level == 'slide':
            self._items = [[s] for s in slides if s in self._paths]
            self._item_patient = [cohort.patient_of(s[0]) for s in self._items]
        else:
            by_patient: Dict[str, List[str]] = {}
            for s in slides:
                if s in self._paths:
                    by_patient.setdefault(cohort.patient_of(s), []).append(s)
            self._item_patient = list(by_patient)
            self._items = [by_patient[p] for p in self._item_patient]

        # Targets.
        self.classes: Optional[List[Any]] = None
        self.target_type: Optional[str] = None
        self._targets: Optional[List[Any]] = None
        if target is not None:
            if level == 'slide':
                values = [cohort.labels(target)[item[0]] for item in self._items]
            else:
                values = [cohort.patient_labels(target)[p] for p in self._item_patient]
            self._targets, self.classes, self.target_type = _encode_targets(
                values, target_type, classes)

    # --- Dataset interface ---------------------------------------------------

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        slides = self._items[index]
        parts = [self._load(s) for s in slides]
        features = torch.cat([f for f, _ in parts]) if len(parts) > 1 else parts[0][0]
        if any(c is None for _, c in parts):
            coords = None
        else:
            coords = torch.cat([c for _, c in parts]) if len(parts) > 1 else parts[0][1]
        if self.max_tiles is not None and features.shape[0] > self.max_tiles:
            keep = torch.randperm(features.shape[0])[:self.max_tiles]
            features = features[keep]
            if coords is not None:
                coords = coords[keep]
        item: Dict[str, Any] = {
            'features': features,
            'coords': coords,
            'slide': slides[0] if self.level == 'slide' else list(slides),
            'patient': self._item_patient[index],
        }
        if self._targets is not None:
            item['target'] = self._targets[index]
        if self.transform is not None:
            item = self.transform(item)
        return item

    # --- Information ----------------------------------------------------------

    @property
    def slides(self) -> List[str]:
        """Slides included, in item order (flattened for patient-level items)."""
        return [s for item in self._items for s in item]

    @property
    def patients(self) -> List[str]:
        """Patient of each item."""
        return list(self._item_patient)

    @property
    def num_features(self) -> int:
        """Feature dimension, read from the first bag."""
        return int(self._load(self._items[0][0])[0].shape[1])

    def __repr__(self) -> str:
        target = f", target={self.target!r} ({self.target_type})" if self.target else ''
        return (f"BagDataset({len(self)} {self.level}s{target}, "
                f"bags={self.directories})")

    # --- Helpers ---------------------------------------------------------------

    def _find(self, slide: str) -> Optional[str]:
        for directory in self.directories:
            path = join(str(directory), f'{slide}.pt')
            if exists(path):
                return path
        return None

    def _load(self, slide: str):
        if slide in self._cache:
            return self._cache[slide]
        from histox.bag import Bag
        bag = Bag.from_path(self._paths[slide])
        coords = None if bag.coords is None else torch.as_tensor(
            np.asarray(bag.coords)[:, :2], dtype=torch.int64)
        out = (bag.features, coords)
        if self.preload:
            self._cache[slide] = out
        return out


def _encode_targets(values: List[Any], target_type: str,
                    classes: Optional[Sequence[Any]]):
    """Encode targets as tensors; return (targets, classes, target_type)."""
    if target_type == 'auto':
        try:
            [float(v) for v in values]
            numeric = classes is None
        except (TypeError, ValueError):
            numeric = False
        target_type = 'numeric' if numeric else 'categorical'
    if target_type == 'numeric':
        return [torch.tensor(float(v), dtype=torch.float32) for v in values], None, 'numeric'
    if classes is None:
        classes = sorted(set(values), key=lambda v: (str(type(v)), v))
    classes = list(classes)
    lookup = {c: i for i, c in enumerate(classes)}
    unknown = sorted({str(v) for v in values if v not in lookup})
    if unknown:
        raise ValueError(f"Target values not in classes {classes}: {unknown[:5]}")
    return ([torch.tensor(lookup[v], dtype=torch.int64) for v in values],
            classes, 'categorical')


def collate_bags(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Batch bag items of different lengths, for a ``DataLoader``.

    Bags are padded with zeros to the longest bag in the batch.

    Returns:
        dict: ``features`` ``(B, N_max, F)``; ``mask`` ``(B, N_max)``, True for
        real tiles; ``lens`` ``(B,)`` int64, the number of real tiles, for
        models that take lengths; ``coords`` ``(B, N_max, 2)`` padded with -1,
        or None if any bag lacks coordinates; ``target`` stacked, if present;
        other keys as lists.
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    lens = torch.tensor([b['features'].shape[0] for b in batch], dtype=torch.int64)
    n_max = int(lens.max())
    first = batch[0]['features']
    features = first.new_zeros((len(batch), n_max, first.shape[1]))
    mask = torch.zeros((len(batch), n_max), dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b['features'].shape[0]
        features[i, :n] = b['features']
        mask[i, :n] = True
    out: Dict[str, Any] = {'features': features, 'mask': mask, 'lens': lens}
    if all(b.get('coords') is not None for b in batch):
        coords = torch.full((len(batch), n_max, 2), -1, dtype=torch.int64)
        for i, b in enumerate(batch):
            coords[i, :b['coords'].shape[0]] = b['coords']
        out['coords'] = coords
    else:
        out['coords'] = None
    if all('target' in b for b in batch):
        out['target'] = torch.stack([torch.as_tensor(b['target']) for b in batch])
    for key in batch[0]:
        if key not in out:
            out[key] = [b.get(key) for b in batch]
    return out
