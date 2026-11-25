"""Workflow adapter for tile-level supervised training (``Project.train``)."""

import glob
import os
from os.path import join
from typing import Any, Dict, List, Optional

from .._base import Method
from .._registry import register_method
from .._types import DataSpec, RunResult

_LOSS_FOR_TASK = {'classification': 'CrossEntropy', 'regression': 'MSE'}


def _new_dirs(root: str, before: set) -> List[str]:
    if not os.path.isdir(root):
        return []
    return sorted(join(root, d) for d in set(os.listdir(root)) - before)


def _listing(root: str) -> set:
    return set(os.listdir(root)) if os.path.isdir(root) else set()


@register_method("tile_classifier")
class TileClassifierMethod(Method):
    """Train a tile-level CNN with :meth:`histox.Project.train`.

    Tiles inherit their slide's label during training; evaluation reports
    tile-, slide-, and patient-level results. The stages call
    :meth:`histox.Project.train` and :meth:`histox.Project.evaluate`, so an
    experiment trains exactly as those methods do.

    Configuration keys (``method_config``):

    * ``project``: the :class:`histox.Project` (or its root directory) that
      holds the TFRecords and receives the models. Required.
    * ``params``: a :class:`histox.ModelParams`. Alternatively, give its
      keyword arguments directly (``tile_px``, ``tile_um``, ``model``,
      ``epochs``, ``batch_size``, ...). ``loss`` defaults to
      ``'CrossEntropy'`` for classification and ``'MSE'`` for regression.

    The experiment's ``cohort`` should be a :class:`histox.Dataset`; a
    :class:`histox.splits.Split` passed to a stage selects its slides.
    """

    name = "tile_classifier"
    supported_tasks = ("classification", "regression")
    required_modalities = ("histology",)

    # --- Configuration --------------------------------------------------------

    def project(self):
        """The :class:`histox.Project` this method trains in."""
        import histox as hx
        project = self.config.get('project')
        if project is None:
            raise ValueError("Pass project=... (a histox.Project or its root) in method_config")
        return hx.Project(project) if isinstance(project, str) else project

    def model_params(self, task=None):
        """The :class:`histox.ModelParams` for this method."""
        import histox as hx
        if self.config.get('params') is not None:
            return self.config['params']
        kwargs = {k: v for k, v in self.config.items() if k not in ('project', 'params', 'n_out')}
        if 'loss' not in kwargs and task is not None and task.name in _LOSS_FOR_TASK:
            kwargs['loss'] = _LOSS_FOR_TASK[task.name]
        return hx.ModelParams(**kwargs)

    def build_model(self, data_spec: DataSpec):
        """Build an untrained model; the number of classes is needed."""
        n_out = self.config.get('n_out')
        if n_out is None:
            raise ValueError(
                "The output size is inferred from the data during fit(); to build "
                "a model before training, pass n_out in method_config")
        return self.model_params().build_model(num_classes=int(n_out))

    # --- Helpers ----------------------------------------------------------------

    def _outcome(self, experiment) -> str:
        return experiment.data_spec.fields[experiment.task.label_field]

    @staticmethod
    def _select(experiment, dataset, split, side):
        if dataset is not None:
            return dataset
        if split is None:
            return experiment.cohort
        if experiment.cohort is None:
            raise ValueError("A split needs the experiment's cohort (a histox.Dataset)")
        return experiment.cohort.filter(filters={'slide': list(getattr(split, f'{side}_slides'))})

    @staticmethod
    def _final_model(model_dir: str) -> Optional[str]:
        """The saved model of the last epoch in a model directory."""
        saved = glob.glob(join(model_dir, '*_epoch*'))
        saved = [p for p in saved if not p.endswith(('.parquet.gzip', '.png', '.csv'))]

        def epoch(path):
            tail = os.path.basename(path).rsplit('_epoch', 1)[-1]
            digits = ''.join(ch for ch in tail if ch.isdigit())
            return int(digits) if digits else -1

        return max(saved, key=epoch) if saved else None

    # --- Stages -----------------------------------------------------------------

    def stage_fit(
        self,
        experiment,
        train=None,
        val=None,
        *,
        split=None,
        exp_label: Optional[str] = None,
        **train_kwargs
    ) -> RunResult:
        """Train with :meth:`histox.Project.train`.

        Args:
            train, val (histox.Dataset, optional): Training and validation
                data. Default to the experiment's cohort, or to the sides of
                ``split``. Without validation data, trains with
                ``val_strategy='none'``.
            split (histox.splits.Split, optional): Train/validation split.
            exp_label (str, optional): Label added to the model name.
            **train_kwargs: Passed to :meth:`histox.Project.train`.
        """
        project = self.project()
        train_ds = self._select(experiment, train, split, 'train')
        if train_ds is None:
            raise ValueError("No training data: pass train=, split=, or set the experiment cohort")
        val_ds = self._select(experiment, val, split, 'val') if (val is not None or split is not None) else None
        if val_ds is not None:
            train_kwargs['val_dataset'] = val_ds
        elif not any(k.startswith('val_') for k in train_kwargs):
            train_kwargs['val_strategy'] = 'none'
        hp = self.model_params(experiment.task)
        before = _listing(project.models_dir)
        results = project.train(self._outcome(experiment), hp, dataset=train_ds,
                                exp_label=exp_label, **train_kwargs)
        model_dirs = _new_dirs(project.models_dir, before)
        if not model_dirs:
            raise RuntimeError("Training produced no model directory")
        model_dir = model_dirs[-1]
        model_path = self._final_model(model_dir)
        self._model_path = model_path
        result = RunResult(stage='fit', metadata={'model_dir': model_dir,
                                                  'model_path': model_path,
                                                  'results': _plain(results)})
        result.add_artifact('model', kind='tile_classifier', path=model_path or model_dir)
        return result

    def stage_evaluate(
        self,
        experiment,
        dataset=None,
        *,
        split=None,
        model: Optional[str] = None,
        level: str = 'slide',
        **eval_kwargs
    ) -> RunResult:
        """Evaluate with :meth:`histox.Project.evaluate` and score with the task.

        Args:
            dataset (histox.Dataset, optional): Evaluation data. Defaults to
                the validation side of ``split``, or the experiment's cohort.
            model (str, optional): Saved model; defaults to the one from fit.
            level (str): ``'slide'``, ``'patient'``, or ``'tile'``: which
                predictions the task metrics are computed on. Defaults to
                ``'slide'``.
            **eval_kwargs: Passed to :meth:`histox.Project.evaluate`.
        """
        import pandas as pd
        if level not in ('slide', 'patient', 'tile'):
            raise ValueError("level must be 'slide', 'patient', or 'tile'")
        project = self.project()
        ds = self._select(experiment, dataset, split, 'val')
        model = model or getattr(self, '_model_path', None)
        if model is None:
            raise ValueError("No trained model: run fit() first or pass model=")
        outcome = self._outcome(experiment)
        before = _listing(project.eval_dir)
        trainer_metrics = project.evaluate(model, outcome, dataset=ds, **eval_kwargs)
        eval_dirs = _new_dirs(project.eval_dir, before)
        predictions = None
        metrics: Dict[str, Any] = {}
        if eval_dirs:
            path = join(eval_dirs[-1], f'{level}_predictions_eval.parquet.gzip')
            if os.path.exists(path):
                predictions = pd.read_parquet(path)
                metrics = self._score(experiment.task, predictions, outcome)
        result = RunResult(stage='evaluate', metrics=metrics, predictions=predictions,
                           metadata={'model': str(model), 'level': level,
                                     'eval_dir': eval_dirs[-1] if eval_dirs else None,
                                     'trainer_metrics': _plain(trainer_metrics)})
        return result

    @staticmethod
    def _score(task, df, outcome) -> Dict[str, Any]:
        true_col = f'{outcome}-y_true'
        pred_cols = sorted((c for c in df.columns if c.startswith(f'{outcome}-y_pred')),
                           key=lambda c: int(c.rsplit('y_pred', 1)[-1] or 0))
        if true_col not in df.columns or not pred_cols:
            return {}
        if task.name == 'classification':
            return task.score(df[true_col].values.astype(int), df[pred_cols].values)
        if task.name == 'regression':
            return task.score(df[true_col].values.astype(float), df[pred_cols[0]].values)
        return {}


def _plain(obj: Any) -> Any:
    """Convert nested results (DictProxy, numpy) to plain Python for metadata."""
    import numpy as np
    if hasattr(obj, 'items'):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
