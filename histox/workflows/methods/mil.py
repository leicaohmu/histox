"""Workflow adapter for multiple-instance learning (MIL) on feature bags."""

from pathlib import Path
from typing import Any, Dict

from .._base import Method
from .._registry import register_method
from .._types import DataSpec, RunResult

_LOSS_FOR_TASK = {'classification': 'cross_entropy', 'regression': 'mse'}


@register_method("mil")
class MILMethod(Method):
    """Train and evaluate a MIL model with :mod:`histox.mil`.

    The stages call :func:`histox.mil.train_mil` and
    :func:`histox.mil.eval_mil`, so an experiment trains exactly as those
    functions do.

    Configuration (``method_config``):
        model (str or callable): MIL model, e.g. ``'attention_mil'``,
            ``'transmil'``. Defaults to ``'attention_mil'``.
        bags (str): Directory of feature bags (see :class:`histox.Bag`).
        n_feats, n_out (int, optional): Needed only for :meth:`build_model`
            before training; ``fit`` infers them from the data.
        Any other keys are passed to :func:`histox.mil.mil_config`
        (``epochs``, ``lr``, ``batch_size``, ``loss``, ...). ``loss``
        defaults to ``'cross_entropy'`` for classification and ``'mse'`` for
        regression.

    The experiment's ``cohort`` should be a :class:`histox.Dataset`; a
    :class:`histox.splits.Split` passed to a stage selects its slides.
    """

    name = "mil"
    supported_tasks = ("classification", "regression")
    required_modalities = ("histology",)
    supported_explainers = ("attention",)

    _own_keys = ('model', 'bags', 'n_feats', 'n_out')

    # --- Configuration --------------------------------------------------------

    def trainer_config(self, task=None):
        """The :class:`histox.mil.TrainerConfig` for this method."""
        from histox.mil import mil_config
        kwargs = {k: v for k, v in self.config.items() if k not in self._own_keys}
        if 'loss' not in kwargs and task is not None and task.name in _LOSS_FOR_TASK:
            kwargs['loss'] = _LOSS_FOR_TASK[task.name]
        return mil_config(self.config.get('model', 'attention_mil'), **kwargs)

    def build_model(self, data_spec: DataSpec):
        """Build an untrained model (requires ``n_feats`` and ``n_out``)."""
        if 'n_feats' not in self.config or 'n_out' not in self.config:
            raise ValueError(
                "MIL model size is inferred from the data during fit(); to build "
                "a model before training, pass n_feats and n_out in method_config")
        return self.trainer_config().build_model(int(self.config['n_feats']),
                                                 int(self.config['n_out']))

    # --- Helpers ----------------------------------------------------------------

    def _outcome(self, experiment) -> str:
        field = experiment.task.label_field
        return experiment.data_spec.fields[field]

    def _bags(self, bags):
        bags = bags if bags is not None else self.config.get('bags')
        if bags is None:
            raise ValueError("Pass bags=... to the stage or in method_config")
        return str(bags)

    @staticmethod
    def _select(experiment, dataset, split, side):
        if dataset is not None:
            return dataset
        if split is None:
            return experiment.cohort
        if experiment.cohort is None:
            raise ValueError("A split needs the experiment's cohort (a histox.Dataset)")
        slides = list(getattr(split, f'{side}_slides'))
        return experiment.cohort.filter(filters={'slide': slides})

    def _outdir(self, experiment, outdir, name):
        if outdir is not None:
            return str(outdir)
        if experiment.output is not None:
            return str(Path(experiment.output) / name)
        return name

    # --- Stages -----------------------------------------------------------------

    def stage_fit(
        self,
        experiment,
        train=None,
        val=None,
        *,
        split=None,
        bags=None,
        outdir=None,
        **train_kwargs
    ) -> RunResult:
        """Train with :func:`histox.mil.train_mil`.

        Args:
            train, val (histox.Dataset, optional): Training and validation
                data. Default to the experiment's cohort, or to the sides of
                ``split``.
            split (histox.splits.Split, optional): Train/validation split of
                the experiment's cohort.
            bags (str, optional): Bag directory; defaults to the config.
            outdir (str, optional): Where to write the model directory.
            **train_kwargs: Passed to :func:`histox.mil.train_mil`.
        """
        from histox.mil import train_mil
        train_ds = self._select(experiment, train, split, 'train')
        val_ds = self._select(experiment, val, split, 'val') if (val is not None or split is not None) else None
        if train_ds is None:
            raise ValueError("No training data: pass train=, split=, or set the experiment cohort")
        config = self.trainer_config(experiment.task)
        learner = train_mil(config, train_ds, val_ds, outcomes=self._outcome(experiment),
                            bags=self._bags(bags),
                            outdir=self._outdir(experiment, outdir, 'mil'), **train_kwargs)
        model_dir = str(learner.path)
        self._model_dir = model_dir
        experiment._model = learner.model
        result = RunResult(stage='fit', metadata={'model_dir': model_dir,
                                                  'trainer_config': config.to_dict()})
        result.add_artifact('model', kind='mil_model', path=model_dir)
        return result

    def stage_evaluate(
        self,
        experiment,
        dataset=None,
        *,
        split=None,
        bags=None,
        model_dir=None,
        outdir=None,
        **eval_kwargs
    ) -> RunResult:
        """Evaluate with :func:`histox.mil.eval_mil` and score with the task.

        Args:
            dataset (histox.Dataset, optional): Evaluation data. Defaults to
                the validation side of ``split``, or the experiment's cohort.
            model_dir (str, optional): Trained model; defaults to the one
                from ``fit``.
        """
        from histox.mil import eval_mil
        ds = self._select(experiment, dataset, split, 'val')
        model_dir = model_dir or getattr(self, '_model_dir', None)
        if model_dir is None:
            raise ValueError("No trained model: run fit() first or pass model_dir=")
        df = eval_mil(model_dir, ds, outcomes=self._outcome(experiment),
                      bags=self._bags(bags),
                      outdir=self._outdir(experiment, outdir, 'mil_eval'), **eval_kwargs)
        metrics = self._score(experiment.task, df)
        result = RunResult(stage='evaluate', metrics=metrics, predictions=df,
                           metadata={'model_dir': str(model_dir)})
        return result

    @staticmethod
    def _score(task, df) -> Dict[str, Any]:
        pred_cols = [c for c in df.columns if c.startswith('y_pred')]
        if 'y_true' not in df.columns or not pred_cols:
            return {}
        if task.name == 'classification':
            return task.score(df['y_true'].values, df[pred_cols].values)
        if task.name == 'regression':
            return task.score(df['y_true'].values.astype(float), df[pred_cols[0]].values)
        return {}
