"""Built-in HistoX workflow tasks."""

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
