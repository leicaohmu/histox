"""Exceptions raised by the HistoX workflow API."""


class WorkflowError(Exception):
    """Base class for workflow errors."""


class RegistryError(WorkflowError):
    """Base class for task and method registry errors."""


class DuplicateRegistrationError(RegistryError):
    """Raised when a registry name is reused without explicit replacement."""


class UnknownTaskError(RegistryError):
    """Raised when a requested task is not registered."""


class UnknownMethodError(RegistryError):
    """Raised when a requested method is not registered."""


class InvalidDataSpecError(WorkflowError):
    """Raised when a declared cohort does not satisfy a task or method."""


class IncompatibleWorkflowError(WorkflowError):
    """Raised when a method does not support the selected task."""


class UnsupportedStageError(WorkflowError):
    """Raised when a workflow method does not implement a lifecycle stage."""
