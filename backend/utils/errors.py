"""Shared exceptions and error helpers."""


class DatasetValidationError(Exception):
    """Raised when an uploaded dataset fails validation."""


class DatasetLoadingError(Exception):
    """Raised when a dataset cannot be loaded."""


class ToolError(Exception):
    """Raised when a controlled tool execution fails."""


class UnknownToolError(ToolError):
    """Raised when an agent requests a tool that does not exist."""


class InvalidToolArgumentsError(ToolError):
    """Raised when tool arguments fail validation."""


class LLMError(Exception):
    """Base error for OpenRouter failures."""


class LLMTimeoutError(LLMError):
    """Raised when the LLM call times out."""


class LLMConfigError(LLMError):
    """Raised when the LLM is not configured (missing API key)."""


class LLMQuotaError(LLMError):
    """Raised for 402/429/403 credit or quota failures (not worth retrying)."""


class WorkflowLimitError(Exception):
    """Raised when agent step limits are exceeded."""