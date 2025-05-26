class AppError(Exception):
    status_code: int = 500
    code: str = "internal_error"
    message: str = "An unexpected error occurred."
    details: dict[str, str] | None = None

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        status_code: int | None = None,
        details: dict[str, str] | None = None,
    ) -> None:
        self.message = message or self.message
        self.code = code or self.code
        self.status_code = status_code or self.status_code
        self.details = details
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"{self.code} ({self.status_code}): {self.message}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code={self.code!r}, status_code={self.status_code}, message={self.message!r})"


class ServiceError(AppError):
    """An error raised when a service operation fails."""

    status_code = 500
    code = "service_error"
    message = "Service operation failed."


class InfrastructureError(AppError):
    """Raised when an infrastructure service (DB, S3, etc.) fails."""


class VectorDatabaseError(InfrastructureError):
    """Raised when a DB operation fails."""

    status_code = 500
    code = "vectorDB_failed"
    message = "Vector DB failed."


class SemanticSearchError(VectorDatabaseError):
    """Raised when a semantic search operation fails."""

    status_code = 500
    code = "semantic_search_failed"
    message = "Semantic search failed."


class ConversationalSearchError(VectorDatabaseError):
    """Raised when a semantic search operation fails."""

    status_code = 500
    code = "semantic_search_failed"
    message = "Semantic search failed."


class IndexingError(VectorDatabaseError):
    """Raised when a index operation fails."""

    status_code = 500
    code = "indexing_failed"
    message = "Indexing Failed."


class EmbedderError(ServiceError):
    """Raised when a LLM operation fails."""

    status_code = 500
    code = "LMM_failed"
    message = "Failed to generate embedding."


class SimilarSearchError(VectorDatabaseError):
    """Raised when a similar search operation fails."""

    status_code = 500
    code = "similar_search_failed"
    message = "Similar search failed."
