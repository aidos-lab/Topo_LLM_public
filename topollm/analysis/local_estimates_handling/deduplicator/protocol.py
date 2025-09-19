"""Protocol for deduplicating prepared data."""

from typing import Protocol

from topollm.embeddings_data_prep.prepared_data_containers import PreparedData


class PreparedDataDeduplicator(Protocol):
    """Protocol for deduplicating prepared data."""

    def filter_data(
        self,
        prepared_data: PreparedData,
    ) -> PreparedData:
        """Apply the selected method."""
        ...  # pragma: no cover
