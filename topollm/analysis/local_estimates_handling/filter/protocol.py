"""Protocol for filtering the data for local estimates computation."""

from typing import Protocol

from topollm.embeddings_data_prep.prepared_data_containers import PreparedData


class LocalEstimatesFilter(Protocol):
    """Protocol for filtering the data for local estimates computation."""

    def filter_data(
        self,
        prepared_data: PreparedData,
    ) -> PreparedData:
        """Filter the data for local estimates computation."""
        ...  # pragma: no cover
