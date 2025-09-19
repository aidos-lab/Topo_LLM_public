"""Protocol for adding noise to prepared data."""

from typing import Protocol

from topollm.embeddings_data_prep.prepared_data_containers import PreparedData


class PreparedDataNoiser(Protocol):
    """Protocol for adding noise to prepared data."""

    def apply_noise_to_data(
        self,
        prepared_data: PreparedData,
    ) -> PreparedData:
        """Apply the selected method."""
        ...  # pragma: no cover
