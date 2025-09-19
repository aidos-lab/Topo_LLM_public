"""Identity noiser, i.e., do nothing."""

from topollm.embeddings_data_prep.prepared_data_containers import PreparedData


class IdentityNoiser:
    """Identity noiser, i.e., do nothing."""

    def apply_noise_to_data(
        self,
        prepared_data: PreparedData,
    ) -> PreparedData:
        """Return unmodified data."""
        return prepared_data
