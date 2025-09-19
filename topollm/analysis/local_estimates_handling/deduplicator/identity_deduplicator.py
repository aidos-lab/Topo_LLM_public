from topollm.embeddings_data_prep.prepared_data_containers import PreparedData


class IdentityDeduplicator:
    """Identity deduplicator, i.e., do nothing."""

    def filter_data(
        self,
        prepared_data: PreparedData,
    ) -> PreparedData:
        """Return unmodified data."""
        return prepared_data
