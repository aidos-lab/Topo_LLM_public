from topollm.embeddings_data_prep.prepared_data_containers import PreparedData


class IdentityFilter:
    """Identity filter, i.e., no filtering is applied."""

    def filter_data(
        self,
        prepared_data: PreparedData,
    ) -> PreparedData:
        """Filter the data for local estimates computation."""
        return prepared_data
