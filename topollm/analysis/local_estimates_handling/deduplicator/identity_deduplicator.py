# Copyright 2024
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
# AUTHOR_2 (author2@example.com)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#


from topollm.embeddings_data_prep.prepared_data_containers import PreparedData


class IdentityDeduplicator:
    """Identity deduplicator, i.e., do nothing."""

    def filter_data(
        self,
        prepared_data: PreparedData,
    ) -> PreparedData:
        """Return unmodified data."""
        return prepared_data
