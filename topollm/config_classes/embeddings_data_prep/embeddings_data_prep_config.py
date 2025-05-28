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


"""Configuration class for embedding data preparation."""

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.config_classes.embeddings_data_prep.filter_tokens_config import FilterTokensConfig
from topollm.config_classes.embeddings_data_prep.sampling_config import (
    EmbeddingsDataPrepSamplingConfig,
)


class EmbeddingsDataPrepConfig(ConfigBaseModel):
    """Configurations for specifying data preparation."""

    sampling: EmbeddingsDataPrepSamplingConfig = Field(
        default=EmbeddingsDataPrepSamplingConfig(),
        title="Sampling configurations.",
        description="Configurations for specifying sampling.",
    )

    filter_tokens: FilterTokensConfig = Field(
        default=FilterTokensConfig(),
        title="Filter tokens.",
        description="Configurations for filtering tokens.",
    )

    # Note: We use a feature flag in a different config group to enable or disable saving of the metadata sentences.
    @property
    def config_description(
        self,
    ) -> str:
        """Get the description of the config."""
        desc = f"{self.sampling.config_description}"

        return desc
