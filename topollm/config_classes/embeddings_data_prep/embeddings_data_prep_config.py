"""Configuration class for embedding data preparation."""

import pathlib

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.config_classes.embeddings_data_prep.filter_tokens_config import FilterTokensConfig
from topollm.config_classes.embeddings_data_prep.sampling_config import (
    EmbeddingsDataPrepSamplingConfig,
)


class EmbeddingsDataPrepConfig(ConfigBaseModel):
    """Configurations for specifying data preparation.

    Notes:
    - We use a feature flag in a different config group to enable or disable saving of the metadata sentences.
    - This config does not provide a config description, since the file paths in the path managers
    are assembled from the config descriptions of certain components of this config.

    """

    filter_tokens: FilterTokensConfig = Field(
        default=FilterTokensConfig(),
        title="Filter tokens.",
        description="Configurations for filtering tokens.",
    )

    sampling: EmbeddingsDataPrepSamplingConfig = Field(
        default=EmbeddingsDataPrepSamplingConfig(),
        title="Sampling configurations.",
        description="Configurations for specifying sampling.",
    )

    def get_partial_path(
        self,
    ) -> pathlib.Path:
        """Get the partial path for the embeddings data preparation."""
        path: pathlib.Path = pathlib.Path(
            # TODO: We will add additional config information here once we implement the token masking
            self.sampling.config_description,
        )

        return path
