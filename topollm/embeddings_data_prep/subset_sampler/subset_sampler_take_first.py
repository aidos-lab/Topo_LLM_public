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


"""Implementation of the SubsetSampler protocol for random sampling."""

import logging

import numpy as np

from topollm.config_classes.embeddings_data_prep.sampling_config import EmbeddingsDataPrepSamplingConfig
from topollm.embeddings_data_prep.prepared_data_containers import PreparedData
from topollm.logging.log_array_info import log_array_info
from topollm.typing.enums import Verbosity

default_logger = logging.getLogger(__name__)


class SubsetSamplerTakeFirst:
    """Implementation of the SubsetSampler protocol for random sampling."""

    def __init__(
        self,
        embeddings_data_prep_sampling_config: EmbeddingsDataPrepSamplingConfig,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Initialize the SubsetSamplerTakeFirst."""
        self.embeddings_data_prep_sampling_config = embeddings_data_prep_sampling_config

        self.verbosity = verbosity
        self.logger = logger

    def sample_subsets(
        self,
        input_data: PreparedData,
    ) -> tuple[
        PreparedData,
        np.ndarray,
    ]:
        """Take the first num_samples many rows from the array and meta_df."""
        array = input_data.array
        meta_df = input_data.meta_df

        requested_num_samples = self.embeddings_data_prep_sampling_config.num_samples

        if array.shape[0] < requested_num_samples:
            actual_num_samples = array.shape[0]
            self.logger.warning(
                f"{requested_num_samples = } is larger than the number of available samples {array.shape[0] = }.",  # noqa: G004 - low overhead
            )
        else:
            actual_num_samples = requested_num_samples

        subsampled_array = array[:actual_num_samples]
        subsampled_df = meta_df.iloc[:actual_num_samples]
        subsample_idx_vector = np.arange(actual_num_samples)

        sampled_data = PreparedData(
            array=subsampled_array,
            meta_df=subsampled_df,
        )

        if self.verbosity >= Verbosity.NORMAL:
            log_array_info(
                subsample_idx_vector,
                array_name="subsample_idx_vector",
                logger=self.logger,
            )
            self.logger.info(
                f"{subsampled_array.shape = }",  # noqa: G004 - low overhead
            )
            self.logger.info(
                f"Expected sample size: {requested_num_samples = }",  # noqa: G004 - low overhead
            )

        return sampled_data, subsample_idx_vector
