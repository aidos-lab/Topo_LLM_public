# Copyright 2024-2025
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


"""Sample subsets of the arrays and metadata."""

import logging

import numpy as np
from pandas import DataFrame

from topollm.config_classes.data_processing_column_names.data_processing_column_names import DataProcessingColumnNames
from topollm.config_classes.embeddings_data_prep.sampling_config import (
    EmbeddingsDataPrepSamplingConfig,
)
from topollm.embeddings_data_prep.prepared_data_containers import PreparedData
from topollm.embeddings_data_prep.subset_sampler.factory import get_subset_sampler
from topollm.embeddings_data_prep.subset_sampler.protocol import SubsetSampler
from topollm.typing.enums import Verbosity

default_data_processing_column_names = DataProcessingColumnNames()

default_logger: logging.Logger = logging.getLogger(name=__name__)


def sample_subsets_of_array_and_meta_df(
    input_data: PreparedData,
    embeddings_data_prep_sampling_config: EmbeddingsDataPrepSamplingConfig,
    data_processing_column_names: DataProcessingColumnNames = default_data_processing_column_names,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> tuple[
    PreparedData,
    np.ndarray,
]:
    """Sample subsets of the arrays and metadata."""
    subset_sampler: SubsetSampler = get_subset_sampler(
        embeddings_data_prep_sampling_config=embeddings_data_prep_sampling_config,
        verbosity=verbosity,
        logger=logger,
    )
    (
        sampled_data,
        subsample_idx_vector,
    ) = subset_sampler.sample_subsets(
        input_data=input_data,
    )

    # # # #
    # Add the subsample index to the metadata DataFrame
    subsampled_df: DataFrame = sampled_data.meta_df
    subsampled_df[data_processing_column_names.subsample_idx] = list(
        subsample_idx_vector,
    )

    output_data = PreparedData(
        array=sampled_data.array,
        meta_df=subsampled_df,
    )

    return_value: tuple[
        PreparedData,
        np.ndarray,
    ] = (
        output_data,
        subsample_idx_vector,
    )

    return return_value
