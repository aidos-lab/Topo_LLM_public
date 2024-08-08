# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (ruppik@hhu.de)
# Julius von Rohrscheidt (julius.rohrscheidt@helmholtz-muenchen.de)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import numpy as np
import pandas as pd

from topollm.config_classes.data_processing_column_names.data_processing_column_names import DataProcessingColumnNames
from topollm.config_classes.embeddings_data_prep.embeddings_data_prep_config import EmbeddingsDataPrepSamplingConfig
from topollm.embeddings_data_prep.prepared_data_containers import PreparedData
from topollm.logging.log_array_info import log_array_info
from topollm.typing.enums import Verbosity

default_data_processing_column_names = DataProcessingColumnNames()

default_logger = logging.getLogger(__name__)


def select_subsets_of_array_and_meta_df(
    input_data: PreparedData,
    embeddings_data_prep_sampling_config: EmbeddingsDataPrepSamplingConfig,
    data_processing_column_names: DataProcessingColumnNames = default_data_processing_column_names,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> tuple[
    PreparedData,
    np.ndarray,
]:
    """Select subsets of the arrays and metadata."""
    # TODO: Make the sampling method configurable
    # TODO: i.e., allow for sampling via the first sequences instead of random sampling
    array = input_data.array
    meta_df = input_data.meta_df

    rng = np.random.default_rng(
        seed=embeddings_data_prep_sampling_config.seed,
    )
    if len(array) >= embeddings_data_prep_sampling_config.num_samples:
        subsample_idx_vector: np.ndarray = rng.choice(
            range(len(array)),
            replace=False,
            size=embeddings_data_prep_sampling_config.num_samples,
        )
    else:
        subsample_idx_vector: np.ndarray = rng.choice(
            range(len(array)),
            replace=False,
            size=len(array),
        )

    subsampled_array = array[subsample_idx_vector]
    without_array_subsampled_df = meta_df.iloc[subsample_idx_vector]

    if verbosity >= Verbosity.NORMAL:
        log_array_info(
            subsample_idx_vector,
            array_name="subsample_idx_vector",
            logger=logger,
        )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"{subsampled_array.shape = }",  # noqa: G004 - low overhead
        )
        logger.info(
            f"Expected sample size: {embeddings_data_prep_sampling_config.num_samples = }",  # noqa: G004 - low overhead
        )

    # # # #
    # Add the subsample index to the metadata DataFrame
    without_array_subsampled_df[data_processing_column_names.subsample_idx] = list(
        subsample_idx_vector,
    )

    output_data = PreparedData(
        array=subsampled_array,
        meta_df=without_array_subsampled_df,
    )

    return_value = (
        output_data,
        subsample_idx_vector,
    )

    return return_value
