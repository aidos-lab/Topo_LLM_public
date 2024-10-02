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

"""Sample subsets of the arrays and metadata."""

import logging

import numpy as np

from topollm.config_classes.data_processing_column_names.data_processing_column_names import DataProcessingColumnNames
from topollm.config_classes.embeddings_data_prep.sampling_config import (
    EmbeddingsDataPrepSamplingConfig,
)
from topollm.embeddings_data_prep.prepared_data_containers import PreparedData
from topollm.embeddings_data_prep.subset_sampler.factory import get_subset_sampler
from topollm.typing.enums import Verbosity

default_data_processing_column_names = DataProcessingColumnNames()

default_logger = logging.getLogger(__name__)


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
    subset_sampler = get_subset_sampler(
        embeddings_data_prep_sampling_config=embeddings_data_prep_sampling_config,
        verbosity=verbosity,
        logger=logger,
    )
    sampled_data, subsample_idx_vector = subset_sampler.sample_subsets(
        input_data=input_data,
    )

    # # # #
    # Add the subsample index to the metadata DataFrame
    subsampled_df = sampled_data.meta_df
    subsampled_df[data_processing_column_names.subsample_idx] = list(
        subsample_idx_vector,
    )

    output_data = PreparedData(
        array=sampled_data.array,
        meta_df=subsampled_df,
    )

    return_value = (
        output_data,
        subsample_idx_vector,
    )

    return return_value
