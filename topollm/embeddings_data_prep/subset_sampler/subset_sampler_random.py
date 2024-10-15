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

"""Implementation of the SubsetSampler protocol for random sampling."""

import logging

import numpy as np

from topollm.config_classes.embeddings_data_prep.sampling_config import EmbeddingsDataPrepSamplingConfig
from topollm.embeddings_data_prep.prepared_data_containers import PreparedData
from topollm.logging.log_array_info import log_array_info
from topollm.typing.enums import Verbosity

default_logger = logging.getLogger(__name__)


class SubsetSamplerRandom:
    """Implementation of the SubsetSampler protocol for random sampling."""

    def __init__(
        self,
        embeddings_data_prep_sampling_config: EmbeddingsDataPrepSamplingConfig,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Initialize the SubsetSamplerRandom."""
        self.embeddings_data_prep_sampling_config: EmbeddingsDataPrepSamplingConfig = (
            embeddings_data_prep_sampling_config
        )

        self.rng = np.random.default_rng(
            seed=embeddings_data_prep_sampling_config.seed,
        )

        self.verbosity: Verbosity = verbosity
        self.logger: logging.Logger = logger

    def sample_subsets(
        self,
        input_data: PreparedData,
    ) -> tuple[
        PreparedData,
        np.ndarray,
    ]:
        """Sample a random subset of the rows of the array and the corresponding metadata."""
        array = input_data.array
        meta_df = input_data.meta_df

        requested_num_samples = self.embeddings_data_prep_sampling_config.num_samples

        if len(array) >= requested_num_samples:
            subsample_idx_vector: np.ndarray = self.rng.choice(
                range(len(array)),
                replace=False,
                size=requested_num_samples,
            )
        else:
            self.logger.warning(
                f"{requested_num_samples = } is larger than the number of available samples {array.shape[0] = }.",  # noqa: G004 - low overhead
            )
            subsample_idx_vector: np.ndarray = self.rng.choice(
                range(len(array)),
                replace=False,
                size=len(array),
            )

        subsampled_array = array[subsample_idx_vector]
        subsampled_df = meta_df.iloc[subsample_idx_vector]

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
