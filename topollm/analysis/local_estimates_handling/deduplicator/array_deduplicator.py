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

from topollm.embeddings_data_prep.prepared_data_containers import PreparedData
from topollm.logging.log_array_info import log_array_info
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


class ArrayDeduplicator:
    """Remove duplicate vectors and corresponding metadata from the prepared data."""

    def __init__(
        self,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Initialize the deduplicator."""
        self.verbosity: Verbosity = verbosity
        self.logger: logging.Logger = logger

    def filter_data(
        self,
        prepared_data: PreparedData,
    ) -> PreparedData:
        """Applay numpy.unique function to the array and align metadata."""
        input_array = prepared_data.array
        input_meta_frame = prepared_data.meta_df

        unique_vectors, indices_of_original_array = np.unique(
            ar=input_array,
            axis=0,
            return_index=True,
        )

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg=f"unique_vectors.shape = {unique_vectors.shape}",  # noqa: G004 - low overhead
            )
            # Log if duplicates were removed
            if len(unique_vectors) < len(input_array):
                self.logger.info(
                    msg=f"Removed {len(input_array) - len(unique_vectors) = } duplicate vectors.",  # noqa: G004 - low overhead
                )
        if self.verbosity >= Verbosity.DEBUG:
            log_array_info(
                array_=indices_of_original_array,
                array_name="indices_of_original_array",
                logger=self.logger,
            )

        # Keep same order of original vectors by sorting the indices
        sorted_indices_of_original_array = np.sort(
            indices_of_original_array,
        )

        output_array = input_array[sorted_indices_of_original_array]
        output_meta_frame: pd.DataFrame = input_meta_frame.iloc[sorted_indices_of_original_array]

        output_data = PreparedData(
            array=output_array,
            meta_df=output_meta_frame,
        )

        return output_data
