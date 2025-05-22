# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Matthias Ruppik (mail@ruppik.net)
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

from topollm.model_inference.perplexity.saving.save_perplexity_results_list import (
    save_perplexity_array_as_zarr,
    save_perplexity_df_as_csv,
)
from topollm.path_management.embeddings.protocol import EmbeddingsPathManager
from topollm.typing.enums import PerplexityContainerSaveFormat, Verbosity

default_logger = logging.getLogger(__name__)


def save_concatenated_perplexity_results(
    token_perplexities_df: pd.DataFrame,
    token_perplexities_array: np.ndarray,
    embeddings_path_manager: EmbeddingsPathManager,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Save the concatenated perplexity results to a zarr array and a csv file."""
    for perplexity_container_save_format in [
        PerplexityContainerSaveFormat.CONCATENATED_ARRAY_AS_ZARR,
        PerplexityContainerSaveFormat.CONCATENATED_DATAFRAME_AS_CSV,
    ]:
        save_file_path = embeddings_path_manager.get_perplexity_container_save_file_absolute_path(
            perplexity_container_save_format=perplexity_container_save_format,
        )

        save_file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        match perplexity_container_save_format:
            case PerplexityContainerSaveFormat.CONCATENATED_ARRAY_AS_ZARR:
                save_perplexity_array_as_zarr(
                    perplexities_array=token_perplexities_array,
                    save_file_path=save_file_path,
                    verbosity=verbosity,
                    logger=logger,
                )
            case PerplexityContainerSaveFormat.CONCATENATED_DATAFRAME_AS_CSV:
                save_perplexity_df_as_csv(
                    perplexities_df=token_perplexities_df,
                    save_file_path=save_file_path,
                    verbosity=verbosity,
                    logger=logger,
                )
            case _:
                msg = "Unsupported perplexity container save format for this script."
                raise ValueError(msg)
