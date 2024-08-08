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

"""Prepare the embedding data of a model and its metadata for further analysis."""

import logging
from typing import TYPE_CHECKING

import pandas as pd
import torch

from topollm.config_classes.main_config import MainConfig
from topollm.embeddings_data_prep.add_additional_metadata_to_meta_df import (
    add_additional_metadata_to_meta_df,
    add_token_name_column_to_meta_frame,
)
from topollm.embeddings_data_prep.load_and_stack_embedding_data import load_and_stack_embedding_data
from topollm.embeddings_data_prep.prepared_data_containers import PreparedData
from topollm.embeddings_data_prep.remove_padding_and_extra_tokens import remove_padding_and_extra_tokens
from topollm.embeddings_data_prep.save_prepared_data import save_prepared_data
from topollm.embeddings_data_prep.select_subsets_of_arrays_and_meta import select_subsets_of_arrays_and_meta
from topollm.logging.log_array_info import log_array_info
from topollm.logging.log_dataframe_info import log_dataframe_info
from topollm.model_handling.tokenizer.load_modified_tokenizer_from_main_config import (
    load_modified_tokenizer_from_main_config,
)
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    from topollm.path_management.embeddings.protocol import EmbeddingsPathManager

default_logger = logging.getLogger(__name__)


def embeddings_data_prep_worker(
    main_config: MainConfig,
    device: torch.device,  # noqa: ARG001 - placeholder for future use
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Prepare the embedding data of a model and its metadata for further analysis."""
    embeddings_path_manager: EmbeddingsPathManager = get_embeddings_path_manager(
        main_config=main_config,
        logger=logger,
    )

    full_df: pd.DataFrame = load_and_stack_embedding_data(
        embeddings_path_manager=embeddings_path_manager,
        data_processing_column_names=main_config.data_processing_column_names,
        verbosity=verbosity,
        logger=logger,
    )

    tokenizer, _ = load_modified_tokenizer_from_main_config(
        main_config=main_config,
        verbosity=verbosity,
        logger=logger,
    )

    filtered_array, filtered_without_array_df = remove_padding_and_extra_tokens(
        full_df=full_df,
        tokenizer=tokenizer,
        filter_tokens_config=main_config.embeddings_data_prep.filter_tokens,
        data_processing_column_names=main_config.data_processing_column_names,
        verbosity=verbosity,
        logger=logger,
    )

    filtered_subsampled_array, filtered_subsampled_without_array_df, subsample_idx_vector = (
        select_subsets_of_arrays_and_meta(
            array=filtered_array,
            without_array_df=filtered_without_array_df,
            embeddings_data_prep_sampling_config=main_config.embeddings_data_prep.sampling,
            verbosity=verbosity,
            logger=logger,
        )
    )

    # # # #
    # Add the subsample index to the metadata DataFrame
    filtered_subsampled_without_array_df[main_config.data_processing_column_names.subsample_idx] = list(
        subsample_idx_vector,
    )

    # Add the token names to the metadata DataFrame
    filtered_subsampled_augmented_without_array_df = add_token_name_column_to_meta_frame(
        input_df=filtered_subsampled_without_array_df,
        tokenizer=tokenizer,
        data_processing_column_names=main_config.data_processing_column_names,
    )

    # # # #
    # Optionally add sentence information to the metadata
    if main_config.feature_flags.embeddings_data_prep.add_additional_metadata:
        filtered_subsampled_augmented_without_array_df = add_additional_metadata_to_meta_df(
            full_df=full_df,
            meta_df_to_modify=filtered_subsampled_augmented_without_array_df,
            tokenizer=tokenizer,
            data_processing_column_names=main_config.data_processing_column_names,
            write_tokens_list_to_meta=main_config.feature_flags.embeddings_data_prep.write_tokens_list_to_meta,
            write_concatenated_tokens_to_meta=main_config.feature_flags.embeddings_data_prep.write_concatenated_tokens_to_meta,
        )

    if verbosity >= Verbosity.NORMAL:
        log_array_info(
            filtered_subsampled_array,
            array_name="filtered_subsampled_array",
            logger=logger,
        )
        log_dataframe_info(
            filtered_subsampled_augmented_without_array_df,
            df_name="filtered_subsampled_augmented_without_array_df",
            logger=logger,
        )

    # # # #
    # Save the prepared data
    prepared_data = PreparedData(
        array=filtered_subsampled_array,
        meta_df=filtered_subsampled_augmented_without_array_df,
    )

    save_prepared_data(
        embeddings_path_manager=embeddings_path_manager,
        prepared_data=prepared_data,
        verbosity=verbosity,
        logger=logger,
    )
