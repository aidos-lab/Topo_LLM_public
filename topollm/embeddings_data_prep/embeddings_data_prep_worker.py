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

import numpy as np
import pandas as pd
import torch

from topollm.config_classes.main_config import MainConfig
from topollm.embeddings_data_prep.load_and_stack_embedding_data import load_and_stack_embedding_data
from topollm.embeddings_data_prep.prepared_data_containers import PreparedData
from topollm.embeddings_data_prep.remove_padding_and_extra_tokens import remove_padding_and_extra_tokens
from topollm.embeddings_data_prep.save_prepared_data import save_prepared_data
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
        verbosity=verbosity,
        logger=logger,
    )

    tokenizer, _ = load_modified_tokenizer_from_main_config(
        main_config=main_config,
        verbosity=verbosity,
        logger=logger,
    )

    arr_no_pad, meta_no_pad, sentence_idx_no_pad = remove_padding_and_extra_tokens(
        full_df=full_df,
        tokenizer=tokenizer,
        filter_tokens_config=main_config.embeddings_data_prep.filter_tokens,
        verbosity=verbosity,
        logger=logger,
    )

    # Choose size of a meta sample which is used to take subsets for a point-wise
    # comparison of local estimators.
    # Sample size of the arrays
    sample_size = main_config.embeddings_data_prep.num_samples

    arr_no_pad_subsampled, meta_no_pad_subsampled, sentence_idx_no_pad_subsampled, subsample_idx = (
        select_subsets_of_arrays_and_meta(
            arr_no_pad=arr_no_pad,
            meta_no_pad=meta_no_pad,
            sentence_idx_no_pad=sentence_idx_no_pad,
            sample_size=sample_size,
            verbosity=verbosity,
            logger=logger,
        )
    )

    # # # #
    # Convert the metadata to a DataFrame

    # x of type 'numpy.int64' needs to be explicitly converted to an integer,
    # otherwise the convert_ids_to_tokens() method will raise the error:
    # TypeError: 'numpy.int64' object is not iterable

    token_names_no_pad_subsampled = [tokenizer.convert_ids_to_tokens(int(x)) for x in meta_no_pad_subsampled]

    meta_frame_no_pad_subsampled: pd.DataFrame = pd.DataFrame(
        {
            "token_id": list(meta_no_pad_subsampled),
            "token_name": list(token_names_no_pad_subsampled),
            "sentence_idx": list(sentence_idx_no_pad_subsampled),
            "subsample_idx": list(subsample_idx),
        },
    )

    # # # #
    # Optionally add sentence information to the metadata
    if main_config.feature_flags.embeddings_data_prep.write_sentences_to_meta:
        # Decode the token ids to tokens
        meta_tokens_column_name: str = main_config.embeddings_data_prep.meta_tokens_column_name
        meta_tokens = [tokenizer.convert_ids_to_tokens(int(x)) for x in list(full_df.meta)]
        full_df[meta_tokens_column_name] = meta_tokens  # Add the decoded tokens to the DataFrame

        grouped_df = (
            full_df.iloc[:, 1:]
            .groupby(
                by="sentence_idx",
                sort=False,
            )[meta_tokens_column_name]
            .apply(" ".join)
            .reset_index()
        )

        # grouped_df["meta_tokens_joined"] = grouped_df[meta_tokens_column_name].apply(" ".join)

        meta_frame_no_pad_subsampled = meta_frame_no_pad_subsampled.merge(
            grouped_df,
            on="sentence_idx",
        )

        pass

    if verbosity >= Verbosity.NORMAL:
        log_array_info(
            arr_no_pad_subsampled,
            array_name="arr_no_pad_subsampled",
            logger=logger,
        )
        log_dataframe_info(
            meta_frame_no_pad_subsampled,
            df_name="meta_frame",
            logger=logger,
        )

    # # # #
    # Save the prepared data
    prepared_data = PreparedData(
        arr_no_pad=arr_no_pad_subsampled,
        meta_frame=meta_frame_no_pad_subsampled,
    )

    save_prepared_data(
        embeddings_path_manager=embeddings_path_manager,
        prepared_data=prepared_data,
        verbosity=verbosity,
        logger=logger,
    )


def select_subsets_of_arrays_and_meta(
    arr_no_pad: np.ndarray,
    meta_no_pad: np.ndarray,
    sentence_idx_no_pad: np.ndarray,
    sample_size: int,
    seed: int = 42,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Select subsets of the arrays and metadata."""
    # TODO: Make the sampling method configurable
    # TODO: i.e., allow for sampling via the first sequences instead of random sampling
    rng = np.random.default_rng(
        seed=seed,
    )
    if len(arr_no_pad) >= sample_size:
        subsample_idx: np.ndarray = rng.choice(
            range(len(arr_no_pad)),
            replace=False,
            size=sample_size,
        )
    else:
        subsample_idx: np.ndarray = rng.choice(
            range(len(arr_no_pad)),
            replace=False,
            size=len(arr_no_pad),
        )

    if verbosity >= Verbosity.NORMAL:
        log_array_info(
            subsample_idx,
            array_name="subsample_idx",
            logger=logger,
        )

    arr_no_pad_subsampled = arr_no_pad[subsample_idx]
    meta_no_pad_subsampled = meta_no_pad[subsample_idx]
    sentence_idx_no_pad_subsampled = sentence_idx_no_pad[subsample_idx]

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"{arr_no_pad_subsampled.shape = }",  # noqa: G004 - low overhead
        )
        logger.info(
            f"Expected sample size: {sample_size = }",  # noqa: G004 - low overhead
        )

    return_value = (
        arr_no_pad_subsampled,
        meta_no_pad_subsampled,
        sentence_idx_no_pad_subsampled,
        subsample_idx,
    )

    return return_value
