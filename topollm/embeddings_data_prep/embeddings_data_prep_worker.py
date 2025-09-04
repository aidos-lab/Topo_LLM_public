"""Prepare the embedding data of a model and its metadata for further analysis."""

import logging
from typing import TYPE_CHECKING

from topollm.config_classes.main_config import MainConfig
from topollm.embeddings_data_prep.add_additional_metadata_to_meta_df import (
    add_additional_metadata_to_meta_df,
    add_token_name_column_to_meta_frame,
)
from topollm.embeddings_data_prep.load_and_stack_embedding_data import load_and_stack_embedding_data
from topollm.embeddings_data_prep.mask_tokens_of_arrays_and_meta import mask_tokens_of_arrays_and_meta
from topollm.embeddings_data_prep.prepared_data_containers import PreparedData
from topollm.embeddings_data_prep.remove_padding_and_extra_tokens import remove_padding_and_extra_tokens
from topollm.embeddings_data_prep.sample_subsets_of_arrays_and_meta import sample_subsets_of_array_and_meta_df
from topollm.embeddings_data_prep.save_prepared_data import save_prepared_data
from topollm.model_handling.tokenizer.load_modified_tokenizer_from_main_config import (
    load_modified_tokenizer_from_main_config,
)
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    import pandas as pd

    from topollm.path_management.embeddings.protocol import EmbeddingsPathManager

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def embeddings_data_prep_worker(
    main_config: MainConfig,
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

    (
        tokenizer,
        _,
    ) = load_modified_tokenizer_from_main_config(
        main_config=main_config,
        verbosity=verbosity,
        logger=logger,
    )

    (
        filtered_array,
        filtered_without_array_df,
    ) = remove_padding_and_extra_tokens(
        full_df=full_df,
        tokenizer=tokenizer,
        filter_tokens_config=main_config.embeddings_data_prep.filter_tokens,
        data_processing_column_names=main_config.data_processing_column_names,
        verbosity=verbosity,
        logger=logger,
    )

    filtered_data = PreparedData(
        array=filtered_array,
        meta_df=filtered_without_array_df,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Logging information about `filtered_data`:",
        )
        filtered_data.log_info(
            logger=logger,
        )

    # # # #
    # Filter vectors based on a token mask (which is part of the metadata).

    (
        filtered_masked_data,
        _,
    ) = mask_tokens_of_arrays_and_meta(
        input_data=filtered_data,
        token_masking_config=main_config.embeddings_data_prep.token_masking,
        verbosity=verbosity,
        logger=logger,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Logging information about `filtered_masked_data`:",
        )
        filtered_masked_data.log_info(
            logger=logger,
        )

    # # # #
    # Take token subsample

    (
        filtered_subsampled_data,
        _,
    ) = sample_subsets_of_array_and_meta_df(
        input_data=filtered_masked_data,
        embeddings_data_prep_sampling_config=main_config.embeddings_data_prep.sampling,
        data_processing_column_names=main_config.data_processing_column_names,
        verbosity=verbosity,
        logger=logger,
    )

    # Add the token names to the metadata DataFrame
    filtered_subsampled_augmented_without_array_df: pd.DataFrame = add_token_name_column_to_meta_frame(
        input_df=filtered_subsampled_data.meta_df,
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

    # # # #
    # Save the prepared data
    filtered_subsampled_prepared_data = PreparedData(
        array=filtered_subsampled_data.array,
        meta_df=filtered_subsampled_augmented_without_array_df,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Logging information about `filtered_subsampled_prepared_data`:",
        )
        filtered_subsampled_prepared_data.log_info(
            logger=logger,
        )

    save_prepared_data(
        embeddings_path_manager=embeddings_path_manager,
        prepared_data=filtered_subsampled_prepared_data,
        verbosity=verbosity,
        logger=logger,
    )
