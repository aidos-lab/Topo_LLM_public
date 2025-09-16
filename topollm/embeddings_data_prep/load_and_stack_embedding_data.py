"""Load and stack embedding data from precomputed files."""

import logging
import pathlib
import pprint

import numpy as np
import pandas as pd
import zarr

from topollm.config_classes.data_processing_column_names.data_processing_column_names import DataProcessingColumnNames
from topollm.embeddings_data_prep.load_pickle_files_from_meta_path import load_pickle_files_from_meta_path
from topollm.path_management.embeddings.protocol import EmbeddingsPathManager
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)
default_data_processing_column_names = DataProcessingColumnNames()


def load_and_stack_embedding_data(
    embeddings_path_manager: EmbeddingsPathManager,
    data_processing_column_names: DataProcessingColumnNames = default_data_processing_column_names,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> pd.DataFrame:
    """Load the embedding data and metadata."""
    # Path for loading the precomputed embeddings
    array_path: pathlib.Path = embeddings_path_manager.array_dir_absolute_path

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{array_path = }",  # noqa: G004 - low overhead
        )

    if not array_path.exists():
        msg: str = f"{array_path = } does not exist."
        raise FileNotFoundError(
            msg,
        )

    # Path for loading the precomputed metadata
    meta_path = pathlib.Path(
        embeddings_path_manager.metadata_dir_absolute_path,
        "pickle_chunked_metadata_storage",
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{meta_path = }",  # noqa: G004 - low overhead
        )

    if not meta_path.exists():
        msg = f"{meta_path = } does not exist."
        raise FileNotFoundError(
            msg,
        )

    array_zarr = zarr.open(
        store=str(array_path),
        mode="r",
    )

    array_np: np.ndarray = np.array(
        array_zarr,
    )
    number_of_sentences: int = array_np.shape[0]
    number_of_tokens_per_sentence: int = array_np.shape[1]

    array_np = array_np.reshape(
        array_np.shape[0] * array_np.shape[1],
        array_np.shape[2],
    )

    loaded_metadata_chunks: list = load_pickle_files_from_meta_path(
        meta_path=meta_path,
    )

    if verbosity >= Verbosity.DEBUG:
        logger.info(
            msg=f"Loaded pickle files loaded_metadata_chunks:\n{loaded_metadata_chunks}",  # noqa: G004 - low overhead
        )

    # Note: This assumes that the batches saved in the embedding data computation are a dict which contains
    # different keys for the model_inputs and the metadata.
    input_ids_collection: list[list] = [
        metadata_chunk["model_inputs"][data_processing_column_names.input_ids].tolist()
        for metadata_chunk in loaded_metadata_chunks
    ]

    stacked_input_ids: np.ndarray = np.vstack(
        tup=input_ids_collection,
    )
    stacked_input_ids: np.ndarray = stacked_input_ids.reshape(
        stacked_input_ids.shape[0] * stacked_input_ids.shape[1],
    )

    sentence_idx: np.ndarray = np.array(
        [np.ones(shape=number_of_tokens_per_sentence) * i for i in range(number_of_sentences)],
    ).reshape(
        number_of_sentences * number_of_tokens_per_sentence,
    )

    # Note:
    # The "metadata" key in the batch saved in a metadata_chunk is currently lost.
    # You would need to add it to the full_df here if you want to use it in the downstream pipeline.

    full_data_dict: dict = {
        data_processing_column_names.embedding_vectors: list(array_np),
        data_processing_column_names.input_ids: list(stacked_input_ids),
        data_processing_column_names.sentence_idx: [int(x) for x in sentence_idx],
    }

    # # # #
    # Process the sequence-level metadata
    sequence_level_metadata_column_names: list[str] = [
        "dialogue_id",
        "turn_index",
    ]
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Processing the following sequence-level metadata columns:\n"  # noqa: G004 - low overhead
            f"{pprint.pformat(object=sequence_level_metadata_column_names)}",
        )

    for column_name in sequence_level_metadata_column_names:
        full_data_dict = process_sequence_level_metadata(
            column_name=column_name,
            loaded_metadata_chunks=loaded_metadata_chunks,
            number_of_tokens_per_sentence=number_of_tokens_per_sentence,
            full_data_dict=full_data_dict,
            verbosity=verbosity,
            logger=logger,
        )

    # # # #
    # Manually concatenate the elements in the token-level metadata (if it exist)

    # Determine the token-level metadata columns to process.
    # Check the first entry in the loaded metadata chunks to see which columns are present
    available_metadata_keys: list[str] = list(loaded_metadata_chunks[0]["metadata"].keys())

    # Column names which start with 'mask_' contain token-level masks, and should be processed
    token_level_mask_columns: list[str] = [
        column_name for column_name in available_metadata_keys if column_name.startswith("mask_")
    ]

    # Select the token-level metadata columns which we want to process:
    # - BIO-tags and POS-tags are always processed if they exist
    # - Any column which starts with 'mask_' is processed
    token_level_metadata_column_names: list[str] = [
        data_processing_column_names.bio_tags_name,
        data_processing_column_names.pos_tags_name,
        *token_level_mask_columns,
    ]
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Processing the following token-level metadata columns:\n"  # noqa: G004 - low overhead
            f"{pprint.pformat(object=token_level_metadata_column_names)}",
        )

    for column_name in token_level_metadata_column_names:
        full_data_dict = process_token_level_metadata(
            column_name=column_name,
            loaded_metadata_chunks=loaded_metadata_chunks,
            full_data_dict=full_data_dict,
            verbosity=verbosity,
            logger=logger,
        )

    full_df = pd.DataFrame(
        data=full_data_dict,
    )

    return full_df


def process_sequence_level_metadata(
    column_name: str,
    loaded_metadata_chunks: list,
    number_of_tokens_per_sentence: int,
    full_data_dict: dict,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> dict:
    """Process sequence-level metadata and add it to the full data dictionary."""
    if column_name in loaded_metadata_chunks[0]["metadata"]:
        # The metadata is a list of batches,
        # where each batch is a list of sentences,
        # where each sentence is a list of sequence-level tags
        # (for example, dialogue_id or turn_index).
        sequence_level_collection: list[list] = [
            metadata_chunk["metadata"][column_name] for metadata_chunk in loaded_metadata_chunks
        ]

        # Concatenate the list of sequence-level tags
        concatenated_sequence_level_tags: list = [tag for tags in sequence_level_collection for tag in tags]

        # Duplicate the sequence-level tags to match the number of tokens per sentence
        concatenated_sequence_level_tags_one_for_each_token = [
            tag for tag in concatenated_sequence_level_tags for _ in range(number_of_tokens_per_sentence)
        ]

        # Add the concatenated sequence tags to the full_data_dict
        full_data_dict[column_name] = concatenated_sequence_level_tags_one_for_each_token
    elif verbosity >= Verbosity.NORMAL:
        # If the column is not found in the metadata, log a message
        logger.info(
            msg=f"Column {column_name=} not found in metadata. "  # noqa: G004 - low overhead
            f"Skipping this column's sequence-level metadata processing.",
        )

    return full_data_dict


def process_token_level_metadata(
    column_name: str,
    loaded_metadata_chunks: list,
    full_data_dict: dict,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> dict:
    """Process token-level metadata and add it to the full data dictionary."""
    if column_name in loaded_metadata_chunks[0]["metadata"]:
        # The metadata is a list of batches,
        # where each batch is a list of sentences,
        # where each sentence is a list of token-wise tags (for example, POS tags or BIO-tags).
        token_level_collection: list[list[list]] = [
            metadata_chunk["metadata"][column_name] for metadata_chunk in loaded_metadata_chunks
        ]

        concatenated_token_level_batches: list = [
            token_batch for token_batches in token_level_collection for token_batch in token_batches
        ]

        # Concatenate the list of token-wise tags
        concatenated_token_level_tags: list = [
            token_tag for token_tags in concatenated_token_level_batches for token_tag in token_tags
        ]

        # Add the concatenated token tags to the full_data_dict
        full_data_dict[column_name] = concatenated_token_level_tags
    elif verbosity >= Verbosity.NORMAL:
        # If the column is not found in the metadata, log a message
        logger.info(
            msg=f"Column {column_name=} not found in metadata. Skipping this column's token-level metadata processing.",  # noqa: G004 - low overhead
        )

    return full_data_dict
