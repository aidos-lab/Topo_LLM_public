# Copyright 2024-2025
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#


"""Script for loading cached features and saving them into a format for the Topo_LLM repository."""

import argparse
import logging
import os
import pathlib
import sys
from dataclasses import dataclass
from enum import StrEnum, auto

import click
import pandas as pd
import torch
import transformers
from tqdm import tqdm

from topollm.logging.create_and_configure_global_logger import create_and_configure_global_logger
from topollm.typing.enums import Verbosity

TOPO_LLM_REPOSITORY_BASE_PATH: str = os.path.expandvars(
    path=os.getenv(
        key="TOPO_LLM_REPOSITORY_BASE_PATH",
        default="${HOME}/git-source/Topo_LLM",
    ),
)

global_logger: logging.Logger = create_and_configure_global_logger(
    name=__name__,
    file=__file__,
)
default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


class DataMode(StrEnum):
    """Enum for the data modes."""

    TRIPPY = auto()
    TRIPPY_R = auto()


class MetadataHandlingMode(StrEnum):
    """Enum for the metadata handling modes."""

    IGNORE = auto()
    CREATE_AND_SAVE_BIO_TAGS = auto()


@dataclass
class ProcessedDataPathsCollection:
    """Class to store the input and output paths for the processed data."""

    checkpoints_root_dir: pathlib.Path
    post_processed_cached_features_dir: pathlib.Path


def get_processed_data_paths_collection(
    data_mode: DataMode,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> ProcessedDataPathsCollection:
    """Get the paths for the processed data."""
    match data_mode:
        case DataMode.TRIPPY:
            checkpoints_root_dir = pathlib.Path(
                TOPO_LLM_REPOSITORY_BASE_PATH,
                "data/models/trippy_checkpoints/",
            )
        case DataMode.TRIPPY_R:
            checkpoints_root_dir = pathlib.Path(
                TOPO_LLM_REPOSITORY_BASE_PATH,
                "data/models/trippy_r_checkpoints/",
                "multiwoz21/all_checkpoints",
            )
        case _:
            msg: str = f"Unknown data mode: {data_mode=}"
            raise ValueError(
                msg,
            )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{data_mode=}",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{checkpoints_root_dir=}",  # noqa: G004 - low overhead
        )

    post_processed_cached_features_dir = pathlib.Path(
        checkpoints_root_dir,
        "post_processed_cached_features",
        "multiwoz21",
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{post_processed_cached_features_dir=}",  # noqa: G004 - low overhead
        )
    post_processed_cached_features_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed_data_path_collection = ProcessedDataPathsCollection(
        checkpoints_root_dir=checkpoints_root_dir,
        post_processed_cached_features_dir=post_processed_cached_features_dir,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"processed_data_path_collection:\n{processed_data_path_collection}",  # noqa: G004 - low overhead
        )

    return processed_data_path_collection


@click.command(
    help="Load cached features and save them into a format for the Topo_LLM repository.",
)
@click.option(
    "--data-mode",
    type=click.Choice(choices=list(DataMode)),
    default=DataMode.TRIPPY_R,
    show_default=True,
    help="Data mode to use.",
)
@click.option(
    "--metadata-handling-mode",
    type=click.Choice(choices=list(MetadataHandlingMode)),
    default=MetadataHandlingMode.CREATE_AND_SAVE_BIO_TAGS,
    show_default=True,
    help="Metadata handling mode.",
)
@click.option(
    "--number-of-elements-to-log",
    type=int,
    default=5,
    show_default=True,
    help="Number of elements to log.",
)
@click.option(
    "--verbosity",
    type=Verbosity,
    # Note: Do not add `type=click.Choice()` parameter here, as this is problematic with IntEnum in Click.
    default=Verbosity.NORMAL,
    show_default=True,
    help="Verbosity level.",
)
def main(
    data_mode: DataMode = DataMode.TRIPPY_R,
    metadata_handling_mode: MetadataHandlingMode = MetadataHandlingMode.CREATE_AND_SAVE_BIO_TAGS,
    number_of_elements_to_log: int = 5,
    verbosity: Verbosity = Verbosity.NORMAL,
) -> None:
    """Load cached features and save them into a format for the Topo_LLM repository."""
    logger: logging.Logger = global_logger

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{TOPO_LLM_REPOSITORY_BASE_PATH=}",  # noqa: G004 - low overhead
        )

    # ======================================================== #
    # File paths
    # ======================================================== #

    processed_data_path_collection: ProcessedDataPathsCollection = get_processed_data_paths_collection(
        data_mode=data_mode,
        verbosity=verbosity,
        logger=logger,
    )

    # Add folder to Python path so that the custom modules which are necessary for loading are found.
    # This is necessary to import modules from the project, such as the 'utils_dst' module.
    match data_mode:
        case DataMode.TRIPPY:
            # For Trippy, add the project root directory to Python path.
            sys.path.insert(
                0,
                str(pathlib.Path(__file__).resolve().parent.parent),
            )
        case DataMode.TRIPPY_R:
            sys.path.insert(
                0,
                str(
                    pathlib.Path(
                        pathlib.Path(__file__).resolve().parent.parent,
                        "trippy_r_code",
                    ),
                ),
            )
        case _:
            msg: str = f"Unknown data mode: {data_mode=}"
            raise ValueError(
                msg,
            )

    # Select the splits to process
    match data_mode:
        case DataMode.TRIPPY:
            splits_to_process: list[str] = [
                "train",
                "dev",
                "test",
            ]
        case DataMode.TRIPPY_R:
            splits_to_process: list[str] = [
                "train",
                "dev",
                "test",
            ]
        case _:
            msg: str = f"Unknown data mode: {data_mode=}"
            raise ValueError(
                msg,
            )

    # ======================================================== #
    # Prepare additional objects
    # ======================================================== #

    # The tokenizer is not strictly necessary for this script,
    # but it is useful for debugging and understanding the data.
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="roberta-base",
    )

    # ======================================================== #
    # Process data
    # ======================================================== #

    for split_identifier in tqdm(
        iterable=splits_to_process,
        desc="Processing cached feature splits",
    ):
        file_path = pathlib.Path(
            processed_data_path_collection.checkpoints_root_dir,
            f"cached_{split_identifier}_features",  # Note: File has no extension
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"{file_path=}",  # noqa: G004 - low overhead
            )

        if not file_path.exists():
            logger.warning(
                msg=f"File does not exist: {file_path=}",  # noqa: G004 - low overhead
            )
            continue

        # Load the cached features.
        # Notes:
        # - The module 'utils_dst' needs to be accessible for this loading to work.
        # - `weights_only=False` is necessary because the saved object needs to execute code.
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Loading cached features from {file_path=} ... (This might take some time)",  # noqa: G004 - low overhead
            )
        loaded_features: list = torch.load(
            f=file_path,
            weights_only=False,
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Loading cached features from {file_path=} DONE",  # noqa: G004 - low overhead
            )

        # Additional processing of the loaded features
        match data_mode:
            case DataMode.TRIPPY:
                # - For Trippy, the loaded features are already in the correct format,
                #   i.e., a list of utils_dst.InputFeatures objects
                # - Here, utils_dst is from the Trippy codebase.
                cached_features: list = loaded_features
            case DataMode.TRIPPY_R:
                # - For Trippy-R, the loaded features are a tuple of length two,
                #   where the first element is a list of utils_dst.InputFeatures objects
                # - Here, utils_dst is from the Trippy-R codebase.
                # - The second element is a list of utils_dst.DSTExample objects.
                # - We only need the first element.
                cached_features: list = loaded_features[0]

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Loaded object of {type(cached_features)=} with {len(cached_features)=}",  # noqa: G004 - low overhead
            )

        if len(cached_features) == 0:
            logger.warning(
                msg=f"Loaded object is empty: {file_path=}",  # noqa: G004 - low overhead
            )
            continue

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"List items are of type {type(cached_features[0])=}",  # noqa: G004 - low overhead
            )

            logger.info(
                msg=f"-------- First {number_of_elements_to_log} elements --------",  # noqa: G004 - low overhead
            )
            for i in range(min(len(cached_features), number_of_elements_to_log)):
                logger.info(
                    msg=f"cached_features[{i}]:\n{cached_features[i]}",  # noqa: G004 - low overhead
                )
            logger.info(
                msg=f"-------- Last {number_of_elements_to_log} elements --------",  # noqa: G004 - low overhead
            )
            for i in range(len(cached_features) - number_of_elements_to_log, len(cached_features)):
                logger.info(
                    msg=f"cached_features[{i}]:\n{cached_features[i]}",  # noqa: G004 - low overhead
                )

        input_ids_list: list = []
        attention_mask_list: list = []
        dialogue_ids_list: list = []

        for i, cached_feature in enumerate(
            iterable=tqdm(
                cached_features,
                desc=f"Iterating over {split_identifier=} cached features",
            ),
        ):
            # Note:
            # - `cached_feature` as an 'InputFeatures' object is not subscriptable,
            #   so we need to access via attributes and not via dictionary keys.
            #
            # > value: <utils_dst.InputFeatures object at 0x31b2ec4a0>
            # > type: utils_dst.InputFeatures
            # >
            # > Public attributes:
            # >     class_label_id: dict = {'attraction-area': 0, 'attraction-name': 0, …
            # >     diag_state: dict = {'attraction-area': 0, 'attraction-name': 0, …
            # >     guid: str = 'multiwoz21-train-0-0'
            # >     hst_boundaries: list = []
            # >     inform: dict = {'attraction-area': 'none', 'attraction-name': 'n…
            # >     inform_slot: dict = {'attraction-area': 0, 'attraction-name': 0, …
            # >     input_ids: list = [0, 524, 546, 13, 10, 317…
            # >     input_mask: list = [1, 1, 1, 1, 1, 1, …
            # >     refer_id: dict = {'attraction-area': 30, 'attraction-name': 30…
            # >     segment_ids: list = [0, 0, 0, 0, 0, 0, …
            # >     start_pos: dict = {'attraction-area': [0, 0, 0, …
            # >     usr_mask: list = [0, 1, 1, 1, 1, 1, …
            # >     values: dict = {'attraction-area': 'none', 'attraction-name': 'n…
            current_input_ids: torch.Tensor = torch.IntTensor(
                cached_feature.input_ids,
            )
            current_attention_mask: torch.Tensor = torch.IntTensor(
                cached_feature.input_mask,
            )
            current_dialogue_id: str = cached_feature.guid

            input_ids_list.append(
                current_input_ids,
            )
            attention_mask_list.append(
                current_attention_mask,
            )
            dialogue_ids_list.append(
                current_dialogue_id,
            )

            match metadata_handling_mode:
                case MetadataHandlingMode.IGNORE:
                    # Do nothing, just ignore the metadata.
                    pass
                case MetadataHandlingMode.CREATE_AND_SAVE_BIO_TAGS:
                    # Create and save BIO tags if available.

                    # Notes:
                    # - For decoding of the current input ids, use:
                    # > decoded_text: str = tokenizer.decode(current_input_ids, skip_special_tokens=False)

                    # Create a dataframe for the current feature, with one column filled with the input_ids
                    single_feature_df = pd.DataFrame(
                        data={
                            "input_ids": current_input_ids.tolist(),
                            "attention_mask": current_attention_mask.tolist(),
                            "usr_mask": cached_feature.usr_mask,
                        },
                    )
                    # Add column of decoded input ids to the dataframe.
                    single_feature_df["input_ids_decoded"] = single_feature_df["input_ids"].apply(
                        func=lambda x: tokenizer.convert_ids_to_tokens(
                            ids=x,
                        ),
                    )

                    # Add columns which contain the values of the lists in the cached_feature.start_pos
                    for (
                        slot_name,
                        start_positions,
                    ) in cached_feature.start_pos.items():
                        # Add a column for the slot name, with the start positions as values.
                        single_feature_df["start_pos_" + slot_name] = start_positions

                    # TODO: Save example single_feature_df as csv files for debugging (use the current_dialogue_id for the file name).
                    # TODO: Log the single_feature_df via a rich table.
                    # TODO: Convert the start positions to BIO tags.
                    # TODO: Save token-level BIO-tags into the post-processed cached features.

                    pass  # TODO: This is here for setting breakpoints, remove in production code.

        # Stack the tensors of the individual input data features into a single tensor.
        input_ids_stacked: torch.Tensor = torch.stack(
            tensors=input_ids_list,
            dim=0,
        )
        attention_masks_stacked: torch.Tensor = torch.stack(
            tensors=attention_mask_list,
            dim=0,
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Stacked tensors: {input_ids_stacked.shape=}, {attention_masks_stacked.shape=}",  # noqa: G004 - low overhead
            )

        result: dict[str, torch.Tensor | list] = {
            "input_ids": input_ids_stacked,
            "attention_mask": attention_masks_stacked,
            "dialogue_ids": dialogue_ids_list,
        }

        # ======================================================== #
        # Save processed cached features to disk
        # ======================================================== #

        output_path = pathlib.Path(
            processed_data_path_collection.post_processed_cached_features_dir,
            f"{split_identifier}_processed_cached_features.pt",
        )
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving processed cached features to {output_path=} ...",  # noqa: G004 - low overhead
            )

        torch.save(
            obj=result,
            f=output_path,
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving processed cached features to {output_path=} DONE",  # noqa: G004 - low overhead
            )

    # ======================================================== #

    logger.info(
        msg="Script finished.",
    )


if __name__ == "__main__":
    main()
