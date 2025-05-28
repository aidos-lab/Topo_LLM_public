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

"""Script for loading cached features and saving them into a format for the Topo_LLM repository."""

import argparse
import logging
import os
import pathlib
import sys
from dataclasses import dataclass
from enum import StrEnum, auto

import torch
from tqdm import tqdm

from topollm.logging.create_and_configure_global_logger import create_and_configure_global_logger

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


@dataclass
class ProcessedDataPathsCollection:
    """Class to store the input and output paths for the processed data."""

    checkpoints_root_dir: pathlib.Path
    post_processed_cached_features_dir: pathlib.Path


def get_processed_data_paths_collection(
    data_mode: DataMode,
    verbosity: int = 1,
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

    if verbosity > 0:
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
    if verbosity > 0:
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

    if verbosity > 0:
        logger.info(
            msg=f"processed_data_path_collection:\n{processed_data_path_collection}",  # noqa: G004 - low overhead
        )

    return processed_data_path_collection


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Load cached features and save them into a format for the Topo_LLM repository.",
    )

    parser.add_argument(
        "--data_mode",
        type=DataMode,
        choices=list(DataMode),
        default=DataMode.TRIPPY_R,
        help="Data mode to use.",
    )

    parser.add_argument(
        "--verbosity",
        type=int,
        default=1,
        help="Verbosity level (0: no output, 1: info, 2: debug)",
    )

    args: argparse.Namespace = parser.parse_args()

    return args


def main() -> None:
    """Load cached features and save them into a format for the Topo_LLM repository."""
    logger: logging.Logger = global_logger

    args: argparse.Namespace = parse_arguments()
    data_mode: DataMode = args.data_mode
    verbosity: int = args.verbosity

    number_of_elements_to_log: int = 5

    if verbosity > 0:
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

        if verbosity > 0:
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
        if verbosity > 0:
            logger.info(
                msg=f"Loading cached features from {file_path=} ... (This might take some time)",  # noqa: G004 - low overhead
            )
        loaded_features: list = torch.load(
            f=file_path,
            weights_only=False,
        )
        if verbosity > 0:
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

        if verbosity > 0:
            logger.info(
                msg=f"Loaded object of {type(cached_features)=} with {len(cached_features)=}",  # noqa: G004 - low overhead
            )

        if len(cached_features) == 0:
            logger.warning(
                msg=f"Loaded object is empty: {file_path=}",  # noqa: G004 - low overhead
            )
            continue

        if verbosity > 0:
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

        for cached_feature in tqdm(
            cached_features,
            desc=f"Iterating over {split_identifier=} cached features",
        ):
            # Note:
            # - `cached_feature` as an 'InputFeatures' object is not subscriptable,
            #   so we need to access via attributes and not via dictionary keys.
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

        # Stack the tensors of the individual input data features into a single tensor.
        input_ids_stacked: torch.Tensor = torch.stack(
            tensors=input_ids_list,
            dim=0,
        )
        attention_masks_stacked: torch.Tensor = torch.stack(
            tensors=attention_mask_list,
            dim=0,
        )

        if verbosity > 0:
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

        if verbosity > 0:
            logger.info(
                msg=f"Saving processed cached features to {output_path=} ...",  # noqa: G004 - low overhead
            )

        torch.save(
            obj=result,
            f=output_path,
        )
        if verbosity > 0:
            logger.info(
                msg=f"Saving processed cached features to {output_path=} DONE",  # noqa: G004 - low overhead
            )

    # ======================================================== #

    logger.info(
        msg="Script finished.",
    )


if __name__ == "__main__":
    main()
