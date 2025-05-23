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
import pprint

import torch
import transformers
from tqdm import tqdm

from topollm.experiments.local_dimensions_detect_exhaustion_of_training_capabilities.data_post_processing.load_cached_features_and_save_into_format_for_topo_llm import (
    DataMode,
    ProcessedDataPathsCollection,
    get_processed_data_paths_collection,
)
from topollm.experiments.local_dimensions_detect_exhaustion_of_training_capabilities.logging.create_and_configure_global_logger import (
    create_and_configure_global_logger,
)

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

block_separator_line = "=" * 80


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Load cached features and save them into a format for the Topo_LLM repository.",
    )

    parser.add_argument(
        "--data_mode",
        type=DataMode,
        choices=list(DataMode),
        default=DataMode.TRIPPY,
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
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="roberta-base",
    )

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
        desc="Processing splits",
    ):
        # ======================================================== #
        # Load post processed cached features
        # ======================================================== #

        output_path = pathlib.Path(
            processed_data_path_collection.post_processed_cached_features_dir,
            f"{split_identifier}_processed_cached_features.pt",
        )

        if verbosity > 0:
            logger.info(
                msg=f"Loading processed cached features from {output_path=} ...",  # noqa: G004 - low overhead
            )

        loaded_processed_cached_features = torch.load(
            f=output_path,
            map_location=torch.device(device="cpu"),
        )

        if verbosity > 0:
            logger.info(
                msg=f"Loading processed cached features from {output_path=} DONE",  # noqa: G004 - low overhead
            )

        if verbosity > 0:
            logger.info(
                msg=f"{split_identifier=}",  # noqa: G004 - low overhead
            )
            logger.info(
                msg=f"{type(loaded_processed_cached_features) = }",  # noqa: G004 - low overhead
            )

            logger.info(
                msg=f"{len(loaded_processed_cached_features) = }",  # noqa: G004 - low overhead
            )
            # The keys should be:
            # > dict_keys(['input_ids', 'attention_mask', 'dialogue_ids'])
            logger.info(
                msg=f"loaded_processed_cached_features.keys():\n"  # noqa: G004 - low overhead
                f"{pprint.pformat(object=loaded_processed_cached_features.keys())}",
            )

        if verbosity > 0:
            log_selected_loaded_processed_cached_features(
                features_dict=loaded_processed_cached_features,
                num_elements_to_log=number_of_elements_to_log,
                tokenizer=tokenizer,
                logger=logger,
            )

    # ======================================================== #

    logger.info(
        msg="Script finished.",
    )


def log_selected_loaded_processed_cached_features(
    features_dict: dict,
    num_elements_to_log: int = 5,
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast | None = None,
    logger: logging.Logger = default_logger,
) -> None:
    """Log selected loaded processed cached features."""
    logger.info(
        msg=block_separator_line,
    )

    logger.info(
        msg=f"{features_dict['input_ids'].shape = }",  # noqa: G004 - low overhead
    )
    logger.info(
        msg=f"{features_dict['attention_mask'].shape = }",  # noqa: G004 - low overhead
    )
    logger.info(
        msg=f"{len(features_dict['dialogue_ids'])= }",  # noqa: G004 - low overhead
    )

    for i in range(num_elements_to_log):
        logger.info(
            msg=f"{i=}",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{features_dict['input_ids'][i] = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{features_dict['attention_mask'][i] = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{features_dict['dialogue_ids'][i] = }",  # noqa: G004 - low overhead
        )

        if tokenizer is not None:
            decoded_input_ids = tokenizer.decode(
                token_ids=features_dict["input_ids"][i],
                skip_special_tokens=False,
            )

            logger.info(
                msg=f"{decoded_input_ids = }",  # noqa: G004 - low overhead
            )

    logger.info(
        msg=block_separator_line,
    )


if __name__ == "__main__":
    main()
