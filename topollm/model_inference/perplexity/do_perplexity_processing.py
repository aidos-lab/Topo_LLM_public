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

"""Perform the perplexity computation based on the MainConfig object."""

import logging
import os
import pathlib
import pickle
from typing import TYPE_CHECKING

from topollm.config_classes.main_config import MainConfig
from topollm.data_handling.dataset_preparer.factory import get_dataset_preparer
from topollm.model_handling.prepare_loaded_model_container import prepare_device_and_tokenizer_and_model
from topollm.model_inference.perplexity.compute_perplexity_over_dataset import (
    compute_perplexity_over_dataset,
)
from topollm.model_inference.perplexity.sentence_perplexity_container import SentencePerplexityContainer
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.typing.enums import Verbosity
from topollm.typing.types import PerplexityResultsList

if TYPE_CHECKING:
    import datasets

    from topollm.model_handling.loaded_model_container import LoadedModelContainer

default_logger = logging.getLogger(__name__)


def do_perplexity_computation(
    main_config: MainConfig,
    logger: logging.Logger = default_logger,
) -> None:
    """Run the perplexity computation."""
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Prepare device, tokenizer, model
    loaded_model_container: LoadedModelContainer = prepare_device_and_tokenizer_and_model(
        main_config=main_config,
        logger=logger,
    )
    model = loaded_model_container.model
    # Put model in evaluation mode
    model.eval()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Prepare dataset
    dataset_preparer = get_dataset_preparer(
        data_config=main_config.data,
        verbosity=main_config.verbosity,
        logger=logger,
    )
    dataset: datasets.Dataset = dataset_preparer.prepare_dataset()

    embeddings_path_manager = get_embeddings_path_manager(
        main_config=main_config,
        logger=logger,
    )
    perplexity_dir = embeddings_path_manager.perplexity_dir_absolute_path
    # Create the directory if it does not exist
    perplexity_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    perplexity_results_list: PerplexityResultsList = compute_perplexity_over_dataset(
        loaded_model_container=loaded_model_container,
        dataset=dataset,
        column_name=main_config.data.column_name,
        verbosity=main_config.verbosity,
        logger=logger,
    )

    save_perplexity_results_list(
        perplexity_results_list=perplexity_results_list,
        perplexity_dir=perplexity_dir,
        verbosity=main_config.verbosity,
        logger=logger,
    )


def save_perplexity_results_list(
    perplexity_results_list: PerplexityResultsList,
    perplexity_dir: os.PathLike,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Save the perplexity results list to a file."""
    # # # #
    # Save in pickle format
    save_file_path_pickle = pathlib.Path(
        perplexity_dir,
        "perplexity_results_list_new_format.pkl",
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(f"Saving perplexity results to {save_file_path_pickle = } ...")  # noqa: G004 - low overhead
    with pathlib.Path(save_file_path_pickle).open(
        mode="wb",
    ) as file:
        pickle.dump(
            obj=perplexity_results_list,
            file=file,
        )
    if verbosity >= Verbosity.NORMAL:
        logger.info(f"Saving perplexity results to {save_file_path_pickle = } DONE")  # noqa: G004 - low overhead

    # # # #
    # Save in jsonl format
    save_file_path_josnl = pathlib.Path(
        perplexity_dir,
        "perplexity_results_list.jsonl",
    )
    # Iterate over the list and save each item as a jsonl line
    if verbosity >= Verbosity.NORMAL:
        logger.info(f"Saving perplexity results to {save_file_path_josnl = } ...")  # noqa: G004 - low overhead
    with save_file_path_josnl.open(
        mode="w",
    ) as file:
        for _, sentence_perplexity_container in perplexity_results_list:
            if not isinstance(
                sentence_perplexity_container,
                SentencePerplexityContainer,
            ):
                msg = "Expected a SentencePerplexityContainer."
                raise TypeError(msg)

            model_dump: str = sentence_perplexity_container.model_dump_json()
            file.write(model_dump)
            file.write("\n")
    if verbosity >= Verbosity.NORMAL:
        logger.info(f"Saving perplexity results to {save_file_path_josnl = } DONE")  # noqa: G004 - low overhead
