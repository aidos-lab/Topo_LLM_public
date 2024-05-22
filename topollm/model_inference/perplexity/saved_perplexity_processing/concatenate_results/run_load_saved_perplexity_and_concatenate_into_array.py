# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (ruppik@hhu.de)
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

"""Load computed perplexity and concatente sequences into single array and df."""

import logging
import pathlib
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import omegaconf
import torch
from tqdm import tqdm

from topollm.config_classes.setup_OmegaConf import setup_OmegaConf
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.log_list_info import log_list_info
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.model_inference.perplexity.saved_perplexity_processing.load_perplexity_containers_from_jsonl_files import (
    load_perplexity_containers_from_jsonl_files,
)
from topollm.model_inference.perplexity.saved_perplexity_processing.load_perplexity_containers_from_pickle_files import (
    load_perplexity_containers_from_pickle_files,
)
from topollm.model_inference.perplexity.sentence_perplexity_container import SentencePerplexityContainer
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.typing.enums import PerplexityContainerSaveFormat, Verbosity

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig

default_device = torch.device("cpu")
default_logger = logging.getLogger(__name__)

global_logger = logging.getLogger(__name__)

setup_exception_logging(
    logger=global_logger,
)


setup_OmegaConf()


@hydra.main(
    config_path="../../../../../configs",
    config_name="main_config",
    version_base="1.2",
)
def main(
    config: omegaconf.DictConfig,
) -> None:
    """Run the script."""
    logger = global_logger
    logger.info("Running script ...")

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=logger,
    )
    verbosity = main_config.verbosity

    data_dir = main_config.paths.data_dir
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "data_dir:\n%s",
            data_dir,
        )

    # # # #
    # Get save paths
    embeddings_path_manager = get_embeddings_path_manager(
        main_config=main_config,
        logger=logger,
    )
    perplexity_dir = embeddings_path_manager.perplexity_dir_absolute_path

    save_file_path_josnl = pathlib.Path(
        perplexity_dir,
        "perplexity_results_list.jsonl",
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "save_file_path_josnl:\n%s",
            save_file_path_josnl,
        )

    loaded_data_list = load_perplexity_containers_from_jsonl_files(
        path_list=[
            save_file_path_josnl,
        ],
        verbosity=verbosity,
        logger=logger,
    )

    # Since we are only loading one container, we can directly access the first element
    loaded_data = loaded_data_list[0]

    # Empty lists for holding the concatenated data
    token_ids_list: list[int] = []
    token_strings_list: list[str] = []
    perplexity_list: list[float] = []

    for _, sentence_perplexity_container in tqdm(
        loaded_data,
        desc="Iterating over loaded_data",
    ):
        sentence_perplexity_container: SentencePerplexityContainer

        token_ids_list.extend(
            sentence_perplexity_container.token_ids,
        )
        token_strings_list.extend(
            sentence_perplexity_container.token_strings,
        )
        perplexity_list.extend(
            sentence_perplexity_container.token_perplexities,
        )

    if verbosity >= Verbosity.NORMAL:
        log_list_info(
            token_strings_list,
            list_name="token_strings_list",
            logger=logger,
        )

    # TODO: Continue here

    logger.info("Running script DONE")


if __name__ == "__main__":
    main()
