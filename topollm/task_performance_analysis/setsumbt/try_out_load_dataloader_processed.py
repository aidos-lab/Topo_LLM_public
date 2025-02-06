# Copyright 2024-2025
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (mail@ruppik.net)
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

"""Example script to load a dataloader processed file and check the content."""

import logging
import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import omegaconf
import torch

from topollm.config_classes.constants import (
    HYDRA_CONFIGS_BASE_PATH,
)
from topollm.config_classes.setup_OmegaConf import setup_omega_conf
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.log_recursive_dict_info import log_recursive_dict_info
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.model_handling.loaded_model_container import LoadedModelContainer
from topollm.model_handling.prepare_loaded_model_container import (
    prepare_device_and_tokenizer_and_model_from_main_config,
)
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig
    from topollm.path_management.embeddings.protocol import EmbeddingsPathManager


try:
    from hydra_plugins import hpc_submission_launcher

    hpc_submission_launcher.register_plugin()
except ImportError:
    pass

# logger for this file
global_logger: logging.Logger = logging.getLogger(
    name=__name__,
)
default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)

setup_exception_logging(
    logger=global_logger,
)


@hydra.main(
    config_path=f"{HYDRA_CONFIGS_BASE_PATH}",
    config_name="main_config",
    version_base="1.3",
)
def main(
    config: omegaconf.DictConfig,
) -> None:
    """Run the script."""
    logger: logging.Logger = global_logger
    logger.info(
        msg="Running script ...",
    )

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=logger,
    )
    verbosity: Verbosity = main_config.verbosity

    # The embeddings path manager will be used to get the data directory
    embeddings_path_manager: EmbeddingsPathManager = get_embeddings_path_manager(
        main_config=main_config,
        logger=logger,
    )

    # # # # # # # # # # # #
    # Load the dataloaders

    dataloaders_processed_root_directory: pathlib.Path = pathlib.Path(
        embeddings_path_manager.data_dir,
        "models",
        "setsumbt_checkpoints",
        "multiwoz21",
        "dataloaders_processed",
    )

    # selected_dataloader_processed_name = "train_0.data"
    selected_dataloader_processed_name = "train_1.data"

    selected_dataloader_processed_path = pathlib.Path(
        dataloaders_processed_root_directory,
        selected_dataloader_processed_name,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Loading from {selected_dataloader_processed_path = } ...",  # noqa: G004 - low overhead
        )

    dataloader_processed = torch.load(
        f=selected_dataloader_processed_path,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Loading from {selected_dataloader_processed_path = } DONE",  # noqa: G004 - low overhead
        )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{type(dataloader_processed) = }",  # noqa: G004 - low overhead
        )

        if isinstance(
            dataloader_processed,
            dict,
        ):
            logger.info(
                msg="Logging additional information about the dictionary.",
            )

            log_recursive_dict_info(
                dictionary=dataloader_processed,
                dictionary_name="dataloader_processed",
                logger=logger,
            )

    # # # # # # # #
    # Decode some of the input_ids to text to check the content

    loaded_model_container: LoadedModelContainer = prepare_device_and_tokenizer_and_model_from_main_config(
        main_config=main_config,
        verbosity=verbosity,
        logger=logger,
    )
    tokenizer = loaded_model_container.tokenizer

    # Example for 'train_0.data':
    # > dataloader_processed["input_ids"].shape = torch.Size([8438, 12, 64])
    #
    # Example: Select the first sequence of the first dialogue:
    # > dataloader_processed["input_ids"][0][0]

    for sequence_to_decode in dataloader_processed["input_ids"][0]:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"{sequence_to_decode.shape = }",  # noqa: G004 - low overhead
            )

        # Decode the sequence
        decoded_sequence = tokenizer.decode(
            token_ids=sequence_to_decode,
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"{sequence_to_decode = }",  # noqa: G004 - low overhead
            )
            logger.info(
                msg=f"{decoded_sequence = }",  # noqa: G004 - low overhead
            )

    # TODO: Implement concatenating the tensors of the individual dialogues (select only those where the attention mask is non-zero)

    logger.info(
        msg="Running script DONE",
    )


if __name__ == "__main__":
    setup_omega_conf()

    main()
