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

"""Create a config file for the finetuned language model."""

import logging
import os
import pathlib
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import omegaconf
import transformers
import wandb

from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.config_classes.setup_OmegaConf import setup_OmegaConf
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.model_finetuning.do_finetuning_process import do_finetuning_process
from topollm.model_finetuning.prepare_finetuned_model_dir import prepare_finetuned_model_dir
from topollm.model_handling.get_torch_device import get_torch_device
from topollm.path_management.finetuning.factory import get_finetuning_path_manager
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig

global_logger = logging.getLogger(__name__)

setup_exception_logging(
    logger=global_logger,
)

# Set the transformers logging level
transformers.logging.set_verbosity_info()

setup_OmegaConf()


@hydra.main(
    config_path=f"{HYDRA_CONFIGS_BASE_PATH}",
    config_name="main_config",
    version_base="1.2",
)
def main(
    config: omegaconf.DictConfig,
) -> None:
    """Run the script."""
    logger = global_logger
    logger.info(
        "Running script ...",
    )

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=logger,
    )

    create_language_model_hydra_config(
        main_config=main_config,
        logger=logger,
    )

    logger.info(
        "Running script DONE",
    )


default_logger = logging.getLogger(__name__)


def create_language_model_hydra_config(
    main_config: MainConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
):
    finetuning_path_manager = get_finetuning_path_manager(
        main_config=main_config,
        logger=logger,
    )

    finetuned_model_dir = prepare_finetuned_model_dir(
        finetuning_path_manager=finetuning_path_manager,
        logger=logger,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"{finetuned_model_dir = }",  # noqa: G004 - low overhead
        )

    # TODO(Ben): Continue here


if __name__ == "__main__":
    main()
