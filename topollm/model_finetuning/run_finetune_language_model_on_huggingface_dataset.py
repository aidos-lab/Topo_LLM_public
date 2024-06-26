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

"""Script for fine-tuning language model on huggingface datasets."""

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
from topollm.config_classes.language_model.language_model_config import LanguageModelConfig
from topollm.config_classes.main_config import MainConfig
from topollm.config_classes.setup_OmegaConf import setup_OmegaConf
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.model_finetuning.do_finetuning_process import do_finetuning_process
from topollm.model_finetuning.initialize_wandb import initialize_wandb
from topollm.model_finetuning.prepare_finetuned_model_dir import prepare_finetuned_model_dir
from topollm.model_handling.get_torch_device import get_torch_device
from topollm.path_management.finetuning.factory import get_finetuning_path_manager
from topollm.path_management.finetuning.protocol import FinetuningPathManager
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    pass

# Increase the wandb service wait time to prevent errors.
# https://github.com/wandb/wandb/issues/5214
os.environ["WANDB__SERVICE_WAIT"] = "300"
wandb.require(
    "core",
)

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

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Initialize wandb
    if main_config.feature_flags.finetuning.use_wandb:
        initialize_wandb(
            main_config=main_config,
            config=config,
            logger=logger,
        )
    else:
        os.environ["WANDB_MODE"] = "disabled"
        # Note: Do not set `os.environ["WANDB_DISABLED"] = "true"` because this will raise the error
        # `RuntimeError: WandbCallback requires wandb to be installed. Run `pip install wandb`.`

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Use accelerator if available
    device = get_torch_device(
        preferred_torch_backend=main_config.preferred_torch_backend,
        verbosity=main_config.verbosity,
        logger=logger,
    )

    if not main_config.feature_flags.finetuning.skip_finetuning:
        logger.info(
            "Calling do_finetuning_process ...",
        )

        do_finetuning_process(
            main_config=main_config,
            device=device,
            logger=logger,
        )

        logger.info(
            "Calling do_finetuning_process DONE",
        )

    if main_config.feature_flags.finetuning.use_wandb:
        # We need to manually finish the wandb run
        # so that the hydra multi-run submissions are not summarized in the same run
        wandb.finish()

    if main_config.feature_flags.finetuning.do_create_finetuned_language_model_config:
        logger.info(
            "Calling create_finetuned_language_model_config ...",
        )

        create_finetuned_language_model_config(
            main_config=main_config,
            verbosity=main_config.verbosity,
            logger=logger,
        )

        logger.info(
            "Calling create_finetuned_language_model_config DONE",
        )

    logger.info(
        "Running script DONE",
    )


default_logger = logging.getLogger(__name__)


def create_finetuned_language_model_config(
    main_config: MainConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Create the config for the language model resulting from fine-tuning.

    This config can be used for further processing, e.g. for the embedding and data generation.
    """
    finetuning_path_manager: FinetuningPathManager = get_finetuning_path_manager(
        main_config=main_config,
        logger=logger,
    )

    finetuned_model_relative_dir = finetuning_path_manager.get_finetuned_model_relative_dir()

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"{finetuned_model_relative_dir = }",  # noqa: G004 - low overhead
        )

    # TODO: Get this from the parameters
    checkpoint_no = 31200

    base_language_model_config = main_config.language_model
    new_language_model_config = update_language_model_config(
        base_language_model_config=base_language_model_config,
        finetuned_model_relative_dir=finetuned_model_relative_dir,
        checkpoint_no=checkpoint_no,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "base_language_model_config:%s",
            base_language_model_config,
        )
        logger.info(
            "new_language_model_config:%s",
            new_language_model_config,
        )

    # TODO(Ben): Continue here: Save the config to a file
    pass


def update_language_model_config(
    base_language_model_config: LanguageModelConfig,
    finetuned_model_relative_dir: pathlib.Path,
    checkpoint_no: int,
) -> LanguageModelConfig:
    """Update the language model config with the new finetuned model path and short model name."""
    new_pretrained_model_path = (
        r"${paths.data_dir}/" + str(finetuned_model_relative_dir) + r"/checkpoint-${language_model.checkpoint_no}"
    )

    # TODO: Update this with a better way to get the short model name
    new_short_model_name = str(base_language_model_config.short_model_name) + r"_ckpt-${language_model.checkpoint_no}"

    updated_config = base_language_model_config.model_copy(
        update={
            "pretrained_model_name_or_path": new_pretrained_model_path,
            "short_model_name": new_short_model_name,
            "checkpoint_no": checkpoint_no,
        },
        deep=True,
    )

    return updated_config


if __name__ == "__main__":
    main()
