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
from topollm.config_classes.setup_OmegaConf import setup_OmegaConf
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.model_finetuning.do_finetuning_process import do_finetuning_process
from topollm.model_handling.get_torch_device import get_torch_device

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig

# Increase the wandb service wait time to prevent errors.
# https://github.com/wandb/wandb/issues/5214
os.environ["WANDB__SERVICE_WAIT"] = "300"

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

    wandb_dir = pathlib.Path(
        main_config.wandb.dir,
    )
    logger.info(
        f"{wandb_dir = }",  # noqa: G004 - low overhead
    )
    # Create the wandb directory if it does not exist
    wandb_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    wandb.init(
        dir=main_config.wandb.dir,
        entity=main_config.wandb.entity,  # Note: To make this None, use null in the hydra config
        project=main_config.wandb.project,
        settings=wandb.Settings(
            start_method="thread",  # Note: https://docs.wandb.ai/guides/integrations/hydra#troubleshooting-multiprocessing
            _service_wait=300,
            init_timeout=300,
        ),
        tags=main_config.wandb.tags,
    )

    # Note: Convert OmegaConf to dict to avoid issues with wandb
    # https://docs.wandb.ai/guides/integrations/hydra#track-hyperparameters
    omegaconf_converted_to_dict = omegaconf.OmegaConf.to_container(
        cfg=config,
        resolve=True,
        throw_on_missing=True,
    )

    # Add the hydra config to the wandb config
    # (so that they are tracked in the wandb run)
    wandb.config.hydra = omegaconf_converted_to_dict

    # Add information about the wandb run to the logger
    if wandb.run is not None:
        logger.info(
            f"{wandb.run.dir = }",  # noqa: G004 - low overhead
        )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Use accelerator if available
    device = get_torch_device(
        preferred_torch_backend=main_config.preferred_torch_backend,
        verbosity=main_config.verbosity,
        logger=logger,
    )

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

    # We need to manually finish the wandb run so that the hydra multi-run submissions are not summarized in the same run
    wandb.finish()

    logger.info(
        "Running script DONE",
    )


if __name__ == "__main__":
    main()
