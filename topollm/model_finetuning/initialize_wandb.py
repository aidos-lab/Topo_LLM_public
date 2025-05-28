# Copyright 2024
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
# AUTHOR_2 (author2@example.com)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#


import logging
import os
import pathlib

import omegaconf
import wandb

from topollm.config_classes.main_config import MainConfig

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def initialize_wandb(
    main_config: MainConfig,
    config: omegaconf.DictConfig,
    logger: logging.Logger = default_logger,
) -> None:
    """Initialize wandb."""
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

    os.environ["WANDB_WATCH"] = "all"

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
