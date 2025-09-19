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
        # Notes:
        # - Previously, the `start_method` was set to "thread" to avoid issues with hydra multiprocessing, see
        # https://docs.wandb.ai/guides/integrations/hydra#troubleshooting-multiprocessing
        # But now, the `start_method` argument will be deprecated in the future versions of wandb,
        # so we remove `start_method="thread"` from the `wandb.init` call.
        settings=wandb.Settings(
            _service_wait=300,  # type: ignore - there is a problem with recognizing this argument
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
