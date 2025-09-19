"""Compute the (pseudo-)perplexity of a masked language model and save to disk."""

import logging
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import omegaconf
import torch

from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.model_inference.perplexity.do_perplexity_processing import do_perplexity_computation

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig

default_device = torch.device("cpu")
default_logger = logging.getLogger(__name__)

global_logger = logging.getLogger(__name__)

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
    logger.info("Running script ...")

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=logger,
    )

    do_perplexity_computation(
        main_config=main_config,
        logger=logger,
    )

    logger.info("Running script DONE")


if __name__ == "__main__":
    main()
