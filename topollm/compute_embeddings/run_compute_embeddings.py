"""Run script to create embedding vectors from dataset based on config."""

import logging
from typing import TYPE_CHECKING

import hydra
import omegaconf

from topollm.compute_embeddings.compute_and_store_embeddings import (
    compute_and_store_embeddings,
)
from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig

# logger for this file
global_logger: logging.Logger = logging.getLogger(
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
    global_logger.info(
        msg="Running script ...",
    )

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=global_logger,
    )

    compute_and_store_embeddings(
        main_config=main_config,
        logger=global_logger,
    )

    global_logger.info(
        msg="Running script DONE",
    )


if __name__ == "__main__":
    main()
