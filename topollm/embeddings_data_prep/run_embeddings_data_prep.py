"""Prepare the embedding data of a model and its metadata for further analysis.

Since paddings are removed from the embeddings, the resulting size of the arrays will usually be
significantly lower than the specified sample size
(often ~5% of the specified size).
"""

import logging
from typing import TYPE_CHECKING

import hydra
import omegaconf

from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.embeddings_data_prep.embeddings_data_prep_worker import embeddings_data_prep_worker
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
    global_logger.info(msg="Running script ...")

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=global_logger,
    )

    embeddings_data_prep_worker(
        main_config=main_config,
        verbosity=main_config.verbosity,
        logger=global_logger,
    )

    global_logger.info(msg="Script finished.")


if __name__ == "__main__":
    main()
