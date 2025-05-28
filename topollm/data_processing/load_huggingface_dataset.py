# Copyright 2024-2025
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
from typing import TYPE_CHECKING

import datasets
import hydra
import omegaconf

from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.data_handling.dataset_preparer.factory import get_dataset_preparer
from topollm.data_handling.dataset_preparer.protocol import DatasetPreparer
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig


# Logger for this file
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
    logger: logging.Logger = global_logger
    logger.info(
        msg="Running script ...",
    )

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=logger,
    )
    verbosity: Verbosity = main_config.verbosity

    dataset_preparer: DatasetPreparer = get_dataset_preparer(
        data_config=main_config.data,
        verbosity=verbosity,
        logger=logger,
    )

    dataset: datasets.Dataset = dataset_preparer.prepare_dataset()

    pass  # Note: You can place a breakpoint here to inspect the dataset

    logger.info(
        msg="Running script DONE",
    )


if __name__ == "__main__":
    main()
