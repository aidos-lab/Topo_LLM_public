"""Run the twonn analysis."""

import logging
from typing import TYPE_CHECKING

import hydra
import omegaconf

from topollm.analysis.local_estimates_computation.global_and_pointwise_local_estimates_worker import (
    global_and_pointwise_local_estimates_worker,
)
from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.model_handling.get_torch_device import get_torch_device

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig

try:
    from hydra_plugins import hpc_submission_launcher

    hpc_submission_launcher.register_plugin()
except ImportError:
    pass

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

    device = get_torch_device(
        preferred_torch_backend=main_config.preferred_torch_backend,
        logger=logger,
    )

    global_and_pointwise_local_estimates_worker(
        main_config=main_config,
        device=device,
        verbosity=main_config.verbosity,
        logger=logger,
    )

    logger.info(
        msg="Script finished.",
    )


if __name__ == "__main__":
    main()
