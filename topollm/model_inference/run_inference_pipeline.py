"""Run model inference on example data."""

import logging
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import omegaconf

from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.model_handling.set_seed import set_seed
from topollm.model_inference.do_inference import do_inference
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.path_management.embeddings.protocol import EmbeddingsPathManager
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig

global_logger: logging.Logger = logging.getLogger(
    name=__name__,
)

setup_exception_logging(
    logger=global_logger,
)


@hydra.main(
    config_path="../../configs",
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

    set_seed(
        seed=main_config.global_seed,
        logger=logger,
    )

    embeddings_path_manager: EmbeddingsPathManager = get_embeddings_path_manager(
        main_config=main_config,
        verbosity=verbosity,
        logger=logger,
    )

    _ = do_inference(
        main_config=main_config,
        embeddings_path_manager=embeddings_path_manager,
        verbosity=verbosity,
        logger=logger,
    )

    logger.info(
        msg="Running script DONE",
    )


if __name__ == "__main__":
    main()
