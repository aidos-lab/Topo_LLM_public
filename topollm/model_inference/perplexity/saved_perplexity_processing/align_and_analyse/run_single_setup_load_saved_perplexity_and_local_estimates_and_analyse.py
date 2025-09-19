"""Load computed perplexity and concatente sequences into single array and df."""

import logging
import sys
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import omegaconf
import torch

from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.model_inference.perplexity.saved_perplexity_processing.align_and_analyse.load_perplexity_and_local_estimates_and_align import (
    load_perplexity_and_local_estimates_and_align,
)

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig
    from topollm.model_inference.perplexity.saved_perplexity_processing.align_and_analyse.aligned_local_estimates_data_container import (
        AlignedLocalEstimatesDataContainer,
    )

default_device = torch.device(device="cpu")
default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)

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
    logger = global_logger
    logger.info("Running script ...")

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=logger,
    )
    verbosity = main_config.verbosity

    # # # #
    main_config_for_local_estimates = main_config

    # # # # # # # # # # # # # # # # # # # #
    # Set the parameters so that the correct local estimates and perplexities are loaded.
    # Note also that the number of sequences for the perplexity computation
    # might be different from the number of sequences for the local estimates computation.

    # Make a configuration for the perplexity paths.
    main_config_for_perplexity = main_config_for_local_estimates.model_copy(
        deep=True,
    )
    # For the perplexity paths, we use layer index -1, so we need to set the layer index to -1 here.
    main_config_for_perplexity.embeddings.embedding_extraction.layer_indices = [-1]

    try:
        aligned_local_estimates_data_container: AlignedLocalEstimatesDataContainer | None = (
            load_perplexity_and_local_estimates_and_align(
                main_config_for_perplexity=main_config_for_perplexity,
                main_config_for_local_estimates=main_config_for_local_estimates,
                verbosity=verbosity,
                logger=logger,
            )
        )
    except FileNotFoundError as e:
        msg = f"FileNotFoundError: {e}"
        logger.exception(msg)
        logger.info("Running script FAILED. Exiting.")
        return

    if aligned_local_estimates_data_container is None:
        msg = "aligned_local_estimates_data_container is None"
        raise ValueError(msg)

    aligned_local_estimates_data_container.run_analysis_and_save_results(
        display_plots=False,
    )

    logger.info("Running script DONE")
    # # # # # # # # # # # # # # # # # # # #


if __name__ == "__main__":
    main()
