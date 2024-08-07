# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (ruppik@hhu.de)
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

"""Load computed perplexity and concatente sequences into single array and df."""

import logging
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import omegaconf
import torch

from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.config_classes.setup_OmegaConf import setup_omega_conf
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.model_inference.perplexity.saved_perplexity_processing.load_perplexity_and_local_estimates_and_align import (
    load_perplexity_and_local_estimates_and_align,
)

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig
    from topollm.model_inference.perplexity.saved_perplexity_processing.aligned_local_estimates_data_container import (
        AlignedLocalEstimatesDataContainer,
    )

default_device = torch.device("cpu")
default_logger = logging.getLogger(__name__)

global_logger = logging.getLogger(__name__)

setup_exception_logging(
    logger=global_logger,
)


setup_omega_conf()

# TODO: Use the function on multiple language models and datasets


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
    main_config_for_perplexity = main_config

    # # # # # # # # # # # # # # # # # # # #
    # Set the parameters so that the correct local estimates are loaded.
    # Note that we have to do this because the number of sequences for the perplexity computation
    # might be different from the number of sequences for the local estimates computation.

    local_estimates_layer_indices = None

    if local_estimates_layer_indices is None:
        local_estimates_layer_indices = [-1]
        # local_estimates_layer_indices = [-5]

    # Make a configuration for the local estimates
    main_config_for_local_estimates = main_config_for_perplexity.model_copy(
        deep=True,
    )
    main_config_for_local_estimates.embeddings.embedding_extraction.layer_indices = local_estimates_layer_indices

    if main_config_for_local_estimates.data.dataset_description_string == "multiwoz21":
        main_config_for_local_estimates.data.number_of_samples = 3000
    else:
        main_config_for_local_estimates.data.number_of_samples = -1

    aligned_local_estimates_data_container: AlignedLocalEstimatesDataContainer | None = (
        load_perplexity_and_local_estimates_and_align(
            main_config_for_perplexity=main_config_for_perplexity,
            main_config_for_local_estimates=main_config_for_local_estimates,
            verbosity=verbosity,
            logger=logger,
        )
    )

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
