# Copyright 2024-2025
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (mail@ruppik.net)
# Julius von Rohrscheidt (julius.rohrscheidt@helmholtz-muenchen.de)
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

"""Create plots of the local estimates and compare with other task performance measures."""

import itertools
import logging
import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import hydra
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
from tqdm import tqdm

from topollm.analysis.local_estimates_handling.saving.local_estimates_containers import LocalEstimatesContainer
from topollm.analysis.local_estimates_handling.saving.local_estimates_saving_manager import LocalEstimatesSavingManager
from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.config_classes.setup_OmegaConf import setup_omega_conf
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.path_management.embeddings.protocol import EmbeddingsPathManager
from topollm.plotting.line_plot_grouped_by_categorical_column import PlotSizeConfig
from topollm.task_performance_analysis.plotting.distribution_violinplots_and_distribution_boxplots import (
    TicksAndLabels,
    make_distribution_violinplots_from_extracted_arrays,
)
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig

# Logger for this file
global_logger: logging.Logger = logging.getLogger(
    name=__name__,
)
default_logger: logging.Logger = logging.getLogger(
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
    """Run main function."""
    logger: logging.Logger = global_logger
    logger.info(
        msg="Running script ...",
    )

    # ================================================== #
    # Load configuration and initialize path manager
    # ================================================== #

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=logger,
    )
    verbosity: Verbosity = main_config.verbosity

    embeddings_path_manager: EmbeddingsPathManager = get_embeddings_path_manager(
        main_config=main_config,
        logger=logger,
    )

    # ================================================== #
    # Load data
    # ================================================== #

    local_estimates_pointwise_dir_absolute_path = (
        embeddings_path_manager.get_local_estimates_pointwise_dir_absolute_path()
    )

    local_estimates_saving_manager: LocalEstimatesSavingManager = (
        LocalEstimatesSavingManager.from_local_estimates_pointwise_dir_absolute_path(
            local_estimates_pointwise_dir_absolute_path=local_estimates_pointwise_dir_absolute_path,
            verbosity=verbosity,
            logger=logger,
        )
    )
    local_estimates_container: LocalEstimatesContainer = local_estimates_saving_manager.load_local_estimates()
    local_estimates_container.log_info(
        verbosity=verbosity,
        logger=logger,
    )

    # # # #
    # For reference, create the violin plots corresponding to this data

    (
        fig,
        ax,
    ) = make_distribution_violinplots_from_extracted_arrays(
        extracted_arrays=[local_estimates_container.pointwise_results_array_np],
        ticks_and_labels=TicksAndLabels(
            xlabel="main_config.language_model.checkpoint_no",
            ylabel="pointwise_results_array_np",
            xticks_labels=[str(object=main_config.language_model.checkpoint_no)],
        ),
        plot_size_config=PlotSizeConfig(),
        verbosity=verbosity,
        logger=logger,
    )

    fig.show()

    # TODO: Implement the clustering in value space and analysis of the token distribution
    # TODO: Implement the code for the additional analyis

    logger.info(
        msg="Running script DONE",
    )


if __name__ == "__main__":
    setup_omega_conf()

    main()
