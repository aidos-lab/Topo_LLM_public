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

"""Run script to compare computed Hausdorff distances with the local estimates."""

import logging
import pathlib
import pprint
from collections.abc import Generator
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import joblib
import numpy as np
import omegaconf
import pandas as pd
from scipy import stats
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

from topollm.analysis.local_estimates_handling.saving.local_estimates_containers import LocalEstimatesContainer
from topollm.analysis.local_estimates_handling.saving.local_estimates_saving_manager import LocalEstimatesSavingManager
from topollm.config_classes.constants import (
    HYDRA_CONFIGS_BASE_PATH,
    NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS,
    TOPO_LLM_REPOSITORY_BASE_PATH,
)
from topollm.config_classes.main_config import MainConfig
from topollm.config_classes.setup_OmegaConf import setup_omega_conf
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.log_dataframe_info import log_dataframe_info
from topollm.logging.log_list_info import log_list_info
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.typing.enums import ArtificialNoiseMode, Verbosity

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig
    from topollm.path_management.embeddings.protocol import EmbeddingsPathManager

try:
    from hydra_plugins import hpc_submission_launcher

    hpc_submission_launcher.register_plugin()
except ImportError:
    pass

# logger for this file
global_logger: logging.Logger = logging.getLogger(
    name=__name__,
)
default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)

setup_exception_logging(
    logger=global_logger,
)

setup_omega_conf()


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

    # # # # # # # # # # # # # # # # # # # # #
    # START Global settings

    # END Global settings
    # # # # # # # # # # # # # # # # # # # # #

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=logger,
    )
    verbosity: Verbosity = main_config.verbosity

    embeddings_path_manager_for_base_data: EmbeddingsPathManager = get_embeddings_path_manager(
        main_config=main_config,
        logger=logger,
    )
    data_dir: pathlib.Path = embeddings_path_manager_for_base_data.data_dir

    # ================================================== #
    # Non-noised data
    # ================================================== #

    local_estimates_saving_manager_for_base_data = LocalEstimatesSavingManager(
        embeddings_path_manager=embeddings_path_manager_for_base_data,
        verbosity=verbosity,
        logger=logger,
    )

    local_estimates_container_base_data: LocalEstimatesContainer = (
        local_estimates_saving_manager_for_base_data.load_local_estimates()
    )

    # ================================================== #
    # Load artificial noise data
    # ================================================== #

    artificial_noise_mode = ArtificialNoiseMode.GAUSSIAN
    artificial_noise_distortion_parameter = 0.01
    artificial_noise_seed = 4

    main_config_for_noised_data: MainConfig = main_config.model_copy(
        deep=True,
    )
    main_config_for_noised_data.local_estimates.noise.artificial_noise_mode = artificial_noise_mode
    main_config_for_noised_data.local_estimates.noise.distortion_parameter = artificial_noise_distortion_parameter
    main_config_for_noised_data.local_estimates.noise.seed = artificial_noise_seed

    embeddings_path_manager_for_noised_data: EmbeddingsPathManager = get_embeddings_path_manager(
        main_config=main_config_for_noised_data,
        logger=logger,
    )

    local_estimates_saving_manager_for_noised_data = LocalEstimatesSavingManager(
        embeddings_path_manager=embeddings_path_manager_for_noised_data,
        verbosity=verbosity,
        logger=logger,
    )

    local_estimates_container_noised_data: LocalEstimatesContainer = (
        local_estimates_saving_manager_for_noised_data.load_local_estimates()
    )

    # # # #
    # Compute differences between the local estimates

    # TODO: Implement the analysis here

    # TODO(Ben): Implement iteration over different noise levels and noise seeds,
    # to make a plot of Hausdorff distances vs. local estimates for each noise level and seed.

    # TODO: Create analysis of twoNN measure for individual tokens under different noise distortions
    # TODO: (currently, we plan to create an extra script for the token-level analysis)

    array_base_data = local_estimates_container_base_data.array_for_estimator_np
    if array_base_data is None:
        msg = "The array for the estimator is None."
        raise ValueError(
            msg,
        )

    # Generate toy data of size (2000, 768), random normal distribution
    toy_data = np.random.randn(2000, 768)

    # data_to_analyze: np.ndarray = toy_data
    data_to_analyze: np.ndarray = array_base_data

    # # # # # # # #
    # Version 1:
    #
    # Implementation via scipy.stats

    kernel = stats.gaussian_kde(
        dataset=data_to_analyze.T,  # Transpose to match the expected shape for multivariate KDE
        # bw_method=1.0,
    )
    result = kernel(
        data_to_analyze[:100].T,
    )
    logger.info(
        msg=f"result:\n{result}",  # noqa: G004 - low overhead
    )

    # # # # # # # #
    # Version 2:
    #
    # Implementation via sklearn.neighbors

    # Initialize and fit the KDE model
    kde = KernelDensity(
        kernel="gaussian",
        bandwidth="scott",
    )
    kde.fit(
        X=data_to_analyze,
    )

    # Evaluate the density on the dataset
    log_density: np.ndarray = kde.score_samples(
        X=data_to_analyze[:100],
    )  # Log density values
    density: np.ndarray = np.exp(log_density)  # Convert to actual density values

    logger.info(
        msg=f"log_density:\n{log_density}",  # noqa: G004 - low overhead
    )
    logger.info(
        msg=f"density:\n{density}",  # noqa: G004 - low overhead
    )

    # ================================================== #
    # Note: You can add additional analysis steps here
    # ================================================== #

    logger.info(
        msg="Running script DONE",
    )


def pairwise_distances(
    X,
) -> np.ndarray:
    """Calculate pairwise distance matrix of a given data matrix and return said matrix."""
    D = np.sum((X[None, :] - X[:, None]) ** 2, -1) ** 0.5
    return D


def get_neighbours_and_ranks(
    X,
    k,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the neighbourhoods and the ranks of a given space `X`, and return the corresponding tuple.

    An additional parameter $k$,
    the size of the neighbourhood, is required.
    """
    X = pairwise_distances(X)

    # Warning: this is only the ordering of neighbours that we need to
    # extract neighbourhoods below. The ranking comes later!
    X_ranks = np.argsort(X, axis=-1, kind="stable")

    # Extract neighbourhoods.
    X_neighbourhood = X_ranks[:, 1 : k + 1]

    # Convert this into ranks (finally)
    X_ranks = X_ranks.argsort(axis=-1, kind="stable")

    return X_neighbourhood, X_ranks


def MRRE_pointwise(
    X,
    Z,
    k,
) -> np.ndarray:
    """Calculate the pointwise mean rank distortion for each data point in the data space `X` with respect to the latent space `Z`.

    Inputs:
        - X: array of shape (m, n) (data space)
        - Z: array of shape (m, l) (latent space)
        - k: number of nearest neighbors to consider
    Output:
        - mean_rank_distortions: array of length m
    """
    X_neighbourhood, X_ranks = get_neighbours_and_ranks(X, k)
    Z_neighbourhood, Z_ranks = get_neighbours_and_ranks(Z, k)

    n = X.shape[0]
    mean_rank_distortions = np.zeros(n)

    for row in range(n):
        rank_differences = []
        for neighbour in Z_neighbourhood[row]:
            rx = X_ranks[row, neighbour]
            rz = Z_ranks[row, neighbour]
            rank_differences.append(abs(rx - rz) / rz)
        mean_rank_distortions[row] = np.mean(rank_differences)

    return mean_rank_distortions


if __name__ == "__main__":
    main()
