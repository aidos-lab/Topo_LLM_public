# Copyright 2024-2025
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Matthias Ruppik (mail@ruppik.net)
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

"""Kernel-based density estimation."""

import logging

import numpy as np
from scipy import stats
from sklearn.neighbors import KernelDensity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def test_kernel_based_density_estimation(
    logger: logging.Logger = default_logger,
) -> None:
    """Test kernel-based density estimation.

    Note:
    Kernel based density estimation methods do not appear to work in our higher-dimensional setting
    (for example, the roberta-base embeddings have 768 dimensions).
    We either get very small values or NaNs.

    """
    # Generate toy data of size (2000, 768), random normal distribution
    toy_data: np.ndarray = np.random.randn(2000, 768)

    data_to_analyze: np.ndarray = toy_data

    # # # # # # # #
    # Version 1:
    #
    # Implementation via scipy.stats

    kernel = stats.gaussian_kde(
        dataset=data_to_analyze.T,  # Transpose to match the expected shape for multivariate KDE
        bw_method=1.0,
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
