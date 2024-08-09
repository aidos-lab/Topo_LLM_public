# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (ruppik@hhu.de)
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

"""Compute twoNN estimates from prepared embeddings."""

import logging

import numpy as np
import skdim

from topollm.logging.log_array_info import log_array_info
from topollm.typing.enums import Verbosity

default_logger = logging.getLogger(__name__)


def twonn_local_estimates_computation(
    array_for_estimator: np.ndarray,
    discard_fraction: float = 0.1,
    n_jobs: int = 1,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> np.ndarray:
    """Run the local estimates computation."""
    # Number of neighbors which are used for the computation
    n_neighbors = round(len(array_for_estimator) * 0.8)
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"{n_neighbors = }",  # noqa: G004 - low overhead
        )

    estimator = skdim.id.TwoNN(
        discard_fraction=discard_fraction,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info("Calling estimator.fit_pw() ...")

    fitted_estimator = estimator.fit_pw(
        X=array_for_estimator,
        precomputed_knn=None,
        smooth=False,
        n_neighbors=n_neighbors,
        n_jobs=n_jobs,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info("Calling estimator.fit_pw() DONE")

    results_array = list(fitted_estimator.dimension_pw_)

    results_array_np = np.array(
        results_array,
    )

    if verbosity >= Verbosity.NORMAL:
        log_array_info(
            results_array_np,
            array_name="results_array_np",
            log_array_size=True,
            log_row_l2_norms=False,  # Note: This is a one-dimensional array, so the l2-norms are not meaningful
            logger=logger,
        )

        # Log the mean and standard deviation of the local estimates
        logger.info(
            f"Mean of local estimates: {results_array_np.mean() = }",  # noqa: G004 - low overhead
        )
        logger.info(
            f"Standard deviation of local estimates: {results_array_np.std() = }",  # noqa: G004 - low overhead
        )

    return results_array_np
