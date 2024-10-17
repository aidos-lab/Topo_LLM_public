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

from topollm.analysis.local_estimates_computation.get_n_neighbors_from_array_len_and_pointwise_config import (
    get_n_neighbors_from_array_len_and_pointwise_config,
)
from topollm.config_classes.local_estimates.local_estimates_config import LocalEstimatesConfig
from topollm.logging.log_array_info import log_array_info
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def global_and_pointwise_local_estimates_computation(
    array_for_estimator: np.ndarray,
    local_estimates_config: LocalEstimatesConfig,
    twonn_discard_fraction: float = 0.1,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> tuple[
    np.ndarray | None,
    np.ndarray,
]:
    """Run the local estimates computation."""
    # # # #
    # Number of neighbors which are used for the computation of the pointwise local estimates
    n_neighbors: int = get_n_neighbors_from_array_len_and_pointwise_config(
        array_len=len(array_for_estimator),
        pointwise_config=local_estimates_config.pointwise,
        verbosity=verbosity,
        logger=logger,
    )

    # # # #
    # Estimator setup
    estimator = skdim.id.TwoNN(
        discard_fraction=twonn_discard_fraction,
    )

    # # # #
    # Pointwise estimates computation
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Pointwise computation.",
        )
        logger.info(
            msg="Calling estimator.fit_pw() ...",
        )

    fitted_pw_estimator = estimator.fit_pw(
        X=array_for_estimator,
        precomputed_knn=None,
        smooth=False,
        n_neighbors=n_neighbors,
        n_jobs=local_estimates_config.pointwise.n_jobs,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Calling estimator.fit_pw() DONE",
        )

    pointwise_results_array = list(
        fitted_pw_estimator.dimension_pw_,
    )

    pointwise_results_array_np = np.array(
        pointwise_results_array,
    )

    if verbosity >= Verbosity.NORMAL:
        log_array_info(
            array_=pointwise_results_array_np,
            array_name="pointwise_results_array_np",
            log_array_size=True,
            log_row_l2_norms=False,  # Note: This is a one-dimensional array, so the l2-norms are not meaningful
            logger=logger,
        )

        # Log the mean and standard deviation of the local estimates
        logger.info(
            msg=f"{pointwise_results_array_np.mean() = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{pointwise_results_array_np.std() = }",  # noqa: G004 - low overhead
        )

    # # # #
    # Global estimate computation
    if local_estimates_config.compute_global_estimates:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Global computation.",
            )
            logger.info(
                msg="Calling estimator.fit_transform() ...",
            )

        global_dimension = estimator.fit_transform(
            X=array_for_estimator,
        )

        global_estimate_array_np = np.array(
            [global_dimension],
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Calling estimator.fit_transform() DONE",
            )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                "global_estimate_array_np:\n%s",
                global_estimate_array_np,
            )
    else:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Skipping global computation.",
            )
        global_estimate_array_np = None

    return global_estimate_array_np, pointwise_results_array_np
