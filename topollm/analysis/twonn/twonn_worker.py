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

"""Run script to compute twoNN estimates from prepared embeddings."""

import logging
from typing import TYPE_CHECKING

import numpy as np
import skdim
import torch

from topollm.analysis.local_estimates.filter.get_local_estimates_filter import get_local_estimates_filter
from topollm.analysis.twonn.save_local_estimates import save_local_estimates
from topollm.config_classes.main_config import MainConfig
from topollm.embeddings_data_prep.prepared_data_containers import PreparedData
from topollm.embeddings_data_prep.save_prepared_data import load_prepared_data
from topollm.logging.log_array_info import log_array_info
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    from topollm.analysis.local_estimates.filter.protocol import LocalEstimatesFilter

default_device = torch.device("cpu")
default_logger = logging.getLogger(__name__)


def twonn_worker(
    main_config: MainConfig,
    device: torch.device = default_device,  # noqa: ARG001 - placeholder for future use
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Prepare the embedding data of a model and its metadata for further analysis."""
    embeddings_path_manager = get_embeddings_path_manager(
        main_config=main_config,
        logger=logger,
    )

    # # # #
    # Load the prepared data.
    # Logging of the loaded data is handled in the loading function.
    prepared_data: PreparedData = load_prepared_data(
        embeddings_path_manager=embeddings_path_manager,
        verbosity=verbosity,
        logger=logger,
    )

    arr_no_pad = prepared_data.arr_no_pad

    if verbosity >= Verbosity.NORMAL:
        log_array_info(
            arr_no_pad,
            array_name="arr_no_pad",
            log_array_size=True,
            log_row_l2_norms=True,
            logger=logger,
        )

    # # # #
    if verbosity >= Verbosity.NORMAL:
        logger.info("Filtering local estimates and truncating to first vectors ...")

    local_estimates_filter: LocalEstimatesFilter = get_local_estimates_filter(
        local_estimates_filtering_config=main_config.local_estimates.filtering,
        verbosity=verbosity,
        logger=logger,
    )

    # Filter the array, for example, by potentially removing zero vectors
    arr_no_pad_filtered = local_estimates_filter.filter_data(
        input_array=arr_no_pad,
    )

    # Restrict to the first `local_estimates_sample_size` samples
    local_estimates_sample_size = main_config.local_estimates.filtering.num_samples
    arr_no_pad_truncated = truncate_data(
        arr_no_pad_filtered=arr_no_pad_filtered,
        local_estimates_sample_size=local_estimates_sample_size,
    )

    if verbosity >= Verbosity.NORMAL:
        log_array_info(
            arr_no_pad_truncated,
            array_name="arr_no_pad_truncated",
            log_array_size=True,
            log_row_l2_norms=True,
            logger=logger,
        )

    if verbosity >= Verbosity.NORMAL:
        logger.info("Filtering local estimates and truncating to first vectors DONE")

    # # # #
    # Local estimates computation

    # provide number of jobs for the computation
    n_jobs = 1

    # provide number of neighbors which are used for the computation
    n_neighbors = round(len(arr_no_pad_truncated) * 0.8)
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"{n_neighbors = }",  # noqa: G004 - low overhead
        )

    estimator = skdim.id.TwoNN(
        discard_fraction=0.1,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info("Calling estimator.fit_pw() ...")

    fitted_estimator = estimator.fit_pw(
        X=arr_no_pad_truncated,
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

    # # # #
    # Save the results
    save_local_estimates(
        embeddings_path_manager=embeddings_path_manager,
        results_array_np=results_array_np,
        verbosity=verbosity,
        logger=logger,
    )


def truncate_data(
    arr_no_pad_filtered: np.ndarray,
    local_estimates_sample_size: int,
) -> np.ndarray:
    """Truncate the data to the first `local_estimates_sample_size` samples."""
    local_estimates_sample_size = min(
        local_estimates_sample_size,
        arr_no_pad_filtered.shape[0],
    )

    arr_no_pad_truncated = arr_no_pad_filtered[:local_estimates_sample_size]
    return arr_no_pad_truncated
