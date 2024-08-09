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

import torch

from topollm.analysis.local_estimates.filter.get_local_estimates_filter import get_local_estimates_filter
from topollm.analysis.local_estimates.saving.local_estimates_containers import LocalEstimatesContainer
from topollm.analysis.local_estimates.saving.save_local_estimates import save_local_estimates
from topollm.analysis.twonn.truncate_prepared_data import truncate_prepared_data
from topollm.analysis.twonn.twonn_local_estimates_computation import twonn_local_estimates_computation
from topollm.analysis.visualization.create_projected_data import create_projected_data
from topollm.analysis.visualization.create_projection_plot import create_projection_plot, save_projection_plot
from topollm.config_classes.main_config import MainConfig
from topollm.embeddings_data_prep.save_prepared_data import load_prepared_data
from topollm.logging.log_array_info import log_array_info
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    import numpy as np

    from topollm.analysis.local_estimates.filter.protocol import LocalEstimatesFilter
    from topollm.embeddings_data_prep.prepared_data_containers import PreparedData
    from topollm.path_management.embeddings.protocol import EmbeddingsPathManager

default_device = torch.device("cpu")
default_logger = logging.getLogger(__name__)


def twonn_worker(
    main_config: MainConfig,
    device: torch.device = default_device,  # noqa: ARG001 - placeholder for future use
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Prepare the embedding data of a model and its metadata for further analysis."""
    embeddings_path_manager: EmbeddingsPathManager = get_embeddings_path_manager(
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

    # # # #
    if verbosity >= Verbosity.NORMAL:
        logger.info("Filtering local estimates and truncating to first vectors ...")

    local_estimates_filter: LocalEstimatesFilter = get_local_estimates_filter(
        local_estimates_filtering_config=main_config.local_estimates.filtering,
        verbosity=verbosity,
        logger=logger,
    )

    # Filter the array, for example, by potentially removing zero vectors
    prepared_data_filtered: PreparedData = local_estimates_filter.filter_data(
        prepared_data=prepared_data,
    )

    # Restrict to the first `local_estimates_sample_size` samples
    local_estimates_sample_size = main_config.local_estimates.filtering.num_samples
    prepared_data_filtered_truncated: PreparedData = truncate_prepared_data(
        prepared_data=prepared_data_filtered,
        local_estimates_sample_size=local_estimates_sample_size,
    )

    array_for_estimator = prepared_data_filtered_truncated.array

    if verbosity >= Verbosity.NORMAL:
        log_array_info(
            array_for_estimator,
            array_name="array_for_estimator",
            log_array_size=True,
            log_row_l2_norms=True,
            logger=logger,
        )

    if verbosity >= Verbosity.NORMAL:
        logger.info("Filtering local estimates and truncating to first vectors DONE")

    # # # #
    # Local estimates computation

    # provide number of jobs for the computation
    results_array_np = twonn_local_estimates_computation(
        array_for_estimator=array_for_estimator,
        verbosity=verbosity,
        logger=logger,
    )

    # # # #
    # Save the results
    local_estimates_container = LocalEstimatesContainer(
        results_array_np=results_array_np,
        results_meta_frame=prepared_data_filtered_truncated.meta_df,
    )

    save_local_estimates(
        embeddings_path_manager=embeddings_path_manager,
        local_estimates_container=local_estimates_container,
        verbosity=verbosity,
        logger=logger,
    )

    # # # #
    # Create plots
    if main_config.feature_flags.analysis.create_plots_in_local_estimates_worker:
        local_estimates_plot_config = main_config.local_estimates.plot

        tsne_array: np.ndarray = create_projected_data(
            array=prepared_data_filtered.array,
            pca_n_components=local_estimates_plot_config.pca_n_components,
            tsne_n_components=local_estimates_plot_config.tsne_n_components,
            tsne_random_state=local_estimates_plot_config.tsne_random_state,
        )

        figure, tsne_df = create_projection_plot(
            tsne_result=tsne_array,
            meta_df=prepared_data_filtered.meta_df,
            results_array_np=results_array_np,
            verbosity=verbosity,
            logger=logger,
        )
        save_projection_plot(
            figure=figure,
            tsne_df=tsne_df,
            output_folder=embeddings_path_manager.get_saved_plots_local_estimates_projection_dir_absolute_path(),
            save_html=local_estimates_plot_config.saving.save_html,
            save_pdf=local_estimates_plot_config.saving.save_pdf,
            save_csv=local_estimates_plot_config.saving.save_csv,
            verbosity=verbosity,
            logger=logger,
        )
