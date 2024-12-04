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

"""Run script to compute twoNN estimates from prepared embeddings."""

import logging
import pathlib
from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

from topollm.analysis.local_estimates_computation.global_and_pointwise_local_estimates_computation import (
    global_and_pointwise_local_estimates_computation,
)
from topollm.analysis.local_estimates_computation.truncate_prepared_data import truncate_prepared_data
from topollm.analysis.local_estimates_handling.deduplicator.factory import (
    get_prepared_data_deduplicator,
)
from topollm.analysis.local_estimates_handling.deduplicator.protocol import PreparedDataDeduplicator
from topollm.analysis.local_estimates_handling.filter.factory import get_local_estimates_filter
from topollm.analysis.local_estimates_handling.saving.local_estimates_containers import LocalEstimatesContainer
from topollm.analysis.local_estimates_handling.saving.save_local_estimates import save_local_estimates
from topollm.analysis.visualization.create_projected_data import create_projected_data
from topollm.analysis.visualization.create_projection_plot import create_projection_plot, save_projection_plot
from topollm.config_classes.main_config import MainConfig
from topollm.embeddings_data_prep.save_prepared_data import load_prepared_data
from topollm.logging.log_array_info import log_array_info
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    import numpy as np

    from topollm.analysis.local_estimates_handling.filter.protocol import LocalEstimatesFilter
    from topollm.config_classes.local_estimates.plot_config import LocalEstminatesPlotConfig
    from topollm.embeddings_data_prep.prepared_data_containers import PreparedData
    from topollm.path_management.embeddings.protocol import EmbeddingsPathManager

default_device = torch.device(device="cpu")
default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def global_and_pointwise_local_estimates_worker(
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
        logger.info(
            msg="Filtering prepared data and truncating to first vectors ...",
        )

    # Applay a filter; for example, for removing zero vectors in the array
    local_estimates_filter: LocalEstimatesFilter = get_local_estimates_filter(
        local_estimates_filtering_config=main_config.local_estimates.filtering,
        verbosity=verbosity,
        logger=logger,
    )
    prepared_data_filtered: PreparedData = local_estimates_filter.filter_data(
        prepared_data=prepared_data,
    )

    # Apply a deduplicator; for example, for removing duplicate vectors in the array
    prepared_data_deduplicator: PreparedDataDeduplicator = get_prepared_data_deduplicator(
        local_estimates_filtering_config=main_config.local_estimates.filtering,
        verbosity=verbosity,
        logger=logger,
    )
    prepared_data_filtered_deduplicated: PreparedData = prepared_data_deduplicator.filter_data(
        prepared_data=prepared_data_filtered,
    )

    # Restrict to the first `local_estimates_sample_size` samples
    local_estimates_sample_size: int = main_config.local_estimates.filtering.num_samples
    prepared_data_filtered_deduplicated_truncated: PreparedData = truncate_prepared_data(
        prepared_data=prepared_data_filtered_deduplicated,
        local_estimates_sample_size=local_estimates_sample_size,
    )

    array_for_estimator = prepared_data_filtered_deduplicated_truncated.array

    if verbosity >= Verbosity.NORMAL:
        log_array_info(
            array_=array_for_estimator,
            array_name="array_for_estimator",
            log_array_size=True,
            log_row_l2_norms=True,
            logger=logger,
        )
    if verbosity >= Verbosity.DEBUG:
        prepared_data_filtered_deduplicated_truncated.log_info(
            logger=logger,
        )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Filtering local estimates and truncating to first vectors DONE",
        )

    # # # #
    # Local estimates computation

    global_estimate_array_np, pointwise_results_array_np = global_and_pointwise_local_estimates_computation(
        array_for_estimator=array_for_estimator,
        local_estimates_config=main_config.local_estimates,
        verbosity=verbosity,
        logger=logger,
    )

    # # # #
    # Save the results
    local_estimates_container = LocalEstimatesContainer(
        pointwise_results_array_np=pointwise_results_array_np,
        pointwise_results_meta_frame=prepared_data_filtered_deduplicated_truncated.meta_df,
        global_estimate_array_np=global_estimate_array_np,
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
        local_estimates_plot_config: LocalEstminatesPlotConfig = main_config.local_estimates.plot

        tsne_array: np.ndarray = create_projected_data(
            array=prepared_data_filtered.array,
            pca_n_components=local_estimates_plot_config.pca_n_components,
            tsne_n_components=local_estimates_plot_config.tsne_n_components,
            tsne_random_state=local_estimates_plot_config.tsne_random_state,
            verbosity=verbosity,
            logger=logger,
        )

        for maximum_number_of_points in tqdm(
            iterable=[
                500,
                1_000,
                5_000,
            ],
            desc="Creating projection plots",
        ):
            figure, tsne_df = create_projection_plot(
                tsne_result=tsne_array,
                meta_df=prepared_data_filtered.meta_df,
                results_array_np=pointwise_results_array_np,
                maximum_number_of_points=maximum_number_of_points,
                verbosity=verbosity,
                logger=logger,
            )

            number_of_points_in_plot: int = len(tsne_df)
            output_folder = pathlib.Path(
                embeddings_path_manager.get_saved_plots_local_estimates_projection_dir_absolute_path(),
                f"no-points-in-plot-{number_of_points_in_plot}",
            )
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"Saving projection plot to {output_folder = }",  # noqa: G004 - low overhead
                )

            save_projection_plot(
                figure=figure,
                tsne_df=tsne_df,
                output_folder=output_folder,
                save_html=local_estimates_plot_config.saving.save_html,
                save_pdf=local_estimates_plot_config.saving.save_pdf,
                save_csv=local_estimates_plot_config.saving.save_csv,
                verbosity=verbosity,
                logger=logger,
            )
