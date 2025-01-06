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
import pprint
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from topollm.analysis.local_estimates_computation.constants import (
    APPROXIMATE_HAUSDORFF_VIA_KDTREE_DICT_KEY,
    EXACT_HAUSDORFF_DICT_KEY,
)
from topollm.analysis.local_estimates_computation.global_and_pointwise_local_estimates_computation import (
    global_and_pointwise_local_estimates_computation,
)
from topollm.analysis.local_estimates_computation.truncate_prepared_data import truncate_prepared_data
from topollm.analysis.local_estimates_handling.deduplicator.factory import (
    get_prepared_data_deduplicator,
)
from topollm.analysis.local_estimates_handling.distances.distance_functions import (
    approximate_hausdorff_via_kdtree,
    compute_exact_hausdorff,
    geomloss_sinkhorn_wasserstein,
)
from topollm.analysis.local_estimates_handling.filter.factory import get_local_estimates_filter
from topollm.analysis.local_estimates_handling.noise.factory import get_prepared_data_noiser
from topollm.analysis.local_estimates_handling.saving.local_estimates_containers import LocalEstimatesContainer
from topollm.analysis.local_estimates_handling.saving.local_estimates_saving_manager import LocalEstimatesSavingManager
from topollm.analysis.visualization.create_projected_data import create_projected_data
from topollm.analysis.visualization.create_projection_plot import create_projection_plot, save_projection_plot
from topollm.config_classes.local_estimates.plot_config import LocalEstminatesPlotConfig
from topollm.config_classes.main_config import MainConfig
from topollm.embeddings_data_prep.prepared_data_containers import PreparedData
from topollm.embeddings_data_prep.save_prepared_data import load_prepared_data
from topollm.logging.log_array_info import log_array_info
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.path_management.embeddings.protocol import EmbeddingsPathManager
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    from topollm.analysis.local_estimates_handling.deduplicator.protocol import PreparedDataDeduplicator
    from topollm.analysis.local_estimates_handling.filter.protocol import LocalEstimatesFilter
    from topollm.analysis.local_estimates_handling.noise.protocol import PreparedDataNoiser

default_device = torch.device(
    device="cpu",
)
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
    (
        prepared_data_filtered_deduplicated_truncated,
        prepared_data_filtered_deduplicated_truncated_noised,
    ) = preprocess_prepared_data(
        main_config=main_config,
        prepared_data=prepared_data,
        verbosity=verbosity,
        logger=logger,
    )

    array_for_estimator: np.ndarray = prepared_data_filtered_deduplicated_truncated_noised.array

    if verbosity >= Verbosity.NORMAL:
        log_array_info(
            array_=array_for_estimator,
            array_name="array_for_estimator",
            log_array_size=True,
            log_row_l2_norms=True,
            logger=logger,
        )

    # # # #
    # Local estimates computation

    (
        global_estimate_array_np,
        pointwise_results_array_np,
    ) = global_and_pointwise_local_estimates_computation(
        array_for_estimator=array_for_estimator,
        local_estimates_config=main_config.local_estimates,
        verbosity=verbosity,
        logger=logger,
    )

    # # # #
    # Distance computation between the original and the distorted data

    clean_array: np.ndarray = prepared_data_filtered_deduplicated_truncated.array
    noisy_array: np.ndarray = array_for_estimator

    additional_distance_computations_results: dict = compute_distance_metrics(
        main_config=main_config,
        clean_array=clean_array,
        noisy_array=noisy_array,
        verbosity=verbosity,
        logger=logger,
    )

    # # # #
    # Create additional statistics for easier storage and analysis
    additional_pointwise_results_statistics: dict = create_additional_pointwise_results_statistics(
        pointwise_results_array_np=pointwise_results_array_np,
        verbosity=verbosity,
        logger=logger,
    )

    # Select to save the array for the estimator based on the feature flag
    if main_config.feature_flags.analysis.saving.save_array_for_estimator:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="array_for_estimator will be part of the container and will be saved.",
            )
        array_for_estimator_for_container = array_for_estimator
    else:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="array_for_estimator will NOT be part of the container and will NOT be saved.",
            )
        array_for_estimator_for_container = None

    # # # #
    # Save the results
    local_estimates_container = LocalEstimatesContainer(
        pointwise_results_array_np=pointwise_results_array_np,
        pointwise_results_meta_frame=prepared_data_filtered_deduplicated_truncated_noised.meta_df,
        global_estimate_array_np=global_estimate_array_np,
        array_for_estimator_np=array_for_estimator_for_container,
        additional_distance_computations_results=additional_distance_computations_results,
        additional_pointwise_results_statistics=additional_pointwise_results_statistics,
    )

    local_estimates_save_manager: LocalEstimatesSavingManager = (
        LocalEstimatesSavingManager.from_embeddings_path_manager(
            embeddings_path_manager=embeddings_path_manager,
            verbosity=verbosity,
            logger=logger,
        )
    )

    local_estimates_save_manager.save_local_estimates(
        local_estimates_container=local_estimates_container,
    )

    # # # #
    # Create plots
    if main_config.feature_flags.analysis.create_plots_in_local_estimates_worker:
        local_estimates_plot_config: LocalEstminatesPlotConfig = main_config.local_estimates.plot

        generate_tsne_visualizations(
            embeddings_path_manager=embeddings_path_manager,
            prepared_data_filtered=prepared_data_filtered_deduplicated_truncated_noised,
            pointwise_results_array_np=pointwise_results_array_np,
            local_estimates_plot_config=local_estimates_plot_config,
            verbosity=verbosity,
            logger=logger,
        )


def create_additional_pointwise_results_statistics(
    pointwise_results_array_np: np.ndarray,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> dict:
    """Create additional statistics from the pointwise results array and other computation results."""
    additional_pointwise_results_statistics: dict = {}

    # We collect the statistics of the pointwise results array under a separate key.
    # This allows for a more structured storage of the results and easier extension in the future.
    subkey = "pointwise_results_array_np"
    subdict: dict = make_array_statistics_dict(
        array=pointwise_results_array_np,
        array_name=subkey,
    )

    additional_pointwise_results_statistics[subkey] = subdict

    # Add statistics of truncated pointwise results arrays
    for truncation_size in range(
        5_000,
        60_001,
        5_000,
    ):
        subkey: str = f"pointwise_results_array_np_truncated_first_{truncation_size}"
        subdict: dict = make_array_statistics_dict(
            array=pointwise_results_array_np[:truncation_size],
            array_name=subkey,
        )

        additional_pointwise_results_statistics[subkey] = subdict

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"additional_pointwise_results_statistics:\n"  # noqa: G004 - low overhead
            f"{pprint.pformat(object=additional_pointwise_results_statistics)}",
        )

    return additional_pointwise_results_statistics


def make_array_statistics_dict(
    array: np.ndarray,
    array_name: str,
) -> dict:
    """Create a dictionary with statistics about the array."""
    array_statistics_dict: dict = {}

    array_statistics_dict["array_name"] = array_name
    array_statistics_dict["shape"] = array.shape
    array_statistics_dict["np_mean"] = np.mean(
        a=array,
    )
    array_statistics_dict["np_std"] = np.std(
        a=array,
    )

    # Convert into a pandas DataFrame and save the describe() output.
    # Note that numpy and pandas use different versions of the standard deviation,
    # where pandas is the unbiased estimator with N-1 in the denominator,
    # while numpy uses N.
    pd_describe_df: pd.DataFrame = pd.DataFrame(
        data=array,
    ).describe()
    array_statistics_dict["pd_describe"] = pd_describe_df.to_dict()

    return array_statistics_dict


def compute_distance_metrics(
    main_config: MainConfig,
    clean_array: np.ndarray,
    noisy_array: np.ndarray,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> dict:
    """Compute additional distance metrics between the clean and noisy arrays."""
    # Container for additional distance computations
    additional_distance_computations_results: dict = {}

    if main_config.feature_flags.analysis.distance_functions.use_approximate_hausdorff_via_kdtree:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Computing approximate Hausdorff distance via KDTree ...",
            )

        approximate_hausdorff_distance: float = approximate_hausdorff_via_kdtree(
            array_1=clean_array,
            array_2=noisy_array,
        )

        additional_distance_computations_results[APPROXIMATE_HAUSDORFF_VIA_KDTREE_DICT_KEY] = (
            approximate_hausdorff_distance
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"{approximate_hausdorff_distance = }",  # noqa: G004 - low overhead
            )

    if main_config.feature_flags.analysis.distance_functions.use_exact_hausdorff:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Computing exact Hausdorff distance ...",
            )

        exact_hausdorff_distance: float = compute_exact_hausdorff(
            array_1=clean_array,
            array_2=noisy_array,
        )

        additional_distance_computations_results[EXACT_HAUSDORFF_DICT_KEY] = exact_hausdorff_distance

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"{exact_hausdorff_distance = }",  # noqa: G004 - low overhead
            )

    if main_config.feature_flags.analysis.distance_functions.use_sinkhorn_wasserstein:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Computing Sinkhorn Wasserstein distance ...",
            )

        sinkhorn_wasserstein = geomloss_sinkhorn_wasserstein(
            P_np=clean_array,
            Q_np=noisy_array,
        )

        additional_distance_computations_results["sinkhorn_wasserstein"] = sinkhorn_wasserstein

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"{sinkhorn_wasserstein = }",  # noqa: G004 - low overhead
            )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"additional_distance_computations_results:\n"  # noqa: G004 - low overhead
            f"{pprint.pformat(object=additional_distance_computations_results)}",
        )

    return additional_distance_computations_results


def preprocess_prepared_data(
    main_config: MainConfig,
    prepared_data: PreparedData,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> tuple[
    PreparedData,
    PreparedData,
]:
    """Preprocess the prepared data for local estimates computation."""
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Filtering, deduplicating, truncating, noising the prepared data ...",
        )

    # Apply a filter; for example, for removing zero vectors in the array.
    local_estimates_filter: LocalEstimatesFilter = get_local_estimates_filter(
        local_estimates_filtering_config=main_config.local_estimates.filtering,
        verbosity=verbosity,
        logger=logger,
    )
    prepared_data_filtered: PreparedData = local_estimates_filter.filter_data(
        prepared_data=prepared_data,
    )

    # Apply a deduplicator; for example, for removing duplicate vectors in the array.
    prepared_data_deduplicator: PreparedDataDeduplicator = get_prepared_data_deduplicator(
        local_estimates_filtering_config=main_config.local_estimates.filtering,
        verbosity=verbosity,
        logger=logger,
    )
    prepared_data_filtered_deduplicated: PreparedData = prepared_data_deduplicator.filter_data(
        prepared_data=prepared_data_filtered,
    )

    # Restrict to the first `local_estimates_sample_size` samples.
    local_estimates_sample_size: int = main_config.local_estimates.filtering.num_samples
    prepared_data_filtered_deduplicated_truncated: PreparedData = truncate_prepared_data(
        prepared_data=prepared_data_filtered_deduplicated,
        local_estimates_sample_size=local_estimates_sample_size,
    )

    # Potentially apply noise to the data.
    prepared_data_noiser: PreparedDataNoiser = get_prepared_data_noiser(
        local_estimates_noise_config=main_config.local_estimates.noise,
        verbosity=verbosity,
        logger=logger,
    )
    prepared_data_filtered_deduplicated_truncated_noised: PreparedData = prepared_data_noiser.apply_noise_to_data(
        prepared_data=prepared_data_filtered_deduplicated_truncated,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Filtering, deduplicating, truncating, noising the prepared data DONE",
        )

    if verbosity >= Verbosity.DEBUG:
        prepared_data_filtered_deduplicated_truncated_noised.log_info(
            logger=logger,
        )

    return prepared_data_filtered_deduplicated_truncated, prepared_data_filtered_deduplicated_truncated_noised


def generate_tsne_visualizations(
    embeddings_path_manager: EmbeddingsPathManager,
    prepared_data_filtered: PreparedData,
    pointwise_results_array_np: np.ndarray,
    local_estimates_plot_config: LocalEstminatesPlotConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Generate t-SNE visualizations of the local estimates."""
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
        (
            figure,
            tsne_df,
        ) = create_projection_plot(
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

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving projection plot to {output_folder = } DONE",  # noqa: G004 - low overhead
            )
