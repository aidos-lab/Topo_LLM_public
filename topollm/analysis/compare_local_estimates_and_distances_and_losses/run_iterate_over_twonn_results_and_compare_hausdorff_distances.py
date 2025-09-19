"""Run script to compare computed Hausdorff distances with the local estimates."""

import logging
import os
import pathlib
import pprint
from itertools import product
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import omegaconf
import pandas as pd
from tqdm import tqdm

from topollm.analysis.local_estimates_handling.saving.local_estimates_saving_manager import LocalEstimatesSavingManager
from topollm.config_classes.constants import (
    HYDRA_CONFIGS_BASE_PATH,
)
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.path_management.parse_path_info import parse_local_estimates_info
from topollm.plotting.create_scatter_plot import (
    create_scatter_plot,
)
from topollm.storage.saving_and_loading_functions.saving_and_loading import save_dataframe_as_csv
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    from topollm.analysis.local_estimates_handling.saving.local_estimates_containers import LocalEstimatesContainer
    from topollm.config_classes.main_config import MainConfig
    from topollm.path_management.embeddings.protocol import EmbeddingsPathManager


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


def iterate_and_collect_data(
    base_path: pathlib.Path,
    subdirectory_to_match: str = "n-neighbors-mode=absolute_size_n-neighbors=128",
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> pd.DataFrame:
    """Iterate over experiment directories and collect data into a DataFrame.

    Args:
        base_path:
            Path to the directory containing experiment subfolders.
        subdirectory_to_match:
            Subdirectory name to match.
        verbosity:
            Verbosity level.
        logger:
            Logger instance.

    Returns:
        A pandas DataFrame with the collected data.

    """
    base_path_iterdir_list = list(base_path.iterdir())

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Iterating over experiments in {base_path = } ...",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"Matching {subdirectory_to_match = }",  # noqa: G004 - low overhead
        )

        logger.info(
            msg=f"{len(base_path_iterdir_list) = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"base_path_iterdir_list:\n{pprint.pformat(base_path_iterdir_list)}",  # noqa: G004 - low overhead
        )

    # Container for the collected data
    data: list = []

    experiment_dir_count = 0
    subdirectory_count = 0
    matched_subdirectory_count = 0

    for experiment_dir in tqdm(
        iterable=base_path.iterdir(),
        desc="Iterating over experiments ...",
        total=len(base_path_iterdir_list),
    ):
        if experiment_dir.is_dir():
            experiment_dir_count += 1

            # Iterate over the subdirectories
            for subdirectory in experiment_dir.iterdir():
                if subdirectory.is_dir():
                    subdirectory_count += 1
                    if verbosity >= Verbosity.NORMAL:
                        logger.info(
                            msg=f"{subdirectory = }",  # noqa: G004 - low overhead
                        )

                if subdirectory.is_dir() and subdirectory.name == subdirectory_to_match:
                    matched_subdirectory_count += 1

                    local_estimates_saving_manager: LocalEstimatesSavingManager = (
                        LocalEstimatesSavingManager.from_local_estimates_pointwise_dir_absolute_path(
                            local_estimates_pointwise_dir_absolute_path=subdirectory,
                            verbosity=verbosity,
                            logger=logger,
                        )
                    )
                    local_estimates_container: LocalEstimatesContainer = (
                        local_estimates_saving_manager.load_local_estimates()
                    )

                    local_estimates_info: dict = parse_local_estimates_info(
                        path=subdirectory,
                    )

                    additional_distance_computations_dict: dict | None = (
                        local_estimates_container.additional_distance_computations_results
                    )
                    if additional_distance_computations_dict is None:
                        additional_distance_computations_dict = {}

                    # Add the prefix "additional_distance_" to the keys
                    # of the additional distance computations results
                    # so that it is easier to pick them out later for the comparison scatter plots.
                    modified_keys_additional_distance_computations_dict: dict = {
                        f"additional_distance_{key}": value
                        for key, value in additional_distance_computations_dict.items()
                    }

                    experiment_data: dict = {
                        "experiment_dir_name": experiment_dir.name,
                        "pointwise_results_np_mean": local_estimates_container.get_pointwise_results_np_mean(),
                        "pointwise_results_np_std": local_estimates_container.get_pointwise_results_np_std(),
                        "global_estimate": local_estimates_container.get_global_estimate(),
                        **modified_keys_additional_distance_computations_dict,
                        **local_estimates_info,
                    }

                    if experiment_data:
                        data.append(experiment_data)

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Finished iterating over experiments.",
        )
        logger.info(
            msg=f"{experiment_dir_count = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{subdirectory_count = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{matched_subdirectory_count = }",  # noqa: G004 - low overhead
        )

    collected_data_df = pd.DataFrame(
        data=data,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Collected data for {len(collected_data_df)} experiments.",  # noqa: G004 - low overhead
        )

    return collected_data_df


def iterate_over_different_local_estimates_directories_for_given_base_directory(
    base_dir: os.PathLike,
    output_directory: os.PathLike,
    subdirectory_to_match: str = "n-neighbors-mode=absolute_size_n-neighbors=128",
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Iterate over experiments, save collected data, and create a scatter plot.

    Args:
        base_dir:
            Path to the directory containing experiment subfolders.
        output_directory:
            Root path to save data.
        subdirectory_to_match:
            Subdirectory name to match.
        verbosity:
            Verbosity level.
        logger:
            Logger instance.

    """
    # Convert to pathlib.Path
    base_dir = pathlib.Path(
        base_dir,
    )
    output_directory = pathlib.Path(
        output_directory,
    )

    # Collect data
    collected_data_df: pd.DataFrame = iterate_and_collect_data(
        base_path=base_dir,
        subdirectory_to_match=subdirectory_to_match,
        verbosity=verbosity,
        logger=logger,
    )

    # Save raw data
    dataframe_output_path = pathlib.Path(
        output_directory,
        "raw_data.csv",
    )
    save_dataframe_as_csv(
        dataframe=collected_data_df,
        save_path=dataframe_output_path,
        dataframe_name_for_logging="collected_data_df",
        verbosity=verbosity,
        logger=logger,
    )

    # Check if the DataFrame is empty
    if collected_data_df.empty:
        logger.warning(
            msg="No data collected. Skipping scatter plot creation.",
        )
        return

    # # # #
    # Create scatter plots
    create_multiple_scatter_plots(
        collected_data_df=collected_data_df,
        output_directory=output_directory,
        verbosity=verbosity,
        logger=logger,
    )


def create_multiple_scatter_plots(
    collected_data_df: pd.DataFrame,
    output_directory: os.PathLike,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Create multiple scatter plots for the collected data."""
    additional_distance_column_names: list[str] = [
        column_name for column_name in collected_data_df.columns if column_name.startswith("additional_distance_")
    ]

    x_column_names_for_iteration: list[str] = [
        *additional_distance_column_names,
        "local_estimates_noise_distortion",
    ]
    # We also want to plot the additional distances against each other and the noise distortion,
    # this is why we add the additional distances to the y_column_names_for_iteration.
    y_column_names_for_iteration: list[str] = [
        "pointwise_results_np_mean",
        "pointwise_results_np_std",
        "global_estimate",
        *additional_distance_column_names,
    ]

    y_axes_limits: dict[
        str,
        tuple[
            float | None,
            float | None,
        ],
    ] = {
        "auto": (None, None),
        "multiwoz": (8.5, 11.5),
    }

    for y_min, y_max in y_axes_limits.values():
        for x_column_name, y_column_name in tqdm(
            iterable=product(
                x_column_names_for_iteration,
                y_column_names_for_iteration,
            ),
            desc="Creating scatter plots ...",
        ):
            plot_output_path = pathlib.Path(
                output_directory,
                "scatter_plots",
                f"{y_min=}_{y_max=}",
                f"{x_column_name=}_vs_{y_column_name=}",
            )

            create_scatter_plot(
                df=collected_data_df,
                output_folder=plot_output_path,
                x_column_name=x_column_name,
                y_column_name=y_column_name,
                y_min=y_min,
                y_max=y_max,
                show_plot=False,
                verbosity=verbosity,
                logger=logger,
            )


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

    # ================================================== #
    #
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
    #
    # ================================================== #

    local_estimates_dir_absolute_path: pathlib.Path = embeddings_path_manager.get_local_estimates_dir_absolute_path()
    root_iteration_dir: pathlib.Path = local_estimates_dir_absolute_path.parent

    logger.info(
        msg=f"{local_estimates_dir_absolute_path = }",  # noqa: G004 - low overhead
    )
    logger.info(
        msg=f"{root_iteration_dir = }",  # noqa: G004 - low overhead
    )

    # Get the local estimates config description,
    # our directory iteration will match only these directories.
    local_estimates_pointwise_config_description: str = (
        embeddings_path_manager.get_local_estimates_pointwise_config_description()
    )

    output_directory = pathlib.Path(
        embeddings_path_manager.analysis_dir,
        "noise_distances_versus_estimates",
        f"subdirectory_to_match={local_estimates_pointwise_config_description}",
        embeddings_path_manager.get_local_estimates_subfolder_path().parent,
    )

    iterate_over_different_local_estimates_directories_for_given_base_directory(
        base_dir=root_iteration_dir,
        output_directory=output_directory,
        subdirectory_to_match=local_estimates_pointwise_config_description,
        verbosity=verbosity,
        logger=logger,
    )

    # ================================================== #
    #
    # ================================================== #

    logger.info(
        msg="Running script DONE",
    )


if __name__ == "__main__":
    main()
