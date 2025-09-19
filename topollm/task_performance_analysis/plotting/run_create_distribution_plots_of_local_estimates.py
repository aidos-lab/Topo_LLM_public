"""Create plots of the local estimates and compare with other task performance measures."""

import logging
import pathlib
import pprint
from typing import TYPE_CHECKING

import hydra
import omegaconf
from tqdm import tqdm

from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.data_processing.iteration_over_directories.load_np_arrays_from_folder_structure_into_list_of_dicts import (
    load_np_arrays_from_folder_structure_into_list_of_dicts,
)
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.path_management.embeddings.protocol import EmbeddingsPathManager
from topollm.plotting.plot_size_config import (
    PlotSizeConfigFlat,
)
from topollm.task_performance_analysis.plotting.create_distribution_plots_over_model_checkpoints import (
    create_distribution_plots_over_model_checkpoints,
)
from topollm.task_performance_analysis.plotting.create_distribution_plots_over_model_layers import (
    create_distribution_plots_over_model_layers,
)
from topollm.task_performance_analysis.plotting.create_mean_plots_over_model_checkpoints_with_different_seeds import (
    create_mean_plots_over_model_checkpoints_with_different_seeds,
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

    # The following directory contains the precomputed local estimates.
    # Logging of the directory is done in the function which iterates over the directories.
    iteration_root_dir = pathlib.Path(
        embeddings_path_manager.get_local_estimates_root_dir_absolute_path(),
    )

    patterns_to_iterate_over: list[str] = (
        main_config.analysis.task_performance_analysis.plotting.patterns_to_iterate_over
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{iteration_root_dir = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{len(patterns_to_iterate_over) = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"patterns_to_iterate_over:\n{pprint.pformat(patterns_to_iterate_over)}",  # noqa: G004 - low overhead
        )

    for pattern in tqdm(
        iterable=patterns_to_iterate_over,
        desc="Iterate over patterns",
    ):
        create_plots_for_given_pattern(
            iteration_root_dir=iteration_root_dir,
            pattern=pattern,
            embeddings_path_manager=embeddings_path_manager,
            do_create_distribution_plots_over_model_checkpoints=main_config.feature_flags.task_performance_analysis.plotting_create_distribution_plots_over_model_checkpoints,
            do_create_mean_plots_over_model_checkpoints_with_different_seeds=main_config.feature_flags.task_performance_analysis.plotting_create_mean_plots_over_model_checkpoints_with_different_seeds,
            do_create_distribution_plots_over_model_layers=main_config.feature_flags.task_performance_analysis.plotting_create_distribution_plots_over_model_layers,
            restrict_to_model_seeds=main_config.analysis.task_performance_analysis.plotting.restrict_to_model_seeds,
            maximum_x_value=main_config.analysis.task_performance_analysis.plotting.maximum_x_value,
            publication_ready=main_config.analysis.task_performance_analysis.plotting.publication_ready,
            add_legend=main_config.analysis.task_performance_analysis.plotting.add_legend,
            verbosity=verbosity,
            logger=logger,
        )


def create_plots_for_given_pattern(
    iteration_root_dir: pathlib.Path,
    pattern: str,
    embeddings_path_manager: EmbeddingsPathManager,
    *,
    do_create_distribution_plots_over_model_checkpoints: bool = True,
    do_create_mean_plots_over_model_checkpoints_with_different_seeds: bool = True,
    do_create_distribution_plots_over_model_layers: bool = True,
    restrict_to_model_seeds: list[int] | None = None,
    maximum_x_value: int | None = None,
    publication_ready: bool = False,
    add_legend: bool = True,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Create plots for a given pattern."""
    loaded_data: list[dict] = load_np_arrays_from_folder_structure_into_list_of_dicts(
        iteration_root_dir=iteration_root_dir,
        pattern=pattern,
        verbosity=verbosity,
        logger=logger,
    )

    # # # #
    # Choose which comparisons to make
    array_key_name: str = "file_data"

    output_root_dir: pathlib.Path = (
        embeddings_path_manager.get_saved_plots_distribution_of_local_estimates_dir_absolute_path()
    )
    output_root_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{output_root_dir = }",  # noqa: G004 - low overhead
        )

    # ================================================== #
    # Create plots
    # ================================================== #

    # # # #
    # Common parameters for all plots
    common_output_pdf_width: int = 650
    common_output_pdf_height: int = 300
    plot_size_configs_list: list[PlotSizeConfigFlat] = [
        PlotSizeConfigFlat(
            x_min=None,
            x_max=None,
            y_min=None,
            y_max=None,
            output_pdf_width=common_output_pdf_width,
            output_pdf_height=common_output_pdf_height,
        ),
        PlotSizeConfigFlat(
            x_min=None,
            x_max=None,
            y_min=0.0,
            y_max=10.5,
            output_pdf_width=common_output_pdf_width,
            output_pdf_height=common_output_pdf_height,
        ),
        PlotSizeConfigFlat(
            x_min=None,
            x_max=None,
            y_min=4.0,
            y_max=11.5,
            output_pdf_width=common_output_pdf_width,
            output_pdf_height=common_output_pdf_height,
        ),
        PlotSizeConfigFlat(
            x_min=None,
            x_max=None,
            y_min=5.0,
            y_max=9.0,
            output_pdf_width=common_output_pdf_width,
            output_pdf_height=common_output_pdf_height,
        ),
        PlotSizeConfigFlat(
            x_min=None,
            x_max=None,
            y_min=7.0,
            y_max=10.25,
            output_pdf_width=common_output_pdf_width,
            output_pdf_height=common_output_pdf_height,
        ),
        PlotSizeConfigFlat(
            x_min=None,
            x_max=None,
            y_min=1.0,
            y_max=13.5,
            output_pdf_width=common_output_pdf_width,
            output_pdf_height=common_output_pdf_height,
        ),
    ]

    # # # #
    # Create plots which show the distribution of the local estimates over the checkpoints
    if do_create_distribution_plots_over_model_checkpoints:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Creating the distribution plots over the model checkpoints ...",
            )
        create_distribution_plots_over_model_checkpoints(
            loaded_data=loaded_data,
            array_key_name=array_key_name,
            output_root_dir=output_root_dir,
            plot_size_configs_list=plot_size_configs_list,
            verbosity=verbosity,
            logger=logger,
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Creating the distribution plots over the model checkpoints DONE",
            )
    else:
        logger.info(
            msg="Skipping the creation of the distribution plots over the model checkpoints.",
        )

    # # # #
    # Create plots which show the mean of the local estimates over the checkpoints
    if do_create_mean_plots_over_model_checkpoints_with_different_seeds:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Creating the mean plots over the model checkpoints ...",
            )

        create_mean_plots_over_model_checkpoints_with_different_seeds(
            loaded_data=loaded_data,
            array_key_name=array_key_name,
            output_root_dir=output_root_dir,
            plot_size_configs_list=plot_size_configs_list,
            embeddings_path_manager=embeddings_path_manager,
            restrict_to_model_seeds=restrict_to_model_seeds,
            maximum_x_value=maximum_x_value,
            save_plot_raw_data=True,
            publication_ready=publication_ready,
            add_legend=add_legend,
            verbosity=verbosity,
            logger=logger,
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Creating the mean plots over the model checkpoints DONE",
            )

    # # # #
    # Create plots which show the distribution of the local estimates over different layers of the model
    if do_create_distribution_plots_over_model_layers:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Creating the distribution plots over the model layers ...",
            )
        create_distribution_plots_over_model_layers(
            loaded_data=loaded_data,
            array_key_name=array_key_name,
            output_root_dir=output_root_dir,
            plot_size_configs_list=plot_size_configs_list,
            verbosity=verbosity,
            logger=logger,
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Creating the distribution plots over the model layers DONE",
            )
    else:
        logger.info(
            msg="Skipping the creation of the distribution plots over the model layers.",
        )

    logger.info(
        msg="Script finished.",
    )


if __name__ == "__main__":
    main()
