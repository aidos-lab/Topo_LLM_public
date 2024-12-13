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

"""Run script to create embedding vectors from dataset based on config."""

import logging
import pathlib
import pprint
from collections.abc import Generator
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import joblib
import omegaconf
import pandas as pd
from tqdm import tqdm

from topollm.analysis.compare_sampling_methods.checkpoint_analysis.model_checkpoint_analysis import (
    run_checkpoint_analysis_over_different_data_and_models,
)
from topollm.analysis.compare_sampling_methods.checkpoint_analysis.model_loss_extractor import ModelLossExtractor
from topollm.analysis.compare_sampling_methods.checkpoint_analysis_modes import CheckpointAnalysisModes
from topollm.analysis.compare_sampling_methods.extract_results_from_directory_structure import (
    run_search_on_single_base_directory_and_process_and_save,
)
from topollm.analysis.compare_sampling_methods.load_and_concatenate_saved_dataframes import (
    load_and_concatenate_saved_dataframes,
)
from topollm.analysis.compare_sampling_methods.organize_results_directory_structure import (
    build_results_directory_structure,
)
from topollm.analysis.compare_sampling_methods.sensitivity_to_parameter_choices.data_subsampling_number_of_samples_analysis import (
    run_data_subsampling_number_of_samples_analysis,
)
from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH, TOPO_LLM_REPOSITORY_BASE_PATH
from topollm.config_classes.setup_OmegaConf import setup_omega_conf
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.log_list_info import log_list_info
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.typing.enums import Verbosity

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
    # START Global settings for analysis

    array_truncation_size: int = 5_000

    # END Global settingsn for analysis
    # # # # # # # # # # # # # # # # # # # # #

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=logger,
    )
    verbosity: Verbosity = main_config.verbosity

    embeddings_path_manager: EmbeddingsPathManager = get_embeddings_path_manager(
        main_config=main_config,
        logger=logger,
    )
    data_dir: pathlib.Path = embeddings_path_manager.data_dir

    data_to_analyse_base_path: pathlib.Path = pathlib.Path(
        data_dir,
        "analysis",
        "twonn",
    )
    all_partial_search_base_directories_paths = list(
        collect_all_data_and_model_combination_paths(
            base_path=data_to_analyse_base_path,
        ),
    )

    analysis_output_subdirectory_partial_relative_path = pathlib.Path(
        "sample_sizes",
        "run_general_comparisons",
        f"{array_truncation_size=}",
    )
    analysis_output_subdirectory_absolute_path = pathlib.Path(
        data_dir,
        "analysis",
        analysis_output_subdirectory_partial_relative_path,
    )

    def process_partial_search_base_directory_partial_function(
        partial_search_base_directory_path: pathlib.Path,
    ) -> None:
        process_partial_search_base_directory(
            partial_search_base_directory_path=partial_search_base_directory_path,
            analysis_output_subdirectory_partial_relative_path=analysis_output_subdirectory_partial_relative_path,
            data_dir=data_dir,
            array_truncation_size=array_truncation_size,
            do_analysis_influence_of_local_estimates_n_neighbors=main_config.feature_flags.analysis.compare_sampling_methods.do_analysis_influence_of_local_estimates_n_neighbors,
            do_create_boxplot_of_mean_over_different_sampling_seeds=main_config.feature_flags.analysis.compare_sampling_methods.do_create_boxplot_of_mean_over_different_sampling_seeds,
            verbosity=verbosity,
            logger=logger,
        )

    # ================================================== #
    # Iterate over all partial search base directories and process them.
    #
    # Note: This step might take a while, depending on the number of directories.
    # ================================================== #

    if main_config.feature_flags.analysis.compare_sampling_methods.do_iterate_all_partial_search_base_directories:
        if verbosity >= Verbosity.NORMAL:
            log_list_info(
                list_=all_partial_search_base_directories_paths,
                list_name="all_partial_search_base_directories_paths",
                logger=logger,
            )
            logger.info(
                msg=f"Iterating over {len(all_partial_search_base_directories_paths) = } paths ...",  # noqa: G004 - low overhead
            )

        # The list application is used to evaluate the generator
        _ = list(
            tqdm(
                # Note the new return_as argument here, which requires joblib >= 1.3:
                iterable=joblib.Parallel(
                    return_as="generator",
                    n_jobs=main_config.n_jobs,
                )(
                    joblib.delayed(function=process_partial_search_base_directory_partial_function)(
                        partial_search_base_directory_path=partial_search_base_directory_path,
                    )
                    for partial_search_base_directory_path in all_partial_search_base_directories_paths
                ),
                total=len(all_partial_search_base_directories_paths),
                desc="Processing paths",
            ),
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Iterating over {len(all_partial_search_base_directories_paths) = } paths DONE",  # noqa: G004 - low overhead
            )

    # ================================================== #
    # Concatenate the extracted dataframes
    # ================================================== #

    concatenated_df: pd.DataFrame = load_and_concatenate_saved_dataframes(
        root_dir=analysis_output_subdirectory_absolute_path,
        save_path=pathlib.Path(
            analysis_output_subdirectory_absolute_path,
            "concatenated_full_local_estimates_df.csv",
        ),
        verbosity=verbosity,
        logger=logger,
    )

    # Log a warning of concatenated_df is empty
    if concatenated_df.empty:
        logger.warning(
            msg="The concatenated_df is empty. The subsequent analysis steps will not yield any results.",
        )

    # ================================================== #
    # Checkpoint analysis
    # ================================================== #

    do_checkpoint_analysis(
        concatenated_df=concatenated_df,
        verbosity=verbosity,
        logger=logger,
    )

    # ================================================== #
    # Data subsampling number of samples analysis
    # ================================================== #

    do_data_subsampling_number_of_samples_analysis(
        concatenated_df=concatenated_df,
        verbosity=verbosity,
        logger=logger,
    )

    # ================================================== #
    # Note: You can add additional analysis steps here
    # ================================================== #

    logger.info(
        msg="Running script DONE",
    )


def do_data_subsampling_number_of_samples_analysis(
    concatenated_df: pd.DataFrame,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Run the data subsampling number of samples analysis."""
    if concatenated_df.empty:
        logger.critical(
            msg="@@@ The concatenated_df is empty.\n"
            "@@@ The data subsampling number of samples analysis will not yield any results.\n"
            "@@@ Exiting the function without creating any analysis files.",
        )
        return

    data_full_list_to_process = list(concatenated_df["data_full"].unique())
    data_subsampling_split_list_to_process = list(concatenated_df["data_subsampling_split"].unique())
    data_subsampling_sampling_mode_list_to_process: list[str] = [
        "random",
    ]

    embedding_data_handler_mode_to_process: list[str] = concatenated_df["embedding_data_handler_mode"].unique().tolist()

    model_full_list_to_process: list[str] = [
        "model=roberta-base_task=masked_lm",
    ]

    run_data_subsampling_number_of_samples_analysis(
        concatenated_df=concatenated_df,
        data_full_list_to_process=data_full_list_to_process,
        data_subsampling_split_list_to_process=data_subsampling_split_list_to_process,
        data_subsampling_sampling_mode_list_to_process=data_subsampling_sampling_mode_list_to_process,
        embedding_data_handler_mode_to_process=embedding_data_handler_mode_to_process,
        model_full_list_to_process=model_full_list_to_process,
        verbosity=verbosity,
        logger=logger,
    )


def do_checkpoint_analysis(
    concatenated_df: pd.DataFrame,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Run the checkpoint analysis."""
    if concatenated_df.empty:
        logger.critical(
            msg="@@@ The concatenated_df is empty.\n"
            "@@@ The checkpoint analyiss will not yield useful results.\n"
            "@@@ We still continue executing the function, so that the model loss extractor can save potential values.",
        )

    # Get optional model loss extractor
    model_loss_extractor: ModelLossExtractor | None = create_model_loss_extractor()

    # # # #
    # Select which analysis to run

    # Create the analysis modes configuration
    checkpoint_analysis_modes = CheckpointAnalysisModes()
    checkpoint_analysis_modes.from_concatenated_df(
        concatenated_df=concatenated_df,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"checkpoint_analysis_modes:\n{pprint.pformat(object=checkpoint_analysis_modes)}",  # noqa: G004 - low overhead
        )

    # # # #
    # Run the checkpoint analysis on the selected data
    run_checkpoint_analysis_over_different_data_and_models(
        concatenated_df=concatenated_df,
        checkpoint_analysis_modes=checkpoint_analysis_modes,
        model_loss_extractor=model_loss_extractor,
        verbosity=verbosity,
        logger=logger,
    )


def create_model_loss_extractor(
    logger: logging.Logger = default_logger,
) -> ModelLossExtractor | None:
    """Create a ModelLossExtractor instance."""
    # Try to initialize the class
    finetuning_monitoring_base_directory: pathlib.Path = pathlib.Path(
        TOPO_LLM_REPOSITORY_BASE_PATH,
        "data",
        "models",
        "finetuning_monitoring",
    )

    try:
        model_loss_extractor = ModelLossExtractor(
            train_loss_file_path=pathlib.Path(
                finetuning_monitoring_base_directory,
                "Topo_LLM_finetuning_from_submission_script_DEBUG",
                "wandb_export_2024-11-20T18_47_32.346+01_00_overfitted_models_50_epochs_train_loss.csv",
            ),
            eval_loss_file_path=pathlib.Path(
                finetuning_monitoring_base_directory,
                "Topo_LLM_finetuning_from_submission_script_DEBUG",
                "wandb_export_2024-11-20T19_02_59.541+01_00_overfitted_models_50_epochs_eval_loss.csv",
            ),
        )
    except FileNotFoundError as e:
        logger.exception(
            msg=e,
        )
        logger.info(
            msg="Returning model_loss_extractor=None.",
        )
        model_loss_extractor = None

    return model_loss_extractor


def process_partial_search_base_directory(
    partial_search_base_directory_path: pathlib.Path,
    analysis_output_subdirectory_partial_relative_path: pathlib.Path,
    data_dir: pathlib.Path,
    array_truncation_size: int,
    *,
    do_analysis_influence_of_local_estimates_n_neighbors: bool = True,
    do_create_boxplot_of_mean_over_different_sampling_seeds: bool = True,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Process a single partial search base directory."""
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{partial_search_base_directory_path = }",  # noqa: G004 - low overhead
        )

    search_base_directory: pathlib.Path = pathlib.Path(
        partial_search_base_directory_path,
        "layer=-1_agg=mean",
        "norm=None",
    )
    results_directory: pathlib.Path = build_results_directory_structure(
        analysis_base_directory=search_base_directory,
        data_dir=data_dir,
        analysis_output_subdirectory_partial_relative_path=analysis_output_subdirectory_partial_relative_path,
        verbosity=verbosity,
        logger=logger,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{search_base_directory = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"{results_directory = }",  # noqa: G004 - low overhead
        )

    run_search_on_single_base_directory_and_process_and_save(
        search_base_directory=search_base_directory,
        results_directory=results_directory,
        array_truncation_size=array_truncation_size,
        do_analysis_influence_of_local_estimates_n_neighbors=do_analysis_influence_of_local_estimates_n_neighbors,
        do_create_boxplot_of_mean_over_different_sampling_seeds=do_create_boxplot_of_mean_over_different_sampling_seeds,
        verbosity=verbosity,
        logger=logger,
    )


def collect_all_data_and_model_combination_paths(
    base_path: pathlib.Path,
    pattern: str = "data=*/split=*/edh-mode=*/add-prefix-space=True_max-len=512/model=*",
) -> Generator[
    pathlib.Path,
    None,
    None,
]:
    """Collect all full paths that match a specific nested folder structure.

    Args:
        base_path: The root directory to start the search.
        pattern: The pattern to match the required directory layout.

    Yields:
        Path: Full paths matching the required structure as Path objects.

    """
    # Define the structured path pattern to match the required directory layout

    # Yield each matching directory path
    for path in base_path.glob(
        pattern=pattern,
    ):
        if path.is_dir():
            yield path


if __name__ == "__main__":
    main()
