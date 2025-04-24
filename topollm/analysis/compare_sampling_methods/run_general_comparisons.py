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

"""Run script to load computed estimates from disk and run various comparison analysis steps."""

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

from topollm.analysis.compare_sampling_methods.analysis_modes.checkpoint_analysis_modes import (
    CheckpointAnalysisModes,
)
from topollm.analysis.compare_sampling_methods.analysis_modes.noise_analysis_modes import (
    NoiseAnalysisCombination,
    NoiseAnalysisModes,
)
from topollm.analysis.compare_sampling_methods.checkpoint_analysis.model_checkpoint_analysis import (
    run_checkpoint_analysis_over_different_data_and_models,
)
from topollm.analysis.compare_sampling_methods.checkpoint_analysis.model_loss_extractor import ModelLossExtractor
from topollm.analysis.compare_sampling_methods.extract_results_from_directory_structure import (
    run_search_on_single_base_directory_and_process_and_save,
)
from topollm.analysis.compare_sampling_methods.filter_dataframe_based_on_filters_dict import (
    filter_dataframe_based_on_filters_dict,
)
from topollm.analysis.compare_sampling_methods.load_and_concatenate_saved_dataframes import (
    load_and_concatenate_saved_dataframes,
)
from topollm.analysis.compare_sampling_methods.make_plots import (
    PlotProperties,
    scatterplot_individual_seed_combinations_and_combined,
)
from topollm.analysis.compare_sampling_methods.organize_results_directory_structure import (
    build_results_directory_structure,
)
from topollm.analysis.compare_sampling_methods.sensitivity_to_parameter_choices.data_subsampling_number_of_samples_analysis import (
    run_data_subsampling_number_of_samples_analysis,
)
from topollm.config_classes.constants import (
    HYDRA_CONFIGS_BASE_PATH,
    NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS,
    TOPO_LLM_REPOSITORY_BASE_PATH,
)
from topollm.config_classes.setup_omega_conf import setup_omega_conf
from topollm.data_processing.dictionary_handling import generate_fixed_parameters_text_from_dict
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.log_dataframe_info import log_dataframe_info
from topollm.logging.log_list_info import log_list_info
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig
    from topollm.path_management.embeddings.protocol import EmbeddingsPathManager


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

    array_truncation_size: int = 60_000

    # END Global settings for analysis
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
    # Noise analysis
    # ================================================== #

    if main_config.feature_flags.analysis.compare_sampling_methods.do_noise_analysis:
        do_noise_analysis(
            concatenated_df=concatenated_df,
            verbosity=verbosity,
            logger=logger,
        )

    # ================================================== #
    # Checkpoint analysis
    # ================================================== #

    if main_config.feature_flags.analysis.compare_sampling_methods.do_checkpoint_analysis:
        do_checkpoint_analysis(
            concatenated_df=concatenated_df,
            verbosity=verbosity,
            logger=logger,
        )

    # ================================================== #
    # Data subsampling number of samples analysis
    # ================================================== #

    if main_config.feature_flags.analysis.compare_sampling_methods.do_data_subsampling_number_of_samples_analysis:
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


def do_noise_analysis(
    concatenated_df: pd.DataFrame,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Run the noise analysis."""
    if concatenated_df.empty:
        logger.critical(
            msg="@@@ The concatenated_df is empty.\n"
            "@@@ The noise analysis will not yield useful results.\n"
            "@@@ Exiting the function without creating any analysis files.",
        )
        return

    # # # #
    # Select which analysis to run

    # Create the analysis modes configuration
    analysis_modes = NoiseAnalysisModes()
    analysis_modes.from_concatenated_df(
        concatenated_df=concatenated_df,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"analysis_modes:\n{pprint.pformat(object=analysis_modes)}",  # noqa: G004 - low overhead
        )

    product_to_process: list[NoiseAnalysisCombination] = analysis_modes.all_combinations()

    # This column is used to describe the strength of the artificial noise which was added to the local estimates,
    # usually, it should be equal to "local_estimates_noise_distortion"
    noise_strength_column_name = NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS["local_estimates_distor"]

    for comb in tqdm(
        iterable=product_to_process,
        desc="Processing different combinations of data subsamples and models",
        total=len(product_to_process),
    ):
        concatenated_filters_dict: dict = {
            **analysis_modes.common_filters_dict,
            "data_full": comb.data_full,
            "data_subsampling_split": comb.data_subsampling_split,
            "data_subsampling_sampling_mode": comb.data_subsampling_sampling_mode,
            "model_full": comb.model_full,
            "embedding_data_handler_mode": comb.embedding_data_handler_mode,
        }

        common_prefix_path = pathlib.Path(
            TOPO_LLM_REPOSITORY_BASE_PATH,
            "data",
            "saved_plots",
            "artificial_noise_analysis",
            f"{comb.data_full=}",
            f"{comb.data_subsampling_split=}",
            f"{comb.data_subsampling_sampling_mode=}",
            f"{comb.embedding_data_handler_mode=}",
            f"{comb.model_full=}",
        )

        filtered_concatenated_df: pd.DataFrame = filter_dataframe_based_on_filters_dict(
            df=concatenated_df,
            filters_dict=concatenated_filters_dict,
            verbosity=verbosity,
            logger=logger,
        )

        # This string will be used in the plots
        fixed_params_text: str = generate_fixed_parameters_text_from_dict(
            filters_dict=concatenated_filters_dict,
        )

        if filtered_concatenated_df.empty:
            logger.info(
                msg=f"This combination of filters yielded an empty dataframe: {concatenated_filters_dict = }",  # noqa: G004 - low overhead
            )
            logger.info(
                msg="Skipping this combination of filters ...",
            )
            continue

        # # # #
        # If the value in the 'local_estimates_noise_artificial_noise_mode' column is 'do_nothing',
        # the 'noise_strength_column_name' column should be filled with 0.

        filtered_concatenated_df.loc[
            filtered_concatenated_df["local_estimates_noise_artificial_noise_mode"] == "do_nothing",
            noise_strength_column_name,
        ] = 0.0

        # # # #
        # Cast columns to the correct data types.
        # Note that we can only cast columns where all values are set
        # (so we cannot cast the noise seed column, as it is not set for all rows).

        # Columns that should be of type float
        columns_with_float_type: list[str] = [
            # All rows have been assigned a value in the 'local_estimates_noise_distortion' column above,
            # so we can safely cast it to float.
            noise_strength_column_name,
        ]

        for column_name in columns_with_float_type:
            if column_name in concatenated_df.columns:
                # Check that there are no missing values in the column
                if concatenated_df[column_name].isna().sum() != 0:
                    logger.warning(
                        msg=f"The column '{column_name}' contains missing values. "  # noqa: G004 - low overhead
                        f"Casting to float might lead to errors.",
                    )

                filtered_concatenated_df[column_name] = filtered_concatenated_df[column_name].astype(
                    dtype=float,
                )

        # # # #
        # Save the raw data
        raw_data_path = pathlib.Path(
            common_prefix_path,
            "raw_data",
            "raw_data.csv",
        )
        raw_data_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        filtered_concatenated_df.to_csv(
            path_or_buf=raw_data_path,
        )

        data_df_to_analyze: pd.DataFrame = filtered_concatenated_df.copy()

        # Select which values to analyse.
        # We use the non-truncated array_data_mean and array_data_std values in the noise analysis.
        y_column_names_to_analyze: list[str] = [
            "array_data_mean",
            "array_data_std",
        ]

        for y_column_name in y_column_names_to_analyze:
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"y_column_name = {y_column_name}",  # noqa: G004 - low overhead
                )

            selected_analysis_partial_path = pathlib.Path(
                common_prefix_path,
                f"{y_column_name=}",
            )

            # # # #
            # Create aggregated data by grouping the data by the noise strength.
            #
            # Note: To avoid problems in the grouping by the noise strength column (we encountered duplicate lines),
            # make sure that the values are cast to float.
            # Otherwise, 0.01 might erroneously be treated as the string "0.01".

            # Log unique values in the "local_estimates_noise_distortion" column with full precision
            unique_values = data_df_to_analyze[noise_strength_column_name].unique()
            unique_values_sorted = sorted(
                unique_values,
            )
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"{unique_values_sorted = }",  # noqa: G004 - low overhead
                )
                logger.info(
                    msg=f"{len(unique_values_sorted) = }",  # noqa: G004 - low overhead
                )

            # Check that the values are all floats, so that we can avoid problems in the grouping
            if not all(isinstance(value, float) for value in unique_values_sorted):
                logger.warning(
                    msg="The values in the 'local_estimates_noise_distortion' column are not all floats. "
                    "This might lead to problems in the grouping by the noise strength column.",
                )

            grouped_stats: pd.DataFrame = (
                data_df_to_analyze.groupby(
                    by=noise_strength_column_name,
                    observed=True,
                )[y_column_name]
                .agg(
                    func=[
                        "mean",
                        "std",
                        "count",
                    ],
                )
                .reset_index()
            )
            if verbosity >= Verbosity.NORMAL:
                log_dataframe_info(
                    df=grouped_stats,
                    df_name="grouped_stats",
                    logger=logger,
                )

            # Save the aggregated data
            aggregated_data_path = pathlib.Path(
                selected_analysis_partial_path,
                "aggregated_data",
                "aggregated_data.csv",
            )
            aggregated_data_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            grouped_stats.to_csv(
                path_or_buf=aggregated_data_path,
            )

            # # # #
            # Create plots
            plot_properties_list: list[PlotProperties] = [
                PlotProperties(
                    y_min=0.0,
                    y_max=16.0,
                ),
                PlotProperties(
                    y_min=None,
                    y_max=None,
                ),
            ]

            for plot_properties in plot_properties_list:
                output_dir = pathlib.Path(
                    selected_analysis_partial_path,
                    f"{plot_properties.y_min=}_{plot_properties.y_max=}",
                )

                scatterplot_individual_seed_combinations_and_combined(
                    data=data_df_to_analyze,
                    output_dir=output_dir,
                    plot_properties=plot_properties,
                    y_column_name=y_column_name,
                    x_column_name=noise_strength_column_name,
                    fixed_params_text=fixed_params_text,
                    verbosity=verbosity,
                    logger=logger,
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
            "@@@ The checkpoint analysis will not yield useful results.\n"
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
