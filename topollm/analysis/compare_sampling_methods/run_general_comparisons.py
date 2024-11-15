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

"""Run script to create embedding vectors from dataset based on config."""

import logging
import os
import pathlib
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import hydra
import hydra.core.hydra_config
import numpy as np
import omegaconf
import pandas as pd
from tqdm import tqdm

from topollm.analysis.compare_sampling_methods.analysis_influence_of_local_estimates_n_neighbors import (
    analysis_influence_of_local_estimates_n_neighbors,
)
from topollm.analysis.compare_sampling_methods.make_plots import (
    Y_AXIS_LIMITS,
    create_boxplot_of_mean_over_different_sampling_seeds,
    generate_fixed_params_text,
)
from topollm.analysis.compare_sampling_methods.organize_results_directory_structure import (
    build_results_directory_structure,
)
from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.config_classes.setup_OmegaConf import setup_omega_conf
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.log_dataframe_info import log_dataframe_info
from topollm.logging.log_list_info import log_list_info
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.path_management.parse_path_info import parse_path_info_full
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

    if verbosity >= Verbosity.NORMAL:
        log_list_info(
            list_=all_partial_search_base_directories_paths,
            list_name="all_partial_search_base_directories_paths",
            logger=logger,
        )
        logger.info(
            msg=f"Iterating over {len(all_partial_search_base_directories_paths) = } paths ...",  # noqa: G004 - low overhead
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

    for partial_search_base_directory_path in tqdm(
        iterable=all_partial_search_base_directories_paths,
        desc="Processing paths",
    ):
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
            verbosity=verbosity,
            logger=logger,
        )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Iterating over {len(all_partial_search_base_directories_paths) = } paths DONE",  # noqa: G004 - low overhead
        )

    concatenated_df: pd.DataFrame = load_and_concatenate_saved_dataframes(
        root_dir=analysis_output_subdirectory_absolute_path,
        save_path=pathlib.Path(
            analysis_output_subdirectory_absolute_path,
            "concatenated_full_local_estimates_df.csv",
        ),
        verbosity=verbosity,
        logger=logger,
    )

    # ================================================== #
    # Note: You can add additional analysis steps here
    # ================================================== #

    logger.info(
        msg="Running script DONE",
    )


def collect_all_data_and_model_combination_paths(
    base_path: pathlib.Path,
    pattern: str = "data=*/split=*/lvl=*/add-prefix-space=True_max-len=512/model=*",
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


def load_and_concatenate_saved_dataframes(
    root_dir: pathlib.Path,
    pattern: str = "full_local_estimates_df.csv",
    save_path: pathlib.Path | None = None,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> pd.DataFrame:
    """Load and concatenate saved dataframes from the specified directory."""
    # Initialize an empty list to store dataframes
    dfs = []

    # Traverse the directory structure using pathlib's rglob
    for file_path in root_dir.rglob(
        pattern=pattern,
    ):
        # Load the CSV file into a dataframe
        current_df = None

        try:
            current_df = pd.read_csv(
                filepath_or_buffer=file_path,
                keep_default_na=False,
            )
            dfs.append(
                current_df,
            )
        except FileNotFoundError as e:
            logger.exception(
                msg=f"Error reading {file_path = }: {e}",  # noqa: G004 - low overhead
            )
            logger.warning(
                msg=f"Skipping {file_path = }",  # noqa: G004 - low overhead
            )

        # Append the dataframe to the list
        dfs.append(
            current_df,
        )

    # Concatenate the dataframes
    if dfs:
        concatenated_df: pd.DataFrame = pd.concat(
            objs=dfs,
            ignore_index=True,
        )
    else:
        logger.info(
            msg=f"No files found with pattern {pattern = } in {root_dir = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg="Returning empty dataframe.",
        )
        concatenated_df = pd.DataFrame()  # Empty dataframe if no files found

    # Save the concatenated dataframe
    if save_path is not None:
        logger.info(
            msg=f"Saving concatenated dataframe to {save_path = } ...",  # noqa: G004 - low overhead
        )
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        concatenated_df.to_csv(
            path_or_buf=save_path,
            index=False,
        )
        logger.info(
            msg=f"Saving concatenated dataframe to {save_path = } DONE",  # noqa: G004 - low overhead
        )

    if verbosity >= Verbosity.NORMAL and "model_partial_name" in concatenated_df.columns:
        logger.info(
            msg=f"{concatenated_df['model_partial_name'].unique() = }",  # noqa: G004 - low overhead
        )

    return concatenated_df


def run_search_on_single_base_directory_and_process_and_save(
    search_base_directory: pathlib.Path,
    results_directory: pathlib.Path,
    array_data_column_name: str = "array_data",
    array_truncation_size: int = 5000,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Run the search and analysis on a single base directory."""
    _, full_local_estimates_df = extract_and_preprocess_dataframes(
        search_base_directory=search_base_directory,
        results_directory=results_directory,
        filenames_to_match=None,
        array_truncation_size=array_truncation_size,
        array_data_column_name=array_data_column_name,
        verbosity=verbosity,
        logger=logger,
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # ===== This is the start of the analysis code =====

    analysis_influence_of_local_estimates_n_neighbors(
        full_local_estimates_df=full_local_estimates_df,
        array_data_column_name=array_data_column_name,
        results_directory=results_directory,
        verbosity=verbosity,
        logger=logger,
    )

    # Select a subset of the data with the same parameters.
    # This allows comparing over different seeds.
    #
    # We do not fix the local_estimates_samples,
    # since we want to compare the results for different sample sizes.
    filters_dict_list = [
        {
            "data_prep_sampling_method": "random",
            "deduplication": "array_deduplicator",
            "n_neighbors": 128,
            "data_prep_sampling_samples": 50000,
        },
        {
            "data_prep_sampling_method": "random",
            "deduplication": "array_deduplicator",
            "n_neighbors": 128,
            "data_prep_sampling_samples": 100000,
        },
        {
            "data_prep_sampling_method": "random",
            "deduplication": "array_deduplicator",
            "n_neighbors": 128,
            "data_prep_sampling_samples": 150000,
        },
        {
            "data_prep_sampling_method": "random",
            "deduplication": "array_deduplicator",
            "n_neighbors": 256,
            "data_prep_sampling_samples": 100000,
        },
    ]

    for filters_dict in filters_dict_list:
        subset_local_estimates_df: pd.DataFrame = filter_dataframe_based_on_filters_dict(
            df=full_local_estimates_df,
            filters_dict=filters_dict,
        )

        fixed_params_text: str = generate_fixed_params_text(
            filters_dict=filters_dict,
        )

        for connect_points in [
            True,
            False,
        ]:
            for y_min, y_max in Y_AXIS_LIMITS.values():
                common_prefix_path = pathlib.Path(
                    results_directory,
                    "different_data_prep_sampling_seeds",
                    f"{filters_dict['data_prep_sampling_samples']=}",
                    f"{filters_dict['n_neighbors']=}",
                    "array_data_truncated_mean_boxplot",
                )

                plot_save_path = pathlib.Path(
                    common_prefix_path,
                    "plots",
                    f"y_{y_min}_{y_max}_{connect_points=}.pdf",
                )
                raw_data_save_path = pathlib.Path(
                    common_prefix_path,
                    "raw_data",
                    "raw_data.csv",
                )
                aggregated_results_save_path = pathlib.Path(
                    common_prefix_path,
                    "raw_data",
                    "aggregated_results.csv",
                )

                create_boxplot_of_mean_over_different_sampling_seeds(
                    subset_local_estimates_df=subset_local_estimates_df,
                    plot_save_path=plot_save_path,
                    raw_data_save_path=raw_data_save_path,
                    aggregated_results_save_path=aggregated_results_save_path,
                    fixed_params_text=fixed_params_text,
                    y_min=y_min,
                    y_max=y_max,
                    show_plot=False,
                    connect_points=connect_points,
                    logger=logger,
                )

    pass  # noqa: PIE790 - This is here for setting a breakpoint


def extract_and_preprocess_dataframes(
    search_base_directory: pathlib.Path,
    results_directory: pathlib.Path,
    filenames_to_match: list[str] | None = None,
    array_truncation_size: int = 2_500,
    array_data_column_name: str = "array_data",
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract and preprocess the dataframes from the search base directory."""
    full_loaded_data_df: pd.DataFrame = walk_through_subdirectories_and_load_arrays(
        root_directory=search_base_directory,
        filenames_to_match=filenames_to_match,
        array_data_column_name=array_data_column_name,
        verbosity=verbosity,
        logger=logger,
    )
    if verbosity >= Verbosity.NORMAL:
        log_dataframe_info(
            df=full_loaded_data_df,
            df_name="full_loaded_data_df",
            logger=logger,
        )

    # Apply the updated parsing function to each row in the DataFrame and create new columns
    parsed_columns_full: pd.DataFrame = (
        full_loaded_data_df["path"]
        .apply(
            func=parse_path_info_full,
        )
        .apply(
            func=pd.Series,
        )
    )
    # Note that since `parsed_columns_full` also contains the global estimates,
    # we do not have `n_neighbors` in every row of the dataframe.

    # Concatenate the original DataFrame with the newly parsed columns
    full_loaded_data_df: pd.DataFrame = pd.concat(
        objs=[
            full_loaded_data_df,
            parsed_columns_full,
        ],
        axis=1,
    )

    full_local_estimates_df: pd.DataFrame = extract_and_prepare_local_estimates_data(
        loaded_data_df=full_loaded_data_df,
        array_truncation_size=array_truncation_size,
        verbosity=verbosity,
        logger=logger,
    )

    # If the column 'n_neighbors' exists, make sure it is typed as an integer
    if "n_neighbors" in full_local_estimates_df.columns:
        full_local_estimates_df["n_neighbors"] = full_local_estimates_df["n_neighbors"].astype(
            dtype=int,
        )

    if verbosity >= Verbosity.NORMAL:
        log_dataframe_info(
            df=full_local_estimates_df,
            df_name="full_local_estimates_df",
            logger=logger,
        )

    # Save the collected dataframe to disk
    full_local_estimates_df_save_path: pathlib.Path = pathlib.Path(
        results_directory,
        "full_local_estimates_df.csv",
    )
    full_local_estimates_df.to_csv(
        path_or_buf=full_local_estimates_df_save_path,
    )

    return full_loaded_data_df, full_local_estimates_df


def filter_dataframe_based_on_filters_dict(
    df: pd.DataFrame,
    filters_dict: dict[str, Any],
) -> pd.DataFrame:
    """Filter a DataFrame based on key-value pairs specified in a dictionary.

    Args:
        df:
            The DataFrame to be filtered.
        filters_dict:
            A dictionary of column names and corresponding values to filter by.

    Returns:
        A filtered DataFrame with rows matching all key-value pairs.

    """
    subset_df: pd.DataFrame = df.copy()
    for column, value in filters_dict.items():
        subset_df = subset_df[subset_df[column] == value]
    return subset_df


def extract_and_prepare_local_estimates_data(
    loaded_data_df: pd.DataFrame,
    array_truncation_size: int = 2_500,
    array_data_column_name: str = "array_data",
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> pd.DataFrame:
    """Extract and prepare the local estimates data from the loaded DataFrame."""
    array_name_to_match: str = "local_estimates_pointwise.npy"

    # Filter the DataFrame to only contain the local estimates
    local_estimates_df: pd.DataFrame = loaded_data_df[loaded_data_df["array_name"] == array_name_to_match]
    # Make a copy of the DataFrame to avoid SettingWithCopyWarning
    local_estimates_df = local_estimates_df.copy()

    # Add a column with the number of elements in the array
    local_estimates_df[f"{array_data_column_name}.size"] = local_estimates_df[array_data_column_name].apply(
        func=lambda array: array.size,
    )

    # Add a column with truncated arrays
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Truncating arrays to {array_truncation_size = }",  # noqa: G004 - low overhead
        )
    local_estimates_df[f"{array_data_column_name}_truncated"] = local_estimates_df[array_data_column_name].apply(
        func=lambda array: array[:array_truncation_size],
    )

    # Add a column with the mean and standard deviation of the arrays
    local_estimates_df[f"{array_data_column_name}_mean"] = local_estimates_df[array_data_column_name].apply(
        func=lambda array: array.mean(),
    )
    local_estimates_df[f"{array_data_column_name}_std"] = local_estimates_df[array_data_column_name].apply(
        func=lambda array: array.std(),
    )
    local_estimates_df[f"{array_data_column_name}_truncated_mean"] = local_estimates_df[
        f"{array_data_column_name}_truncated"
    ].apply(
        func=lambda array: array.mean(),
    )
    local_estimates_df[f"{array_data_column_name}_truncated_std"] = local_estimates_df[
        f"{array_data_column_name}_truncated"
    ].apply(
        func=lambda array: array.std(),
    )

    return local_estimates_df


def walk_through_subdirectories_and_load_arrays(
    root_directory: pathlib.Path,
    filenames_to_match: list[str] | None = None,
    array_data_column_name: str = "array_data",
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> pd.DataFrame:
    """Walk through the subdirectories of the root directory and load the arrays from the files."""
    if filenames_to_match is None:
        filenames_to_match = [
            "global_estimate.npy",
            "local_estimates_pointwise.npy",
        ]

    # List to store metadata and loaded arrays
    data = []

    # Walk through the directory structure
    for dirpath, _, filenames in os.walk(
        top=root_directory,
    ):
        for filename in filenames:
            if filename in filenames_to_match:
                # Construct the full path to the file
                file_path = pathlib.Path(
                    dirpath,
                    filename,
                )
                if verbosity >= Verbosity.NORMAL:
                    logger.info(
                        msg=f"Found {file_path = }",  # noqa: G004 - low overhead
                    )

                # Load the numpy array from the file
                try:
                    array = np.load(
                        file=file_path,
                    )
                except Exception as e:
                    logger.exception(
                        msg=f"Failed to load {file_path = }:\n{e}",  # noqa: G004 - low overhead
                    )
                    continue

                # Append metadata and array to the list
                data.append(
                    {
                        "path": file_path,
                        "array_name": filename,
                        array_data_column_name: array,
                    },
                )

    # Create a DataFrame from the collected data
    loaded_data_df = pd.DataFrame(
        data=data,
    )

    return loaded_data_df


if __name__ == "__main__":
    main()
