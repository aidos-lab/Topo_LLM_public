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

import logging
import os
import pathlib

import numpy as np
import pandas as pd

from topollm.analysis.compare_sampling_methods.filter_dataframe_based_on_filters_dict import (
    filter_dataframe_based_on_filters_dict,
)
from topollm.analysis.compare_sampling_methods.make_plots import (
    Y_AXIS_LIMITS,
    PlotSavePathCollection,
    create_boxplot_of_mean_over_different_sampling_seeds,
    generate_fixed_params_text,
)
from topollm.analysis.compare_sampling_methods.sensitivity_to_parameter_choices.analysis_influence_of_local_estimates_n_neighbors import (
    analysis_influence_of_local_estimates_n_neighbors,
)
from topollm.config_classes.constants import NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS
from topollm.logging.log_dataframe_info import log_dataframe_info
from topollm.path_management.parse_path_info import parse_path_info_full
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def extract_and_prepare_local_estimates_data(
    loaded_data_df: pd.DataFrame,
    array_truncation_size: int = 5_000,
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


def extract_and_preprocess_and_save_full_local_estimates_df_dataframes(
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


def run_search_on_single_base_directory_and_process_and_save(
    search_base_directory: pathlib.Path,
    results_directory: pathlib.Path,
    array_data_column_name: str = "array_data",
    array_truncation_size: int = 5000,
    *,
    do_analysis_influence_of_local_estimates_n_neighbors: bool = True,
    do_create_boxplot_of_mean_over_different_sampling_seeds: bool = True,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Run the search and analysis on a single base directory."""
    _, full_local_estimates_df = extract_and_preprocess_and_save_full_local_estimates_df_dataframes(
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

    if do_analysis_influence_of_local_estimates_n_neighbors:
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
    if do_create_boxplot_of_mean_over_different_sampling_seeds:
        filters_dict_list = [
            {
                "data_prep_sampling_method": "random",
                NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS["dedup"]: "array_deduplicator",
                "n_neighbors": 128,
                "data_prep_sampling_samples": 50_000,
            },
            {
                "data_prep_sampling_method": "random",
                NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS["dedup"]: "array_deduplicator",
                "n_neighbors": 128,
                "data_prep_sampling_samples": 100_000,
            },
            {
                "data_prep_sampling_method": "random",
                NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS["dedup"]: "array_deduplicator",
                "n_neighbors": 128,
                "data_prep_sampling_samples": 150_000,
            },
            {
                "data_prep_sampling_method": "random",
                NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS["dedup"]: "array_deduplicator",
                "n_neighbors": 256,
                "data_prep_sampling_samples": 100_000,
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

            common_prefix_path = pathlib.Path(
                results_directory,
                "different_data_prep_sampling_seeds",
                f"{filters_dict['data_prep_sampling_samples']=}",
                f"{filters_dict['n_neighbors']=}",
                "array_data_truncated_mean_boxplot",
            )

            for connect_points in [
                True,
                False,
            ]:
                for y_min, y_max in Y_AXIS_LIMITS.values():
                    plot_save_path_collection: PlotSavePathCollection = (
                        PlotSavePathCollection.create_from_common_prefix_path(
                            common_prefix_path=common_prefix_path,
                            plot_file_name=f"{y_min=}_{y_max=}_{connect_points=}.pdf",
                        )
                    )

                    create_boxplot_of_mean_over_different_sampling_seeds(
                        subset_local_estimates_df=subset_local_estimates_df,
                        plot_save_path_collection=plot_save_path_collection,
                        fixed_params_text=fixed_params_text,
                        y_min=y_min,
                        y_max=y_max,
                        show_plot=False,
                        connect_points=connect_points,
                        verbosity=verbosity,
                        logger=logger,
                    )
