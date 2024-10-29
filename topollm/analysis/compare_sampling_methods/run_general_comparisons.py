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
from itertools import product
from typing import TYPE_CHECKING, Any

import hydra
import hydra.core.hydra_config
import numpy as np
import omegaconf
import pandas as pd
from tqdm import tqdm

from topollm.analysis.compare_sampling_methods.compute_correlations import compute_and_save_correlations
from topollm.analysis.compare_sampling_methods.data_selection_folder_lists import get_data_folder_list
from topollm.analysis.compare_sampling_methods.log_statistics_of_array import log_statistics_of_array
from topollm.analysis.compare_sampling_methods.make_plots import (
    analyze_and_plot_influence_of_local_estimates_samples,
    create_boxplot_of_mean_over_different_sampling_seeds,
    generate_fixed_params_text,
    make_mean_std_plot,
    make_multiple_line_plots,
)
from topollm.analysis.compare_sampling_methods.organize_results_directory_structure import (
    build_results_directory_structure,
)
from topollm.analysis.compare_sampling_methods.parse_path_info_full import parse_path_info_full
from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.config_classes.setup_OmegaConf import setup_omega_conf
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.log_array_info import log_array_info
from topollm.logging.log_dataframe_info import log_dataframe_info
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

Y_AXIS_LIMITS: dict[str, tuple[float | None, float | None]] = {
    "None": (None, None),
    "full": (6.5, 15.5),  # full range
    "lower": (6.5, 10.0),  # lower range
    "upper": (12.0, 16.0),  # upper range
}


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

    data_folder_list: list[str] = get_data_folder_list()

    model_folder_list: list[str] = [
        "model-roberta-base_task-masked_lm",
        "model-model-roberta-base_task-masked_lm_multiwoz21-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1234_ckpt-14400_task-masked_lm",
        "model-model-roberta-base_task-masked_lm_multiwoz21-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1234_ckpt-31200_task-masked_lm",
        "model-model-roberta-base_task-masked_lm_multiwoz21-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1234_ckpt-400_task-masked_lm",
        "model-model-roberta-base_task-masked_lm_one-year-of-tsla-on-reddit-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1234_ckpt-14400_task-masked_lm",
        "model-model-roberta-base_task-masked_lm_one-year-of-tsla-on-reddit-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1234_ckpt-31200_task-masked_lm",
        "model-model-roberta-base_task-masked_lm_one-year-of-tsla-on-reddit-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1234_ckpt-400_task-masked_lm",
    ]

    for data_folder, model_folder in tqdm(
        iterable=product(
            data_folder_list,
            model_folder_list,
        ),
        desc="Iterating over folder choices",
        total=len(data_folder_list) * len(model_folder_list),
    ):
        search_base_directory: pathlib.Path = pathlib.Path(
            data_dir,
            "analysis",
            "twonn",
            data_folder,
            "lvl-token",
            "add-prefix-space-True_max-len-512",
            model_folder,
            "layer--1_agg-mean",
            "norm-None",
        )
        array_truncation_size: int = 5_000

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"{search_base_directory = }",  # noqa: G004 - low overhead
            )

        run_search_on_single_base_directory_and_process_and_save(
            search_base_directory=search_base_directory,
            data_dir=data_dir,
            array_truncation_size=array_truncation_size,
            verbosity=verbosity,
            logger=logger,
        )

    logger.info(
        msg="Running script DONE",
    )


def run_search_on_single_base_directory_and_process_and_save(
    search_base_directory: pathlib.Path,
    data_dir: pathlib.Path,
    array_truncation_size: int = 2_500,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Run the search and analysis on a single base directory."""
    results_directory: pathlib.Path = build_results_directory_structure(
        analysis_base_directory=search_base_directory,
        data_dir=data_dir,
        analysis_subdirectory_partial_path=pathlib.Path(
            "sample_sizes",
            "run_general_comparisons",
            f"{array_truncation_size=}",
        ),
        verbosity=verbosity,
        logger=logger,
    )

    array_data_column_name: str = "array_data"

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

    run_analysis_influence_of_local_estimates_n_neighbors(
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

                create_boxplot_of_mean_over_different_sampling_seeds(
                    subset_local_estimates_df=subset_local_estimates_df,
                    plot_save_path=plot_save_path,
                    raw_data_save_path=raw_data_save_path,
                    fixed_params_text=fixed_params_text,
                    y_min=y_min,
                    y_max=y_max,
                    show_plot=False,
                    connect_points=connect_points,
                )

    # TODO: Continue analysis here

    pass  # noqa: PIE790 - This is here for setting a breakpoint


def run_analysis_influence_of_local_estimates_n_neighbors(
    full_local_estimates_df: pd.DataFrame,
    array_data_column_name: str,
    results_directory: pathlib.Path,
    selected_subsample_dict: dict | None = None,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Run the analysis of local estimates for different values of 'n_neighbors'."""
    if selected_subsample_dict is None:
        selected_subsample_dict = retrieve_most_frequent_values(
            full_local_estimates_df=full_local_estimates_df,
            verbosity=verbosity,
            logger=logger,
        )

    # Run analysis for different values of 'n_neighbors'
    unique_n_neighbors = full_local_estimates_df["n_neighbors"].unique()

    for n_neighbors in unique_n_neighbors:
        for y_min, y_max in Y_AXIS_LIMITS.values():
            common_prefix_path = pathlib.Path(
                results_directory,
                "influence_of_local_estimates_samples",
            )

            plot_save_path: pathlib.Path = pathlib.Path(
                common_prefix_path,
                "plots",
                f"{n_neighbors=}_y_{y_min}_{y_max}.pdf",
            )
            raw_data_save_path: pathlib.Path = pathlib.Path(
                common_prefix_path,
                "raw_data",
                f"{n_neighbors=}.csv",
            )

            analyze_and_plot_influence_of_local_estimates_samples(
                df=full_local_estimates_df,
                n_neighbors=n_neighbors,
                selected_subsample_dict=selected_subsample_dict,
                array_data_column_name=array_data_column_name,
                y_min=y_min,
                y_max=y_max,
                show_plot=False,
                plot_save_path=plot_save_path,
                raw_data_save_path=raw_data_save_path,
                verbosity=verbosity,
                logger=logger,
            )


def retrieve_most_frequent_values(
    full_local_estimates_df: pd.DataFrame,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> dict[str, Any]:
    """Retrieve the most frequent values for fixed parameters.

    This handles cases where mode might be unavailable.
    """
    most_frequent_values = {}
    for column in [
        "data_prep_sampling_method",
        "deduplication",
        "data_prep_sampling_seed",
        "data_prep_sampling_samples",
    ]:
        if full_local_estimates_df[column].notna().any():
            most_frequent_values[column] = (
                full_local_estimates_df[column].mode()[0] if not full_local_estimates_df[column].mode().empty else None
            )
        else:
            most_frequent_values[column] = None

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"most_frequent_values:\n{most_frequent_values}",  # noqa: G004 - low overhead
        )

    return most_frequent_values


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

    # Add a column with the number of elements in the array
    local_estimates_df["num_elements"] = local_estimates_df[array_data_column_name].apply(
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
