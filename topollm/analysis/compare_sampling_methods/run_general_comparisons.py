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
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import pandas as pd
from tqdm import tqdm

from topollm.analysis.compare_sampling_methods.compute_correlations import compute_and_save_correlations
from topollm.analysis.compare_sampling_methods.data_selection_folder_lists import get_data_folder_list
from topollm.analysis.compare_sampling_methods.log_statistics_of_array import log_statistics_of_array
from topollm.analysis.compare_sampling_methods.make_plots import (
    add_subtitle,
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
        ),
        verbosity=verbosity,
        logger=logger,
    )

    filenames_to_match = [
        "global_estimate.npy",
        "local_estimates_pointwise.npy",
    ]

    full_loaded_data_df: pd.DataFrame = walk_through_subdirectories_and_load_arrays(
        root_directory=search_base_directory,
        verbosity=verbosity,
        filenames_to_match=filenames_to_match,
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

    # ===== This is the start of the analysis code =====

    # Determine the most frequent values for fixed parameters, handling cases where mode might be unavailable
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
    logger.info(
        msg=f"full_local_estimates_df:\n{most_frequent_values}",  # noqa: G004 - low overhead
    )

    # Run analysis for different values of 'n_neighbors'
    unique_n_neighbors = full_local_estimates_df["n_neighbors"].unique()

    for n_neighbors in unique_n_neighbors:
        plot_save_path: pathlib.Path = pathlib.Path(
            results_directory,
            "influence_of_local_estimates_samples",
            f"influence_of_local_estimates_samples_{n_neighbors=}.pdf",
        )
        raw_data_save_path: pathlib.Path = pathlib.Path(
            results_directory,
            "influence_of_local_estimates_samples",
            f"influence_of_local_estimates_samples_{n_neighbors=}.csv",
        )

        analyze_and_plot_influence_of_local_estimates_samples(
            df=full_local_estimates_df,
            n_neighbors=n_neighbors,
            most_frequent_values=most_frequent_values,
            show_plot=False,
            plot_save_path=plot_save_path,
            raw_data_save_path=raw_data_save_path,
            verbosity=verbosity,
            logger=logger,
        )

    # TODO: Continue analysis here

    pass  # Note: This is here for setting break points


def analyze_and_plot_influence_of_local_estimates_samples(
    df: pd.DataFrame,
    n_neighbors: int,
    most_frequent_values: dict,
    additional_title: str | None = None,
    *,
    show_plot: bool = False,
    plot_save_path: pathlib.Path | None = None,
    raw_data_save_path: pathlib.Path | None = None,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Analyze and visualize the influence of 'local_estimates_samples'.

    We plot array_data_truncated_mean and array_data_truncated_std
    for a given value of 'n_neighbors', while keeping other parameters fixed.
    """
    # Update the fixed parameter values to the chosen 'n_neighbors'
    updated_values = most_frequent_values.copy()
    updated_values["n_neighbors"] = n_neighbors

    # Filter the DataFrame with the updated fixed values,
    # handling cases where deduplication is None
    filtered_df = df[
        (df["data_prep_sampling_method"] == updated_values["data_prep_sampling_method"])
        & (
            df["deduplication"].isna()
            if updated_values["deduplication"] is None
            else df["deduplication"] == updated_values["deduplication"]
        )
        & (df["n_neighbors"] == updated_values["n_neighbors"])
        & (df["data_prep_sampling_seed"] == updated_values["data_prep_sampling_seed"])
        & (df["data_prep_sampling_samples"] == updated_values["data_prep_sampling_samples"])
    ]

    if verbosity >= Verbosity.NORMAL:
        log_dataframe_info(
            df=filtered_df,
            df_name="filtered_df",
            logger=logger,
        )

    # Ensure there are enough valid data points to proceed
    if filtered_df.empty:
        logger.warning(
            msg=f"No valid data to plot for {n_neighbors = }",  # noqa: G004 - low overhead
        )
        return

    # Sort by 'local_estimates_samples' for consistent plotting
    sorted_df = filtered_df.sort_values(
        by="local_estimates_samples",
    )

    # Convert columns to NumPy arrays for plotting
    sample_size_array = sorted_df["local_estimates_samples"].to_numpy(dtype=float)
    mean_array = sorted_df["array_data_truncated_mean"].to_numpy(dtype=float)
    std_array = sorted_df["array_data_truncated_std"].to_numpy(dtype=float)

    # Plotting the analysis
    plt.figure(figsize=(10, 6))
    plt.plot(
        sample_size_array,
        mean_array,
        color="b",
        label="Truncated Mean",
        marker="o",
    )
    plt.fill_between(
        x=sample_size_array,
        y1=mean_array - std_array,
        y2=mean_array + std_array,
        color="b",
        alpha=0.2,
        label="Standard Deviation",
    )

    # Add horizontal and vertical grid lines for better readability
    plt.grid(
        visible=True,
        which="both",
        linestyle="--",
        linewidth=0.5,
        color="gray",
    )

    # Label the axes and add title
    plt.xlabel("local_estimates_samples")
    plt.ylabel("Truncated Mean")
    plt.title(
        label=f"Influence of 'local_estimates_samples' on Truncated Mean and Std (n_neighbors={n_neighbors})",
    )

    if additional_title:
        add_subtitle(
            additional_title=additional_title,
        )
    else:
        # If no additional title is provided, use the first file path as a subtitle
        add_subtitle(
            additional_title=str(object=sorted_df["path"].iloc[0]),
        )

    # Adding additional information about the fixed parameters in the plot
    fixed_params_text = "\n".join([f"{key}: {value}" for key, value in updated_values.items()])
    plt.text(
        x=0.02,
        y=0.95,
        s=f"Fixed Parameters:\n{fixed_params_text}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox={
            "boxstyle": "round",
            "facecolor": "wheat",
            "alpha": 0.5,
        },
    )

    # Add a legend
    plt.legend()
    plt.tight_layout()

    # Save the plot if save_path is provided
    if plot_save_path:
        plot_save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        plt.savefig(
            plot_save_path,
            format="pdf",
        )
    # Save the raw data if save_path is provided
    if raw_data_save_path:
        raw_data_save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        sorted_df.to_csv(
            path_or_buf=raw_data_save_path,
        )

    # Show the plot if requested
    if show_plot:
        plt.show()


def extract_and_prepare_local_estimates_data(
    loaded_data_df: pd.DataFrame,
    array_truncation_size: int = 2_500,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> pd.DataFrame:
    array_name_to_match: str = "local_estimates_pointwise.npy"

    # Filter the DataFrame to only contain the local estimates
    local_estimates_df: pd.DataFrame = loaded_data_df[loaded_data_df["array_name"] == array_name_to_match]

    # Add a column with the number of elements in the array
    local_estimates_df["num_elements"] = local_estimates_df["array_data"].apply(
        func=lambda array: array.size,
    )

    # Add a column with truncated arrays
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Truncating arrays to {array_truncation_size = }",  # noqa: G004 - low overhead
        )
    local_estimates_df["array_data_truncated"] = local_estimates_df["array_data"].apply(
        func=lambda array: array[:array_truncation_size],
    )

    # Add a column with the mean and standard deviation of the arrays
    local_estimates_df["array_data_mean"] = local_estimates_df["array_data"].apply(
        func=lambda array: array.mean(),
    )
    local_estimates_df["array_data_std"] = local_estimates_df["array_data"].apply(
        func=lambda array: array.std(),
    )
    local_estimates_df["array_data_truncated_mean"] = local_estimates_df["array_data_truncated"].apply(
        func=lambda array: array.mean(),
    )
    local_estimates_df["array_data_truncated_std"] = local_estimates_df["array_data_truncated"].apply(
        func=lambda array: array.std(),
    )

    return local_estimates_df


def walk_through_subdirectories_and_load_arrays(
    root_directory: pathlib.Path,
    filenames_to_match: list[str] | None = None,
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
                        "array_data": array,
                    },
                )

    # Create a DataFrame from the collected data
    loaded_data_df = pd.DataFrame(
        data=data,
    )

    return loaded_data_df


if __name__ == "__main__":
    main()
