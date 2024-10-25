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
import re
from itertools import product
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import numpy as np
import omegaconf
import pandas as pd
from tqdm import tqdm

from topollm.analysis.compare_sampling_methods.compute_correlations import compute_and_save_correlations
from topollm.analysis.compare_sampling_methods.data_selection_folder_lists import get_data_folder_list
from topollm.analysis.compare_sampling_methods.log_statistics_of_array import log_statistics_of_array
from topollm.analysis.compare_sampling_methods.make_plots import make_mean_std_plot, make_multiple_line_plots
from topollm.analysis.compare_sampling_methods.organize_results_directory_structure import (
    build_results_directory_structure,
)
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
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"{search_base_directory = }",  # noqa: G004 - low overhead
            )

        run_search_on_single_base_directory_and_process_and_save(
            search_base_directory=search_base_directory,
            data_dir=data_dir,
            verbosity=verbosity,
            logger=logger,
        )

        # TODO: Continue analysis here
        # TODO: Make plots for the different comparisons

    logger.info(
        msg="Running script DONE",
    )


def run_search_on_single_base_directory_and_process_and_save(
    search_base_directory: pathlib.Path,
    data_dir: pathlib.Path,
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
        [full_loaded_data_df, parsed_columns_full],
        axis=1,
    )

    full_local_estimates_df: pd.DataFrame = extract_and_prepare_local_estimates_data(
        loaded_data_df=full_loaded_data_df,
    )
    if verbosity >= Verbosity.NORMAL:
        log_dataframe_info(
            df=full_local_estimates_df,
            df_name="full_local_estimates_df",
            logger=logger,
        )

    full_local_estimates_df_save_path: pathlib.Path = pathlib.Path(
        results_directory,
        "full_local_estimates_df.csv",
    )
    full_local_estimates_df.to_csv(
        path_or_buf=full_local_estimates_df_save_path,
    )


def parse_path_info_full(
    path: str | pathlib.Path,
):
    """Parse the information from the path.

    Example path:
    /Users/USER_NAME/git-source/Topo_LLM/
    data/
    analysis/
    twonn/
    data-multiwoz21_split-train_ctxt-dataset_entry_samples-10000_feat-col-ner_tags/
    lvl-token/
    add-prefix-space-True_max-len-512/
    model-model-roberta-base_task-masked_lm_one-year-of-tsla-on-reddit-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1234_ckpt-31200_task-masked_lm/
    layer--1_agg-mean/
    norm-None/
    sampling-take_first_seed-42_samples-30000/
    desc-twonn_samples-12500_zerovec-keep/
    n-neighbors-mode-absolute_size_n-neighbors-128/
    local_estimates_pointwise.npy
    """
    # Convert the path to a string
    path_str = str(path)

    # Initialize an empty dictionary to hold parsed values
    parsed_info: dict = {}

    # Extract sampling information
    sampling_match = re.search(
        pattern=r"sampling-(\w+)_seed-(\d+)_samples-(\d+)",
        string=path_str,
    )
    if sampling_match:
        parsed_info["data_prep_sampling_method"] = sampling_match.group(1)
        parsed_info["data_prep_sampling_seed"] = sampling_match.group(2)
        parsed_info["data_prep_sampling_samples"] = sampling_match.group(3)

    # Extract local estimates information
    desc_match = re.search(
        pattern=r"desc-(\w+)_samples-(\d+)_zerovec-(\w+)(?:_dedup-(\w+))?",
        string=path_str,
    )
    if desc_match:
        parsed_info["local_estimates_description"] = desc_match.group(1)
        parsed_info["local_estimates_samples"] = desc_match.group(2)
        parsed_info["zerovec"] = desc_match.group(3)
        # TODO: The deduplication parsing does not appear to work correctly currently
        if desc_match.group(4):
            parsed_info["deduplication"] = desc_match.group(4)
        else:
            parsed_info["deduplication"] = None

    # Extract neighbors information
    neighbors_match = re.search(
        pattern=r"n-neighbors-mode-(\w+)_size_n-neighbors-(\d+)",
        string=path_str,
    )
    if neighbors_match:
        parsed_info["neighbors_mode"] = neighbors_match.group(1)
        parsed_info["n_neighbors"] = neighbors_match.group(2)

    # Extract model information
    model_match = re.search(
        pattern=r"model-(\w+-[\w-]+)_seed-(\d+)_ckpt-(\d+)",
        string=path_str,
    )
    if model_match:
        parsed_info["model_name"] = model_match.group(1)
        parsed_info["model_seed"] = model_match.group(3)
        parsed_info["checkpoint"] = model_match.group(4)

    # Extract layer and aggregation information
    layer_match = re.search(
        pattern=r"layer--(\d+)_agg-(\w+)_norm-(\w+)",
        string=path_str,
    )
    if layer_match:
        parsed_info["model_layer"] = layer_match.group(1)
        parsed_info["aggregation"] = layer_match.group(2)
        parsed_info["normalization"] = layer_match.group(3)

    return parsed_info


def extract_and_prepare_local_estimates_data(
    loaded_data_df: pd.DataFrame,
    array_truncation_size: int = 2_500,
):
    array_name_to_match: str = "local_estimates_pointwise.npy"

    # Filter the DataFrame to only contain the local estimates
    local_estimates_df: pd.DataFrame = loaded_data_df[loaded_data_df["array_name"] == array_name_to_match]

    # Add a column with the number of elements in the array
    local_estimates_df["num_elements"] = local_estimates_df["array_data"].apply(
        func=lambda array: array.size,
    )

    # Add a column with truncated arrays
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
