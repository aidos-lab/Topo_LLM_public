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
from topollm.analysis.compare_sampling_methods.log_statistics_of_array import log_statistics_of_array
from topollm.analysis.compare_sampling_methods.make_plots import make_mean_std_plot, make_multiple_line_plots
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

    data_folder = "data-multiwoz21_split-validation_ctxt-dataset_entry_samples-3000_feat-col-ner_tags"
    model_folder = "model-roberta-base_task-masked_lm"

    search_base_directory: pathlib.Path = pathlib.Path(
        embeddings_path_manager.data_dir,
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

    filenames_to_match = [
        "global_estimate.npy",
        "local_estimates_pointwise.npy",
    ]

    loaded_data_df = walk_through_subdirectories_and_load_arrays(
        root_directory=search_base_directory,
        verbosity=verbosity,
        filenames_to_match=filenames_to_match,
        logger=logger,
    )
    if verbosity >= Verbosity.NORMAL:
        log_dataframe_info(
            df=loaded_data_df,
            df_name="loaded_data_df",
            logger=logger,
        )

    # TODO: Continue analysis here
    # TODO: Parse the information from the file names
    # TODO: Make plots for the different comparisons

    logger.info(
        msg="Running script DONE",
    )


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
    df = pd.DataFrame(
        data=data,
    )

    return df


if __name__ == "__main__":
    main()
