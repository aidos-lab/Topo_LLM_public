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
import pathlib
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import pandas as pd

from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.config_classes.setup_OmegaConf import setup_omega_conf
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig

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
    logger = global_logger
    logger.info("Running script ...")

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=logger,
    )

    base_directory: pathlib.Path = pathlib.Path(
        "/Users/ruppik/git-source/Topo_LLM/data/analysis/twonn/data-multiwoz21_split-test_ctxt-dataset_entry_samples-3000_feat-col-ner_tags/lvl-token/add-prefix-space-True_max-len-512/model-roberta-base_task-masked_lm/layer--1_agg-mean/norm-None/sampling-take_first_seed-42_samples-30000",
    )

    sample_sizes_list: list[int] = [
        2500,
        5000,
        7500,
        10000,
        12500,
        15000,
    ]
    arrays_truncated_list = []
    mean_list = []
    std_list = []
    truncation_size: int = 2500

    # Go through the folders in this directory for different sample sizes
    for sample_size in sample_sizes_list:
        current_array_path = pathlib.Path(
            base_directory,
            f"desc-twonn_samples-{sample_size}_zerovec-keep",
            "local_estimates_paddings_removed.npy",
        )
        current_array = np.load(
            file=current_array_path,
        )

        # Truncate the arrays to the first common elements, so that we can compare them
        current_array_truncated = current_array[:truncation_size,]

        arrays_truncated_list.append(
            current_array_truncated,
        )
        mean_list.append(
            np.mean(current_array_truncated),
        )
        std_list.append(
            np.std(current_array_truncated),
        )

    arrays_truncated_stacked = np.stack(
        arrays=arrays_truncated_list,
        axis=0,
    )

    # Make a dataframe of the results
    results_df = pd.DataFrame(
        data={
            "sample_size": sample_sizes_list,
            "mean": mean_list,
            "std": std_list,
        },
    )

    make_multiple_line_plots(
        array=arrays_truncated_stacked,
    )

    # =============================================== #
    # Manually loaded debug data with explicit paths

    small_sample_size = 2500
    small_sample_size_meta_pkl_path = pathlib.Path(
        base_directory,
        f"desc-twonn_samples-{small_sample_size}_zerovec-keep",
        "local_estimates_paddings_removed_meta.pkl",
    )
    small_sample_size_array_npy_path = pathlib.Path(
        base_directory,
        f"desc-twonn_samples-{small_sample_size}_zerovec-keep",
        "local_estimates_paddings_removed.npy",
    )
    small_sample_size_meta_df = pd.read_pickle(  # noqa: S301 - we trust the data
        filepath_or_buffer=small_sample_size_meta_pkl_path,
    )
    small_sample_size_array = np.load(
        file=small_sample_size_array_npy_path,
    )

    large_sample_size = 15_000
    large_sample_size_meta_pkl_path = pathlib.Path(
        base_directory,
        f"desc-twonn_samples-{large_sample_size}_zerovec-keep",
        "local_estimates_paddings_removed_meta.pkl",
    )
    large_sample_size_array_npy_path = pathlib.Path(
        base_directory,
        f"desc-twonn_samples-{large_sample_size}_zerovec-keep",
        "local_estimates_paddings_removed.npy",
    )
    large_sample_size_meta_df = pd.read_pickle(  # noqa: S301 - we trust the data
        filepath_or_buffer=large_sample_size_meta_pkl_path,
    )
    large_sample_size_array = np.load(
        file=large_sample_size_array_npy_path,
    )

    # # # #
    # Log some statistics of the debug data

    logger.info("Running script DONE")


def make_multiple_line_plots(
    array: np.ndarray,
) -> None:
    # Create a plot
    plt.figure(figsize=(10, 6))

    # Loop over the 2500 data points and plot each as a line graph
    for i in range(array.shape[1]):
        plt.plot(
            array[:, i],
            alpha=0.5,
            linewidth=0.5,
        )  # Plot each line with some transparency

    # Label the axes
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.title("Development of 2500 data points over 6 time steps")

    # Show the plot
    plt.show()


def log_statistics_of_array(
    array: np.ndarray,
    array_name: str = "default_array_name",
    logger: logging.Logger = default_logger,
) -> None:
    """Log some statistics of the array."""
    logger.info(
        msg=f"Statistics of array: {array_name = }",  # noqa: G004 - low overhead
    )
    logger.info(
        msg=f"Shape:\t{array.shape = }",  # noqa: G004 - low overhead
    )
    logger.info(
        msg=f"Min:\t{np.min(array) = }",  # noqa: G004 - low overhead
    )
    logger.info(
        msg=f"Max:\t{np.max(array) = }",  # noqa: G004 - low overhead
    )
    logger.info(
        msg=f"Mean:\t{np.mean(array) = }",  # noqa: G004 - low overhead
    )
    logger.info(
        msg=f"Std:\t{np.std(array) =}",  # noqa: G004 - low overhead
    )


if __name__ == "__main__":
    main()
