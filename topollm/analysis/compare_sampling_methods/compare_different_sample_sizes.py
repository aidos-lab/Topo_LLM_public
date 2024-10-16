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
import re
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import pandas as pd

from topollm.analysis.compare_sampling_methods.log_statistics_of_array import log_statistics_of_array
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

    analysis_base_directory: pathlib.Path = pathlib.Path(
        embeddings_path_manager.data_dir,
        "analysis/twonn/",
        "data-multiwoz21_split-test_ctxt-dataset_entry_samples-3000_feat-col-ner_tags/",
        "lvl-token/add-prefix-space-True_max-len-512/model-roberta-base_task-masked_lm/layer--1_agg-mean/norm-None/",
        "sampling-take_first_seed-42_samples-30000",
    )

    arrays_truncated_list = []
    mean_list = []
    std_list = []
    sample_sizes_list: list[int] = []
    truncation_size: int = 2500

    # Discover directories matching the expected pattern (e.g., "desc-twonn_samples-<sample_size>_zerovec-keep")
    pattern = re.compile(
        pattern=r"desc-twonn_samples-(\d+)_zerovec-keep",
    )

    # Iterate through the folders in the base directory
    for subdirectory in analysis_base_directory.iterdir():
        if subdirectory.is_dir():
            match = pattern.match(
                string=subdirectory.name,
            )
            if match:
                if verbosity >= Verbosity.NORMAL:
                    logger.info(
                        msg=f"{match = }",  # noqa: G004 - low overhead
                    )
                    logger.info(
                        msg=f"Processing subdirectory: {subdirectory = }",  # noqa: G004 - low overhead
                    )

                sample_size = int(match.group(1))
                sample_sizes_list.append(
                    sample_size,
                )

                # Load the array from the current directory
                current_array_path: pathlib.Path = subdirectory / "local_estimates_paddings_removed.npy"
                current_array = np.load(
                    file=current_array_path,
                )

                if verbosity >= Verbosity.NORMAL:
                    log_statistics_of_array(
                        array=current_array,
                        array_name=f"current_array {subdirectory = }",
                        logger=logger,
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

    # Create a DataFrame to store results
    results_df = pd.DataFrame(
        data={
            "sample_size": sample_sizes_list,
            "mean": mean_list,
            "std": std_list,
        },
    )

    if verbosity >= Verbosity.NORMAL:
        log_dataframe_info(
            df=results_df,
            df_name="results_df (before sorting)",
            logger=logger,
        )

    # Sort the DataFrame by sample_size.
    # Do not reset the index, as we want to know the original order of the arrays.
    sorted_df: pd.DataFrame = results_df.sort_values(
        by="sample_size",
    )

    if verbosity >= Verbosity.NORMAL:
        log_dataframe_info(
            df=sorted_df,
            df_name="sorted_df",
            logger=logger,
        )

    # Sort the stacked arrays accordingly
    sorted_indices = sorted_df.index.to_numpy()
    arrays_truncated_sorted = [arrays_truncated_list[i] for i in sorted_indices]
    arrays_truncated_stacked = np.stack(
        arrays_truncated_sorted,
        axis=0,
    )

    if verbosity >= Verbosity.NORMAL:
        log_array_info(
            array_=arrays_truncated_stacked,
            array_name="arrays_truncated_stacked",
            logger=logger,
        )

    # Define the new base directory for saving results
    results_base_directory = pathlib.Path(
        embeddings_path_manager.data_dir,
        "analysis/sample_sizes/",
        analysis_base_directory.relative_to(
            embeddings_path_manager.data_dir,
        ),
    )
    results_base_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save the results DataFrame to a CSV file for traceability
    sorted_df_save_path = results_base_directory / "sorted_df.csv"
    sorted_df.to_csv(
        path_or_buf=sorted_df_save_path,
        index=False,
    )

    plot_save_path = results_base_directory / "arrays_truncated_stacked.pdf"

    make_multiple_line_plots(
        array=arrays_truncated_stacked,
        sample_sizes=sorted_df["sample_size"].to_numpy(),
        show_plot=False,
        save_path=plot_save_path,
    )

    # TODO: Compute and save Kendall rank correlation coefficient between the different sample sizes

    # # # #
    # Log some statistics of the debug data

    logger.info(
        msg="Running script DONE",
    )


def make_multiple_line_plots(
    array: np.ndarray,
    sample_sizes: np.ndarray,
    *,
    show_plot: bool = False,
    save_path: pathlib.Path | None = None,
) -> None:
    """Create multiple line plots of the data points over different sample sizes.

    Args:
    ----
        array:
            The data array to plot, with shape (num_samples, num_points).
        sample_sizes:
            The sample sizes corresponding to each row in the array.
        show_plot:
            Whether to display the plot.
        save_path:
            The path to save the plot to.

    """
    # Create a plot
    plt.figure(
        figsize=(10, 6),
    )

    # Loop over the 2500 data points and plot each as a line graph
    for i in range(array.shape[1]):
        plt.plot(
            sample_sizes,
            array[:, i],
            alpha=0.5,
            linewidth=0.5,
        )  # Plot each line with some transparency

    # Label the axes
    plt.xlabel(
        xlabel="Sample Size",
    )
    plt.ylabel(
        ylabel="Value",
    )
    plt.title(
        label=f"Development of data points over different sample sizes; {array.shape = }",
    )

    if show_plot:
        plt.show()

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(
            save_path,
            format="pdf",
        )


if __name__ == "__main__":
    main()
