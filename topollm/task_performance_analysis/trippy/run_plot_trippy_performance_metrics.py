# Copyright 2024-2025
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
# AUTHOR_2 (author2@example.com)
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

"""Create plots of the Trippy performance metrics."""

import json
import logging
import pathlib
import pickle
from typing import TYPE_CHECKING, Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import pandas as pd

from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.config_classes.setup_omega_conf import setup_omega_conf
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.task_performance_analysis.plotting.parameter_combinations_and_loaded_data_handling import (
    construct_plots_over_checkpoints_output_dir_from_filter_key_value_pairs,
)
from topollm.task_performance_analysis.plotting.plot_performance_metrics import plot_performance_metrics
from topollm.typing.enums import DescriptionType, Verbosity

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig
    from topollm.path_management.embeddings.protocol import EmbeddingsPathManager

# Logger for this file
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
    """Run main function."""
    logger: logging.Logger = global_logger
    logger.info(
        msg="Running script ...",
    )

    # ================================================== #
    # Load configuration and initialize path manager
    # ================================================== #

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=logger,
    )
    verbosity: Verbosity = main_config.verbosity

    embeddings_path_manager: EmbeddingsPathManager = get_embeddings_path_manager(
        main_config=main_config,
        logger=logger,
    )

    array_key_name: str = "file_data"

    # ================================================== #
    # Load performance log files
    # ================================================== #

    if main_config.language_model.model_log_file_path is None:
        logger.error(
            msg="The path to the log file is not set in the configuration.",
        )
        return

    # Construct paths to the log files.
    # Note: For the Trippy models, we have separate log files
    # for the dev and test sets.
    log_filepath_dict: dict[str, pathlib.Path] = {
        "dev": pathlib.Path(
            main_config.language_model.model_log_file_path,
            "eval_res.dev.json",
        ),
        "test": pathlib.Path(
            main_config.language_model.model_log_file_path,
            "eval_res.test.json",
        ),
    }

    try:
        loaded_logfile_data: dict[str, list[dict]] = {}

        for split_name, log_filepath in log_filepath_dict.items():
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"Loading log file for {split_name = }: {log_filepath = } ...",  # noqa: G004 - low overhead
                )

            # Load the log file (which is a list of dictionaries).
            with log_filepath.open(
                mode="r",
            ) as f:
                log_data: list[dict] = json.load(
                    fp=f,
                )
                loaded_logfile_data[split_name] = log_data

            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"Loading log file for {split_name = }: {log_filepath = } DONE",  # noqa: G004 - low overhead
                )
    except FileNotFoundError as e:
        logger.exception(
            msg=f"Error reading log file: {e}",  # noqa: G004 - low overhead
        )
        return

    # # # # # # # # # # # # # # #
    # Convert loaded data

    # Create the joined DataFrame
    logfile_data_df: pd.DataFrame = create_evaluation_dataframe(
        loaded_logfile_data=loaded_logfile_data,
    )

    # Filter to keep only rows where global_step is numeric
    logfile_data_df = logfile_data_df[
        logfile_data_df["global_step"].apply(
            func=lambda x: str(x).isdigit(),
        )
    ].copy()

    # Convert global_step to integer and sort by it
    logfile_data_df["global_step"] = logfile_data_df["global_step"].astype(
        dtype=int,
    )
    logfile_data_df = logfile_data_df.sort_values(
        by="global_step",
    )

    if logfile_data_df.empty:
        global_logger.warning(
            msg="The processed logfile_data_df is empty after filtering.",
        )

    # Create a debug plot and save to the log file directory.
    # This is useful for quickly checking the loaded data.
    debug_plot_save_file_path: pathlib.Path = pathlib.Path(
        main_config.language_model.model_log_file_path,
        "plots",
        "debug_plot_evaluation_metrics.pdf",
    )
    debug_plot_evaluation_metrics(
        df=logfile_data_df,
        plot_config=None,
        show_plot=False,
        save_file_path=debug_plot_save_file_path,
    )

    # ================================================== #
    # Optional:
    # Load the local estimates distributions
    # which can be added to the performance plots.
    # ================================================== #

    # Note:
    # In the future, we might want to fill in the values in the dict here from the config.
    filter_key_value_pairs: dict = {
        "data_full": "data=trippy_dataloaders_processed_multiwoz21_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags",
        "data_subsampling_full": "split=dev_samples=10000_sampling=random_sampling-seed=778",
        "data_dataset_seed": "None",
        "model_layer": -1,
        "model_partial_name": "model=roberta-base-trippy_multiwoz21",
        "model_seed": main_config.language_model.seed,
        "local_estimates_desc_full": "desc=twonn_samples=60000_zerovec=keep_dedup=array_deduplicator_noise=do_nothing",
        "tokenizer_add_prefix_space": "False",
    }

    # We saved the local estimates distributions in the plots over checkpoints directory.
    plots_over_checkpoints_output_dir: pathlib.Path = (
        construct_plots_over_checkpoints_output_dir_from_filter_key_value_pairs(
            output_root_dir=embeddings_path_manager.get_saved_plots_distribution_of_local_estimates_dir_absolute_path(),
            filter_key_value_pairs=filter_key_value_pairs,
            verbosity=verbosity,
            logger=logger,
        )
    )

    sorted_data_output_file_path: pathlib.Path = pathlib.Path(
        plots_over_checkpoints_output_dir,
        "sorted_data_list_of_dicts_with_arrays.pkl",
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Trying to load sorted data from {sorted_data_output_file_path = } ...",  # noqa: G004 - low overhead
        )

    try:
        with sorted_data_output_file_path.open(
            mode="rb",
        ) as f:
            loaded_sorted_local_estimates_data = pickle.load(  # noqa: S301 - we only use this for trusted data
                file=f,
            )
    except FileNotFoundError as e:
        logger.warning(
            msg=f"Error reading local estimates file: {e}",  # noqa: G004 - low overhead
        )
        logger.warning(
            msg="The local estimates distribution will not be added to the performance plots.",
        )
        loaded_sorted_local_estimates_data = None

    if loaded_sorted_local_estimates_data is not None:
        model_checkpoint_str_list: list[str] = [
            str(object=single_dict["model_checkpoint"]) for single_dict in loaded_sorted_local_estimates_data
        ]

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"{model_checkpoint_str_list=}",  # noqa: G004 - low overhead
            )

    # ================================================== #
    # Create plot of the performance
    # ================================================== #

    # Specify which columns to plot on the primary and secondary axes.
    primary_metrics: list[str] = [
        "dev_eval_accuracy_goal",
        "test_eval_accuracy_goal",
    ]
    secondary_metrics: list[str] = [
        "dev_loss",
        "test_loss",
    ]

    highlight_columns: list[str] = [
        "dev_eval_accuracy_goal",
        "test_eval_accuracy_goal",
        "dev_loss",
        "test_loss",
    ]

    # Call the plotting function.
    # Leave primary_ylim and secondary_ylim as None for auto-scaling,
    # or supply a tuple (min, max) if you want fixed limits.
    plot_performance_metrics(
        df=logfile_data_df,
        x_col="global_step",
        primary_y_cols=primary_metrics,
        secondary_y_cols=secondary_metrics,
        title="Development of Performance Metrics through Checkpoints",
        xlabel="Checkpoint",
        primary_ylabel="Metric (Primary Scale)",
        secondary_ylabel="Metric (Secondary Scale)",
        primary_ylim=None,  # e.g., None for auto
        secondary_ylim=None,  # e.g., None for auto
        output_root_dir=plots_over_checkpoints_output_dir,
        figsize=(
            20,
            14,
        ),
        highlight_best=highlight_columns,
        loaded_sorted_local_estimates_data=loaded_sorted_local_estimates_data,
        array_key_name=array_key_name,
        local_estimates_limits=(
            0.0,
            15.0,
        ),
        verbosity=verbosity,
        logger=logger,
    )

    logger.info(
        msg="Running script DONE",
    )


def create_evaluation_dataframe(
    loaded_logfile_data: dict[str, Any],
) -> pd.DataFrame:
    """Create a merged evaluation DataFrame from dev and test logs.

    The function assumes that loaded_logfile_data is a dictionary with keys 'dev' and 'test',
    where each value is a list of evaluation dictionaries. Each dictionary must have a 'global_step' key.
    It checks that the set of global steps in both splits match, prefixes the metric columns
    with 'dev_' or 'test_' as appropriate, and merges the DataFrames on 'global_step'.

    Args:
        loaded_logfile_data:
            Dictionary with keys 'dev' and 'test' containing
            lists of evaluation dictionaries.

    Returns:
        pd.DataFrame: A DataFrame with one row per global step and prefixed metric columns.

    Raises:
        ValueError: If the global steps do not match between the dev and test splits.

    """
    # Convert each list of dictionaries into a DataFrame
    df_dev = pd.DataFrame(
        data=loaded_logfile_data["dev"],
    )
    df_test = pd.DataFrame(
        data=loaded_logfile_data["test"],
    )

    # Check that the same global steps exist in both splits
    steps_dev = set(df_dev["global_step"])
    steps_test = set(df_test["global_step"])
    if steps_dev != steps_test:
        msg = "Global steps do not match between dev and test splits."
        raise ValueError(msg)

    # Rename columns for dev: add prefix 'dev_' to every column except 'global_step'
    df_dev_renamed: pd.DataFrame = df_dev.rename(
        columns=lambda col: f"dev_{col}" if col != "global_step" else col,
    )
    # Rename columns for test: add prefix 'test_' to every column except 'global_step'
    df_test_renamed: pd.DataFrame = df_test.rename(
        columns=lambda col: f"test_{col}" if col != "global_step" else col,
    )

    # Merge on 'global_step'
    merged_df: pd.DataFrame = df_dev_renamed.merge(
        right=df_test_renamed,
        on="global_step",
        how="inner",
    )

    return merged_df


def debug_plot_evaluation_metrics(
    df: pd.DataFrame,
    plot_config: list[dict[str, Any]] | None = None,
    *,
    show_plot: bool = False,
    save_file_path: pathlib.Path | None = None,
) -> None:
    """Plot evaluation metrics based on a configurable plot specification.

    This function is designed for quickly visualizing the loaded metrics.
    For the actual plotting, we use our more general function plot_performance_metrics,
    which in addition to the metrics plots the local estimates distributions.

    The function assumes that the DataFrame 'df' has a numeric 'global_step' column.
    The plot_config argument allows you to specify multiple subplots. Each configuration
    dictionary should contain:
      - 'title': Title of the subplot.
      - 'ylabel': Label for the y-axis.
      - 'columns': A list of tuples, each tuple containing (column_name, label) to be plotted.

    By default, it plots 'dev_loss' vs 'global_step' and 'test_loss' vs 'global_step' in one subplot,
    and 'dev_eval_accuracy_goal' vs 'global_step' and 'test_eval_accuracy_goal' vs 'global_step' in another subplot.

    Args:
        df (pd.DataFrame): Merged evaluation DataFrame with a numeric 'global_step' column.
        plot_config (Optional[List[Dict[str, Any]]]): Configuration for the subplots.
            Defaults to plotting losses and eval_accuracy_goal.
        filename (str): Filename to save the plot. Default is 'plot.png'.

    Returns:
        None: The plot is saved to disk and displayed.

    """
    # Define default configuration if none provided
    if plot_config is None:
        plot_config = [
            {
                "title": "Loss vs Global Step",
                "ylabel": "Loss",
                "columns": [("dev_loss", "dev_loss"), ("test_loss", "test_loss")],
            },
            {
                "title": "Eval Accuracy Goal vs Global Step",
                "ylabel": "Eval Accuracy Goal",
                "columns": [
                    ("dev_eval_accuracy_goal", "dev_eval_accuracy_goal"),
                    ("test_eval_accuracy_goal", "test_eval_accuracy_goal"),
                ],
            },
        ]

    n_plots = len(plot_config)
    (
        fig,
        axes,
    ) = plt.subplots(
        nrows=1,
        ncols=n_plots,
        figsize=(7 * n_plots, 6),
    )

    # Ensure axes is a list (even if only one subplot is created)
    if n_plots == 1:
        axes = [axes]

    for ax, config in zip(
        axes,
        plot_config,
        strict=True,
    ):
        for col, label in config["columns"]:
            ax.plot(
                df["global_step"],
                df[col],
                label=label,
                marker="o",
            )
        ax.set_xlabel("Global Step")
        ax.set_ylabel(config["ylabel"])
        ax.set_title(config["title"])
        ax.legend()

    plt.tight_layout()

    if save_file_path is not None:
        save_file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        plt.savefig(
            save_file_path,
        )

    if show_plot:
        plt.show()


if __name__ == "__main__":
    # Note: See the VSCode launch configurations for an example of how to run this script.

    setup_omega_conf()

    main()
