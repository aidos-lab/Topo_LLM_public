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

"""Parse a SETSUMBT JSONL log file and create a plot of performance metrics versus checkpoints."""

import json
import logging
import os
import pathlib
from typing import TYPE_CHECKING, Any

import hydra
import matplotlib.pyplot as plt
import omegaconf
import pandas as pd

from tests.conftest import language_model_config
from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.config_classes.main_config import MainConfig
from topollm.config_classes.setup_OmegaConf import setup_omega_conf
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.log_dataframe_info import log_dataframe_info
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.typing.enums import DescriptionType, Verbosity

if TYPE_CHECKING:
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


def parse_log_file(
    file_path: os.PathLike,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> pd.DataFrame:
    """Parse a SETSUMBT JSONL log file containing training checkpoints and evaluation statistics.

    The log file is expected to contain JSON objects on each line. The function looks for
    two types of log entries:
      1. A checkpoint entry with a message like "Step <number> complete" and a key
         "Loss since last update". This entry gives the checkpoint (step) and loss.
      2. An evaluation entry with the message "Validation set statistics" that includes
         various validation metrics (e.g., "Joint Goal Accuracy", "Slot F1 Score", etc.).

    The function pairs each checkpoint entry with the following validation entry and returns
    a DataFrame where each row corresponds to a checkpoint with its associated loss and validation
    metrics.

    Args:
        file_path:
            The path to the JSONL log file.
        verbosity:
            The verbosity level for logging and output.
        logger:
            The logger to use for output.


    Returns:
        pd.DataFrame: A DataFrame with columns for checkpoint number, loss, timestamps, and
                      validation metrics.

    """
    results: list[dict[str, Any]] = []
    current_checkpoint: dict[str, Any] = {}

    # Use Path to allow easy file handling
    log_file = pathlib.Path(
        file_path,
    )
    if not log_file.is_file():
        msg: str = f"File not found: {file_path = }"
        raise FileNotFoundError(
            msg,
        )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Reading {log_file = } ...",  # noqa: G004 - low overhead
        )

    # Open the file and process line by line
    with log_file.open(
        mode="r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line: str = line.strip()
            if not line:
                continue  # Skip empty lines
            try:
                log_entry = json.loads(
                    s=line,
                )
            except json.JSONDecodeError as e:
                # If a line is not valid JSON, skip it or handle as needed.
                logger.info(
                    msg=f"Skipping invalid JSON line: {line}\nError: {e}",  # noqa: G004 - low overhead
                )
                continue

            message: str = log_entry.get("message", "")

            # Look for checkpoint entries: message starts with "Step" and contains "complete"
            # and the entry includes a "Loss since last update"
            if message.startswith("Step") and "complete" in message and "Loss since last update" in log_entry:
                # Extract the step number from the message, e.g., "Step 2813 complete"
                parts: list[str] = message.split()
                try:
                    step_number = int(parts[1])
                except (IndexError, ValueError):
                    step_number = None

                # Build a dictionary for this checkpoint
                current_checkpoint = {
                    "checkpoint": step_number,
                    "loss": log_entry.get("Loss since last update"),
                    "step_timestamp": log_entry.get("timestamp"),
                }

            # Look for validation entries with evaluation metrics
            elif "Validation set statistics" in message:
                # Extract all keys except the basic ones (like "message" and "timestamp")
                validation_metrics = {
                    key: value for key, value in log_entry.items() if key not in {"message", "timestamp"}
                }
                # Optionally, you might store the timestamp of the validation entry separately.
                validation_metrics["validation_timestamp"] = log_entry.get("timestamp")

                # Merge the checkpoint info (if any) with the validation metrics.
                # If no checkpoint was found before, you could either skip or record a row with NaNs.
                if current_checkpoint:
                    # Combine the dictionaries; if the same key exists, validation metrics overwrite.
                    combined_entry = {
                        **current_checkpoint,
                        **validation_metrics,
                    }
                    results.append(combined_entry)
                    # Clear current_checkpoint so that a new pair is started for the next checkpoint.
                    current_checkpoint = {}
                else:
                    # In case a validation entry occurs without a preceding checkpoint entry,
                    # record it with a missing checkpoint.
                    row = {
                        "checkpoint": None,
                        "loss": None,
                        "step_timestamp": None,
                    }
                    row.update(validation_metrics)
                    results.append(row)

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Reading {log_file = } DONE",  # noqa: G004 - low overhead
        )

    # Convert the list of dictionaries into a pandas DataFrame.
    parsed_df = pd.DataFrame(
        data=results,
    )

    if verbosity >= Verbosity.NORMAL:
        log_dataframe_info(
            df=parsed_df,
            df_name="parsed_df",
            logger=logger,
        )

    return parsed_df


def plot_performance_metrics(
    df: pd.DataFrame,
    x_col: str = "checkpoint",
    primary_y_cols: list[str] | None = None,
    secondary_y_cols: list[str] | None = None,
    title: str = "Performance Metrics vs Checkpoint",
    xlabel: str = "Checkpoint",
    primary_ylabel: str = "Primary Metric Value",
    secondary_ylabel: str = "Secondary Metric Value",
    primary_ylim: tuple[float, float] | None = None,
    secondary_ylim: tuple[float, float] | None = None,
    figsize: tuple[int, int] = (20, 8),
    output_root_dir: pathlib.Path | None = None,
    highlight_best: list[str] | None = None,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Plot performance metrics versus checkpoints using configurable y-axis groups and scales.

    This function creates a line plot from the given DataFrame showing the development of
    performance measures through training checkpoints. You can configure which metrics
    (columns) are plotted on the primary y‑axis and which (if any) on a secondary y‑axis.
    In addition, you can optionally specify fixed y‑axis limits. If set to None, the scale
    is determined automatically.

    Args:
        df (pd.DataFrame): DataFrame containing the log data. Must include the x-axis column.
        x_col (str): Name of the column to use as the x-axis (default "checkpoint").
        primary_y_cols: List of column names to plot on the primary y-axis.
            If None, no primary lines are plotted.
        secondary_y_cols: List of column names to plot on the secondary y-axis.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        primary_ylabel (str): Label for the primary y-axis.
        secondary_ylabel (str): Label for the secondary y-axis.
        primary_ylim:
            Fixed y-axis limits for the primary y-axis.
            If None, the limits are set automatically.
        secondary_ylim:
            Fixed y-axis limits for the secondary y-axis.
            If None, the limits are set automatically.
        figsize:
            Figure size.

    """
    if primary_y_cols is None and secondary_y_cols is None:
        msg = "At least one of primary_y_cols or secondary_y_cols must be provided."
        raise ValueError(msg)

    # Create the figure and the primary axis.
    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot metrics for the primary y-axis.
    if primary_y_cols:
        for col in primary_y_cols:
            if col not in df.columns:
                logger.info(
                    msg=f"Warning: '{col}' not found in DataFrame; skipping.",  # noqa: G004 - low overhead
                )
                continue
            ax1.plot(df[x_col], df[col], marker="o", label=col)
        ax1.set_ylabel(primary_ylabel)
        if primary_ylim is not None:
            ax1.set_ylim(primary_ylim)

    # Plot metrics for the secondary y-axis if provided.
    if secondary_y_cols:
        ax2 = ax1.twinx()
        for col in secondary_y_cols:
            if col not in df.columns:
                logger.info(
                    msg=f"Warning: '{col}' not found in DataFrame; skipping.",  # noqa: G004 - low overhead
                )
                continue
            ax2.plot(df[x_col], df[col], marker="s", linestyle="--", label=col)
        ax2.set_ylabel(secondary_ylabel)
        if secondary_ylim is not None:
            ax2.set_ylim(secondary_ylim)
    else:
        ax2 = None

    # Highlight both the maximum and minimum points for selected columns.
    if highlight_best is not None:
        for col in highlight_best:
            if col not in df.columns:
                logger.info(f"Warning: '{col}' not found in DataFrame for highlighting; skipping.")
                continue

            # Determine which axis to use.
            if primary_y_cols is not None and col in primary_y_cols:
                axis = ax1
            elif secondary_y_cols is not None and ax2 is not None and col in secondary_y_cols:
                axis = ax2
            else:
                axis = ax1

            # Mark maximum point.
            max_idx = df[col].idxmax()
            max_x = df.loc[max_idx, x_col]
            max_y = df.loc[max_idx, col]

            # Make sure max_x and max_y are numbers.
            max_x = float(
                max_x,  # type: ignore - typing problem with pandas Scalar
            )
            max_y = float(
                max_y,  # type: ignore - typing problem with pandas Scalar
            )

            # Check that max_x and max_y are valid numbers.
            axis.plot(
                max_x,
                max_y,
                marker="*",
                markersize=14,
                color="red",
                label=f"Max {col}",
            )

            # Compute an offset (5% of the current y-axis range) for the maximum annotation.
            current_ylim = axis.get_ylim()
            offset_max = (current_ylim[1] - current_ylim[0]) * 0.05
            axis.annotate(
                f"{max_y:.2f}",
                xy=(max_x, max_y),
                xytext=(max_x, max_y + offset_max),
                arrowprops={"arrowstyle": "->", "color": "red"},
                color="red",
                fontsize=10,
            )

            # Mark minimum point.
            min_idx = df[col].idxmin()
            min_x = df.loc[min_idx, x_col]
            min_y = df.loc[min_idx, col]

            # Make sure min_x and min_y are numbers.
            min_x = float(
                min_x,  # type: ignore - typing problem with pandas Scalar
            )
            min_y = float(
                min_y,  # type: ignore - typing problem with pandas Scalar
            )
            axis.plot(
                min_x,
                min_y,
                marker="*",
                markersize=14,
                color="blue",
                label=f"Min {col}",
            )

            # Compute an offset for the minimum annotation (placing text below the marker).
            offset_min = (current_ylim[1] - current_ylim[0]) * 0.05
            axis.annotate(
                f"{min_y:.2f}",
                xy=(min_x, min_y),
                xytext=(min_x, min_y - offset_min),
                arrowprops={"arrowstyle": "->", "color": "blue"},
                color="blue",
                fontsize=10,
            )

    ax1.set_xlabel(
        xlabel=xlabel,
    )
    ax1.set_title(
        label=title,
    )

    # Combine legends from both axes.
    handles1, labels1 = ax1.get_legend_handles_labels()
    if ax2 is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
    else:
        handles2, labels2 = [], []

    # Place the legend below the plot.
    ax1.legend(
        handles=handles1 + handles2,
        labels=labels1 + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=True,
    )

    ax1.grid(visible=True)
    fig.tight_layout()

    if output_root_dir is not None:
        output_file_path: pathlib.Path = pathlib.Path(
            output_root_dir,
            f"{title.replace(' ', '_').lower()}.pdf",
        )
        fig.savefig(
            fname=output_file_path,
            dpi=300,
            bbox_inches="tight",
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

    # ================================================== #
    # Parse the SetSUMBT log file and print a summary of the DataFrame.
    # ================================================== #

    if main_config.language_model.model_log_file_path is None:
        logger.error(
            msg="The path to the SETSUMBT log file is not set in the configuration.",
        )
        return

    log_filepath = pathlib.Path(
        main_config.language_model.model_log_file_path,
    )

    try:
        parsed_df: pd.DataFrame = parse_log_file(
            file_path=log_filepath,
        )
    except FileNotFoundError as e:
        logger.exception(
            msg=f"Error reading log file: {e}",  # noqa: G004 - low overhead
        )
        return

    # ================================================== #
    # Saved parsed DataFrame
    # ================================================== #

    output_root_dir: pathlib.Path = pathlib.Path(
        embeddings_path_manager.saved_plots_dir_absolute_path,
        "task_performance_analysis",
        "model_performance_metrics",
        main_config.language_model.get_config_description(
            description_type=DescriptionType.LONG,
        ),
    )
    output_root_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file_path: pathlib.Path = pathlib.Path(
        output_root_dir,
        "parsed_setsumbt_run_df.csv",
    )
    parsed_df.to_csv(
        path_or_buf=output_file_path,
        index=False,
    )

    # ================================================== #
    # Create plot of the performance
    # ================================================== #

    # Specify which columns to plot on the primary and secondary axes.
    primary_metrics: list[str] = [
        "loss",
        "Joint Goal Accuracy",
        "Slot F1 Score",
        "Slot Precision",
        "Slot Recall",
    ]
    secondary_metrics: list[str] = [
        "Joint Goal ECE",
        "Joint Goal L2-Error",
        "Joint Goal L2-Error Ratio",
    ]

    highlight_columns: list[str] = [
        "Joint Goal Accuracy",
        "Slot F1 Score",
        "Slot Precision",
        "Slot Recall",
    ]

    # Call the plotting function.
    # Leave primary_ylim and secondary_ylim as None for auto-scaling,
    # or supply a tuple (min, max) if you want fixed limits.
    plot_performance_metrics(
        df=parsed_df,
        x_col="checkpoint",
        primary_y_cols=primary_metrics,
        secondary_y_cols=secondary_metrics,
        title="Development of Performance Metrics through Checkpoints",
        xlabel="Checkpoint",
        primary_ylabel="Metric (Primary Scale)",
        secondary_ylabel="Metric (Secondary Scale)",
        primary_ylim=None,  # e.g., None for auto
        secondary_ylim=None,  # e.g., None for auto
        output_root_dir=output_root_dir,
        figsize=(20, 10),
        highlight_best=highlight_columns,
        verbosity=verbosity,
        logger=logger,
    )


if __name__ == "__main__":
    # Note: See the VSCode launch configurations for an example of how to run this script.

    setup_omega_conf()

    main()
