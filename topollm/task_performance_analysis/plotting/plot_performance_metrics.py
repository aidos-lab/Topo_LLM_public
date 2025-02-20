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

"""Plot performance metrics versus checkpoints using configurable y-axis groups and scales."""

import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


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
    loaded_sorted_local_estimates_data: list[dict] | None = None,
    array_key_name: str = "file_data",
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Plot performance metrics versus checkpoints using configurable y-axis groups and scales.

    This function creates a line plot from the given DataFrame showing the development of
    performance measures through training checkpoints. You can configure which metrics
    (columns) are plotted on the primary y-axis and which (if any) on a secondary y-axis.
    In addition, you can optionally specify fixed y-axis limits. If set to None, the scale
    is determined automatically.

    Args:
        df :
            DataFrame containing the log data. Must include the x-axis column.
        x_col:
            Name of the column to use as the x-axis (default "checkpoint").
        primary_y_cols:
            List of column names to plot on the primary y-axis.
            If None, no primary lines are plotted.
        secondary_y_cols:
            List of column names to plot on the secondary y-axis.
        title:
            Title of the plot.
        xlabel:
            Label for the x-axis.
        primary_ylabel:
            Label for the primary y-axis.
        secondary_ylabel:
            Label for the secondary y-axis.
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
        raise ValueError(
            msg,
        )

    # Create the figure and the primary axis.
    (
        fig,
        ax1,
    ) = plt.subplots(
        figsize=figsize,
    )

    # -------------------------------------------------------------
    # Plot metrics for the primary y-axis.
    # -------------------------------------------------------------
    if primary_y_cols:
        for col in primary_y_cols:
            if col not in df.columns:
                logger.info(
                    msg=f"Warning: '{col}' not found in DataFrame; skipping.",  # noqa: G004 - low overhead
                )
                continue
            ax1.plot(
                df[x_col],
                df[col],
                marker="o",
                label=col,
            )
        ax1.set_ylabel(
            ylabel=primary_ylabel,
        )
        if primary_ylim is not None:
            ax1.set_ylim(
                bottom=primary_ylim,
            )

    # -------------------------------------------------------------
    # Plot metrics for the secondary y-axis if provided.
    # -------------------------------------------------------------
    if secondary_y_cols:
        ax2 = ax1.twinx()
        for col in secondary_y_cols:
            if col not in df.columns:
                logger.info(
                    msg=f"Warning: '{col}' not found in DataFrame; skipping.",  # noqa: G004 - low overhead
                )
                continue
            ax2.plot(
                df[x_col],
                df[col],
                marker="s",
                linestyle="--",
                label=col,
            )
        ax2.set_ylabel(secondary_ylabel)
        if secondary_ylim is not None:
            ax2.set_ylim(secondary_ylim)
    else:
        ax2 = None

    # -------------------------------------------------------------
    # Highlight both the maximum and minimum points for selected columns.
    # -------------------------------------------------------------
    if highlight_best is not None:
        for col in highlight_best:
            if col not in df.columns:
                logger.info(
                    msg=f"Warning: '{col}' not found in DataFrame for highlighting; skipping.",  # noqa: G004 - low overhead
                )
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

    # -------------------------------------------------------------
    # Add violin plots for the local estimates, if provided.
    # -------------------------------------------------------------
    if loaded_sorted_local_estimates_data is not None:
        for single_dict in loaded_sorted_local_estimates_data:
            ckpt_raw = single_dict["model_checkpoint"]
            local_estimates = single_dict[array_key_name]
            # Convert checkpoint to float if it's stored as a string.
            try:
                ckpt_val = float(ckpt_raw)
            except ValueError:
                # If it's not parseable, skip or handle differently
                logger.info(f"Skipping checkpoint {ckpt_raw} - cannot convert to float.")
                continue

            # Create a violin plot at the x-position = ckpt_val
            # You can tweak 'widths', 'showmeans', 'showextrema', etc. as needed:
            #
            # Note: We need to set the widths to a value in the thousands,
            # because the x-axis is the global step of the checkpoints,
            # and otherwise the violins would not be visible.
            parts: dict = ax1.violinplot(
                dataset=local_estimates,
                positions=[ckpt_val],
                widths=2000,
                showmeans=True,
                showextrema=True,
                showmedians=False,
            )

            # Make all the violin plots the same color (e.g., black).
            # Note that since we are creating each violin plot separately,
            # by default matplotlib will cycle through colors.
            # We can override this by setting the color of the parts of the violin plot.
            if "cmeans" in parts:
                parts["cmeans"].set_color("black")
                parts["cmeans"].set_linewidth(2.0)

            if "cmins" in parts:
                parts["cmins"].set_color("black")
            if "cmaxes" in parts:
                parts["cmaxes"].set_color("black")
            if "cbars" in parts:
                parts["cbars"].set_color("black")

            # Optionally unify the violin body color/alpha:
            for pc in parts["bodies"]:
                pc.set_facecolor("gray")
                pc.set_alpha(0.8)

        # Note: We do not want to rescale the x-axis,
        # since the violin plot of the base model will be placed at x-coordinate -1.

        pass  # TODO: Here for setting breakpoints in debugging, remove later.

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

    ax1.grid(
        visible=True,
    )
    fig.tight_layout()

    if output_root_dir is not None:
        output_file_name: str = f"{title.replace(' ', '_').lower()}.pdf"
        output_file_path: pathlib.Path = pathlib.Path(
            output_root_dir,
            output_file_name,
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving plot to {output_file_path=} ...",  # noqa: G004 - low overhead
            )
        fig.savefig(
            fname=output_file_path,
            dpi=300,
            bbox_inches="tight",
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving plot to {output_file_path=} DONE",  # noqa: G004 - low overhead
            )
