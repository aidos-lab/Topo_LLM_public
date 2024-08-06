# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (ruppik@hhu.de)
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

"""Plotting functions for histograms."""

import logging
import os
import pathlib
from dataclasses import dataclass

import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd

from topollm.typing.enums import Verbosity

default_logger = logging.getLogger(__name__)


@dataclass
class HistogramSettings:
    """Settings for plotting histograms."""

    scale: tuple[float | None, float | None] | None = None
    bins: int | None = 30


def plot_histograms(
    df: pd.DataFrame,
    settings: dict[str, HistogramSettings] | None = None,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> matplotlib.figure.Figure | None:
    """Plot histograms for specified columns of a dataframe with optional manual scaling and configurable bins.

    Args:
    ----
        df:
            The dataframe containing the data.
        settings:
            Dictionary specifying the settings for each column.
            Each setting includes
            'scale' (optional tuple of min and max for x-axis)
            and 'bins' (int for number of bins).
            If None, the histograms will be automatically scaled and use default bin count of 30.
        verbosity:
            The verbosity level.
        logger:
            The logger.

    """
    columns_to_plot = df.select_dtypes(include="number").columns.tolist() if settings is None else list(settings.keys())

    num_columns = len(columns_to_plot)
    num_cols = 3  # Number of columns per row
    num_rows = (num_columns + num_cols - 1) // num_cols  # Calculate the number of rows needed

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"Number of columns to plot: {num_columns = }",  # noqa: G004 - low overhead
        )
        logger.info(
            f"Number of rows: {num_rows = }",  # noqa: G004 - low overhead
        )
        logger.info(
            f"Number of columns per row: {num_cols = }",  # noqa: G004 - low overhead
        )

    fig, axs = plt.subplots(
        num_rows,
        num_cols,
        figsize=(18, 6 * num_rows),
    )
    axs = axs.flatten()  # Flatten in case of multiple rows

    if columns_to_plot == []:
        logger.warning(
            "No columns to plot. The function will return None.",
        )
        return None

    i = 0  # Initialize index i to avoid error in case of empty columns_to_plot
    for i, column in enumerate(columns_to_plot):
        ax = axs[i]
        if settings and column in settings:
            scale = settings[column].scale
            bins = settings[column].bins
            if scale is not None:
                ax.hist(
                    df[column],
                    bins=bins,
                    alpha=0.7,
                    color="blue",
                    edgecolor="black",
                    range=scale,
                )
                ax.set_xlim(scale)  # Set the x-axis scale
            else:
                ax.hist(
                    df[column],
                    bins=bins,
                    alpha=0.7,
                    color="blue",
                    edgecolor="black",
                )
        else:
            ax.hist(
                df[column],
                bins=30,
                alpha=0.7,
                color="blue",
                edgecolor="black",
            )
        ax.set_title(f"Histogram of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")

    # Remove any unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    return fig


@dataclass
class ScatterPlotSettings:
    """Settings for plotting scatter plots."""

    x_scale: tuple[float, float] | None = None
    y_scale: tuple[float, float] | None = None


def create_scatter_plot(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    settings: ScatterPlotSettings | None = None,
) -> matplotlib.figure.Figure | None:
    """Create a scatter plot comparing two columns of a dataframe with optional manual scaling.

    Args:
    ----
        df (pd.DataFrame): The dataframe containing the data.
        x_column (str): The column name to be used for the x-axis.
        y_column (str): The column name to be used for the y-axis.
        settings (ScatterPlotSettings | None): Settings for manual scaling of the axes.

    Returns:
    -------
        plt.Figure: The matplotlib figure object containing the scatter plot.

    """
    fig, ax = plt.subplots(
        figsize=(10, 6),
    )

    if settings:
        x_scale = settings.x_scale
        y_scale = settings.y_scale

        if x_scale is not None:
            ax.set_xlim(x_scale)
        if y_scale is not None:
            ax.set_ylim(y_scale)

    ax.scatter(
        df[x_column],
        df[y_column],
        alpha=0.7,
        color="blue",
        edgecolor="black",
    )
    ax.set_title(f"Scatter Plot of {x_column} vs {y_column}")
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)

    return fig


def save_plot(
    figure: matplotlib.figure.Figure,
    path: os.PathLike,
) -> None:
    """Save the given plot to a specified path.

    Args:
    ----
        figure: The matplotlib figure object to save.
        path: The path where the plot should be saved.

    """
    pathlib.Path(path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        path,
    )
