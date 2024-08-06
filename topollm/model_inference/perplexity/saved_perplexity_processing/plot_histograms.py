import logging
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
            Each setting includes 'scale' (optional tuple of min and max for x-axis) and 'bins' (int for number of bins).
            If None, the histograms will be automatically scaled and use default bin count of 30.

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


def save_plot(
    figure: matplotlib.figure.Figure,
    path: pathlib.Path,
) -> None:
    """Save the given plot to a specified path.

    Args:
    ----
        figure: The matplotlib figure object to save.
        path: The path where the plot should be saved.

    """
    figure.savefig(
        path,
    )
