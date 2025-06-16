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


"""Plotting functions for violin plots."""

import logging
import pathlib
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from topollm.plotting.plot_size_config import PlotSizeConfigFlat
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


@dataclass
class TicksAndLabels:
    """Container for ticks and labels."""

    xlabel: str
    ylabel: str
    xticks_labels: list[str]


def make_distribution_violinplots_from_extracted_arrays(
    extracted_arrays: list[np.ndarray],
    ticks_and_labels: TicksAndLabels,
    plot_size_config: PlotSizeConfigFlat,
    *,
    print_means_and_medians_and_stds: bool = True,
    fixed_params_text: str | None = None,
    base_model_model_partial_name: str | None = None,
    plots_output_dir: pathlib.Path | None = None,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> tuple:
    """Create a violin plot."""
    (
        fig,
        ax,
    ) = plt.subplots(
        figsize=(
            plot_size_config.output_pdf_width / 100,
            plot_size_config.output_pdf_height / 100,
        ),
    )

    # # # #
    # Plot violin plot
    ax.violinplot(
        dataset=extracted_arrays,
        showmeans=True,
        showextrema=True,
        showmedians=True,
    )
    ax.set_title(
        label="Violin plot",
    )

    # Add text to the plot with means and medians
    if print_means_and_medians_and_stds:
        for i, extracted_array in enumerate(extracted_arrays):
            mean: np.floating = np.mean(extracted_array)
            median: np.floating = np.median(extracted_array)
            std: np.floating = np.std(extracted_array, ddof=1)

            ax.text(
                x=i + 1,
                y=float(mean),
                s=f"Mean: {mean:.2f}\nMedian: {median:.2f}\nStd(ddof=1): {std:.2f}",
                fontsize=6,
                verticalalignment="bottom",
                horizontalalignment="center",
            )

    # Add info about the base model if available into the bottom left corner of the plot
    if base_model_model_partial_name is not None:
        ax.text(
            x=0.01,
            y=0.01,
            s=f"{base_model_model_partial_name=}",
            transform=plt.gca().transAxes,
            fontsize=6,
            verticalalignment="bottom",
            horizontalalignment="left",
            bbox={
                "boxstyle": "round",
                "facecolor": "wheat",
                "alpha": 0.3,
            },
        )

    # Adding horizontal grid lines
    ax.yaxis.grid(
        visible=True,
    )

    # Use the model checkpoints to set the xticks
    ax.set_xticks(
        ticks=[y + 1 for y in range(len(extracted_arrays))],
        labels=ticks_and_labels.xticks_labels,
    )

    ax.set_xlabel(
        xlabel=ticks_and_labels.xlabel,
    )
    ax.set_ylabel(
        ylabel=ticks_and_labels.ylabel,
    )

    # Set the y-axis limits
    if plot_size_config.y_min is not None:
        ax.set_ylim(
            bottom=plot_size_config.y_min,
        )
    if plot_size_config.y_max is not None:
        ax.set_ylim(
            top=plot_size_config.y_max,
        )

    if fixed_params_text is not None:
        ax.text(
            x=1.05,
            y=0.25,
            s=f"Fixed Parameters:\n{fixed_params_text}",
            transform=plt.gca().transAxes,
            fontsize=6,
            verticalalignment="top",
            bbox={
                "boxstyle": "round",
                "facecolor": "wheat",
                "alpha": 0.3,
            },
        )

    # Saving the plot
    if plots_output_dir is not None:
        plot_name: str = f"violinplot_{plot_size_config.y_min}_{plot_size_config.y_max}"
        plot_output_path: pathlib.Path = pathlib.Path(
            plots_output_dir,
            f"{plot_name}.pdf",
        )
        plot_output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving plot to {plot_output_path = } ...",  # noqa: G004 - low overhead
            )
        fig.savefig(
            fname=plot_output_path,
            bbox_inches="tight",
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving plot to {plot_output_path = } DONE",  # noqa: G004 - low overhead
            )

    return fig, ax


def make_distribution_boxplots_from_extracted_arrays(
    extracted_arrays: list[np.ndarray],
    ticks_and_labels: TicksAndLabels,
    fixed_params_text: str,
    plots_output_dir: pathlib.Path,
    plot_size_config: PlotSizeConfigFlat,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Create a boxplot."""
    (
        fig,
        ax,
    ) = plt.subplots(
        figsize=(
            plot_size_config.output_pdf_width / 100,
            plot_size_config.output_pdf_height / 100,
        ),
    )

    # Plot box plot
    ax.boxplot(
        x=extracted_arrays,
    )
    ax.set_title(
        label="Box plot",
    )

    # adding horizontal grid lines
    ax.yaxis.grid(
        visible=True,
    )

    # Use the model checkpoints to set the xticks
    ax.set_xticks(
        ticks=[y + 1 for y in range(len(extracted_arrays))],
        labels=ticks_and_labels.xticks_labels,
    )

    ax.set_xlabel(
        xlabel=ticks_and_labels.xlabel,
    )
    ax.set_ylabel(
        ylabel=ticks_and_labels.ylabel,
    )

    # Set the y-axis limits
    if plot_size_config.y_min is not None:
        ax.set_ylim(
            bottom=plot_size_config.y_min,
        )
    if plot_size_config.y_max is not None:
        ax.set_ylim(
            top=plot_size_config.y_max,
        )

    if fixed_params_text is not None:
        ax.text(
            x=1.05,
            y=0.25,
            s=f"Fixed Parameters:\n{fixed_params_text}",
            transform=plt.gca().transAxes,
            fontsize=6,
            verticalalignment="top",
            bbox={
                "boxstyle": "round",
                "facecolor": "wheat",
                "alpha": 0.3,
            },
        )

    plot_name: str = f"boxplot_{plot_size_config.y_min}_{plot_size_config.y_max}"
    plot_output_path: pathlib.Path = pathlib.Path(
        plots_output_dir,
        f"{plot_name}.pdf",
    )
    plot_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Saving plot to {plot_output_path = } ...",  # noqa: G004 - low overhead
        )
    fig.savefig(
        fname=plot_output_path,
        bbox_inches="tight",
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Saving plot to {plot_output_path = } DONE",  # noqa: G004 - low overhead
        )
