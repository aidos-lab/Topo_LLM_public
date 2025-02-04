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

"""Plot the development of a selected y-axis column over a selected x-axis column, grouping by a categorical column."""

import logging
import pathlib

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def generate_color_mapping(
    df: pd.DataFrame,
    group_column: str,
) -> dict:
    """Generate a consistent color mapping for unique values in a categorical column."""
    unique_groups: list = sorted(df[group_column].unique())  # Sort for consistent ordering
    color_list = list(
        mcolors.TABLEAU_COLORS.values(),
    )  # Use Matplotlib's Tableau colors
    color_mapping: dict = {group: color_list[i % len(color_list)] for i, group in enumerate(iterable=unique_groups)}
    return color_mapping


def line_plot_grouped_by_categorical_column(
    df: pd.DataFrame,
    output_folder: pathlib.Path | None = None,
    *,
    plot_name: str = "line_plot",
    subtitle_text: str | None = None,
    x_column: str = "model_checkpoint",
    y_column: str = "loss_mean",
    group_column: str = "data_full",
    color_mapping: dict | None = None,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    output_pdf_width: int = 2500,
    output_pdf_height: int = 1500,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Plot the development of a selected y-axis column over a selected x-axis column, grouping by a categorical column.

    This can for example be used to plot the development of the loss over the model checkpoints for different datasets.
    """
    # Ensure x_column exists and fill missing values with -1
    df[x_column] = df.get(
        key=x_column,
        default=pd.Series([-1] * len(df)),
    )
    df[x_column] = df[x_column].fillna(
        value=-1,
    )

    # Sort by x_column for better visualization
    df = df.sort_values(
        by=x_column,
    )

    # Plotting
    fig = plt.figure(
        figsize=(output_pdf_width / 100, output_pdf_height / 100),  # Convert pixels to inches
    )

    # Generate color mapping if not provided
    if color_mapping is None:
        color_mapping = generate_color_mapping(
            df=df,
            group_column=group_column,
        )

    # Plot each group separately
    for group_value in df[group_column].unique():
        subset = df[df[group_column] == group_value]
        plt.plot(
            subset[x_column],
            subset[y_column],
            marker="o",
            linestyle="-",
            color=color_mapping.get(
                group_value,
                "black",
            ),  # Use mapped color or default to black
            label=f"{group_column}={group_value}",
        )

    # Labels and title
    plt.xlabel(
        xlabel=x_column,
    )
    plt.ylabel(
        ylabel=y_column,
    )
    plt.title(
        label=f"Development of {y_column = } over {x_column = } grouped by {group_column = }",
    )

    # Set the x-axis limits
    if x_min is not None and x_max is not None:
        plt.xlim(
            x_min,
            x_max,
        )

    # Set the y-axis limits
    if y_min is not None and y_max is not None:
        plt.ylim(
            y_min,
            y_max,
        )

    if subtitle_text is not None:
        plt.suptitle(
            t=subtitle_text,
        )

    # Legend at the bottom
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
    )

    # Grid for better readability
    plt.grid(
        visible=True,
    )

    # # # # # # # # # # # # # #
    # Save plot and raw data
    if output_folder is not None:
        output_folder = pathlib.Path(
            output_folder,
        )
        output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Save as PDF
        output_file = pathlib.Path(
            output_folder,
            f"{plot_name}.pdf",
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving plot to {output_file} ...",  # noqa: G004 - low overhead
            )
        plt.savefig(
            output_file,
            bbox_inches="tight",
            format="pdf",
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving plot to {output_file} DONE",  # noqa: G004 - low overhead
            )

        # Save the raw data
        output_file_raw_data = pathlib.Path(
            output_folder,
            f"{plot_name}_raw_data.csv",
        )
        output_file_raw_data.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving raw data to {output_file_raw_data} ...",  # noqa: G004 - low overhead
            )
        df.to_csv(
            path_or_buf=output_file_raw_data,
            index=False,
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving raw data to {output_file_raw_data} DONE",  # noqa: G004 - low overhead
            )
