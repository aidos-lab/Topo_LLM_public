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

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_multiple_line_plots(
    array: np.ndarray,
    sample_sizes: np.ndarray,
    additional_title: str = "",
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

    # Add horizontal and vertical grid lines for better readability
    plt.grid(
        visible=True,
        which="both",
        linestyle="--",
        linewidth=0.5,
        color="gray",
    )

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

    add_subtitle(
        additional_title=additional_title,
    )

    if show_plot:
        plt.show()

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(
            save_path,
            format="pdf",
        )


def make_mean_std_plot(
    sorted_df: pd.DataFrame,
    additional_title: str = "",
    *,
    show_plot: bool = False,
    save_path: pathlib.Path | None = None,
) -> None:
    """Create a plot of the means with a standard deviation band.

    Args:
    ----
        sorted_df:
            DataFrame containing 'sample_size', 'mean', and 'std' columns.
        additional_title:
            Additional title to add to the plot.
        show_plot:
            Whether to display the plot.
        save_path:
            The path to save the plot as a PDF.

    """
    # Convert columns to NumPy arrays
    sample_size_array = sorted_df["sample_size"].to_numpy(
        dtype=float,
    )
    mean_array = sorted_df["mean"].to_numpy(
        dtype=float,
    )
    std_array = sorted_df["std"].to_numpy(
        dtype=float,
    )

    # Create the plot
    plt.figure(
        figsize=(10, 6),
    )
    plt.plot(
        sample_size_array,
        mean_array,
        color="b",
        label="Mean",
        marker="o",
    )
    plt.fill_between(
        x=sample_size_array,
        y1=mean_array - std_array,
        y2=mean_array + std_array,
        color="b",
        alpha=0.2,
        label="Standard Deviation",
    )

    # Add horizontal and vertical grid lines for better readability
    plt.grid(
        visible=True,
        which="both",
        linestyle="--",
        linewidth=0.5,
        color="gray",
    )

    # Label the axes and add title
    plt.xlabel(
        xlabel="Sample Size",
    )
    plt.ylabel(
        ylabel="Mean Value",
    )
    plt.title(
        label="Mean and Standard Deviation of Estimates over Sample Sizes",
    )

    add_subtitle(
        additional_title=additional_title,
    )

    # Add a legend
    plt.legend()

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(
            save_path,
            format="pdf",
        )

    # Show the plot if requested
    if show_plot:
        plt.show()


def add_subtitle(
    additional_title: str,
) -> None:
    """Add a subtitle to the current plot."""
    formatted_subtitle: str = "\n".join(
        [
            additional_title[i : i + 100]
            for i in range(
                0,
                len(additional_title),
                100,
            )
        ],
    )

    # Add the subtitle with line breaks and smaller font
    plt.suptitle(
        t=formatted_subtitle,
        fontsize=8,
        wrap=True,
    )
