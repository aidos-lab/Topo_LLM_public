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

"""Functions to create plots for the sampling method comparison analysis."""

import logging
import pathlib
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import AutoLocator, MultipleLocator

from topollm.logging.log_dataframe_info import log_dataframe_info
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def make_multiple_line_plots(
    array: np.ndarray,
    sample_sizes: np.ndarray,
    additional_title: str | None = None,
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
        additional_title:
            Additional title to add to the plot.
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

    if additional_title:
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
    additional_title: str | None = None,
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

    if additional_title:
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


def create_boxplot_of_mean_over_different_sampling_seeds(
    subset_local_estimates_df: pd.DataFrame,
    plot_save_path: pathlib.Path | None = None,
    raw_data_save_path: pathlib.Path | None = None,
    x_column_name: str = "local_estimates_samples",
    y_column_name: str = "array_data_truncated_mean",
    seed_column_name: str = "data_prep_sampling_seed",
    fixed_params_text: str | None = None,
    additional_title: str | None = None,
    *,
    y_min: float | None = 6.5,
    y_max: float | None = 15.5,
    show_plot: bool = False,
    connect_points: bool = False,
    logger: logging.Logger = default_logger,
) -> None:
    """Create a boxplot of the the measurement over different sampling seeds."""
    # If the DataFrame is empty, return early
    if subset_local_estimates_df.empty:
        logger.warning(
            msg="Empty DataFrame provided. Cannot create boxplot.",
        )
        return

    plt.figure(figsize=(10, 6))

    # Set the fixed y-axis limits
    if y_min is not None and y_max is not None:
        plt.ylim(
            y_min,
            y_max,
        )
    else:
        # Automatically adjust the y-axis limits
        plt.autoscale(
            axis="y",
        )

    # Automatically set major and minor tick locators
    plt.gca().yaxis.set_major_locator(
        locator=AutoLocator(),
    )  # Auto-adjust major ticks
    plt.gca().yaxis.set_minor_locator(
        locator=MultipleLocator(0.1),
    )  # Set minor ticks for finer grid

    # Enable the grid with different styling for major and minor lines
    plt.grid(
        which="major",
        axis="y",
        color="gray",
        linestyle="-",
        linewidth=0.6,
        alpha=0.5,
    )  # Major grid lines
    plt.grid(
        which="minor",
        axis="y",
        color="gray",
        linestyle="--",
        linewidth=0.3,
        alpha=0.3,
    )  # Minor grid lines

    # Create boxplot and stripplot
    sns.boxplot(
        x=x_column_name,
        y=y_column_name,
        data=subset_local_estimates_df,
    )
    sns.stripplot(
        x=x_column_name,
        y=y_column_name,
        data=subset_local_estimates_df,
        color="red",
        jitter=False,
        dodge=True,
        marker="o",
        alpha=0.5,
    )

    # Convert the x-axis column to categorical for proper ordering
    subset_local_estimates_df[x_column_name] = pd.Categorical(
        subset_local_estimates_df[x_column_name],
        ordered=True,
    )

    # Connect the points from the same seed across different samples if requested
    if connect_points:
        unique_seeds = subset_local_estimates_df[seed_column_name].unique()
        # Use modern colormap access without resampling argument
        colormap = plt.colormaps.get_cmap("tab20")

        for idx, seed in enumerate(unique_seeds):
            seed_data = subset_local_estimates_df[subset_local_estimates_df[seed_column_name] == seed]
            # Sort seed_data by 'local_estimates_samples' for consistent plotting
            seed_data = seed_data.sort_values(by=x_column_name)

            # Plot lines connecting the same seed points
            plt.plot(
                seed_data[x_column_name].cat.codes,  # Using categorical codes for proper x ordering
                seed_data[y_column_name],
                linestyle="-",
                linewidth=1,
                alpha=0.7,
                color=colormap(idx / len(unique_seeds)),  # Use a different color for each seed
                label=f"Seed {seed}" if idx < 2 else "",  # Labeling only the first few for readability
            )

    # Adding additional information about the fixed parameters in the plot
    if fixed_params_text is not None:
        plt.text(
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

    if additional_title:
        add_subtitle(
            additional_title=additional_title,
        )
    elif "path" in subset_local_estimates_df.columns:
        # If no additional title is provided, use the first file path as a subtitle
        add_subtitle(
            additional_title=str(object=subset_local_estimates_df["path"].iloc[0]),
        )
    else:
        logger.warning(
            msg="No 'path' column found in the DataFrame. Could not add additional_title.",
        )

    # Save plot to the specified path if provided
    if plot_save_path is not None:
        plot_save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        plt.savefig(
            plot_save_path,
            bbox_inches="tight",
        )
    if raw_data_save_path is not None:
        raw_data_save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        subset_local_estimates_df.to_csv(
            path_or_buf=raw_data_save_path,
        )

    # Show plot if needed
    if show_plot:
        plt.show()


def generate_fixed_params_text(
    filters_dict: dict[str, Any],
) -> str:
    """Generate a string representation of the fixed parameters used for filtering.

    Args:
        filters_dict:
            A dictionary of column names and corresponding values used for filtering.

    Returns:
        str:
            A formatted string suitable for display in the plot.

    """
    return "\n".join([f"{key}: {value}" for key, value in filters_dict.items()])


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


def analyze_and_plot_influence_of_local_estimates_samples(
    df: pd.DataFrame,
    n_neighbors: int,
    selected_subsample_dict: dict,
    array_data_column_name: str,
    additional_title: str | None = None,
    *,
    y_min: float | None = 6.5,
    y_max: float | None = 15.5,
    show_plot: bool = False,
    plot_save_path: pathlib.Path | None = None,
    raw_data_save_path: pathlib.Path | None = None,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Analyze and visualize the influence of 'local_estimates_samples'.

    We plot array_data_truncated_mean and array_data_truncated_std
    for a given value of 'n_neighbors', while keeping other parameters fixed.
    """
    # Update the fixed parameter values to the chosen 'n_neighbors'
    updated_selected_subsample_dict = selected_subsample_dict.copy()
    updated_selected_subsample_dict["n_neighbors"] = n_neighbors

    # Filter the DataFrame with the updated fixed values,
    # handling cases where deduplication is None
    filtered_df = df[
        (df["data_prep_sampling_method"] == updated_selected_subsample_dict["data_prep_sampling_method"])
        & (
            df["deduplication"].isna()
            if updated_selected_subsample_dict["deduplication"] is None
            else df["deduplication"] == updated_selected_subsample_dict["deduplication"]
        )
        & (df["n_neighbors"] == updated_selected_subsample_dict["n_neighbors"])
        & (df["data_prep_sampling_seed"] == updated_selected_subsample_dict["data_prep_sampling_seed"])
        & (df["data_prep_sampling_samples"] == updated_selected_subsample_dict["data_prep_sampling_samples"])
    ]

    if verbosity >= Verbosity.NORMAL:
        log_dataframe_info(
            df=filtered_df,
            df_name="filtered_df",
            logger=logger,
        )

    # Ensure there are enough valid data points to proceed
    if filtered_df.empty:
        logger.warning(
            msg=f"No valid data to plot for {n_neighbors = }",  # noqa: G004 - low overhead
        )
        return

    # Sort by 'local_estimates_samples' for consistent plotting
    sorted_df = filtered_df.sort_values(
        by="local_estimates_samples",
    )

    # Convert columns to NumPy arrays for plotting
    sample_size_array = sorted_df["local_estimates_samples"].to_numpy(dtype=float)
    mean_array = sorted_df[f"{array_data_column_name}_truncated_mean"].to_numpy(dtype=float)
    std_array = sorted_df[f"{array_data_column_name}_truncated_std"].to_numpy(dtype=float)

    # Plotting the analysis
    plt.figure(
        figsize=(10, 6),
    )

    # Set the fixed y-axis limits
    if y_min is not None and y_max is not None:
        if verbosity >= Verbosity.VERBOSE:
            logger.debug(
                msg="Setting fixed y-axis limits",
            )
        plt.ylim(
            y_min,
            y_max,
        )
    else:
        # Automatically adjust the y-axis limits
        if verbosity >= Verbosity.VERBOSE:
            logger.debug(
                msg="Automatically adjusting y-axis limits",
            )
        plt.autoscale(
            axis="y",
        )

    plt.plot(
        sample_size_array,
        mean_array,
        color="b",
        label="Truncated Mean",
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
        xlabel="local_estimates_samples",
    )
    plt.ylabel(
        ylabel="Truncated Mean",
    )
    plt.title(
        label=f"Influence of 'local_estimates_samples' on Truncated Mean and Std (n_neighbors={n_neighbors})",
    )

    if additional_title:
        add_subtitle(
            additional_title=additional_title,
        )
    else:
        # If no additional title is provided, use the first file path as a subtitle
        add_subtitle(
            additional_title=str(object=sorted_df["path"].iloc[0]),
        )

    # Adding additional information about the fixed parameters in the plot
    fixed_params_text = "\n".join([f"{key}: {value}" for key, value in updated_selected_subsample_dict.items()])
    plt.text(
        x=0.02,
        y=0.95,
        s=f"Fixed Parameters:\n{fixed_params_text}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox={
            "boxstyle": "round",
            "facecolor": "wheat",
            "alpha": 0.5,
        },
    )

    # Add a legend
    plt.legend()
    plt.tight_layout()

    # Save the plot if save_path is provided
    if plot_save_path:
        plot_save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        plt.savefig(
            plot_save_path,
            format="pdf",
        )
    # Save the raw data if save_path is provided
    if raw_data_save_path:
        raw_data_save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        sorted_df.to_csv(
            path_or_buf=raw_data_save_path,
        )

    # Show the plot if requested
    if show_plot:
        plt.show()
