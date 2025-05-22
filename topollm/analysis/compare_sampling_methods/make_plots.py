# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Matthias Ruppik (mail@ruppik.net)
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
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import AutoLocator, MultipleLocator

from topollm.config_classes.constants import NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS
from topollm.logging.log_dataframe_info import log_dataframe_info
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)

Y_AXIS_LIMITS: dict[
    str,
    tuple[float | None, float | None],
] = {
    "None": (None, None),
    "full_low": (6.5, 18.0),  # full range
    "lower": (6.5, 10.0),  # lower range
    "upper": (12.0, 18.0),  # upper range
    "extended": (6.5, 23.0),  # extended range
}

Y_AXIS_LIMITS_ONLY_FULL: dict[
    str,
    tuple[float | None, float | None],
] = {
    "full": (6.5, 18.0),  # full range
}


@dataclass
class PlotProperties:
    """Dataclass to store properties for the plots."""

    figsize: tuple[float, float] = (26, 10)
    y_min: float | None = 6.5
    y_max: float | None = 15.5


@dataclass
class PlotSavePathCollection:
    """Dataclass to hold the paths for saving plots and raw data."""

    plot: pathlib.Path | None = None
    raw_data: pathlib.Path | None = None
    aggregated_results: pathlib.Path | None = None

    @staticmethod
    def create_from_common_prefix_path(
        common_prefix_path: pathlib.Path | None = None,
        plot_file_name: str = "plot.pdf",
    ) -> "PlotSavePathCollection":
        """Create a PlotSavePathCollection from a common prefix path."""
        if common_prefix_path is not None:
            return PlotSavePathCollection(
                plot=pathlib.Path(
                    common_prefix_path,
                    "plots",
                    plot_file_name,
                ),
                raw_data=pathlib.Path(
                    common_prefix_path,
                    "raw_data",
                    "raw_data.csv",
                ),
                aggregated_results=pathlib.Path(
                    common_prefix_path,
                    "raw_data",
                    "aggregated_results.csv",
                ),
            )
        return PlotSavePathCollection()


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
    plot_save_path_collection: PlotSavePathCollection | None = None,
    x_column_name: str = "local_estimates_samples",
    y_column_name: str = "array_data_truncated_mean",
    seed_column_name: str = "data_prep_sampling_seed",
    array_data_size_column_name: str = "array_data.size",
    fixed_params_text: str | None = None,
    additional_title: str | None = None,
    # Additional optional data points to plot
    model_losses_df: pd.DataFrame | None = None,
    loss_x_column_name: str = "model_checkpoint",
    loss_y_column_name: str = "loss",
    *,
    figsize: tuple[float, float] = (12, 8),
    y_min: float | None = 6.5,
    y_max: float | None = 15.5,
    y_min_additional_plot: float | None = 0.0,
    y_max_additional_plot: float | None = 3.0,
    show_plot: bool = False,
    connect_points: bool = False,
    show_aggregated_stats: bool = True,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Create a boxplot of the the measurement over different sampling seeds."""
    # If the DataFrame is empty, return early
    if subset_local_estimates_df.empty:
        logger.warning(
            msg="Empty DataFrame provided. Cannot create boxplot.",
        )
        return

    # Set default save paths if not provided
    if plot_save_path_collection is None:
        plot_save_path_collection = PlotSavePathCollection()

    # # # #
    # Save the raw data to a CSV if a path is provided.
    # Note: We place this early in the function to ensure the raw data is saved even if the plot fails.
    if plot_save_path_collection.raw_data is not None:
        plot_save_path_collection.raw_data.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        subset_local_estimates_df.to_csv(
            path_or_buf=plot_save_path_collection.raw_data,
        )

    # # # #
    # Aggregating the results

    # Calculate mean and standard deviation for each unique value in the x-axis column
    grouped_stats: pd.DataFrame = (
        subset_local_estimates_df.groupby(
            by=x_column_name,
            observed=True,
        )
        .agg(
            mean_value=(y_column_name, "mean"),
            std_value=(y_column_name, "std"),
            array_data_size=(array_data_size_column_name, lambda x: x.min()),
            number_of_seeds=(seed_column_name, "nunique"),
            number_of_elements=("path", "count"),
        )
        .reset_index()
    )

    # Log the calculated values
    if verbosity >= Verbosity.NORMAL:
        log_dataframe_info(
            df=grouped_stats,
            df_name="grouped_stats",
            logger=logger,
        )

    # Save the aggregated results to a CSV if a path is provided
    if plot_save_path_collection.aggregated_results is not None:
        plot_save_path_collection.aggregated_results.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        grouped_stats.to_csv(
            path_or_buf=plot_save_path_collection.aggregated_results,
            index=False,
        )

    # Extract unique x-axis checkpoints from the subset_local_estimates_df and convert to categorical with sorted order
    # Convert the x-axis column to categorical for proper ordering
    unique_checkpoints: list = sorted(
        subset_local_estimates_df[x_column_name].unique(),
    )
    subset_local_estimates_df[x_column_name] = pd.Categorical(
        values=subset_local_estimates_df[x_column_name],
        categories=unique_checkpoints,
        ordered=True,
    )

    # Sort subset_local_estimates_df by the ordered categories
    subset_local_estimates_df = subset_local_estimates_df.sort_values(
        by=x_column_name,
    )

    # # # #
    # Plotting
    # Plotting with a secondary y-axis for loss values
    (
        fig,
        ax1,
    ) = plt.subplots(
        figsize=figsize,
    )

    # Plot boxplot and stripplot on the primary y-axis
    sns.boxplot(
        x=x_column_name,
        y=y_column_name,
        data=subset_local_estimates_df,
        ax=ax1,
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
        ax=ax1,
    )

    # Set y-axis limits for the boxplot
    if y_min is not None and y_max is not None:
        # Set the fixed y-axis limits
        ax1.set_ylim(
            bottom=y_min,
            top=y_max,
        )
    else:
        # Automatically adjust the y-axis limits
        ax1.autoscale(
            axis="y",
        )

    ax1.set_xlabel(
        xlabel=x_column_name,
    )
    ax1.set_ylabel(
        ylabel=y_column_name,
    )
    # Rotate x-axis labels 45 degrees for better readability
    ax1.tick_params(
        axis="x",
        rotation=45,
    )

    # Enable grid on primary y-axis.
    # Automatically set major and minor tick locators.
    ax1.yaxis.set_major_locator(
        locator=AutoLocator(),
    )  # Auto-adjust major ticks
    ax1.yaxis.set_minor_locator(
        locator=MultipleLocator(base=0.1),
    )  # Set minor ticks for finer grid
    # Enable the grid with different styling for major and minor lines
    ax1.grid(
        which="major",
        axis="y",
        color="gray",
        linestyle="-",
        linewidth=0.6,
        alpha=0.5,
    )  # Major grid lines
    ax1.grid(
        which="minor",
        axis="y",
        color="gray",
        linestyle="--",
        linewidth=0.3,
        alpha=0.3,
    )  # Minor grid lines

    # Add secondary y-axis for filtered loss values if provided
    if model_losses_df is not None:
        add_secondary_loss_plot(
            ax1=ax1,
            model_losses_df=model_losses_df,
            unique_checkpoints=unique_checkpoints,
            loss_x_column_name=loss_x_column_name,
            loss_y_column_name=loss_y_column_name,
            y_min_additional_plot=y_min_additional_plot,
            y_max_additional_plot=y_max_additional_plot,
        )

    # Connect the points from the same seed across different samples if requested
    if connect_points:
        number_of_seeds_to_label = 2

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
                label=f"Seed {seed}"
                if idx < number_of_seeds_to_label
                else "",  # Labeling only the first few for readability
            )

    # Display aggregated statistics as text boxes aligned to the x-axis
    if show_aggregated_stats:
        for _, row in grouped_stats.iterrows():
            x_position = subset_local_estimates_df[x_column_name].cat.categories.get_loc(
                key=row[x_column_name],
            )
            # Check that the x_position is an integer
            if not isinstance(
                x_position,
                int,
            ):
                logger.warning(
                    msg=f"Could not find x_position for {row[x_column_name] = }. Skipping stats display.",  # noqa: G004 - low overhead
                )
                continue

            # Dynamically build the text to include all stats columns
            stats_text = "\n".join(
                [
                    f"{col}: {row[col]:.2f}"
                    if isinstance(
                        row[col],
                        int | float,
                    )
                    else f"{col}: {row[col]}"
                    for col in grouped_stats.columns
                    if col != x_column_name
                ],
            )

            # Set y_max if it is None
            y_max = y_max if y_max is not None else ax1.get_ylim()[1]

            ax1.text(
                x=x_position,
                y=y_max - 0.2,  # Place it near the top of the y-axis
                s=stats_text,
                ha="center",
                va="top",
                fontsize=6,  # Set font size to smaller for compact display
                bbox={
                    "boxstyle": "round",
                    "facecolor": "wheat",
                    "alpha": 0.3,
                },
            )

    # # # #
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
    if plot_save_path_collection.plot is not None:
        plot_save_path_collection.plot.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        plt.savefig(
            plot_save_path_collection.plot,
            bbox_inches="tight",
        )

    # Show plot if needed
    if show_plot:
        plt.show()


def add_secondary_loss_plot(
    ax1: plt.Axes,  # type: ignore - no type checking for matplotlib
    model_losses_df: pd.DataFrame,
    unique_checkpoints: list[str],
    loss_x_column_name: str = "model_checkpoint",
    loss_y_column_name: str = "loss",
    y_min_additional_plot: float | None = 0.0,
    y_max_additional_plot: float | None = 3.0,
) -> plt.Axes:  # type: ignore - no type checking for matplotlib
    # Filter the model_losses_df to only include the checkpoints present in subset_local_estimates_df
    filtered_losses_df = model_losses_df[model_losses_df[loss_x_column_name].isin(values=unique_checkpoints)]

    # Convert the `loss_x_column_name` column to categorical using the same categories as subset_local_estimates_df
    filtered_losses_df[loss_x_column_name] = pd.Categorical(
        filtered_losses_df[loss_x_column_name],
        categories=unique_checkpoints,
        ordered=True,
    )

    # Sort filtered_losses_df by the ordered categories
    filtered_losses_df = filtered_losses_df.sort_values(
        by=loss_x_column_name,
    )

    # Create a secondary y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel(
        ylabel=loss_y_column_name,
        color="blue",
    )

    # Plot filtered loss values on the secondary y-axis using categorical values directly without conversion
    ax2.plot(
        filtered_losses_df[loss_x_column_name].astype(str),
        filtered_losses_df[loss_y_column_name],
        linestyle="--",
        linewidth=2,
        color="blue",
        alpha=0.7,
        label="Loss (Filtered Checkpoints)",
    )

    # Set y-axis properties for loss axis
    ax2.tick_params(
        axis="y",
        labelcolor="blue",
    )

    # Add legend for the secondary y-axis line
    ax2.legend(
        loc="upper right",
    )

    # Set y-axis limits for the loss values
    if y_min_additional_plot is not None and y_max_additional_plot is not None:
        # Set the fixed y-axis limits
        ax2.set_ylim(
            bottom=y_min_additional_plot,
            top=y_max_additional_plot,
        )
    else:
        # Automatically adjust the y-axis limits
        ax2.autoscale(
            axis="y",
        )

    return ax2


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
            df[NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS["local_estimates_dedup"]].isna()
            if updated_selected_subsample_dict[NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS["local_estimates_dedup"]]
            is None
            else df[NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS["local_estimates_dedup"]]
            == updated_selected_subsample_dict[NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS["local_estimates_dedup"]]
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


def scatterplot_individual_seed_combination(
    data: pd.DataFrame,
    seed_combination: str,
    output_dir: pathlib.Path,
    y_column_name: str,
    plot_properties: PlotProperties,
    x_column_name: str = "local_estimates_noise_distortion",
    fixed_params_text: str | None = None,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Create and save an individual plot for a specific seed combination.

    Parameters
    ----------
    data:
        The dataframe filtered for the specific seed combination.
    seed_combination:
        The unique seed combination identifier.
    output_dir:
        Directory to save the plot.

    """
    (
        fig,
        ax,
    ) = plt.subplots(
        figsize=(18, 10),
    )
    ax.scatter(
        x=data[x_column_name],
        y=data[y_column_name],
        alpha=0.5,
        label="Data Points",
    )

    configure_y_limits(
        ax=ax,
        plot_properties=plot_properties,
    )

    # Add labels and title
    ax.set_xlabel(
        xlabel="Local Estimates Noise Distortion",
    )
    ax.set_ylabel(
        ylabel="Array Data Mean",
    )
    ax.set_title(
        label=f"Noise Analysis for {seed_combination = }",
    )
    ax.legend()
    ax.grid(
        visible=True,
    )

    add_fixed_params_text_to_plot(
        ax=ax,
        fixed_params_text=fixed_params_text,
    )

    # Save the plot
    save_path = pathlib.Path(
        output_dir,
        f"individual_plot_{seed_combination}.pdf",
    )
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    fig.savefig(
        fname=save_path,
        bbox_inches="tight",
    )


def scatterplot_individual_seed_combinations_and_combined(
    data: pd.DataFrame,
    output_dir: pathlib.Path,
    plot_properties: PlotProperties,
    y_column_name: str,
    x_column_name: str = "local_estimates_noise_distortion",
    fixed_params_text: str | None = None,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Create individual plots for each combination of seeds.

    Combinations are taken from `data_subsampling_sampling_seed` and
    `data_prep_sampling_seed` and a combined plot showing `array_data_mean` vs.
    `x_column_name` with unique colors for each seed combination.

    For the noise analysis, the `x_column_name` is the local estimates noise distortion.

    Parameters
    ----------
    data:
        The dataframe containing relevant columns.
    output_dir:
        Directory to save the plots.

    """
    # Add a unique identifier for each combination of sampling seeds
    data["seed_combination"] = (
        "data-sub-seed="
        + data["data_subsampling_sampling_seed"].astype(dtype=str)
        + "_"
        + "data-prep-seed="
        + data["data_prep_sampling_seed"].astype(dtype=str)
    )

    # Ensure numeric columns for plotting
    data[x_column_name] = pd.to_numeric(
        arg=data[x_column_name],
        errors="coerce",
    )
    data[y_column_name] = pd.to_numeric(
        arg=data[y_column_name],
        errors="coerce",
    )

    data["marker_type"] = data["data_subsampling_sampling_seed"].astype(
        dtype=str,
    )

    # Filter data to remove rows with missing values
    filtered_data: pd.DataFrame = data.dropna(
        subset=[
            x_column_name,
            y_column_name,
            "seed_combination",
        ],
    )

    # Get unique seed combinations
    unique_seeds = filtered_data["seed_combination"].unique()

    # Individual plots for each seed combination
    for seed in unique_seeds:
        subset = filtered_data[filtered_data["seed_combination"] == seed]
        if not subset.empty:
            scatterplot_individual_seed_combination(
                data=subset,
                seed_combination=seed,
                output_dir=output_dir,
                y_column_name=y_column_name,
                plot_properties=plot_properties,
                x_column_name=x_column_name,
                fixed_params_text=fixed_params_text,
                verbosity=verbosity,
                logger=logger,
            )

    # Combined plot with unique colors for each seed combination
    (
        fig,
        ax,
    ) = plt.subplots(
        figsize=plot_properties.figsize,
    )
    sns.scatterplot(
        data=filtered_data,
        x=x_column_name,
        y=y_column_name,
        hue="seed_combination",
        style="marker_type",  # Different marker shapes based on `data_subsampling_sampling_seed`
        alpha=0.6,
        ax=ax,
    )

    configure_y_limits(
        ax=ax,
        plot_properties=plot_properties,
    )

    ax.set_title(
        label="Combined Noise Analysis with Seed Combinations",
    )
    ax.set_xlabel(
        xlabel=x_column_name,
    )
    ax.set_ylabel(
        ylabel=y_column_name,
    )
    ax.legend(
        title="Seed Combinations",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0,
    )
    ax.grid(
        visible=True,
    )

    add_fixed_params_text_to_plot(
        ax=ax,
        fixed_params_text=fixed_params_text,
    )

    # Save the combined plot
    save_path = pathlib.Path(
        output_dir,
        "combined_plot.pdf",
    )
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    fig.savefig(
        fname=save_path,
        bbox_inches="tight",
    )


def add_fixed_params_text_to_plot(
    ax: plt.Axes,  # type: ignore - no type checking for matplotlib
    fixed_params_text: str | None = None,
) -> None:
    """Add a text box with fixed parameters to the plot."""
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


def configure_y_limits(
    ax: plt.Axes,  # type: ignore - no type checking for matplotlib
    plot_properties: PlotProperties,
) -> None:
    """Configure the y-axis limits for the plot."""
    if plot_properties.y_min is not None and plot_properties.y_max is not None:
        # Set the fixed y-axis limits
        ax.set_ylim(
            bottom=plot_properties.y_min,
            top=plot_properties.y_max,
        )
    else:
        # Automatically adjust the y-axis limits
        ax.autoscale(
            axis="y",
        )
