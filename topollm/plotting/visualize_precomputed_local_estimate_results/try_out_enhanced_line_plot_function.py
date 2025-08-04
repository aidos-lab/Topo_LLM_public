#!/usr/bin/env python3
"""
A self-contained script that demonstrates the enhanced plotting function.
It uses dataclasses to group plot dimensions and column names for clarity,
and tests the function on toy data.
"""

import logging
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class PlotConfig:
    """Configuration for axis limits and output figure dimensions.

    Attributes:
        x_min: Minimum value for the x-axis (or None to auto-scale).
        x_max: Maximum value for the x-axis (or None to auto-scale).
        y_min: Minimum value for the y-axis (or None to auto-scale).
        y_max: Maximum value for the y-axis (or None to auto-scale).
        output_pdf_width: Figure width in pixels when saving to PDF.
        output_pdf_height: Figure height in pixels when saving to PDF.
    """

    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    output_pdf_width: int = 2500
    output_pdf_height: int = 1500


@dataclass
class ColumnConfig:
    """Configuration for column names used in the plot.

    Attributes:
        x_column: Name of the column for x-axis values.
        y_column: Name of the column for y-axis values.
        group_column: Name of the categorical column to group the data.
        std_column: Optional name of the column for standard deviation values.
                   If provided and found in the DataFrame, an error band will be plotted.
    """

    x_column: str = "model_checkpoint"
    y_column: str = "loss_mean"
    group_column: str = "data_full"
    std_column: Optional[str] = None


def generate_color_mapping(df: pd.DataFrame, group_column: str) -> Dict[Any, str]:
    """Generate a color mapping for each unique value in the group column.

    Args:
        df: DataFrame containing the data.
        group_column: Column name by which to group.

    Returns:
        A dictionary mapping each unique group to a color.
    """
    unique_groups = sorted(df[group_column].unique())
    colors = plt.cm.tab10.colors  # Use the tab10 colormap as an example.
    return {group: colors[i % len(colors)] for i, group in enumerate(unique_groups)}


def line_plot_grouped_by_categorical_column(
    df: pd.DataFrame,
    output_folder: Optional[pathlib.Path] = None,
    *,
    plot_name: str = "line_plot",
    subtitle_text: Optional[str] = None,
    columns: ColumnConfig = ColumnConfig(),
    plot_config: PlotConfig = PlotConfig(),
    color_mapping: Optional[Dict[Any, str]] = None,
    verbosity: int = 1,  # Replace with your Verbosity enum if available.
    logger: logging.Logger = logging.getLogger(__name__),
) -> None:
    """Plot the development of a selected y-axis column over a selected x-axis column, grouping by a categorical column.

    Optionally, if a standard deviation column is provided (via columns.std_column),
    a band representing the central value ± the standard deviation will be plotted.

    Args:
        df: Input data as a pandas DataFrame.
        output_folder: Optional folder where the plot and raw data will be saved.
        plot_name: Base name for the output files.
        subtitle_text: Optional subtitle text for the plot.
        columns: Dataclass containing the column names for the x-axis, y-axis, grouping, and standard deviation.
        plot_config: Dataclass containing the axis limits and output dimensions.
        color_mapping: Optional mapping from group values to colors.
        verbosity: Verbosity level for logging.
        logger: Logger for output messages.
    """
    # Ensure the x_column exists; fill missing values with -1.
    df[columns.x_column] = df.get(columns.x_column, pd.Series([-1] * len(df)))
    df[columns.x_column] = df[columns.x_column].fillna(-1)

    # Sort the DataFrame by the x_column for better visualization.
    df = df.sort_values(by=columns.x_column)

    # Create a figure with the desired dimensions (convert pixels to inches).
    fig = plt.figure(
        figsize=(
            plot_config.output_pdf_width / 100,
            plot_config.output_pdf_height / 100,
        )
    )

    # Generate a color mapping if none is provided.
    if color_mapping is None:
        color_mapping = generate_color_mapping(df=df, group_column=columns.group_column)

    # Plot each group separately.
    for group_value in df[columns.group_column].unique():
        subset = df[df[columns.group_column] == group_value]
        group_color = color_mapping.get(group_value, "black")
        plt.plot(
            subset[columns.x_column],
            subset[columns.y_column],
            marker="o",
            linestyle="-",
            color=group_color,
            label=f"{columns.group_column}={group_value}",
        )
        # If a standard deviation column is provided and exists in the subset, plot an error band.
        if columns.std_column is not None and columns.std_column in subset.columns:
            lower_bound = subset[columns.y_column] - subset[columns.std_column]
            upper_bound = subset[columns.y_column] + subset[columns.std_column]
            plt.fill_between(
                subset[columns.x_column],
                lower_bound,
                upper_bound,
                color=group_color,
                alpha=0.2,  # Transparency for the error band.
            )

    # Set the axis labels and title.
    plt.xlabel(columns.x_column)
    plt.ylabel(columns.y_column)
    plt.title(f"Development of {columns.y_column} over {columns.x_column} grouped by {columns.group_column}")

    # Set the axis limits if specified.
    if plot_config.x_min is not None and plot_config.x_max is not None:
        plt.xlim(plot_config.x_min, plot_config.x_max)
    if plot_config.y_min is not None and plot_config.y_max is not None:
        plt.ylim(plot_config.y_min, plot_config.y_max)

    # Optionally set a subtitle.
    if subtitle_text is not None:
        plt.suptitle(subtitle_text)

    # Add a legend at the bottom of the plot.
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

    # Add a grid for better readability.
    plt.grid(True)

    # Save the plot and raw data if an output folder is provided.
    if output_folder is not None:
        output_folder = pathlib.Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        # Save the plot as a PDF.
        output_file = output_folder / f"{plot_name}.pdf"
        if verbosity >= Verbosity.NORMAL:
            logger.info(f"Saving plot to {output_file} ...")
        plt.savefig(output_file, bbox_inches="tight", format="pdf")
        if verbosity >= Verbosity.NORMAL:
            logger.info(f"Saving plot to {output_file} DONE")

        # Save the raw data as a CSV.
        output_file_raw_data = output_folder / f"{plot_name}_raw_data.csv"
        output_file_raw_data.parent.mkdir(parents=True, exist_ok=True)
        if verbosity >= Verbosity.NORMAL:
            logger.info(f"Saving raw data to {output_file_raw_data} ...")
        df.to_csv(output_file_raw_data, index=False)
        if verbosity >= Verbosity.NORMAL:
            logger.info(f"Saving raw data to {output_file_raw_data} DONE")


if __name__ == "__main__":
    # Configure the logger.
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test_line_plot")

    # Create some toy data.
    np.random.seed(42)  # For reproducibility.
    data = []
    for group in ["A", "B"]:
        for i in range(10):
            x_val = i
            # Different ranges for different groups.
            y_val = np.random.uniform(0.5, 1.0) if group == "A" else np.random.uniform(0.8, 1.3)
            std_val = np.random.uniform(0.05, 0.15)
            data.append(
                {
                    "model_checkpoint": x_val,
                    "loss_mean": y_val,
                    "data_full": group,
                    "loss_std": std_val,
                }
            )
    df = pd.DataFrame(data)

    # Define column configuration (including the standard deviation column) and plot configuration.
    columns = ColumnConfig(
        x_column="model_checkpoint",
        y_column="loss_mean",
        group_column="data_full",
        std_column="loss_std",  # Set to None if you do not want to plot the error band.
    )
    plot_config = PlotConfig(
        x_min=0,
        x_max=9,
        y_min=0.4,
        y_max=1.5,
        output_pdf_width=1200,
        output_pdf_height=800,
    )

    # Call the plotting function.
    line_plot_grouped_by_categorical_column(
        df,
        output_folder=None,  # Change to a valid Path to save the plot and data.
        plot_name="toy_line_plot",
        subtitle_text="Toy data example",
        columns=columns,
        plot_config=plot_config,
        verbosity=1,
        logger=logger,
    )

    # Display the plot.
    plt.show()
