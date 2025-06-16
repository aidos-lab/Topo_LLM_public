# Copyright 2024
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#


"""Plot with error bands for standard deviation."""

import logging
import os
import pathlib

import hydra
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import pandas as pd
from matplotlib import cm
from tqdm import tqdm

from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH, TOPO_LLM_REPOSITORY_BASE_PATH
from topollm.logging.setup_exception_logging import setup_exception_logging

global_logger: logging.Logger = logging.getLogger(
    name=__name__,
)
default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)

setup_exception_logging(
    logger=global_logger,
)


def create_color_map(unique_models):
    """Create a color map for the given models.

    Args:
        unique_models (list): List of unique model identifiers.

    Returns:
        dict: A dictionary mapping each model to a specific color.
    """
    print("Creating color map for models...")
    colors = cm.get_cmap("tab10", len(unique_models))
    return {model: colors(i) for i, model in enumerate(unique_models)}


def filter_base_model_data(model_data):
    """Filter the base model data from the given model data.

    Args:
        model_data (DataFrame): Data for the specific model.

    Returns:
        DataFrame: Filtered base model data.
    """
    return model_data[
        model_data["checkpoint"].isna() & (model_data["model_without_checkpoint"] == "roberta-base_task-masked_lm")
    ]


def align_dataframes(
    mean_df: pd.DataFrame,
    std_df: pd.DataFrame,
    logger: logging.Logger = default_logger,
) -> pd.DataFrame:
    """Align mean and standard deviation DataFrames by merging them on 'model_without_checkpoint' and 'checkpoint'.

    Args:
    ----
        mean_df (DataFrame): DataFrame containing mean values.
        std_df (DataFrame): DataFrame containing standard deviation values.

    Returns:
    -------
        DataFrame: Merged DataFrame with aligned mean and standard deviation values.

    """
    logger.info(
        "Aligning mean and standard deviation DataFrames ...",
    )
    merged_df: pd.DataFrame = pd.merge(
        left=mean_df,
        right=std_df,
        on=["model_without_checkpoint", "checkpoint"],
        suffixes=("_mean", "_std"),
    )
    merged_df_clean = merged_df.dropna(
        subset=[
            "token_perplexity_mean",
            "token_perplexity_std",
        ],
    )
    merged_df_clean = merged_df_clean.sort_values(
        by=["model_without_checkpoint", "checkpoint"],
    ).reset_index(
        drop=True,
    )
    logger.info(
        "Alignment complete. Returning cleaned DataFrame.",
    )
    return merged_df_clean


def plot_metrics(
    ax,
    model_data: pd.DataFrame,
    color,
    model,
    base_model_data,
):
    """Plot perplexity and log perplexity metrics on the given axis.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        model_data (DataFrame): Data for the specific model.
        color (str): Color to use for plotting.
        model (str): Model identifier.
        base_model_data (DataFrame): Data for the base model.
    """
    print(f"Plotting metrics for model: {model}")

    # Ensure data is sorted by checkpoint
    model_data = model_data.sort_values(
        by="checkpoint",
    )

    checkpoints = model_data["checkpoint"].to_numpy()
    mean_perplexity = model_data["token_perplexity_mean"].to_numpy()
    std_perplexity = model_data["token_perplexity_std"].to_numpy()
    mean_perplexity = np.where(
        mean_perplexity > 0,
        mean_perplexity,
        np.nan,
    )  # Ensure positive values for log
    log_perplexity = np.log(mean_perplexity)

    if len(checkpoints) > 0:
        print(f"Checkpoints available for model {model}: {len(checkpoints)}")
        ax.plot(
            checkpoints,
            mean_perplexity,
            linestyle="-",
            label=f"Perplexity ({model})",
            color=color,
        )
        ax.fill_between(
            checkpoints,
            mean_perplexity - std_perplexity,
            mean_perplexity + std_perplexity,
            alpha=0.1,
            color=color,
        )
        ax.plot(
            checkpoints,
            log_perplexity,
            linestyle="--",
            label=f"Log Perplexity ({model})",
            color=color,
        )
        # Plotting the base model point if exists
        if not base_model_data.empty:
            print(f"Base model data found for model: {model}")
            base_perplexity = base_model_data["token_perplexity_mean"].values[0]
            ax.scatter(
                0,
                base_perplexity,
                color=color,
                marker="o",
                label=f"Base Model Perplexity ({model})",
            )


def plot_local_estimates(
    ax,
    model_local_data,
    color,
    model,
    base_model_data,
):
    """Plot local estimates on the given axis.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        model_local_data (DataFrame): Data for the specific model's local estimates.
        color (str): Color to use for plotting.
        model (str): Model identifier.
        base_model_data (DataFrame): Data for the base model.
    """
    print(f"Plotting local estimates for model: {model}")
    checkpoints = model_local_data["checkpoint"].values
    local_estimates = model_local_data["local_estimate_mean"].values
    local_std = model_local_data["local_estimate_std"].values

    if len(checkpoints) > 0:
        print(f"Checkpoints available for local estimates of model {model}: {len(checkpoints)}")
        ax.plot(checkpoints, local_estimates, linestyle=":", label=f"Local Estimate ({model})", color=color)
        ax.fill_between(checkpoints, local_estimates - local_std, local_estimates + local_std, alpha=0.2, color=color)
        # Plotting the base model point if exists
        if not base_model_data.empty:
            print(f"Base model data found for local estimates of model: {model}")
            base_local_estimate = base_model_data["local_estimate_mean"].values[0]
            ax.scatter(0, base_local_estimate, color=color, marker="o", label=f"Base Model Local Estimate ({model})")


def create_side_by_side_plots(
    merged_df_clean,
    merged_local_df_clean,
    unique_models,
    saved_plot_file_path: os.PathLike = pathlib.Path("side_by_side_plots.pdf"),
    *,
    show_plot: bool = False,
):
    """Create side by side plots for perplexity/log perplexity and local estimates for each model.

    Args:
        merged_df_clean (DataFrame):
            Cleaned DataFrame containing mean perplexity and std.
        merged_local_df_clean (DataFrame):
            Cleaned DataFrame containing local estimates.
        unique_models (list):
            List of unique model identifiers.
        saved_plot_file_path (str):
            Name of the file to save the plot as a PDF.
    """
    print("Creating side by side plots...")
    model_colors = create_color_map(unique_models)

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    for model in unique_models:
        print(f"Processing model: {model}")
        model_data = merged_df_clean[merged_df_clean["model_without_checkpoint"] == model]
        model_local_data = merged_local_df_clean[merged_local_df_clean["model_without_checkpoint"] == model]
        base_model_data = filter_base_model_data(model_data)

        color = model_colors[model]

        # Plot metrics and local estimates
        plot_metrics(axs[0], model_data, color, model, base_model_data)
        plot_local_estimates(axs[1], model_local_data, color, model, base_model_data)

    # Setting labels, titles, and legends for both subplots
    print("Setting labels, titles, and legends...")
    axs[0].set_xlabel("Model Checkpoint")
    axs[0].set_ylabel("Value")
    axs[0].set_title("Perplexity and Log Perplexity Over Model Checkpoints for Each Model")
    axs[0].grid()

    axs[1].set_xlabel("Model Checkpoint")
    axs[1].set_ylabel("Local Estimate")
    axs[1].set_title("Local Estimate Over Model Checkpoints for Each Model")
    axs[1].grid()

    # Setting a common legend below both plots
    handles, labels = [], []
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize="small",
    )

    plt.tight_layout(
        rect=(0, 0, 1, 0.9),
    )
    print("Displaying and saving the plot...")

    if show_plot:
        plt.show()

    # Save the figure as a PDF
    fig.savefig(
        saved_plot_file_path,
        format="pdf",
    )


def load_data_and_create_plots(
    currently_selected_dataset_folder_path: os.PathLike,
    logger: logging.Logger = default_logger,
):
    mean_df_path = pathlib.Path(
        currently_selected_dataset_folder_path,
        "statistic-mean/aggregated_statistics.csv",
    )
    std_df_path = pathlib.Path(
        currently_selected_dataset_folder_path,
        "statistic-std/aggregated_statistics.csv",
    )

    logger.info(
        "Loading data ...",
    )
    mean_df = pd.read_csv(
        filepath_or_buffer=mean_df_path,
    )
    std_df = pd.read_csv(
        filepath_or_buffer=std_df_path,
    )
    logger.info(
        "Loading data DONE",
    )

    # Align the mean and std DataFrames
    merged_df_clean = align_dataframes(
        mean_df=mean_df,
        std_df=std_df,
    )

    # Get the list of unique models
    unique_models = merged_df_clean["model_without_checkpoint"].unique()

    saved_plot_file_path = pathlib.Path(
        currently_selected_dataset_folder_path,
        "side_by_side_plots.pdf",
    )

    # Create side by side plots
    create_side_by_side_plots(
        merged_df_clean=merged_df_clean,
        merged_local_df_clean=merged_df_clean,
        unique_models=unique_models,
        saved_plot_file_path=saved_plot_file_path,
        show_plot=False,
    )


@hydra.main(
    config_path=f"{HYDRA_CONFIGS_BASE_PATH}",
    config_name="main_config",
    version_base="1.3",
)
def main(
    config: omegaconf.DictConfig,
) -> None:
    """Align dataframes and create plots based on the aligned data."""
    logger = global_logger

    # This points to the parent folder containing the dataset folders
    parent_folder_containing_dataset_folders = pathlib.Path(
        TOPO_LLM_REPOSITORY_BASE_PATH,
        "data",
        "saved_plots",
        "correlation_analyis",
    )

    # Iterate over all the dataset folders
    for dataset_folder in tqdm(
        parent_folder_containing_dataset_folders.iterdir(),
        desc="Processing dataset folders",
    ):
        if not dataset_folder.is_dir():
            continue
        if dataset_folder.name == "None":
            continue

        logger.info(
            msg=f"Processing dataset folder: {dataset_folder}",  # noqa: G004 - low overhead
        )
        try:
            load_data_and_create_plots(
                currently_selected_dataset_folder_path=dataset_folder,
                logger=logger,
            )
        except FileNotFoundError as e:
            logger.warning(
                msg=f"File not found: {e}",  # noqa: G004 - low overhead
            )
            logger.warning(
                msg=f"Skipping dataset folder: {dataset_folder}",  # noqa: G004 - low overhead
            )
        logger.info(
            msg=f"Processing dataset folder: {dataset_folder} DONE",  # noqa: G004 - low overhead
        )


if __name__ == "__main__":
    main()
