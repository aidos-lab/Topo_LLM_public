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


"""Create plots of the local estimates and compare with other task performance measures."""

import json
import logging
import pathlib
from collections import Counter
from typing import TYPE_CHECKING

import hydra
import matplotlib.axes
import numpy as np
import omegaconf
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm

from topollm.analysis.local_estimates_handling.saving.local_estimates_saving_manager import LocalEstimatesSavingManager
from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.plotting.plot_size_config import PlotSizeConfigFlat
from topollm.task_performance_analysis.plotting.distribution_violinplots_and_distribution_boxplots import (
    TicksAndLabels,
    make_distribution_violinplots_from_extracted_arrays,
)
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    from topollm.analysis.local_estimates_handling.saving.local_estimates_containers import LocalEstimatesContainer
    from topollm.config_classes.main_config import MainConfig
    from topollm.path_management.embeddings.protocol import EmbeddingsPathManager

# Logger for this file
global_logger: logging.Logger = logging.getLogger(
    name=__name__,
)
default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)

setup_exception_logging(
    logger=global_logger,
)

ESTIMATE_VALUE_COLUMN_NAME = "estimate_value"


@hydra.main(
    config_path=f"{HYDRA_CONFIGS_BASE_PATH}",
    config_name="main_config",
    version_base="1.3",
)
def main(
    config: omegaconf.DictConfig,
) -> None:
    """Run main function."""
    logger: logging.Logger = global_logger
    logger.info(
        msg="Running script ...",
    )

    # ================================================== #
    # Load configuration and initialize path manager
    # ================================================== #

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=logger,
    )
    verbosity: Verbosity = main_config.verbosity

    embeddings_path_manager: EmbeddingsPathManager = get_embeddings_path_manager(
        main_config=main_config,
        logger=logger,
    )

    # ================================================== #
    # Load data
    # ================================================== #

    local_estimates_pointwise_dir_absolute_path = (
        embeddings_path_manager.get_local_estimates_pointwise_dir_absolute_path()
    )

    local_estimates_saving_manager: LocalEstimatesSavingManager = (
        LocalEstimatesSavingManager.from_local_estimates_pointwise_dir_absolute_path(
            local_estimates_pointwise_dir_absolute_path=local_estimates_pointwise_dir_absolute_path,
            verbosity=verbosity,
            logger=logger,
        )
    )
    local_estimates_container: LocalEstimatesContainer = local_estimates_saving_manager.load_local_estimates()
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"local_estimates_container:\n{local_estimates_container.get_summary_string()}",  # noqa: G004 - low overhead
        )

    if local_estimates_container.pointwise_results_meta_frame is None:
        msg = "No metadata for the pointwise results available."
        raise ValueError(
            msg,
        )

    # ================================================== #
    # Output folders
    # ================================================== #

    # We will locate the output of this script nested under the saved plots directory,
    # into a subfolder derived from the local estimates subfolder.
    output_root_dir: pathlib.Path = pathlib.Path(
        embeddings_path_manager.saved_plots_dir_absolute_path,
        "local_estimates_distribution_over_tokens",
        embeddings_path_manager.get_local_estimates_subfolder_path(),
    )

    # Note: We move the creation of the output_root_dir creation after the loading of the local estimates data,
    # so that we do not create a directory if the data loading fails.
    output_root_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ================================================== #
    # For reference, create the violin plot corresponding to this data
    # ================================================== #

    # For the violin plots, the local estimates values are plotted on the y-axis
    plot_size_config_violinplot_choices = [
        PlotSizeConfigFlat(),
        PlotSizeConfigFlat(
            y_min=0,
            y_max=15,
        ),
    ]

    for plot_size_config_violinplot in plot_size_config_violinplot_choices:
        (
            fig,
            ax,
        ) = make_distribution_violinplots_from_extracted_arrays(
            extracted_arrays=[local_estimates_container.pointwise_results_array_np],
            ticks_and_labels=TicksAndLabels(
                xlabel="main_config.language_model.checkpoint_no",
                ylabel="pointwise_results_array_np",
                xticks_labels=[str(object=main_config.language_model.checkpoint_no)],
            ),
            plots_output_dir=output_root_dir,
            plot_size_config=plot_size_config_violinplot,
            verbosity=verbosity,
            logger=logger,
        )

    # ================================================== #
    # Find peaks in the local estimates distribution
    # ================================================== #

    random_state = 42

    num_clusters_options: list[int] = [
        1,
        2,
        3,
        5,
        10,
    ]

    for num_clusters in tqdm(
        iterable=num_clusters_options,
        desc="Iterating over different number of clusters",
    ):
        current_output_dir: pathlib.Path = pathlib.Path(
            output_root_dir,
            f"{num_clusters=}",
        )

        # Run clustering on the synthetic data
        clustered_results_df: pd.DataFrame = cluster_based_on_estimates(
            meta_frame=local_estimates_container.pointwise_results_meta_frame,
            estimates_array=local_estimates_container.pointwise_results_array_np,
            num_clusters=num_clusters,
            random_state=random_state,
        )

        # # # #
        # Plot the cluster distribution.s

        # For the histograms, the local estimate values are on the x-axis,
        # and the frequency on the y-axis.
        plot_size_config_cluster_distribution_choices = [
            PlotSizeConfigFlat(),  # Set axis limits automatically
            PlotSizeConfigFlat(
                x_min=0.0,
                x_max=15.0,
                y_min=0,
                y_max=7_000,
            ),
        ]

        for plot_size_config_cluster_distribution in plot_size_config_cluster_distribution_choices:
            plot_cluster_distribution(
                clustered_df=clustered_results_df,
                plot_size_config=plot_size_config_cluster_distribution,
                plots_output_dir=current_output_dir,
                bins=150,
                num_sample_tokens=40,
                random_state=random_state,
                verbosity=verbosity,
                logger=logger,
            )

        save_cluster_data(
            clustered_df=clustered_results_df,
            output_dir=current_output_dir,
            num_samples=50,
            top_n=30,
            random_state=random_state,
            verbosity=verbosity,
            logger=logger,
        )

        # ================================================== #
        # Additional manual logging of the token distribution
        # ================================================== #

        # Retrieve example tokens
        example_tokens: dict = get_example_tokens(
            clustered_df=clustered_results_df,
            random_state=random_state,
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Example tokens per cluster:",
            )
            for cluster, tokens in example_tokens.items():
                logger.info(
                    msg=f"{cluster}: {tokens}",  # noqa: G004 - low overhead
                )

        # Retrieve most frequent tokens
        most_frequent_tokens: dict = get_most_and_least_frequent_tokens(
            clustered_df=clustered_results_df,
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Most frequent tokens per cluster:",
            )
            for cluster, tokens in most_frequent_tokens.items():
                logger.info(
                    msg=f"{cluster}: {tokens}",  # noqa: G004 - low overhead
                )

    logger.info(
        msg="Running script DONE",
    )


def cluster_based_on_estimates(
    meta_frame: pd.DataFrame,
    estimates_array: np.ndarray,
    num_clusters: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Clusters the local estimates into distinct groups and associates tokens with each cluster."""
    estimates_reshaped = estimates_array.reshape(-1, 1)

    kmeans = KMeans(
        n_clusters=num_clusters,
        random_state=random_state,
        n_init=10,
    )
    cluster_labels: np.ndarray = kmeans.fit_predict(
        X=estimates_reshaped,
    )

    clustered_df: pd.DataFrame = meta_frame.copy()
    clustered_df[ESTIMATE_VALUE_COLUMN_NAME] = estimates_array
    clustered_df["cluster"] = cluster_labels

    return clustered_df


def get_cluster_statistics(
    clustered_df: pd.DataFrame,
) -> dict:
    """Compute summary statistics for each cluster."""
    cluster_stats: dict = {}

    for cluster_id in sorted(clustered_df["cluster"].unique()):
        cluster_data = clustered_df[clustered_df["cluster"] == cluster_id][ESTIMATE_VALUE_COLUMN_NAME]
        cluster_stats[f"{cluster_id=}"] = {
            "min": cluster_data.min(),
            "max": cluster_data.max(),
            "mean": cluster_data.mean(),
            "std": cluster_data.std(),
            # Information about the quantiles
            "q25": cluster_data.quantile(0.25),
            "q50": cluster_data.quantile(0.50),
            "q75": cluster_data.quantile(0.75),
            # Number of tokens in the cluster
            "num": len(cluster_data),
        }

    return cluster_stats


def save_cluster_data(
    clustered_df: pd.DataFrame,
    output_dir: pathlib.Path,
    filename: str = "cluster_data.json",
    num_samples: int = 50,
    top_n: int = 30,
    random_state: int = 42,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Save cluster statistics, example tokens, and frequent tokens to a JSON file."""
    data_to_save: dict[str, dict] = {
        "cluster_statistics": get_cluster_statistics(
            clustered_df=clustered_df,
        ),
        "example_tokens": get_example_tokens(
            clustered_df=clustered_df,
            num_samples=num_samples,
            random_state=random_state,
        ),
        "most_and_least_frequent_tokens": get_most_and_least_frequent_tokens(
            clustered_df=clustered_df,
            top_n=top_n,
        ),
        "extreme_estimate_value_tokens": get_tokens_with_extreme_estimate_value(
            clustered_df=clustered_df,
            num_samples=num_samples,
        ),
    }
    # Note:
    # We do not log the `data_to_save` because this will lead to very large logfiles.
    # The data will be saved to a JSON file instead.

    save_file_path = pathlib.Path(
        output_dir,
        filename,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Saving cluster data to {save_file_path = } ...",  # noqa: G004 - low overhead
        )
    with save_file_path.open(
        mode="w",
    ) as f:
        json.dump(
            obj=data_to_save,
            fp=f,
            indent=4,
        )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Saving cluster data to {save_file_path = } DONE",  # noqa: G004 - low overhead
        )


def plot_cluster_distribution(
    clustered_df: pd.DataFrame,
    plot_size_config: PlotSizeConfigFlat,
    plots_output_dir: pathlib.Path | None = None,
    bins: int = 100,
    num_sample_tokens: int | None = 20,
    random_state: int = 42,
    *,
    plot_extremal_tokens: bool = True,
    num_most_frequent_tokens: int | None = 5,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> tuple:
    """Plot the cluster distribution and highlight mean values."""
    (
        fig,
        ax,
    ) = plt.subplots(
        figsize=(
            plot_size_config.output_pdf_width / 100,
            plot_size_config.output_pdf_height / 100,
        ),
    )
    palette = sns.color_palette(
        palette="husl",
        n_colors=len(clustered_df["cluster"].unique()),
    )

    # Define common bins for all histograms
    all_values: np.ndarray = clustered_df["estimate_value"].to_numpy()
    bin_edges: np.ndarray = np.histogram_bin_edges(
        a=all_values,  # type: ignore - typing problem with numpy
        bins=bins,
    )

    # Iterate over the clusters and plot the histograms
    for i, cluster_id in enumerate(
        iterable=sorted(clustered_df["cluster"].unique()),
    ):
        add_plot_for_single_cluster_id(
            clustered_df=clustered_df,
            i=i,
            cluster_id=cluster_id,
            ax=ax,
            palette=palette,
            plot_size_config=plot_size_config,
            num_sample_tokens=num_sample_tokens,
            num_most_frequent_tokens=num_most_frequent_tokens,
            random_state=random_state,
            bin_edges=bin_edges,
            plot_extremal_tokens=plot_extremal_tokens,
        )

    # # # #
    # General plot setup
    ax.set_title(
        label="Clustering of Local Estimates Values",
    )
    ax.legend()

    ax.set_xlabel(
        xlabel="Local Estimate Values",
    )
    ax.set_ylabel(
        ylabel="Frequency",
    )

    # Set the x-axis limits
    if plot_size_config.x_min is not None:
        ax.set_xlim(
            left=plot_size_config.x_min,
        )
    if plot_size_config.x_max is not None:
        ax.set_xlim(
            right=plot_size_config.x_max,
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

    # # # #
    # Saving the plot
    if plots_output_dir is not None:
        plot_name: str = (
            f"histogram_cluster_distribution"
            f"_x_min={plot_size_config.x_min}_x_max={plot_size_config.x_max}"
            f"_y_min={plot_size_config.y_min}_y_max={plot_size_config.y_max}"
        )
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


def add_plot_for_single_cluster_id(
    clustered_df: pd.DataFrame,
    i: int,
    cluster_id: int,
    ax: matplotlib.axes.Axes,
    palette: list,
    plot_size_config: PlotSizeConfigFlat,
    num_sample_tokens: int | None,
    num_most_frequent_tokens: int | None,
    random_state: int,
    bin_edges: np.ndarray,  # type: ignore - typing problem with numpy
    *,
    plot_extremal_tokens: bool,
) -> None:
    """Add a histogram for a single cluster to the plot."""
    cluster_data: pd.DataFrame = clustered_df[clustered_df["cluster"] == cluster_id]
    ax.hist(
        x=cluster_data[ESTIMATE_VALUE_COLUMN_NAME],
        bins=bin_edges,  # type: ignore - typing problem with numpy
        alpha=0.5,
        label=f"Cluster {cluster_id = }",
        color=palette[i],
    )
    cluster_mean_value: float = cluster_data[ESTIMATE_VALUE_COLUMN_NAME].mean()
    cluster_std_value: float = cluster_data[ESTIMATE_VALUE_COLUMN_NAME].std()
    cluster_count: int = len(cluster_data)
    ax.axvline(
        x=cluster_mean_value,
        linestyle="dashed",
        label=f"Cluster {cluster_id = } (count={cluster_count}; {cluster_mean_value:.2f}±{cluster_std_value:.2f})",
        linewidth=2,
        color=palette[i],
    )

    if num_sample_tokens is not None:
        # Sample tokens from each cluster and plot them with x-coordinates corresponding to the estimate values,
        # and y-coordinates randomly jittered for better visibility.
        sampled_tokens: pd.DataFrame = clustered_df[clustered_df["cluster"] == cluster_id].sample(
            n=num_sample_tokens,
            random_state=random_state,
        )
        if plot_size_config.y_max is not None:
            largest_y_coordinate_for_jitter: float = plot_size_config.y_max * 0.7
        else:
            largest_y_coordinate_for_jitter = 1_500

        rng: np.random.Generator = np.random.default_rng(
            seed=random_state,
        )

        for _, row in sampled_tokens.iterrows():
            plt.text(
                x=row[ESTIMATE_VALUE_COLUMN_NAME],
                y=rng.uniform(
                    low=50,
                    high=largest_y_coordinate_for_jitter,
                ),  # random jitter for better visibility
                s=row["token_name"],
                rotation=90,
                verticalalignment="bottom",
                fontsize=9,
                color=palette[i],
                alpha=1.0,
            )

    if num_most_frequent_tokens is not None:
        # Compute the most common tokens in each cluster and their mean/std estimates
        token_counts = Counter(cluster_data["token_name"])
        common_tokens: list = token_counts.most_common(
            n=5,
        )
        token_summary: list = []
        for token, count in common_tokens:
            token_estimates: pd.Series = cluster_data[cluster_data["token_name"] == token][ESTIMATE_VALUE_COLUMN_NAME]
            token_mean = token_estimates.mean()
            token_std = token_estimates.std()
            token_summary.append(
                f"{token} ({count=})\n{token_mean:.2f}±{token_std:.2f}",
            )
        summary_text = "\n".join(token_summary)

        # Add a transparent box with the common tokens near the cluster mean.
        # Position the text box at the top of the plot.
        y_position_relative_to_top = 0.85
        y: float = (
            plot_size_config.y_max * y_position_relative_to_top
            if plot_size_config.y_max is not None
            else ax.get_ylim()[1] * y_position_relative_to_top
        )
        ax.text(
            x=cluster_mean_value,
            y=y,
            s=summary_text,
            fontsize=7,
            bbox={
                "facecolor": palette[i],
                "alpha": 0.3,
                "edgecolor": "black",
            },
            ha="center",
        )

    if plot_extremal_tokens:
        # # # #
        # Find min and max tokens for each cluster and annotate at the top
        min_row = (
            clustered_df[clustered_df["cluster"] == cluster_id]
            .nsmallest(
                n=1,
                columns="estimate_value",
            )
            .iloc[0]
        )
        max_row = (
            clustered_df[clustered_df["cluster"] == cluster_id]
            .nlargest(
                n=1,
                columns="estimate_value",
            )
            .iloc[0]
        )

        extremal_tokens_y_coordinate = 0.9 * plot_size_config.y_max if plot_size_config.y_max is not None else 1_800
        # Use a small offset in the x-coordinate, so that adjacent extremal values from the clusters do not overlap.
        x_coordinate_offset: float = 0.07

        ax.text(
            x=min_row[ESTIMATE_VALUE_COLUMN_NAME] + x_coordinate_offset,
            y=extremal_tokens_y_coordinate,
            s=f"Min: {min_row['token_name']} ({min_row[ESTIMATE_VALUE_COLUMN_NAME]:.2f})",
            rotation=90,
            verticalalignment="top",
            fontsize=10,
            color=palette[i],
            fontweight="bold",
        )
        ax.text(
            x=max_row[ESTIMATE_VALUE_COLUMN_NAME] - x_coordinate_offset,
            y=extremal_tokens_y_coordinate,
            s=f"Max: {max_row['token_name']} ({max_row[ESTIMATE_VALUE_COLUMN_NAME]:.2f})",
            rotation=90,
            verticalalignment="top",
            fontsize=10,
            color=palette[i],
            fontweight="bold",
        )


def get_example_tokens(
    clustered_df: pd.DataFrame,
    num_samples: int = 10,
    random_state: int = 42,
) -> dict:
    """Retrieve example tokens from each cluster."""
    example_tokens = {}

    for cluster_id in sorted(clustered_df["cluster"].unique()):
        sample_tokens = (
            clustered_df[clustered_df["cluster"] == cluster_id]["token_name"]
            .sample(
                num_samples,
                random_state=random_state,
            )
            .tolist()
        )
        example_tokens[f"{cluster_id = } Randomly Sampled Tokens"] = sample_tokens

    return example_tokens


def get_most_and_least_frequent_tokens(
    clustered_df: pd.DataFrame,
    top_n: int = 10,
) -> dict:
    """Find the most frequent tokens per cluster."""
    results_dict: dict = {}

    for cluster_id in sorted(clustered_df["cluster"].unique()):
        this_cluster_df: pd.DataFrame = clustered_df[clustered_df["cluster"] == cluster_id]

        this_cluster_token_counts = Counter(
            this_cluster_df["token_name"],
        )

        # This will hold the information about this cluster
        results_dict[f"{cluster_id = }"] = {}

        # This will hold the information about the most frequent tokens in this cluster
        results_dict[f"{cluster_id = }"]["Most Frequent Tokens Information"] = {}

        this_cluster_most_frequent_tokens: list[tuple[str, int]] = this_cluster_token_counts.most_common(
            n=top_n,
        )

        for token, count in this_cluster_most_frequent_tokens:
            token_estimates: pd.Series = this_cluster_df[this_cluster_df["token_name"] == token][
                ESTIMATE_VALUE_COLUMN_NAME
            ]
            token_mean = token_estimates.mean()
            token_std = token_estimates.std()
            results_dict[f"{cluster_id = }"]["Most Frequent Tokens Information"][token] = {
                "count": count,
                "mean": token_mean,
                "std": token_std,
            }

        # This will hold the information about the least frequent tokens in this cluster
        results_dict[f"{cluster_id = }"]["Least Frequent Tokens Information"] = {}

        this_cluster_least_frequent_tokens: list[tuple[str, int]] = this_cluster_token_counts.most_common()[
            : -top_n - 1 : -1
        ]

        for token, count in this_cluster_least_frequent_tokens:
            token_estimates: pd.Series = this_cluster_df[this_cluster_df["token_name"] == token][
                ESTIMATE_VALUE_COLUMN_NAME
            ]
            token_mean = token_estimates.mean()
            token_std = token_estimates.std()
            results_dict[f"{cluster_id = }"]["Least Frequent Tokens Information"][token] = {
                "count": count,
                "mean": token_mean,
                "std": token_std,
            }

    return results_dict


def get_tokens_with_extreme_estimate_value(
    clustered_df: pd.DataFrame,
    num_samples: int = 10,
    columns_to_save: list | None = None,
) -> dict:
    """Find tokens with extreme estimate values.

    For each cluster, find the tokens with the lowest and highest estimate values,
    and return them together with the estimate values in a dictionary.
    """
    if columns_to_save is None:
        columns_to_save = [
            "token_id",
            "sentence_idx",
            "subsample_idx",
            "token_name",
            # "tokens_list",
            "concatenated_tokens",
        ]

    results_dict: dict = {}

    for cluster_id in sorted(clustered_df["cluster"].unique()):
        this_cluster_df: pd.DataFrame = clustered_df[clustered_df["cluster"] == cluster_id]

        # This dict will hold the information about this cluster.
        results_dict[f"{cluster_id = }"] = {}

        results_dict[f"{cluster_id = }"]["Lowest Estimate Tokens"] = this_cluster_df.nsmallest(
            num_samples,
            ESTIMATE_VALUE_COLUMN_NAME,
        )[
            [
                *columns_to_save,
                ESTIMATE_VALUE_COLUMN_NAME,
            ]
        ].to_dict(
            orient="records",
        )
        results_dict[f"{cluster_id = }"]["Highest Estimate Tokens"] = this_cluster_df.nlargest(
            num_samples,
            ESTIMATE_VALUE_COLUMN_NAME,
        )[
            [
                *columns_to_save,
                ESTIMATE_VALUE_COLUMN_NAME,
            ]
        ].to_dict(
            orient="records",
        )

    return results_dict


if __name__ == "__main__":
    main()
