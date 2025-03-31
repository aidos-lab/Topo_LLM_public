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

import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from topollm.data_processing.dictionary_handling import (
    filter_list_of_dictionaries_by_key_value_pairs,
    generate_fixed_parameters_text_from_dict,
)
from topollm.logging.log_dataframe_info import log_dataframe_info
from topollm.path_management.embeddings.protocol import EmbeddingsPathManager
from topollm.plotting.plot_size_config import AxisLimits, OutputDimensions, PlotSizeConfigFlat, PlotSizeConfigNested
from topollm.task_performance_analysis.plotting.distribution_violinplots_and_distribution_boxplots import TicksAndLabels
from topollm.task_performance_analysis.plotting.parameter_combinations_and_loaded_data_handling import (
    add_base_model_data,
    construct_mean_plots_over_model_checkpoints_output_dir_from_filter_key_value_pairs,
    derive_base_model_partial_name,
    get_fixed_parameter_combinations,
)
from topollm.task_performance_analysis.plotting.score_loader.score_loader import (
    EmotionClassificationScoreLoader,
    TrippyRScoreLoader,
)
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


@dataclass
class ScoresData:
    """Container for scores data."""

    df: pd.DataFrame | None
    columns_to_plot: list[str] | None

    def save_df(
        self,
        save_dir: pathlib.Path,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        if self.df is None:
            logger.warning(
                msg="No scores data available to save.",
            )
            logger.info(
                msg="Skipping saving scores data.",
            )
            return

        scores_df_save_path = pathlib.Path(
            save_dir,
            "scores_df.csv",
        )
        scores_df_save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving combined scores to {scores_df_save_path = } ...",  # noqa: G004 - low overhead
            )
        self.df.to_csv(
            path_or_buf=scores_df_save_path,
            index=False,
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving combined scores to {scores_df_save_path = } DONE",  # noqa: G004 - low overhead
            )


@dataclass
class PlotInputData:
    """Container for plot input data."""

    local_estimates_df: pd.DataFrame
    scores: ScoresData


@dataclass
class PlotConfig:
    """Container for plot configuration."""

    ticks_and_labels: TicksAndLabels
    plot_size_config_nested: PlotSizeConfigNested
    seeds: np.ndarray
    x_column_name: str = "model_checkpoint"
    filter_key_value_pairs: dict = field(
        default_factory=dict,
    )
    base_model_model_partial_name: str | None = None
    plots_output_dir: pathlib.Path | None = None
    show_plots: bool = False


def create_mean_plots_over_model_checkpoints_with_different_seeds(
    loaded_data: list[dict],
    array_key_name: str,
    output_root_dir: pathlib.Path,
    plot_size_configs_list: list[PlotSizeConfigFlat],
    embeddings_path_manager: EmbeddingsPathManager,
    *,
    fixed_keys: list[str] | None = None,
    additional_fixed_params: dict[str, Any] | None = None,
    save_plot_raw_data: bool = True,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Create mean plots over model checkpoints with different seeds."""
    if fixed_keys is None:
        fixed_keys = [
            "data_full",
            "data_subsampling_full",
            "data_dataset_seed",
            "model_layer",  # model_layer needs to be an integer
            "model_partial_name",
            "local_estimates_desc_full",
            # Note:
            # - Do NOT fix the model seed, as we want to plot the mean over different seeds.
        ]

    if additional_fixed_params is None:
        additional_fixed_params = {
            "tokenizer_add_prefix_space": "False",  # tokenizer_add_prefix_space needs to be a string
        }

    # Iterate over fixed parameter combinations.
    combinations = list(
        get_fixed_parameter_combinations(
            loaded_data=loaded_data,
            fixed_keys=fixed_keys,
            additional_fixed_params=additional_fixed_params,
        ),
    )
    total_combinations = len(combinations)

    if verbosity >= Verbosity.NORMAL:
        # Log available options
        for fixed_param in fixed_keys:
            options = {entry[fixed_param] for entry in loaded_data if fixed_param in entry}
            logger.info(
                msg=f"{fixed_param=} options: {options=}",  # noqa: G004 - low overhead
            )

    for filter_key_value_pairs in tqdm(
        iterable=combinations,
        total=total_combinations,
        desc="Plotting different choices for model checkpoints",
    ):
        filtered_data: list[dict] = filter_list_of_dictionaries_by_key_value_pairs(
            list_of_dicts=loaded_data,
            key_value_pairs=filter_key_value_pairs,
        )

        if len(filtered_data) == 0:
            logger.warning(
                msg=f"No data found for {filter_key_value_pairs = }.",  # noqa: G004 - low overhead
            )
            logger.warning(
                msg="Skipping this combination of parameters.",
            )
            continue

        # The identifier of the base model.
        # This value will be used to select the models for the correlation analysis
        # and add the estimates of the base model for the model checkpoint analysis.
        model_partial_name = filter_key_value_pairs["model_partial_name"]
        base_model_model_partial_name: str = derive_base_model_partial_name(
            model_partial_name=model_partial_name,
        )

        filtered_data_with_added_base_model: list[dict] = add_base_model_data(
            loaded_data=loaded_data,
            base_model_model_partial_name=base_model_model_partial_name,
            filter_key_value_pairs=filter_key_value_pairs,
            filtered_data=filtered_data,
            logger=logger,
        )

        # Sort the arrays by increasing model checkpoint.
        # Then from this point, the list of arrays and list of extracted checkpoints will be in the correct order.
        # 1. Step: Replace None model checkpoints with -1.
        model_checkpoint_column_name = "model_checkpoint"

        for single_dict in filtered_data_with_added_base_model:
            if single_dict[model_checkpoint_column_name] is None:
                single_dict[model_checkpoint_column_name] = -1
        # 2. Step: Call sorting function.
        sorted_data: list[dict] = sorted(
            filtered_data_with_added_base_model,
            key=lambda single_dict: int(single_dict[model_checkpoint_column_name]),
        )

        sorted_data_df = pd.DataFrame(
            data=sorted_data,
        )

        model_checkpoint_str_list: list[str] = [
            str(object=single_dict[model_checkpoint_column_name]) for single_dict in sorted_data
        ]

        # ========================================================== #
        # START: Load the corresponding model performance metrics

        scores_data: ScoresData = load_scores(
            embeddings_path_manager=embeddings_path_manager,
            filter_key_value_pairs=filter_key_value_pairs,
            verbosity=verbosity,
            logger=logger,
        )

        # END: Load the corresponding model performance metrics
        # ========================================================== #

        # # # #
        # Compute means

        # Create column with means
        sorted_data_df[f"{array_key_name}_mean"] = sorted_data_df[array_key_name].apply(
            func=lambda x: np.mean(x),
        )

        if verbosity >= Verbosity.NORMAL:
            log_dataframe_info(
                df=sorted_data_df,
                df_name="sorted_data_df",
                logger=logger,
            )

        # # # #
        # Save locations and saving the data

        plots_output_dir: pathlib.Path = (
            construct_mean_plots_over_model_checkpoints_output_dir_from_filter_key_value_pairs(
                output_root_dir=output_root_dir,
                filter_key_value_pairs=filter_key_value_pairs,
                verbosity=verbosity,
                logger=logger,
            )
        )

        # Save the sorted data list of dicts with the arrays to a pickle file.
        if save_plot_raw_data:
            plot_raw_data_save_dir = pathlib.Path(
                plots_output_dir,
                "raw_data",
            )
            plot_raw_data_save_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            sorted_data_df_save_path = pathlib.Path(
                plot_raw_data_save_dir,
                "sorted_data_df.csv",
            )

            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"Saving sorted data to {sorted_data_df_save_path = } ...",  # noqa: G004 - low overhead
                )
            sorted_data_df.to_csv(
                path_or_buf=sorted_data_df_save_path,
                index=False,
            )
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"Saving sorted data to {sorted_data_df_save_path = } DONE",  # noqa: G004 - low overhead
                )

            # Save the scores data if available
            scores_data.save_df(
                save_dir=plot_raw_data_save_dir,
                verbosity=verbosity,
                logger=logger,
            )

        ticks_and_labels: TicksAndLabels = TicksAndLabels(
            xlabel="checkpoints",
            ylabel=array_key_name,
            xticks_labels=model_checkpoint_str_list,
        )

        # # # #
        # Create plots
        for plot_size_config in plot_size_configs_list:
            # Convert the PlotSizeConfigFlat objects into the new dataclass format.

            secondary_axis_limits_list: list[AxisLimits] = [
                AxisLimits(),  # Automatic scaling
                AxisLimits(
                    y_min=0.0,
                    y_max=1.1,
                ),
            ]

            for secondary_axis_limits in secondary_axis_limits_list:
                plot_size_config_nested = PlotSizeConfigNested(
                    primary_axis_limits=AxisLimits(
                        x_min=plot_size_config.x_min,
                        x_max=plot_size_config.x_max,
                        y_min=plot_size_config.y_min,
                        y_max=plot_size_config.y_max,
                    ),
                    secondary_axis_limits=secondary_axis_limits,
                    output_dimensions=OutputDimensions(
                        output_pdf_width=3_000,
                        output_pdf_height=1_500,
                    ),
                )

                plot_local_estimates_with_individual_seeds_and_aggregated_over_seeds(
                    local_estimates_df=sorted_data_df,
                    ticks_and_labels=ticks_and_labels,
                    plot_size_config_nested=plot_size_config_nested,
                    scores_data=scores_data,
                    x_column_name=model_checkpoint_column_name,
                    filter_key_value_pairs=filter_key_value_pairs,
                    base_model_model_partial_name=base_model_model_partial_name,
                    plots_output_dir=plots_output_dir,
                    verbosity=verbosity,
                    logger=logger,
                )


def load_scores(
    embeddings_path_manager: EmbeddingsPathManager,
    filter_key_value_pairs: dict,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> ScoresData:
    match filter_key_value_pairs["model_partial_name"]:
        case "model=bert-base-uncased-ContextBERT-ERToD_emowoz_basic_setup_debug=-1_use_context=False":
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg="Loading scores for EmoLoop emotion models.",
                )

            seed_dfs: list[pd.DataFrame] = []
            columns_to_plot_set: set[str] = set()

            for seed in range(50, 55):
                parsed_data_path: pathlib.Path = pathlib.Path(
                    embeddings_path_manager.data_dir,
                    f"models/EmoLoop/output_dir/debug=-1/use_context=False/ep=5/seed={seed}/parsed_data/raw_data/parsed_data.csv",
                )

                file_loader = EmotionClassificationScoreLoader(
                    filepath=parsed_data_path,
                )
                scores_df: pd.DataFrame = file_loader.get_scores()
                scores_df["model_seed"] = seed  # Tag the dataframe with the current seed
                seed_dfs.append(scores_df)

                columns_to_plot: list[str] = file_loader.get_columns_to_plot()
                columns_to_plot_set.update(
                    columns_to_plot,
                )

            # Concatenate all seed dataframes into one
            if len(seed_dfs) == 0:
                logger.warning(
                    msg="No seed dataframes found.",
                )
                logger.info(
                    msg="Setting combined_scores_df to None for this model.",
                )
                combined_scores_df: pd.DataFrame | None = None
            else:
                combined_scores_df: pd.DataFrame | None = pd.concat(
                    objs=seed_dfs,
                    ignore_index=True,
                )

            combined_scores_columns_to_plot_list: list[str] | None = list(columns_to_plot_set)
        case "model=roberta-base-trippy_r_multiwoz21":
            seed_dfs: list[pd.DataFrame] = []
            columns_to_plot_set: set[str] = set()

            combined_scores_df = None
            combined_scores_columns_to_plot_list = None

            for seed in range(42, 45):
                results_folder_for_given_seed_path: pathlib.Path = pathlib.Path(
                    embeddings_path_manager.data_dir,
                    f"models/trippy_r_checkpoints/multiwoz21/all_checkpoints/results.{seed}",
                )

                file_loader = TrippyRScoreLoader(
                    results_folder_for_given_seed_path=results_folder_for_given_seed_path,
                    verbosity=verbosity,
                    logger=logger,
                )

                scores_df: pd.DataFrame = file_loader.get_scores()
                scores_df["model_seed"] = seed  # Tag the dataframe with the current seed
                seed_dfs.append(scores_df)

                columns_to_plot: list[str] = file_loader.get_columns_to_plot()
                columns_to_plot_set.update(
                    columns_to_plot,
                )

            # Concatenate all seed dataframes into one
            if len(seed_dfs) == 0:
                logger.warning(
                    msg="No seed dataframes found.",
                )
                logger.info(
                    msg="Setting combined_scores_df to None for this model.",
                )
                combined_scores_df: pd.DataFrame | None = None
            else:
                combined_scores_df: pd.DataFrame | None = pd.concat(
                    objs=seed_dfs,
                    ignore_index=True,
                )

            combined_scores_columns_to_plot_list: list[str] | None = list(columns_to_plot_set)

        # TODO: Implement this for language models (with performance given by loss)

        case _:
            logger.warning(
                msg=f"No specific model performance data loader implemented for "  # noqa: G004 - low overhead
                f"{filter_key_value_pairs['model_partial_name'] = }.",
            )
            logger.info(
                msg="Setting combined_scores_df to None for this model.",
            )
            combined_scores_df: pd.DataFrame | None = None
            combined_scores_columns_to_plot_list: list[str] | None = None

    scores_data = ScoresData(
        df=combined_scores_df,
        columns_to_plot=combined_scores_columns_to_plot_list,
    )

    return scores_data


def get_data_subsampling_split_from_data_subsampling_full(
    data_subsampling_full: str,
) -> str:
    """Extract the split from the full description of the data subsampling.

    For example, from:
    - 'split=test_samples=10000_sampling=random_sampling-seed=778' we extract 'test'.
    """
    split: str = data_subsampling_full.split(
        sep="_",
    )[0].split(
        sep="=",
    )[1]

    return split


def plot_local_estimates_with_individual_seeds_and_aggregated_over_seeds(
    local_estimates_df: pd.DataFrame,
    ticks_and_labels: TicksAndLabels,
    plot_size_config_nested: PlotSizeConfigNested,
    scores_data: ScoresData,
    *,
    x_column_name: str = "model_checkpoint",
    filter_key_value_pairs: dict,
    base_model_model_partial_name: str | None = None,
    plots_output_dir: pathlib.Path | None = None,
    show_plots: bool = False,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Plot local estimates for each model seed and a summary plot.

    Args:
        df:
            Input dataframe with columns:
            - 'model_checkpoint',
            - 'model_seed',
            - 'file_data_mean'

    """
    # # # #
    # Pre-process the local estimates data

    local_estimates_plot_data_df: pd.DataFrame = local_estimates_df[
        [
            x_column_name,
            "model_seed",
            "file_data_mean",
        ]
    ].copy()

    # Separate the checkpoint -1 data (no seeds associated)
    checkpoint_neg1 = local_estimates_plot_data_df[local_estimates_plot_data_df[x_column_name] == -1].iloc[0]
    seeds: np.ndarray = local_estimates_plot_data_df["model_seed"].dropna().unique().astype(dtype=int)

    # Emulate checkpoint -1 data for each seed
    neg1_data_emulated = pd.DataFrame(
        data={
            x_column_name: [-1] * len(seeds),
            "model_seed": seeds,
            "file_data_mean": [checkpoint_neg1["file_data_mean"]] * len(seeds),
        },
    )

    # Drop original -1 checkpoint and append the emulated data
    local_estimates_plot_data_df = local_estimates_plot_data_df[
        local_estimates_plot_data_df[x_column_name] != -1
    ].dropna()
    local_estimates_plot_data_df = pd.concat(
        objs=[
            neg1_data_emulated,
            local_estimates_plot_data_df,
        ],
        ignore_index=True,
    )

    # Ensure correct types
    local_estimates_plot_data_df = local_estimates_plot_data_df.astype(
        dtype={
            x_column_name: int,
            "model_seed": int,
            "file_data_mean": float,
        },
    )

    # # # #
    # Pre-process the scores data

    if "data_subsampling_full" not in filter_key_value_pairs:
        logger.warning(
            msg="No data_subsampling_full key found in filter_key_value_pairs.",
        )
        logger.info(
            msg="Skipping this plot call and returning from function now.",
        )
        return

    data_subsampling_split: str = get_data_subsampling_split_from_data_subsampling_full(
        data_subsampling_full=filter_key_value_pairs["data_subsampling_full"],
    )

    if scores_data.df is not None:
        if "data_subsampling_split" not in scores_data.df.columns:
            logger.warning(
                msg="No data_subsampling_split column found in scores_data.df.",
            )
            logger.info(
                msg="Will not modify the scores_df.",
            )
        else:
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"Filtering scores_df based on {data_subsampling_split = }",  # noqa: G004 - low overhead
                )
                logger.info(
                    msg=f"Shape before filtering: {scores_data.df.shape = }",  # noqa: G004 - low overhead
                )

            # Filter the scores_df based on the data_subsampling_split
            scores_data.df = scores_data.df[scores_data.df["data_subsampling_split"] == data_subsampling_split]

            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"Shape after filtering: {scores_data.df.shape = }",  # noqa: G004 - low overhead
                )

    # # # #
    # Create the containers for the plot data and plot configuration

    plot_input_data = PlotInputData(
        local_estimates_df=local_estimates_plot_data_df,
        scores=scores_data,
    )

    plot_config = PlotConfig(
        ticks_and_labels=ticks_and_labels,
        plot_size_config_nested=plot_size_config_nested,
        seeds=seeds,
        x_column_name=x_column_name,
        filter_key_value_pairs=filter_key_value_pairs,
        base_model_model_partial_name=base_model_model_partial_name,
        plots_output_dir=plots_output_dir,
        show_plots=show_plots,
    )

    # ========================================================= #
    # Plots: Individual by seed using a figure and axis.
    # ========================================================= #

    create_seedwise_estimate_visualization(
        data=plot_input_data,
        config=plot_config,
        verbosity=verbosity,
        logger=logger,
    )

    # ========================================================= #
    # Plots: Aggregated over seeds
    # ========================================================= #

    create_aggregate_estimate_visualization(
        data=plot_input_data,
        config=plot_config,
        verbosity=verbosity,
        logger=logger,
    )


def create_aggregate_estimate_visualization(
    data: PlotInputData,
    config: PlotConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Create a plot of the mean local estimates over checkpoints with standard deviation bands."""
    if "index" in data.local_estimates_df.columns:
        data.local_estimates_df = data.local_estimates_df.drop(
            columns=["index"],
        )

    # Create summary for local estimates with mean and standard deviation
    summary_local_estimates: pd.DataFrame = (
        data.local_estimates_df.groupby(
            by=config.x_column_name,
            as_index=False,
        )["file_data_mean"]
        .agg(
            func=[
                "mean",
                "std",
            ],
        )
        .reset_index()
    )

    if summary_local_estimates.empty:
        logger.warning(
            msg="No data available for plotting.",
        )
        logger.info(
            msg="Skipping this plot call and returning from function now.",
        )
        return

    if "index" in summary_local_estimates.columns:
        summary_local_estimates = summary_local_estimates.drop(
            columns=["index"],
        )

    summary_local_estimates.columns = [
        config.x_column_name,
        "mean",
        "std",
    ]

    # Explicitly handle NaNs in standard deviation (set to 0)
    summary_local_estimates["std"] = summary_local_estimates["std"].fillna(0)

    # Convert to NumPy arrays explicitly for matplotlib
    checkpoints = summary_local_estimates[config.x_column_name].to_numpy()
    means = summary_local_estimates["mean"].to_numpy()
    stds = summary_local_estimates["std"].to_numpy()

    # Create a summary figure and axis
    (
        fig,
        ax1,
    ) = plt.subplots(
        figsize=(
            config.plot_size_config_nested.output_dimensions.output_pdf_width / 100,
            config.plot_size_config_nested.output_dimensions.output_pdf_height / 100,
        ),
    )
    ax1.plot(
        checkpoints,
        means,
        marker="o",
        color="blue",
        label="Mean across seeds",
    )
    ax1.fill_between(
        x=checkpoints,
        y1=means - stds,
        y2=means + stds,
        color="blue",
        alpha=0.2,
        label="Standard Deviation",
    )

    ax1.set_title(
        label="Mean Local Estimates Over Checkpoints with Standard Deviation Band",
    )
    ax1.grid(
        visible=True,
    )

    ax1.set_xlabel(
        xlabel=config.ticks_and_labels.xlabel,
    )
    ax1.set_ylabel(
        ylabel=config.ticks_and_labels.ylabel,
    )

    # # # #
    # Plot the additional data if available

    # Add second y-axis for scores
    ax2 = ax1.twinx()

    if data.scores.df is not None and data.scores.columns_to_plot is not None:
        # Note:
        # - summary_scores is a DataFrame with a multilevel index
        summary_scores: pd.DataFrame = (
            data.scores.df.groupby(
                by=config.x_column_name,
                as_index=False,
            )[data.scores.columns_to_plot]
            .agg(
                func=[
                    "mean",
                    "std",
                ],
            )
            .reset_index()
        )

        if verbosity >= Verbosity.NORMAL:
            log_dataframe_info(
                df=summary_scores,
                df_name="summary_scores",
                logger=logger,
            )

        for column in data.scores.columns_to_plot:
            if column not in summary_scores.columns:
                logger.warning(
                    msg=f"{column=} not found in summary_scores DataFrame.",  # noqa: G004
                )
                continue

            checkpoints = summary_scores[config.x_column_name].to_numpy()
            means = summary_scores[
                column,
                "mean",
            ].to_numpy()
            stds = summary_scores[
                column,
                "std",
            ].to_numpy()

            ax2.plot(
                checkpoints,
                means,
                linestyle="--",
                marker="x",
                label=f"{column} (mean)",
            )
            ax2.fill_between(
                x=checkpoints,
                y1=means - stds,
                y2=means + stds,
                alpha=0.2,
            )

    # Optional: Set axis label once
    ax2.set_ylabel(
        ylabel="Scores",
        color="tab:red",
    )  # Customize label and color if desired
    ax2.tick_params(
        axis="y",
        labelcolor="tab:red",
    )

    # Combine legends from both axes
    (
        lines_1,
        labels_1,
    ) = ax1.get_legend_handles_labels()
    (
        lines_2,
        labels_2,
    ) = ax2.get_legend_handles_labels()
    ax1.legend(
        handles=lines_1 + lines_2,
        labels=labels_1 + labels_2,
        title="Legend",
    )

    # Set the y-axis limits
    ax1 = config.plot_size_config_nested.primary_axis_limits.set_y_axis_limits(
        axis=ax1,
    )
    ax2 = config.plot_size_config_nested.secondary_axis_limits.set_y_axis_limits(
        axis=ax2,
    )

    fixed_params_text: str = generate_fixed_parameters_text_from_dict(
        filters_dict=config.filter_key_value_pairs,
    )

    if fixed_params_text is not None:
        ax1.text(
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

    # Add info about the base model if available into the bottom left corner of the plot
    if config.base_model_model_partial_name is not None:
        ax1.text(
            x=0.01,
            y=0.01,
            s=f"{config.base_model_model_partial_name=}",
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

    fig.tight_layout()

    # Save the figure
    if config.plots_output_dir is not None:
        plot_name: str = f"local_estimates_aggregate_{config.plot_size_config_nested.y_range_description}"
        plot_save_path = pathlib.Path(
            config.plots_output_dir,
            "aggregate",
            f"{plot_name}.pdf",
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving plot to {plot_save_path = } ...",  # noqa: G004 - low overhead
            )

        plot_save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        fig.savefig(
            fname=plot_save_path,
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving plot to {plot_save_path = } DONE",  # noqa: G004 - low overhead
            )

    if config.show_plots:
        fig.show()


def create_seedwise_estimate_visualization(
    data: PlotInputData,
    config: PlotConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Visualize seed-wise estimates over model checkpoints."""
    (
        fig1,
        ax1,
    ) = plt.subplots(
        figsize=(
            config.plot_size_config_nested.output_dimensions.output_pdf_width / 100,
            config.plot_size_config_nested.output_dimensions.output_pdf_height / 100,
        ),
    )

    # # # #
    # Plot the mean of local estimates
    for seed in config.seeds:
        seed_data = data.local_estimates_df[data.local_estimates_df["model_seed"] == seed]
        ax1.plot(
            seed_data[config.x_column_name],
            seed_data["file_data_mean"],
            marker="o",
            label=f"{seed=}",
        )

    ax1.set_xlabel(xlabel="Model Checkpoint")
    ax1.set_ylabel(ylabel="File Data Mean")
    ax1.set_title(label="Local Estimates Over Checkpoints by Model Seed (including checkpoint -1)")

    ax1.grid(
        visible=True,
    )

    ax1.set_xlabel(
        xlabel=config.ticks_and_labels.xlabel,
    )
    ax1.set_ylabel(
        ylabel=config.ticks_and_labels.ylabel,
    )

    # # # #
    # Plot the additional data if available

    # Add second y-axis for scores
    ax2 = ax1.twinx()

    if data.scores.df is not None and data.scores.columns_to_plot is not None:
        for seed in config.seeds:
            seed_scores = data.scores.df[data.scores.df["model_seed"] == seed]
            for column in data.scores.columns_to_plot:
                ax2.plot(
                    seed_scores[config.x_column_name],
                    seed_scores[column],
                    linestyle="--",
                    marker="x",
                    label=f"{column} (seed={seed})",
                )

    # Optional: Set axis label once
    ax2.set_ylabel(
        ylabel="Scores",
        color="tab:red",
    )  # Customize label and color if desired
    ax2.tick_params(
        axis="y",
        labelcolor="tab:red",
    )

    # Combine legends from both axes
    (
        lines_1,
        labels_1,
    ) = ax1.get_legend_handles_labels()
    (
        lines_2,
        labels_2,
    ) = ax2.get_legend_handles_labels()
    ax1.legend(
        handles=lines_1 + lines_2,
        labels=labels_1 + labels_2,
        title="Legend",
    )

    # Set the y-axis limits
    ax1 = config.plot_size_config_nested.primary_axis_limits.set_y_axis_limits(
        axis=ax1,
    )
    ax2 = config.plot_size_config_nested.secondary_axis_limits.set_y_axis_limits(
        axis=ax2,
    )

    fixed_params_text: str = generate_fixed_parameters_text_from_dict(
        filters_dict=config.filter_key_value_pairs,
    )

    if fixed_params_text is not None:
        ax1.text(
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

    # Add info about the base model if available into the bottom left corner of the plot
    if config.base_model_model_partial_name is not None:
        ax1.text(
            x=0.01,
            y=0.01,
            s=f"{config.base_model_model_partial_name=}",
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

    fig1.tight_layout()

    # Save the figure
    if config.plots_output_dir is not None:
        plot_name: str = f"local_estimates_by_model_seed_{config.plot_size_config_nested.y_range_description}"
        plot_save_path = pathlib.Path(
            config.plots_output_dir,
            "separate_seeds",
            f"{plot_name}.pdf",
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving plot to {plot_save_path = } ...",  # noqa: G004 - low overhead
            )

        plot_save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        fig1.savefig(
            fname=plot_save_path,
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving plot to {plot_save_path = } DONE",  # noqa: G004 - low overhead
            )

    if config.show_plots:
        fig1.show()
