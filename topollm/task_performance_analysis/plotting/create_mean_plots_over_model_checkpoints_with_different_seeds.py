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
from topollm.plotting.line_plot_grouped_by_categorical_column import PlotSizeConfig
from topollm.task_performance_analysis.plotting.distribution_violinplots_and_distribution_boxplots import TicksAndLabels
from topollm.task_performance_analysis.plotting.parameter_combinations_and_loaded_data_handling import (
    add_base_model_data,
    construct_mean_plots_over_model_checkpoints_output_dir_from_filter_key_value_pairs,
    derive_base_model_partial_name,
    get_fixed_parameter_combinations,
)
from topollm.task_performance_analysis.plotting.score_loader.score_loader import EmotionClassificationScoreLoader
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def create_mean_plots_over_model_checkpoints_with_different_seeds(
    loaded_data: list[dict],
    array_key_name: str,
    output_root_dir: pathlib.Path,
    plot_size_configs_list: list[PlotSizeConfig],
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
        fixed_params_text: str = generate_fixed_parameters_text_from_dict(
            filters_dict=filter_key_value_pairs,
        )

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
        for single_dict in filtered_data_with_added_base_model:
            if single_dict["model_checkpoint"] is None:
                single_dict["model_checkpoint"] = -1
        # 2. Step: Call sorting function.
        sorted_data: list[dict] = sorted(
            filtered_data_with_added_base_model,
            key=lambda single_dict: int(single_dict["model_checkpoint"]),
        )

        sorted_data_df = pd.DataFrame(
            data=sorted_data,
        )

        model_checkpoint_str_list: list[str] = [
            str(object=single_dict["model_checkpoint"]) for single_dict in sorted_data
        ]

        # ========================================================== #
        # START: Load the corresponding model performance metrics

        match filter_key_value_pairs["model_partial_name"]:
            case "model=bert-base-uncased-ContextBERT-ERToD_emowoz_basic_setup_debug=-1_use_context=False":
                seed_dfs: list[pd.DataFrame] = []
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

                # Concatenate all seed dataframes into one
                combined_scores_df: pd.DataFrame | None = pd.concat(
                    objs=seed_dfs,
                    ignore_index=True,
                )

                pass  # TODO: This is here for setting breakpoints
            case _:
                logger.warning(
                    msg=f"No specific model performance data loader implemented for "  # noqa: G004 - low overhead
                    f"{filter_key_value_pairs['model_partial_name'] = }.",
                )
                logger.info(
                    f"Setting combined_scores_df to None for this model.",
                )
                combined_scores_df: pd.DataFrame | None = None

        # TODO: Load the correct model performance data/losses if it is available (we will use a custom protocol class for this)

        pass  # TODO: This is here for setting breakpoints
        # TODO: We will move this performance metrics loading into a function later

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

            sorted_data_df_save_path = pathlib.Path(
                plot_raw_data_save_dir,
                "sorted_data_df.csv",
            )
            plot_raw_data_save_dir.mkdir(
                parents=True,
                exist_ok=True,
            )
            sorted_data_df.to_csv(
                path_or_buf=sorted_data_df_save_path,
                index=False,
            )

        ticks_and_labels: TicksAndLabels = TicksAndLabels(
            xlabel="checkpoints",
            ylabel=array_key_name,
            xticks_labels=model_checkpoint_str_list,
        )

        # # # #
        # Create plots
        for plot_size_config in plot_size_configs_list:
            plot_local_estimates_for_different_seeds_and_aggregate(
                df=sorted_data_df,
                ticks_and_labels=ticks_and_labels,
                plot_size_config=plot_size_config,
                fixed_params_text=fixed_params_text,
                base_model_model_partial_name=base_model_model_partial_name,
                plots_output_dir=plots_output_dir,
                verbosity=verbosity,
                logger=logger,
            )


def plot_local_estimates_for_different_seeds_and_aggregate(
    df: pd.DataFrame,
    ticks_and_labels: TicksAndLabels,
    plot_size_config: PlotSizeConfig,
    *,
    fixed_params_text: str | None = None,
    base_model_model_partial_name: str | None = None,
    plots_output_dir: pathlib.Path | None = None,
    show_plots: bool = False,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Plot local estimates for each model seed and a summary plot.

    Args:
        df:
            Input dataframe with 'model_checkpoint', 'model_seed', and 'file_data_mean' columns.

    """
    plot_data: pd.DataFrame = df[
        [
            "model_checkpoint",
            "model_seed",
            "file_data_mean",
        ]
    ].copy()

    # Separate the checkpoint -1 data (no seeds associated)
    checkpoint_neg1 = plot_data[plot_data["model_checkpoint"] == -1].iloc[0]
    seeds = plot_data["model_seed"].dropna().unique().astype(int)

    # Emulate checkpoint -1 data for each seed
    neg1_data_emulated = pd.DataFrame(
        data={
            "model_checkpoint": [-1] * len(seeds),
            "model_seed": seeds,
            "file_data_mean": [checkpoint_neg1["file_data_mean"]] * len(seeds),
        },
    )

    # Drop original -1 checkpoint and append the emulated data
    plot_data = plot_data[plot_data["model_checkpoint"] != -1].dropna()
    plot_data = pd.concat([neg1_data_emulated, plot_data], ignore_index=True)

    # Ensure correct types
    plot_data = plot_data.astype(
        dtype={
            "model_checkpoint": int,
            "model_seed": int,
            "file_data_mean": float,
        },
    )

    # # # #
    # Individual plots by seed using a figure and axis

    (
        fig1,
        ax1,
    ) = plt.subplots(
        figsize=(
            plot_size_config.output_pdf_width / 100,
            plot_size_config.output_pdf_height / 100,
        ),
    )

    for seed in seeds:
        seed_data = plot_data[plot_data["model_seed"] == seed]
        ax1.plot(
            seed_data["model_checkpoint"],
            seed_data["file_data_mean"],
            marker="o",
            label=f"Seed {seed}",
        )

    ax1.set_xlabel("Model Checkpoint")
    ax1.set_ylabel("File Data Mean")
    ax1.set_title("Local Estimates Over Checkpoints by Model Seed (including checkpoint -1)")
    ax1.legend(title="Model Seed")
    ax1.grid(
        visible=True,
    )

    ax1.set_xlabel(
        xlabel=ticks_and_labels.xlabel,
    )
    ax1.set_ylabel(
        ylabel=ticks_and_labels.ylabel,
    )

    # Set the y-axis limits
    if plot_size_config.y_min is not None:
        ax1.set_ylim(
            bottom=plot_size_config.y_min,
        )
    if plot_size_config.y_max is not None:
        ax1.set_ylim(
            top=plot_size_config.y_max,
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
    if base_model_model_partial_name is not None:
        ax1.text(
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

    fig1.tight_layout()

    if plots_output_dir is not None:
        # Save the figure
        plot_name: str = f"local_estimates_by_model_seed_{plot_size_config.y_min}_{plot_size_config.y_max}"
        plot_save_path = pathlib.Path(
            plots_output_dir,
            f"{plot_name}.pdf",
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving plot to {plot_save_path = } ...",  # noqa: G004 - low overhead
            )
        fig1.savefig(
            fname=plot_save_path,
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving plot to {plot_save_path = } DONE",  # noqa: G004 - low overhead
            )

    if show_plots:
        plt.show()

    # # # #
    # Aggregated over seeds

    if "index" in plot_data.columns:
        plot_data = plot_data.drop(
            columns=["index"],
        )

    # Summary plot with mean and standard deviation
    summary: pd.DataFrame = (
        plot_data.groupby(
            by="model_checkpoint",
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

    if summary.empty:
        logger.warning(
            msg="No data available for plotting.",
        )
        logger.info(
            msg="Skipping this plot call and returning from function now.",
        )
        return

    if "index" in summary.columns:
        summary = summary.drop(columns=["index"])

    summary.columns = [
        "model_checkpoint",
        "mean",
        "std",
    ]

    # Explicitly handle NaNs in standard deviation (set to 0)
    summary["std"] = summary["std"].fillna(0)

    # Convert to NumPy arrays explicitly for matplotlib
    checkpoints = summary["model_checkpoint"].to_numpy()
    means = summary["mean"].to_numpy()
    stds = summary["std"].to_numpy()

    # Create a summary figure and axis
    (
        fig2,
        ax2,
    ) = plt.subplots(
        figsize=(
            plot_size_config.output_pdf_width / 100,
            plot_size_config.output_pdf_height / 100,
        ),
    )
    ax2.plot(
        checkpoints,
        means,
        marker="o",
        color="blue",
        label="Mean across seeds",
    )
    ax2.fill_between(
        x=checkpoints,
        y1=means - stds,
        y2=means + stds,
        color="blue",
        alpha=0.2,
        label="Standard Deviation",
    )

    ax2.set_xlabel(xlabel="Model Checkpoint")
    ax2.set_ylabel(ylabel="Mean File Data Mean")
    ax2.set_title(
        label="Mean Local Estimates Over Checkpoints with Standard Deviation Band",
    )
    ax2.legend()
    ax2.grid(
        visible=True,
    )

    ax2.set_xlabel(
        xlabel=ticks_and_labels.xlabel,
    )
    ax2.set_ylabel(
        ylabel=ticks_and_labels.ylabel,
    )

    # Set the y-axis limits
    if plot_size_config.y_min is not None:
        ax2.set_ylim(
            bottom=plot_size_config.y_min,
        )
    if plot_size_config.y_max is not None:
        ax2.set_ylim(
            top=plot_size_config.y_max,
        )

    if fixed_params_text is not None:
        ax2.text(
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
    if base_model_model_partial_name is not None:
        ax2.text(
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

    fig2.tight_layout()

    if plots_output_dir is not None:
        # Save the figure
        plot_name: str = f"local_estimates_aggregate_{plot_size_config.y_min}_{plot_size_config.y_max}"
        plot_save_path = pathlib.Path(
            plots_output_dir,
            f"{plot_name}.pdf",
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving plot to {plot_save_path = } ...",  # noqa: G004 - low overhead
            )
        fig2.savefig(
            fname=plot_save_path,
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving plot to {plot_save_path = } DONE",  # noqa: G004 - low overhead
            )

    if show_plots:
        plt.show()
