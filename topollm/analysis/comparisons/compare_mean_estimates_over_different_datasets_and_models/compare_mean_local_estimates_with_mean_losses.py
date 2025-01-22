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

"""Create plots to compare mean local estimates with mean losses for different models."""

import itertools
import json
import logging
import pathlib
import pprint
from typing import TYPE_CHECKING

import hydra
import omegaconf
import pandas as pd
from tqdm import tqdm

from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.config_classes.setup_OmegaConf import setup_omega_conf
from topollm.data_processing.dictionary_handling import flatten_dict
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.log_dataframe_info import log_dataframe_info
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.path_management.parse_path_info import parse_path_info_full
from topollm.plotting.create_scatter_plot import create_scatter_plot
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
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

setup_omega_conf()


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
    #
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

    # The following directory contains the different dataset folders.
    # Logging of the directory is done in the 'load_descriptive_statistics_from_folder_structure' function.
    iteration_root_dir = pathlib.Path(
        embeddings_path_manager.analysis_dir,
        "distances_and_influence_on_losses_and_local_estimates",
        main_config.analysis.investigate_distances.get_config_description(),
        main_config.local_estimates.method_description,  # For example: 'twonn'
    )

    descriptive_statistics_df: pd.DataFrame = load_descriptive_statistics_from_folder_structure(
        iteration_root_dir=iteration_root_dir,
        verbosity=verbosity,
        logger=logger,
    )

    # # # #
    # Filter the DataFrame for selected settings
    tokenizer_add_prefix_space = "False"

    filtered_descriptive_statistics_df: pd.DataFrame = descriptive_statistics_df.copy()
    filtered_descriptive_statistics_df = filtered_descriptive_statistics_df[
        (filtered_descriptive_statistics_df["tokenizer_add_prefix_space"] == tokenizer_add_prefix_space)
    ]

    output_root_dir: pathlib.Path = pathlib.Path(
        embeddings_path_manager.saved_plots_dir_absolute_path,
        "compare_mean_local_estimates_with_mean_losses_for_different_models",
        f"{tokenizer_add_prefix_space=}",
        main_config.analysis.investigate_distances.get_config_description(),
        main_config.local_estimates.method_description,  # For example: 'twonn'
    )
    output_root_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save descriptive_statistics_df and filtered_descriptive_statistics_df to CSV
    descriptive_statistics_df.to_csv(
        path_or_buf=output_root_dir / "descriptive_statistics_df.csv",
        index=False,
    )
    filtered_descriptive_statistics_df.to_csv(
        path_or_buf=output_root_dir / "filtered_descriptive_statistics_df.csv",
        index=False,
    )

    compare_mean_local_estimates_with_mean_losses_for_different_models(
        descriptive_statistics_df=filtered_descriptive_statistics_df,
        output_root_dir=output_root_dir,
        verbosity=verbosity,
        logger=logger,
    )

    logger.info(
        msg="Script finished.",
    )


def load_descriptive_statistics_from_folder_structure(
    iteration_root_dir: pathlib.Path,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> pd.DataFrame:
    """Load descriptive statistics from the folder structure and organize them in a DataFrame."""
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{iteration_root_dir = }",  # noqa: G004 - low overhead
        )

    # Iterate over the different dataset folders in the 'iteration_root_dir' directory
    rootdir: pathlib.Path = iteration_root_dir
    # Only match the 'descriptive_statistics_dict.json' files
    pattern: str = "**/*.json"
    file_path_list: list[pathlib.Path] = [
        f
        for f in rootdir.resolve().glob(
            pattern=pattern,
        )
        if f.is_file()
    ]

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{len(file_path_list) = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"file_list:\n{pprint.pformat(object=file_path_list)}",  # noqa: G004 - low overhead
        )

    # Full path example:
    #
    # data/analysis/distances_and_influence_on_losses_and_local_estimates/a-tr-s=60000/twonn/
    # data=iclr_2024_submissions_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags/
    # split=validation_samples=10000_sampling=random_sampling-seed=777/edh-mode=masked_token_lvl=token/add-prefix-space=True_max-len=512/
    # model=roberta-base-masked_lm-defaults_multiwoz21-rm-empty-True-do_nothing-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5_seed-1234_ckpt-800_task=masked_lm_dr=defaults/
    # layer=-1_agg=mean/norm=None/sampling=random_seed=42_samples=150000/
    # desc=twonn_samples=60000_zerovec=keep_dedup=array_deduplicator_noise=do_nothing/
    # descriptive_statistics_dict.json
    #
    # Example dataset folder:
    #
    # data=one-year-of-tsla-on-reddit_rm-empty=True_spl-mode=proportions_spl-shuf=True_spl-seed=0_tr=0.8_va=0.1_te=0.1_ctxt=dataset_entry_feat-col=ner_tags

    loaded_data_list: list[dict] = []

    for file_path in tqdm(
        iterable=file_path_list,
        desc="Loading data from files",
    ):
        with file_path.open(
            mode="r",
        ) as file:
            file_data: dict = json.load(
                fp=file,
            )

            flattened_file_data: dict = flatten_dict(
                d=file_data,
                separator="_",
            )

            path_info: dict = parse_path_info_full(
                path=file_path,
            )

            # Combine the path information with the flattened file data
            combined_data: dict = {
                **path_info,
                **flattened_file_data,
                "file_path": str(object=file_path),
            }

            loaded_data_list.append(
                combined_data,
            )

    # Convert the list of dictionaries to a DataFrame
    result_df = pd.DataFrame(
        data=loaded_data_list,
    )

    return result_df


def compare_mean_local_estimates_with_mean_losses_for_different_models(
    descriptive_statistics_df: pd.DataFrame,
    output_root_dir: pathlib.Path,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Create plots with compare mean local estimates with mean losses for different models."""
    # # # #
    # Common parameters for all plots
    axes_limits_choices: list[dict] = [
        {
            "x_min": None,
            "x_max": None,
            "y_min": None,
            "y_max": None,
            "output_pdf_width": 2_000,
            "output_pdf_height": 2_000,
        },
        {
            "x_min": 5.5,
            "x_max": 18.5,
            "y_min": 1.0,
            "y_max": 4.5,
            "output_pdf_width": 2_000,
            "output_pdf_height": 2_000,
        },
    ]
    x_column_name = "local_estimates_mean"
    y_column_name = "loss_mean"

    data_full_options: list[str] = descriptive_statistics_df["data_full"].unique().tolist()
    data_subsampling_full_options: list[str] = descriptive_statistics_df["data_subsampling_full"].unique().tolist()
    # > Example: data_subsampling_full = "split=validation_samples=10000_sampling=random_sampling-seed=777"
    model_partial_name_options: list[str] = descriptive_statistics_df["model_partial_name"].unique().tolist()

    # The identifier of the base model.
    # This value will be used to filter the DataFrame for the correlation analysis and for the model checkpoint analysis.
    base_model_model_partial_name = "model=roberta-base"

    # ========================================================== #
    # Create a common plot for
    # - all datasets
    # - all splits
    # - all models together
    # ========================================================== #

    descriptive_statistics_df_copy: pd.DataFrame = descriptive_statistics_df.copy()

    # No filtering in this case
    filtered_df: pd.DataFrame = descriptive_statistics_df_copy

    output_folder = pathlib.Path(
        output_root_dir,
        "plots_for_all_splits_and_all_datasets_and_all_models",
    )
    subtitle_text: str = "all_splits_and_all_datasets_and_all_models"

    # We use the point size to indicate the subsampling.
    # For this to work, we need to create a mapped column that contains the point size.
    size_mapping_dict: dict = {
        "split=train_samples=10000_sampling=take_first": 5,
        "split=validation_samples=10000_sampling=random_sampling-seed=777": 10,
    }
    filtered_df["size_column"] = filtered_df["data_subsampling_full"].map(arg=size_mapping_dict)
    # Fill NaN values with a default value
    filtered_df["size_column"] = filtered_df["size_column"].fillna(
        value=3,
    )

    for axes_limits in axes_limits_choices:
        plot_name: str = (
            f"{x_column_name}_vs_{y_column_name}"
            f"_{axes_limits['x_min']}_{axes_limits['x_max']}"
            f"_{axes_limits['y_min']}_{axes_limits['y_max']}"
        )

        create_scatter_plot(
            df=filtered_df,
            output_folder=output_folder,
            plot_name=plot_name,
            subtitle_text=subtitle_text,
            x_column_name=x_column_name,
            y_column_name=y_column_name,
            color_column_name="data_full",
            symbol_column_name="model_partial_name",
            size_column_name="size_column",
            hover_data=filtered_df.columns.tolist(),
            **axes_limits,
            show_plot=False,
            verbosity=verbosity,
            logger=logger,
        )

    # ========================================================== #
    # Create separate plots for:
    # - individual splits
    # - all datasets together
    # - all models together
    # ========================================================== #

    combinations = itertools.product(
        data_subsampling_full_options,
    )

    # Note: If combinations contains only one element, the unpacked value still needs to be put into a tuple.
    for (data_subsampling_full,) in tqdm(
        iterable=combinations,
        desc="Iterating over different plot combinations",
    ):
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"{data_subsampling_full = }",  # noqa: G004 - low overhead
            )

        output_folder = pathlib.Path(
            output_root_dir,
            "plots_for_individual_splits_and_all_datasets_and_all_models",
            f"{data_subsampling_full=}",
        )
        subtitle_text: str = f"{data_subsampling_full=}"

        # Only filter by the subsampling
        descriptive_statistics_df_copy: pd.DataFrame = descriptive_statistics_df.copy()

        filtered_df: pd.DataFrame = descriptive_statistics_df_copy[
            (descriptive_statistics_df_copy["data_subsampling_full"] == data_subsampling_full)
        ]

        for axes_limits in axes_limits_choices:
            plot_name: str = (
                f"{x_column_name}_vs_{y_column_name}"
                f"_{axes_limits['x_min']}_{axes_limits['x_max']}"
                f"_{axes_limits['y_min']}_{axes_limits['y_max']}"
            )

            create_scatter_plot(
                df=filtered_df,
                output_folder=output_folder,
                plot_name=plot_name,
                subtitle_text=subtitle_text,
                x_column_name=x_column_name,
                y_column_name=y_column_name,
                color_column_name="data_full",
                symbol_column_name="model_partial_name",
                size_column_name=None,
                hover_data=filtered_df.columns.tolist(),
                **axes_limits,
                show_plot=False,
                verbosity=verbosity,
                logger=logger,
            )

    # TODO: Compute correlation between the mean values under comparison

    # ========================================================== #
    # Create separate plots for:
    # - different splits
    # - different datasets
    # - all models together
    # ========================================================== #

    combinations = itertools.product(
        data_full_options,
        data_subsampling_full_options,
    )

    # Note: If combinations contains only one element, the unpacked value still needs to be put into a tuple.
    for (
        data_full,
        data_subsampling_full,
    ) in tqdm(
        iterable=combinations,
        desc="Iterating over different plot combinations",
    ):
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"{data_full = }",  # noqa: G004 - low overhead
            )
            logger.info(
                msg=f"{data_subsampling_full = }",  # noqa: G004 - low overhead
            )

        output_folder = pathlib.Path(
            output_root_dir,
            "plots_for_individual_splits_and_individual_datasets_and_all_models",
            f"{data_full=}",
            f"{data_subsampling_full=}",
        )
        subtitle_text: str = f"{data_full=}, {data_subsampling_full=}"

        # Make a copy of the DataFrame so that we do not modify the original DataFrame
        descriptive_statistics_df_copy: pd.DataFrame = descriptive_statistics_df.copy()

        # Filter the DataFrame:
        # - We want to make separate plots for each dataset and split.
        filtered_df: pd.DataFrame = descriptive_statistics_df_copy[
            (descriptive_statistics_df_copy["data_full"] == data_full)
            & (descriptive_statistics_df_copy["data_subsampling_full"] == data_subsampling_full)
        ]
        if verbosity >= Verbosity.NORMAL:
            log_dataframe_info(
                df=filtered_df,
                df_name="filtered_df",
                logger=logger,
            )

        # Check that certain columns only contain a unique value
        # This is important for consistency in the plots.
        columns_to_check_for_uniqueness: list[str] = [
            "data_full",
            "data_subsampling_full",
            "embedding_data_handler_full",
            "local_estimates_desc_full",
        ]
        for column_name in columns_to_check_for_uniqueness:
            unique_values: pd.Series = filtered_df[column_name].unique()  # type: ignore - problem with pandas typing
            if len(unique_values) == 0:
                logger.warning(
                    msg=f"Column '{column_name = }' does not contain any values.",  # noqa: G004 - low overhead
                )
            elif len(unique_values) != 1:
                msg: str = f"Column '{column_name = }' does not contain a unique value. Found: {unique_values = }"
                raise ValueError(
                    msg,
                )

        for axes_limits in axes_limits_choices:
            plot_name: str = (
                f"{x_column_name}_vs_{y_column_name}"
                f"_{axes_limits['x_min']}_{axes_limits['x_max']}"
                f"_{axes_limits['y_min']}_{axes_limits['y_max']}"
            )
            # - Use the 'model_checkpoint' column for the color
            # - Use the training data description for the model as the symbol
            create_scatter_plot(
                df=filtered_df,
                output_folder=output_folder,
                plot_name=plot_name,
                subtitle_text=subtitle_text,
                x_column_name=x_column_name,
                y_column_name=y_column_name,
                color_column_name="model_checkpoint",
                symbol_column_name="model_partial_name",
                size_column_name=None,
                hover_data=filtered_df.columns.tolist(),
                **axes_limits,
                show_plot=False,
                verbosity=verbosity,
                logger=logger,
            )

    # ========================================================== #
    # Create separate plots for:
    # - different splits
    # - a single model finetuning (plus base model)
    # - all datasets together
    # ========================================================== #

    combinations = itertools.product(
        data_subsampling_full_options,
        model_partial_name_options,
    )

    # Note: If combinations contains only one element, the unpacked value still needs to be put into a tuple.
    for (
        data_subsampling_full,
        model_partial_name,
    ) in tqdm(
        iterable=combinations,
        desc="Iterating over different plot combinations",
    ):
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"{data_subsampling_full = }",  # noqa: G004 - low overhead
            )
            logger.info(
                msg=f"{model_partial_name = }",  # noqa: G004 - low overhead
            )

        output_folder = pathlib.Path(
            output_root_dir,
            "plots_for_individual_splits_and_all_datasets_and_individual_models",
            f"{data_subsampling_full=}",
            f"{model_partial_name=}",
        )
        subtitle_text: str = f"{data_subsampling_full=}, {model_partial_name=}"

        # Make a copy of the DataFrame so that we do not modify the original DataFrame
        descriptive_statistics_df_copy: pd.DataFrame = descriptive_statistics_df.copy()

        # Filter the DataFrame:
        # - We want to make separate plots for each dataset and split.
        filtered_df: pd.DataFrame = descriptive_statistics_df_copy[
            (descriptive_statistics_df_copy["data_subsampling_full"] == data_subsampling_full)
            & (
                (descriptive_statistics_df_copy["model_partial_name"] == model_partial_name)
                | (descriptive_statistics_df_copy["model_partial_name"] == base_model_model_partial_name)
            )
        ]
        if verbosity >= Verbosity.NORMAL:
            log_dataframe_info(
                df=filtered_df,
                df_name="filtered_df",
                logger=logger,
            )

        for axes_limits in axes_limits_choices:
            plot_name: str = (
                f"{x_column_name}_vs_{y_column_name}"
                f"_{axes_limits['x_min']}_{axes_limits['x_max']}"
                f"_{axes_limits['y_min']}_{axes_limits['y_max']}"
            )
            # - Use the 'model_checkpoint' column for the color
            # - Use the training data description for the model as the symbol
            create_scatter_plot(
                df=filtered_df,
                output_folder=output_folder,
                plot_name=plot_name,
                subtitle_text=subtitle_text,
                x_column_name=x_column_name,
                y_column_name=y_column_name,
                color_column_name="model_checkpoint",
                symbol_column_name="data_full",
                size_column_name=None,
                hover_data=filtered_df.columns.tolist(),
                **axes_limits,
                show_plot=False,
                verbosity=verbosity,
                logger=logger,
            )


if __name__ == "__main__":
    main()
