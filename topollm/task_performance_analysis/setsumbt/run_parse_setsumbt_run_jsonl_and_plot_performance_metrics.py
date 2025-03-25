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

"""Parse a SETSUMBT JSONL log file and create a plot of performance metrics versus checkpoints."""

import logging
import pathlib
import pickle
from typing import TYPE_CHECKING

import hydra
import numpy as np
import omegaconf
import pandas as pd

from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.config_classes.setup_OmegaConf import setup_omega_conf
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.task_performance_analysis.plotting.parameter_combinations_and_loaded_data_handling import (
    construct_plots_over_checkpoints_output_dir_from_filter_key_value_pairs,
)
from topollm.task_performance_analysis.plotting.plot_performance_metrics import plot_performance_metrics
from topollm.task_performance_analysis.setsumbt.parse_setsumbt_run_jsonl_log_file import (
    parse_setsumbt_run_jsonl_log_file,
)
from topollm.typing.enums import DescriptionType, Verbosity

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

    array_key_name: str = "file_data"

    # ================================================== #
    # Parse the SetSUMBT log file and print a summary of the DataFrame.
    # ================================================== #

    if main_config.language_model.model_log_file_path is None:
        logger.error(
            msg="The path to the SETSUMBT log file is not set in the configuration.",
        )
        return

    log_filepath = pathlib.Path(
        main_config.language_model.model_log_file_path,
    )

    try:
        parsed_df: pd.DataFrame = parse_setsumbt_run_jsonl_log_file(
            file_path=log_filepath,
        )
    except FileNotFoundError as e:
        logger.exception(
            msg=f"Error reading log file: {e}",  # noqa: G004 - low overhead
        )
        return

    # ================================================== #
    # Saved parsed DataFrame
    # ================================================== #

    output_root_dir: pathlib.Path = pathlib.Path(
        embeddings_path_manager.saved_plots_dir_absolute_path,
        "task_performance_analysis",
        "model_performance_metrics",
        main_config.language_model.get_config_description(
            description_type=DescriptionType.LONG,
        ),
    )
    output_root_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file_path: pathlib.Path = pathlib.Path(
        output_root_dir,
        "parsed_setsumbt_run_df.csv",
    )
    parsed_df.to_csv(
        path_or_buf=output_file_path,
        index=False,
    )

    # ================================================== #
    # Optional:
    # Load the local estimates distributions
    # which can be added to the performance plots.
    # ================================================== #

    # Note:
    # In the future, we might want to fill in the values in the dict here from the config.
    filter_key_value_pairs: dict = {
        "tokenizer_add_prefix_space": "False",
        "data_full": "data=setsumbt_dataloaders_processed_0_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags",
        "data_subsampling_full": "split=dev_samples=10000_sampling=random_sampling-seed=778",
        "model_layer": -1,
        "model_partial_name": "model=roberta-base-setsumbt_multiwoz21",
        "model_seed": main_config.language_model.seed,
    }

    # We saved the local estimates distributions in the plots over checkpoints directory.
    plots_over_checkpoints_output_dir: pathlib.Path = (
        construct_plots_over_checkpoints_output_dir_from_filter_key_value_pairs(
            output_root_dir=embeddings_path_manager.get_saved_plots_distribution_of_local_estimates_dir_absolute_path(),
            filter_key_value_pairs=filter_key_value_pairs,
            verbosity=verbosity,
            logger=logger,
        )
    )

    sorted_data_output_file_path: pathlib.Path = pathlib.Path(
        plots_over_checkpoints_output_dir,
        "sorted_data_list_of_dicts_with_arrays.pkl",
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Trying to load sorted data from {sorted_data_output_file_path = } ...",  # noqa: G004 - low overhead
        )

    try:
        with sorted_data_output_file_path.open(
            mode="rb",
        ) as f:
            loaded_sorted_local_estimates_data = pickle.load(  # noqa: S301 - we only use this for trusted data
                file=f,
            )
    except FileNotFoundError as e:
        logger.warning(
            msg=f"Error reading local estimates file: {e}",  # noqa: G004 - low overhead
        )
        logger.warning(
            msg="The local estimates distribution will not be added to the performance plots.",
        )
        loaded_sorted_local_estimates_data = None

    if loaded_sorted_local_estimates_data is not None:
        model_checkpoint_str_list: list[str] = [
            str(object=single_dict["model_checkpoint"]) for single_dict in loaded_sorted_local_estimates_data
        ]

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"{model_checkpoint_str_list=}",  # noqa: G004 - low overhead
            )

    # ================================================== #
    # Create plot of the performance
    # ================================================== #

    # Specify which columns to plot on the primary and secondary axes.
    primary_metrics: list[str] = [
        "loss",
        "Joint Goal Accuracy",
        "Slot F1 Score",
        "Slot Precision",
        "Slot Recall",
    ]
    secondary_metrics: list[str] = [
        "Joint Goal ECE",
        "Joint Goal L2-Error",
        "Joint Goal L2-Error Ratio",
    ]

    highlight_columns: list[str] = [
        "Joint Goal Accuracy",
        "Slot F1 Score",
        "Slot Precision",
        "Slot Recall",
    ]

    # Call the plotting function.
    # Leave primary_ylim and secondary_ylim as None for auto-scaling,
    # or supply a tuple (min, max) if you want fixed limits.
    plot_performance_metrics(
        df=parsed_df,
        x_col="checkpoint",
        primary_y_cols=primary_metrics,
        secondary_y_cols=secondary_metrics,
        title="Development of Performance Metrics through Checkpoints",
        xlabel="Checkpoint",
        primary_ylabel="Metric (Primary Scale)",
        secondary_ylabel="Metric (Secondary Scale)",
        primary_ylim=None,  # e.g., None for auto
        secondary_ylim=None,  # e.g., None for auto
        output_root_dir=output_root_dir,
        figsize=(
            20,
            14,
        ),
        highlight_best=highlight_columns,
        loaded_sorted_local_estimates_data=loaded_sorted_local_estimates_data,
        array_key_name=array_key_name,
        local_estimates_limits=(
            0.0,
            15.0,
        ),
        verbosity=verbosity,
        logger=logger,
    )

    logger.info(
        msg="Running script DONE",
    )


if __name__ == "__main__":
    # Note: See the VSCode launch configurations for an example of how to run this script.

    setup_omega_conf()

    main()
