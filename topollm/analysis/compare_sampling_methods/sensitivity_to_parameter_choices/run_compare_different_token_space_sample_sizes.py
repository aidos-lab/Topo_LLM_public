"""Run script to create embedding vectors from dataset based on config."""

import logging
import pathlib
import re
from itertools import product
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import numpy as np
import omegaconf
import pandas as pd
from tqdm import tqdm

from topollm.analysis.compare_sampling_methods.compute_correlations import compute_and_save_correlations
from topollm.analysis.compare_sampling_methods.data_selection_folder_lists import get_data_folder_list
from topollm.analysis.compare_sampling_methods.log_statistics import log_statistics_of_array
from topollm.analysis.compare_sampling_methods.make_plots import make_mean_std_plot, make_multiple_line_plots
from topollm.analysis.compare_sampling_methods.organize_results_directory_structure import (
    build_results_directory_structure,
)
from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.log_array_info import log_array_info
from topollm.logging.log_dataframe_info import log_dataframe_info
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig
    from topollm.path_management.embeddings.protocol import EmbeddingsPathManager


# logger for this file
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
    """Run the script."""
    logger: logging.Logger = global_logger
    logger.info(
        msg="Running script ...",
    )

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=logger,
    )
    verbosity: Verbosity = main_config.verbosity

    embeddings_path_manager: EmbeddingsPathManager = get_embeddings_path_manager(
        main_config=main_config,
        logger=logger,
    )

    mode = "create_all_plots"

    if mode == "create_all_plots":
        data_folder_list: list[str] = get_data_folder_list()
        model_folder_list: list[str] = [
            "model-roberta-base_task-masked_lm",
            "model-model-roberta-base_task-masked_lm_multiwoz21-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1234_ckpt-14400_task-masked_lm",
            "model-model-roberta-base_task-masked_lm_multiwoz21-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1234_ckpt-31200_task-masked_lm",
            "model-model-roberta-base_task-masked_lm_multiwoz21-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1234_ckpt-400_task-masked_lm",
            "model-model-roberta-base_task-masked_lm_multiwoz21-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1235_ckpt-14400_task-masked_lm",
            "model-model-roberta-base_task-masked_lm_multiwoz21-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1235_ckpt-31200_task-masked_lm",
            "model-model-roberta-base_task-masked_lm_multiwoz21-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1235_ckpt-400_task-masked_lm",
            "model-model-roberta-base_task-masked_lm_one-year-of-tsla-on-reddit-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1234_ckpt-14400_task-masked_lm",
            "model-model-roberta-base_task-masked_lm_one-year-of-tsla-on-reddit-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1234_ckpt-31200_task-masked_lm",
            "model-model-roberta-base_task-masked_lm_one-year-of-tsla-on-reddit-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1234_ckpt-400_task-masked_lm",
            "model-model-roberta-base_task-masked_lm_one-year-of-tsla-on-reddit-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1235_ckpt-14400_task-masked_lm",
            "model-model-roberta-base_task-masked_lm_one-year-of-tsla-on-reddit-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1235_ckpt-31200_task-masked_lm",
            "model-model-roberta-base_task-masked_lm_one-year-of-tsla-on-reddit-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50_seed-1235_ckpt-400_task-masked_lm",
        ]
        embeddings_data_prep_sampling_folder_list: list[str] = [
            "sampling-random_seed-42_samples-30000",
            "sampling-take_first_seed-42_samples-30000",
        ]
    elif mode == "create_debug_plots":
        data_folder_list: list[str] = [
            "data-multiwoz21_split-train_ctxt-dataset_entry_samples-10000_feat-col-ner_tags",
        ]
        model_folder_list: list[str] = [
            "model-roberta-base_task-masked_lm",
        ]
        embeddings_data_prep_sampling_folder_list: list[str] = [
            "sampling-random_seed-42_samples-30000",
            "sampling-take_first_seed-42_samples-30000",
        ]
    else:
        msg = "Invalid mode"
        raise ValueError(
            msg,
        )

    for data_folder, model_folder, embeddings_data_prep_sampling_folder in tqdm(
        iterable=product(
            data_folder_list,
            model_folder_list,
            embeddings_data_prep_sampling_folder_list,
        ),
        desc="Iterating over folder choices",
    ):
        analysis_base_directory: pathlib.Path = pathlib.Path(
            embeddings_path_manager.data_dir,
            "analysis",
            "twonn",
            data_folder,
            "lvl-token",
            "add-prefix-space-True_max-len-512",
            model_folder,
            "layer--1_agg-mean",
            "norm-None",
            embeddings_data_prep_sampling_folder,
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"{analysis_base_directory = }",  # noqa: G004 - low overhead
            )

        if not analysis_base_directory.exists():
            logger.warning(
                msg=f"Directory does not exist: {analysis_base_directory = }",  # noqa: G004 - low overhead
            )
            continue

        run_comparison_for_analysis_base_directory(
            analysis_base_directory=analysis_base_directory,
            data_dir=embeddings_path_manager.data_dir,
            verbosity=verbosity,
            logger=logger,
        )

    logger.info(
        msg="Running script DONE",
    )


def run_comparison_for_analysis_base_directory(
    analysis_base_directory: pathlib.Path,
    data_dir: pathlib.Path,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    # Define the new base directory for saving results
    results_base_directory: pathlib.Path = build_results_directory_structure(
        analysis_base_directory=analysis_base_directory,
        data_dir=data_dir,
        analysis_output_subdirectory_partial_relative_path=pathlib.Path(
            "sample_sizes",
        ),
        verbosity=verbosity,
        logger=logger,
    )

    array_truncation_size: int = 2500

    # Discover directories matching the expected pattern
    # (e.g., "desc-twonn_samples-<sample_size>_zerovec-keep")
    pattern = re.compile(
        pattern=r"desc-twonn_samples-(\d+)_zerovec-keep",
    )

    # Iterate through the folders in the base directory
    sorted_df, arrays_truncated_stacked = process_subdirectories(
        analysis_base_directory=analysis_base_directory,
        pattern=pattern,
        truncation_size=array_truncation_size,
        verbosity=verbosity,
        logger=logger,
    )

    # Save the results DataFrame to a CSV file for archiving
    sorted_df_save_path: pathlib.Path = results_base_directory / "sorted_df.csv"
    sorted_df.to_csv(
        path_or_buf=sorted_df_save_path,
        index=False,
    )

    arrays_truncated_stacked_plot_save_path: pathlib.Path = results_base_directory / "arrays_truncated_stacked.pdf"
    make_multiple_line_plots(
        array=arrays_truncated_stacked,
        sample_sizes=sorted_df["sample_size"].to_numpy(),
        additional_title=str(analysis_base_directory),
        show_plot=False,
        save_path=arrays_truncated_stacked_plot_save_path,
    )

    compute_and_save_correlations(
        arrays_truncated_stacked=arrays_truncated_stacked,
        results_base_directory=results_base_directory,
        verbosity=verbosity,
        logger=logger,
    )

    sorted_df_plot_save_path: pathlib.Path = results_base_directory / "sorted_df.pdf"
    make_mean_std_plot(
        sorted_df=sorted_df,
        additional_title=str(analysis_base_directory),
        show_plot=False,
        save_path=sorted_df_plot_save_path,
    )


def process_subdirectories(
    analysis_base_directory: pathlib.Path,
    pattern: re.Pattern,
    truncation_size: int,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
]:
    """Process the subdirectories of the analysis base directory."""
    arrays_truncated_list = []
    mean_list = []
    std_list = []
    sample_sizes_list: list[int] = []

    for subdirectory in analysis_base_directory.iterdir():
        if subdirectory.is_dir():
            match = pattern.match(
                string=subdirectory.name,
            )
            if match:
                if verbosity >= Verbosity.NORMAL:
                    logger.info(
                        msg=f"{match = }",  # noqa: G004 - low overhead
                    )
                    logger.info(
                        msg=f"Processing subdirectory: {subdirectory = }",  # noqa: G004 - low overhead
                    )

                sample_size = int(match.group(1))
                sample_sizes_list.append(
                    sample_size,
                )

                # Load the array from the current directory
                current_array_path: pathlib.Path = subdirectory / "local_estimates_paddings_removed.npy"
                current_array = np.load(
                    file=current_array_path,
                )

                if verbosity >= Verbosity.NORMAL:
                    log_statistics_of_array(
                        array=current_array,
                        array_name=f"current_array {subdirectory = }",
                        logger=logger,
                    )

                # Truncate the arrays to the first common elements, so that we can compare them
                current_array_truncated = current_array[:truncation_size,]

                arrays_truncated_list.append(
                    current_array_truncated,
                )
                mean_list.append(
                    np.mean(current_array_truncated),
                )
                std_list.append(
                    np.std(current_array_truncated, ddof=1),
                )

    # Create a DataFrame to store results
    results_df = pd.DataFrame(
        data={
            "sample_size": sample_sizes_list,
            "mean": mean_list,
            "std": std_list,
        },
    )

    if verbosity >= Verbosity.NORMAL:
        log_dataframe_info(
            df=results_df,
            df_name="results_df (before sorting)",
            logger=logger,
        )

    # Sort the DataFrame by sample_size.
    # Do not reset the index, as we want to know the original order of the arrays.
    sorted_df: pd.DataFrame = results_df.sort_values(
        by="sample_size",
    )

    if verbosity >= Verbosity.NORMAL:
        log_dataframe_info(
            df=sorted_df,
            df_name="sorted_df",
            logger=logger,
        )

    # Sort the stacked arrays accordingly
    sorted_indices = sorted_df.index.to_numpy()
    arrays_truncated_sorted = [arrays_truncated_list[i] for i in sorted_indices]
    arrays_truncated_stacked = np.stack(
        arrays_truncated_sorted,
        axis=0,
    )

    if verbosity >= Verbosity.NORMAL:
        log_array_info(
            array_=arrays_truncated_stacked,
            array_name="arrays_truncated_stacked",
            logger=logger,
        )

    return sorted_df, arrays_truncated_stacked


if __name__ == "__main__":
    main()
