# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (ruppik@hhu.de)
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

"""Load computed perplexity and concatente sequences into single array and df."""

import logging
import pathlib
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import numpy as np
import omegaconf
import pandas as pd
import torch
import zarr

from topollm.analysis.local_estimates.saving.local_estimates_containers import LocalEstimatesContainer
from topollm.analysis.local_estimates.saving.save_local_estimates import load_local_estimates
from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.config_classes.setup_OmegaConf import setup_omega_conf
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.log_dataframe_info import log_dataframe_info
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.model_inference.perplexity.saved_perplexity_processing.concatenate_results.convert_perplexity_results_list_to_dataframe import (
    convert_perplexity_results_list_to_dataframe,
)
from topollm.model_inference.perplexity.saved_perplexity_processing.load_perplexity_containers_from_jsonl_files import (
    load_multiple_perplexity_containers_from_jsonl_files,
)
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.typing.enums import Verbosity
from topollm.typing.types import PerplexityResultsList

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig

default_device = torch.device("cpu")
default_logger = logging.getLogger(__name__)

global_logger = logging.getLogger(__name__)

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
    """Run the script."""
    logger = global_logger
    logger.info("Running script ...")

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=logger,
    )
    verbosity = main_config.verbosity

    # # # #
    # Get save paths
    embeddings_path_manager = get_embeddings_path_manager(
        main_config=main_config,
        logger=logger,
    )
    perplexity_dir = embeddings_path_manager.perplexity_dir_absolute_path

    save_file_path_josnl = pathlib.Path(
        perplexity_dir,
        "perplexity_results_list.jsonl",
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "save_file_path_josnl:\n%s",
            save_file_path_josnl,
        )

    loaded_data_list: list[PerplexityResultsList] = load_multiple_perplexity_containers_from_jsonl_files(
        path_list=[
            save_file_path_josnl,
        ],
        verbosity=verbosity,
        logger=logger,
    )

    # Since we are only loading one container, we can directly access the first element
    loaded_data: PerplexityResultsList = loaded_data_list[0]

    # # # #
    # Convert the token perplexities to a pandas dataframe
    token_perplexities_df, token_perplexities_array = convert_perplexity_results_list_to_dataframe(
        loaded_data=loaded_data,
        verbosity=verbosity,
        logger=logger,
    )

    # # # #
    # Save token perplexities as zarr array
    token_perplexities_zarr_array_save_path = pathlib.Path(
        perplexity_dir,
        "perplexity_results_array.zarr",
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"{token_perplexities_zarr_array_save_path = }",  # noqa: G004 - low overhead
        )
        logger.info(
            "Saving token_perplexities_array to zarr file ...",
        )
    zarr.save(
        str(token_perplexities_zarr_array_save_path),
        token_perplexities_array,
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "Saving token_perplexities_array to zarr file DONE",
        )

    # # # #
    # Save token perplexities pandas dataframe as csv

    token_perplexities_df_save_path = pathlib.Path(
        perplexity_dir,
        "perplexity_results_df.csv",
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"{token_perplexities_df_save_path = }",  # noqa: G004 - low overhead
        )
        logger.info(
            "Saving token_perplexities_df to csv file ...",
        )
    token_perplexities_df.to_csv(
        token_perplexities_df_save_path,
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "Saving token_perplexities_df to csv file DONE",
        )

    # # # # # # # # # # # # # # # # # # # #
    # Compute and save summary statistics

    average_perplexity: float = token_perplexities_df["token_perplexity"].mean()
    std_perplexity: float = token_perplexities_df["token_perplexity"].std()
    num_samples: int = len(token_perplexities_df)

    # TODO: This only currently works for the roberta tokenizer

    bos_token_string: str = "<s>"
    eos_token_string: str = "</s>"

    token_perplexities_without_eos_df: pd.DataFrame = token_perplexities_df[
        token_perplexities_df["token_string"] != eos_token_string
    ]
    token_perplexities_without_eos_df["token_log_perplexity"] = token_perplexities_without_eos_df[
        "token_perplexity"
    ].apply(
        lambda x: np.log(x),
    )
    # Replace `-inf` values with `0.0`
    token_perplexities_without_eos_df["token_log_perplexity"] = token_perplexities_without_eos_df[
        "token_log_perplexity"
    ].replace(
        -np.inf,
        0.0,
    )

    num_samples_without_eos = len(token_perplexities_without_eos_df)
    average_perplexity_without_eos = token_perplexities_without_eos_df["token_perplexity"].mean()
    std_perplexity_without_eos = token_perplexities_without_eos_df["token_perplexity"].std()

    average_log_perplexity_without_eos = token_perplexities_without_eos_df["token_log_perplexity"].mean()
    std_log_perplexity_without_eos = token_perplexities_without_eos_df["token_log_perplexity"].std()

    average_perplexity_without_special_tokens = token_perplexities_df[
        ~token_perplexities_df["token_string"].isin([bos_token_string, eos_token_string])
    ]["token_perplexity"].mean()
    std_perplexity_without_special_tokens = token_perplexities_df[
        ~token_perplexities_df["token_string"].isin([bos_token_string, eos_token_string])
    ]["token_perplexity"].std()
    num_samples_without_special_tokens = len(
        token_perplexities_df[~token_perplexities_df["token_string"].isin([bos_token_string, eos_token_string])]
    )

    # Create string with statistics
    perplexities_statistics_string: str = (
        f"{num_samples = }\n"  # noqa: ISC003 - explicit string concatenation to avoid confusion
        + f"{average_perplexity = }\n"
        + f"{std_perplexity = }\n"
        + f"{num_samples_without_eos = }\n"
        + f"{average_perplexity_without_eos = }\n"
        + f"{std_perplexity_without_eos = }\n"
        + f"{average_log_perplexity_without_eos = }\n"
        + f"{std_log_perplexity_without_eos = }\n"
        + f"{num_samples_without_special_tokens = }\n"
        + f"{average_perplexity_without_special_tokens = }\n"
        + f"{std_perplexity_without_special_tokens = }\n"
    )

    # Write statistics to log
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "perplexities_statistics_string:\n%s",
            perplexities_statistics_string,
        )

    # Save statistics to text file in the perplexity directory

    perplexities_statistics_string_save_path = pathlib.Path(
        perplexity_dir,
        "perplexity_statistics.txt",
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"{perplexities_statistics_string_save_path = }",  # noqa: G004 - low overhead
        )
        logger.info(
            "Saving perplexities_statistics_string to text file ...",
        )

    with perplexities_statistics_string_save_path.open(
        mode="w",
    ) as f:
        f.write(
            perplexities_statistics_string,
        )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "Saving perplexities_statistics_string to text file DONE",
        )

    # # # # # # # # # # # # # # # # # # # #

    # Set the parameters so that the correct local estimates are loaded.
    # Note that we have to do this because the number of sequences for the perplexity computation
    # might be different from the number of sequences for the local estimates computation.

    # TODO: Make this more flexible (we will make this an additional function parameter later)

    if main_config.data.dataset_description_string == "multiwoz21":
        main_config.data.number_of_samples = 3000
    else:
        main_config.data.number_of_samples = -1

    # TODO: We currently set this manually here to run the script for different embedding indices
    # TODO: This currently needs to happen after the perplexity loading, because otherwise we pick the wrong path to the perplexity directory (for legacy reasons, this contains the layer index, even though this would not be necessary)
    main_config.embeddings.embedding_extraction.layer_indices = [-9]

    local_estimates_container: LocalEstimatesContainer = load_local_estimates(
        embeddings_path_manager=embeddings_path_manager,
        verbosity=verbosity,
        logger=logger,
    )

    local_estimates_array_np = local_estimates_container.results_array_np

    # Create string with statistics
    local_estimates_statistics_string: str = (
        f"{local_estimates_array_np.shape = }\n"  # noqa: ISC003 - explicit string concatenation to avoid confusion
        + f"{local_estimates_array_np.mean() = }\n"
        + f"{local_estimates_array_np.std() = }\n"
        + f"{main_config.data.number_of_samples = }\n"
        + f"{main_config.embeddings.embedding_extraction.layer_indices = }\n"
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "local_estimates_statistics_string:\n%s",
            local_estimates_statistics_string,
        )

    # Save statistics to text file in the perplexity directory

    local_estimates_string_save_file_name: str = (
        "local_estimates_statistics" + f"layer-{main_config.embeddings.embedding_extraction.layer_indices}" + ".txt"
    )
    local_estimates_string_save_path = pathlib.Path(
        perplexity_dir,
        local_estimates_string_save_file_name,
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"{local_estimates_string_save_path = }",  # noqa: G004 - low overhead
        )
        logger.info(
            "Saving local_estimates_statistics_string to text file ...",
        )

    with local_estimates_string_save_path.open(
        mode="w",
    ) as f:
        f.write(
            local_estimates_statistics_string,
        )
        # Write the main_config to the file as well
        f.write(
            f"\n\nmain_config:\n{main_config}\n",
        )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "Saving local_estimates_statistics_string to text file DONE",
        )

    # TODO: Align the corresponding local estimates of the tokens for which we have both measures,
    # TODO: compute the pairwise correlation of the perplexities with the local estimates

    local_estimates_meta_frame = local_estimates_container.results_meta_frame

    if local_estimates_meta_frame is None:
        logger.info("local_estimates_meta_frame is None.")
        logger.info("The function will return now without computing the correlations.")
        logger.warning("Correlations between perplexities and local estimates cannot be computed.")
        return

    if verbosity >= Verbosity.NORMAL:
        log_dataframe_info(
            df=local_estimates_meta_frame,
            df_name="local_estimates_meta_frame",
            logger=logger,
        )

    # Add the local estimates to the local_estimates_meta_frame
    local_estimates_meta_frame["local_estimate"] = local_estimates_array_np

    corresponding_token_perplexities_df = token_perplexities_without_eos_df.iloc[
        local_estimates_meta_frame["subsample_idx"]
    ]

    # Check that local_estimates_meta_frame["token_name"] and corresponding_token_perplexities_df["token_string"] agree
    discrepancies = compare_columns(
        df1=local_estimates_meta_frame,
        col1="token_name",
        df2=corresponding_token_perplexities_df,
        col2="token_string",
    )

    if not discrepancies.empty:
        logger.error(
            "local_estimates_meta_frame['token_name'] and "
            "corresponding_token_perplexities_df['token_string'] do not agree."
        )
        logger.error("The function will return now without computing the correlations.")
        logger.warning("Correlations between perplexities and local estimates cannot be computed.")
        return

    # Compute the correlation between the 'token_log_perplexity', and 'local_estimate'
    aligned_df = pd.concat(
        [
            corresponding_token_perplexities_df.reset_index(drop=True),
            local_estimates_meta_frame.reset_index(drop=True),
        ],
        axis=1,
    )

    correlation_columns = [
        "token_perplexity",
        "token_log_perplexity",
        "local_estimate",
    ]
    only_correlation_columns_aligned_df = aligned_df[correlation_columns]

    if verbosity >= Verbosity.NORMAL:
        log_dataframe_info(
            df=aligned_df,
            df_name="aligned_df",
            logger=logger,
        )
        log_dataframe_info(
            df=only_correlation_columns_aligned_df,
            df_name="only_correlation_columns_aligned_df",
            logger=logger,
        )

    for method in [
        "pearson",
        "spearman",
        "kendall",
    ]:
        correlation_results_df = only_correlation_columns_aligned_df.corr(
            method=method,  # type: ignore - these methods are available
        )
        logger.info(
            f"Correlation using '{method = }':\n{correlation_results_df}",  # noqa: G004 - low overhead
        )
        logger.info(
            f"{correlation_results_df['local_estimate']['token_log_perplexity'] = }",  # noqa: G004 - low overhead
        )

    # # # # # # # # # # # # # # # # # # # #
    logger.info("Running script DONE")


def compare_columns(
    df1: pd.DataFrame,
    col1: str,
    df2: pd.DataFrame,
    col2: str,
) -> pd.DataFrame:
    """Compare two columns from different DataFrames and return a DataFrame highlighting the differences.

    Args:
        df1 (pd.DataFrame): First DataFrame.
        col1 (str): Column name in the first DataFrame.
        df2 (pd.DataFrame): Second DataFrame.
        col2 (str): Column name in the second DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing the rows where the columns differ.
    """
    # Reset indices to ensure comparison by position
    df1_reset = df1[col1].reset_index(drop=True)
    df2_reset = df2[col2].reset_index(drop=True)

    # Ensure the columns have the same length
    if len(df1_reset) != len(df2_reset):
        msg = "The columns must have the same length to compare."
        raise ValueError(msg)

    # Create a DataFrame to compare the two columns
    comparison_df = pd.DataFrame(
        {
            f"{col1}_df1": df1_reset,
            f"{col2}_df2": df2_reset,
        },
    )

    # Add a column to indicate where the values are not equal
    comparison_df["Equal"] = comparison_df[f"{col1}_df1"] == comparison_df[f"{col2}_df2"]

    # Filter rows where the values are not equal
    discrepancies = comparison_df[~comparison_df["Equal"]]

    return discrepancies


if __name__ == "__main__":
    main()
