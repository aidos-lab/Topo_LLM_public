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
from tqdm import tqdm

from topollm.analysis.twonn.save_local_estimates import load_local_estimates
from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.config_classes.setup_OmegaConf import setup_omega_conf
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.log_dataframe_info import log_dataframe_info
from topollm.logging.log_list_info import log_list_info
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.model_inference.perplexity.saved_perplexity_processing.load_perplexity_containers_from_jsonl_files import (
    load_multiple_perplexity_containers_from_jsonl_files,
)
from topollm.model_inference.perplexity.saved_perplexity_processing.load_perplexity_containers_from_pickle_files import (
    load_perplexity_containers_from_pickle_files,
)
from topollm.model_inference.perplexity.sentence_perplexity_container import SentencePerplexityContainer
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.typing.enums import PerplexityContainerSaveFormat, Verbosity

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

    data_dir = main_config.paths.data_dir
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "data_dir:\n%s",
            data_dir,
        )

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

    loaded_data_list = load_multiple_perplexity_containers_from_jsonl_files(
        path_list=[
            save_file_path_josnl,
        ],
        verbosity=verbosity,
        logger=logger,
    )

    # Since we are only loading one container, we can directly access the first element
    loaded_data = loaded_data_list[0]

    # Empty lists for holding the concatenated data
    token_ids_list: list[int] = []
    token_strings_list: list[str] = []
    token_perplexities_list: list[float] = []

    for _, sentence_perplexity_container in tqdm(
        loaded_data,
        desc="Iterating over loaded_data",
    ):
        sentence_perplexity_container: SentencePerplexityContainer

        token_ids_list.extend(
            sentence_perplexity_container.token_ids,
        )
        token_strings_list.extend(
            sentence_perplexity_container.token_strings,
        )
        token_perplexities_list.extend(
            sentence_perplexity_container.token_perplexities,
        )

    if verbosity >= Verbosity.NORMAL:
        log_list_info(
            token_ids_list,
            list_name="token_ids_list",
            logger=logger,
        )
        log_list_info(
            token_strings_list,
            list_name="token_strings_list",
            logger=logger,
        )
        log_list_info(
            token_perplexities_list,
            list_name="perplexity_list",
            logger=logger,
        )

    # # # #
    # Convert the token perplexities list to a numpy array and save as zarr array

    token_perplexities_array = np.array(
        token_perplexities_list,
    )

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
    # Convert the token perplexities to a pandas dataframe and save as csv

    token_perplexities_df = pd.DataFrame(
        {
            "token_id": token_ids_list,
            "token_string": token_strings_list,
            "token_perplexity": token_perplexities_list,
        },
    )

    if verbosity >= Verbosity.NORMAL:
        log_dataframe_info(
            token_perplexities_df,
            df_name="token_perplexities_df",
            check_for_nan=True,
            logger=logger,
        )

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

    average_perplexity = token_perplexities_df["token_perplexity"].mean()
    std_perplexity = token_perplexities_df["token_perplexity"].std()
    num_samples = len(token_perplexities_df)

    # TODO: This only currently works for the roberta tokenizer

    bos_token_string = "<s>"
    eos_token_string = "</s>"
    average_perplexity_without_eos = token_perplexities_df[token_perplexities_df["token_string"] != eos_token_string][
        "token_perplexity"
    ].mean()
    std_perplexity_without_eos = token_perplexities_df[token_perplexities_df["token_string"] != eos_token_string][
        "token_perplexity"
    ].std()
    num_samples_without_eos = len(token_perplexities_df[token_perplexities_df["token_string"] != eos_token_string])

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
    statistics_string: str = (
        f"{average_perplexity = }\n"
        + f"{std_perplexity = }\n"
        + f"{num_samples = }\n"
        + f"{average_perplexity_without_eos = }\n"
        + f"{std_perplexity_without_eos = }\n"
        + f"{num_samples_without_eos = }\n"
        + f"{average_perplexity_without_special_tokens = }\n"
        + f"{std_perplexity_without_special_tokens = }\n"
        + f"{num_samples_without_special_tokens = }\n"
    )

    # Write statistics to log
    logger.info(
        "statistics_string:\n%s",
        statistics_string,
    )

    # Save statistics to text file in the perplexity directory

    statistics_string_save_path = pathlib.Path(
        perplexity_dir,
        "perplexity_statistics.txt",
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"{statistics_string_save_path = }",  # noqa: G004 - low overhead
        )
        logger.info(
            "Saving statistics_string to text file ...",
        )

    with statistics_string_save_path.open(
        mode="w",
    ) as f:
        f.write(statistics_string)

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "Saving statistics_string to text file DONE",
        )

    # # # # # # # # # # # # # # # # # # # #

    # TODO: Make this more flexible

    main_config.data.number_of_samples = 3000

    local_estimates = load_local_estimates(
        embeddings_path_manager=embeddings_path_manager,
        verbosity=verbosity,
        logger=logger,
    )

    print(local_estimates)
    print(f"{local_estimates.shape = }")
    print(f"{local_estimates.mean() = }")
    print(f"{local_estimates.std() = }")

    # TODO: Load the corresponding local dimension estimates and compute the correlation (naive t-test of unaligned data)
    # TODO: Once we have the indices necessary for alignment, compute the pairwise correlation of the perplexities

    # # # # # # # # # # # # # # # # # # # #
    logger.info("Running script DONE")


if __name__ == "__main__":
    main()
