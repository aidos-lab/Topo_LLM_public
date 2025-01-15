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

# TODO: This script is under development.

import json
import logging
import pathlib
import pprint

import hydra
import omegaconf
import pandas as pd
from tqdm import tqdm

from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.config_classes.main_config import MainConfig
from topollm.config_classes.setup_OmegaConf import setup_omega_conf
from topollm.data_processing.dictionary_handling import flatten_dict
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.path_management.embeddings.protocol import EmbeddingsPathManager
from topollm.typing.enums import Verbosity

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
        "twonn",
    )

    descriptive_statistics_df: pd.DataFrame = load_descriptive_statistics_from_folder_structure(
        iteration_root_dir=iteration_root_dir,
        verbosity=verbosity,
        logger=logger,
    )

    compare_mean_local_estimates_with_mean_losses_for_different_models(
        descriptive_statistics_df=descriptive_statistics_df,
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
    # data/analysis/distances_and_influence_on_losses_and_local_estimates/a-tr-s=60000/twonn/
    # data=iclr_2024_submissions_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags/
    # split=validation_samples=10000_sampling=random_sampling-seed=777/edh-mode=masked_token_lvl=token/add-prefix-space=True_max-len=512/
    # model=roberta-base-masked_lm-defaults_multiwoz21-rm-empty-True-do_nothing-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5_seed-1234_ckpt-800_task=masked_lm_dr=defaults/
    # layer=-1_agg=mean/norm=None/sampling=random_seed=42_samples=150000/
    # desc=twonn_samples=60000_zerovec=keep_dedup=array_deduplicator_noise=do_nothing/
    # descriptive_statistics_dict.json
    #
    # Example dataset folder:
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

            # TODO: Parse the corresponding model and dataset names from the path
            # TODO: Add the parsed information to the dictionary

            loaded_data_list.append(
                flattened_file_data,
            )

    # Convert the list of dictionaries to a DataFrame
    result_df = pd.DataFrame(
        data=loaded_data_list,
    )

    return result_df


def compare_mean_local_estimates_with_mean_losses_for_different_models(
    descriptive_statistics_df: pd.DataFrame,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Create plots with compare mean local estimates with mean losses for different models."""

    # TODO: Make a scatter plot with mean estimates on the x-axis and mean losses on the y-axis

    logger.warning(
        msg="TODO: This function is not yet implemented.",
    )


if __name__ == "__main__":
    main()
