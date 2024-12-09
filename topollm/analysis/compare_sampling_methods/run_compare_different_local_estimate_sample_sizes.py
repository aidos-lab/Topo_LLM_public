# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (ruppik@hhu.de)
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

from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.config_classes.setup_OmegaConf import setup_omega_conf
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.path_management.embeddings.factory import get_embeddings_path_manager
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig
    from topollm.path_management.embeddings.protocol import EmbeddingsPathManager

try:
    from hydra_plugins import hpc_submission_launcher

    hpc_submission_launcher.register_plugin()
except ImportError:
    pass

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

    mode = "create_debug_plots"

    if mode == "create_all_plots":
        data_folder_list: list[str] = [
            "data-multiwoz21_split-train_ctxt-dataset_entry_samples-10000_feat-col-ner_tags",
            "data-multiwoz21_split-validation_ctxt-dataset_entry_samples-3000_feat-col-ner_tags",
            "data-multiwoz21_split-test_ctxt-dataset_entry_samples-3000_feat-col-ner_tags",
            "data-one-year-of-tsla-on-reddit_split-train_ctxt-dataset_entry_samples-10000_feat-col-ner_tags",
            "data-one-year-of-tsla-on-reddit_split-validation_ctxt-dataset_entry_samples-3000_feat-col-ner_tags",
            "data-one-year-of-tsla-on-reddit_split-test_ctxt-dataset_entry_samples-3000_feat-col-ner_tags",
        ]
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
        token_space_sampling_folder_list: list[str] = [
            "desc-twonn_samples-5000_zerovec-keep",
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
        token_space_sampling_folder_list: list[str] = [
            "desc-twonn_samples-10000_zerovec-keep",
        ]
    else:
        msg = "Invalid mode"
        raise ValueError(
            msg,
        )

    for data_folder, model_folder, embeddings_data_prep_sampling_folder, token_space_sampling_folder in tqdm(
        iterable=product(
            data_folder_list,
            model_folder_list,
            embeddings_data_prep_sampling_folder_list,
            token_space_sampling_folder_list,
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
            token_space_sampling_folder,
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

        global_estimate_path: pathlib.Path = analysis_base_directory / "global_estimate.npy"
        global_estimate = np.load(
            global_estimate_path,
        )

        local_estimates_80_percent_path: pathlib.Path = analysis_base_directory / "local_estimates_paddings_removed.npy"
        local_estimates_80_percent = np.load(
            local_estimates_80_percent_path,
        )

        loaded_estimates_list = []

        for n_neighbors in [
            64,
            128,
            256,
            384,
            512,
            1024,
        ]:
            local_estimates_directory = pathlib.Path(
                analysis_base_directory,
                f"n-neighbors-mode-absolute_size_n-neighbors-{n_neighbors}",
            )
            if not local_estimates_directory.exists():
                logger.warning(
                    msg=f"Directory does not exist: {local_estimates_directory = }",  # noqa: G004 - low overhead
                )
                continue

            local_estimates_array_path = pathlib.Path(
                local_estimates_directory,
                "local_estimates_pointwise.npy",
            )
            local_estimates_meta_path = pathlib.Path(
                local_estimates_directory,
                "local_estimates_pointwise.pkl",
            )

            local_estimates_array = np.load(
                local_estimates_array_path,
            )

            current_estimate = {
                "n_neighbors": n_neighbors,
                "mean": np.mean(
                    local_estimates_array,
                ),
                "std": np.std(
                    local_estimates_array,
                ),
            }
            loaded_estimates_list.append(
                current_estimate,
            )

        # Make the loaded estimates into a DataFrame
        loaded_estimates_df = pd.DataFrame(
            loaded_estimates_list,
        )

        pass  # Note: This is a placeholder for adding break points

    logger.info(
        msg="Running script DONE",
    )


if __name__ == "__main__":
    main()
