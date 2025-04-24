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

"""Compute the (pseudo-)perplexity of a masked language model and save to disk."""

import logging
import pathlib
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import omegaconf
import torch

from topollm.config_classes.constants import HYDRA_CONFIGS_BASE_PATH
from topollm.config_classes.get_data_dir import get_data_dir
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.model_inference.perplexity.saved_perplexity_processing.compute_averages.compute_averages_over_loaded_data_list import (
    compute_averages_over_loaded_data_list,
)
from topollm.model_inference.perplexity.saving.load_perplexity_containers_from_jsonl_files import (
    load_multiple_perplexity_containers_from_jsonl_files,
)
from topollm.model_inference.perplexity.saving.load_perplexity_containers_from_pickle_files import (
    load_perplexity_containers_from_pickle_files,
)
from topollm.typing.enums import PerplexityContainerSaveFormat, Verbosity

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig

default_device = torch.device(
    device="cpu",
)
default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)
global_logger: logging.Logger = logging.getLogger(
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
    verbosity = main_config.verbosity

    data_dir: pathlib.Path = get_data_dir(
        main_config=main_config,
        verbosity=main_config.verbosity,
        logger=global_logger,
    )

    # # # #
    # Parameters

    perplexity_container_save_format = PerplexityContainerSaveFormat.LIST_AS_JSONL

    dataset_subdir_list: list[pathlib.Path] = [
        pathlib.Path(
            "data-multiwoz21_split-validation_ctxt-dataset_entry_samples-10000_feat-col-ner_tags",
        ),
        pathlib.Path(
            "data-wikitext_split-validation_ctxt-dataset_entry_samples--1_feat-col-ner_tags",
        ),
        pathlib.Path(
            "data-iclr_2024_submissions_split-validation_ctxt-dataset_entry_samples--1_feat-col-ner_tags",
        ),
    ]

    for dataset_subdir in dataset_subdir_list:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                "============= START dataset_subdir:\n%s",
                dataset_subdir,
            )

        if perplexity_container_save_format == PerplexityContainerSaveFormat.LIST_AS_PICKLE:
            path_list: list[pathlib.Path] = [
                pathlib.Path(
                    data_dir,
                    "embeddings/perplexity/",
                    dataset_subdir,
                    "lvl-token/add-prefix-space-False_max-len-512/",
                    "model-roberta-base_finetuned-on-one-year-of-tsla-on-reddit_ftm-standard_checkpoint-400_mask-no_masking/",
                    "layer-[-1]_agg-mean/norm-None/perplexity_dir/perplexity_results_list_new_format.pkl",
                ),
                pathlib.Path(
                    data_dir,
                    "embeddings/perplexity/",
                    dataset_subdir,
                    "lvl-token/add-prefix-space-False_max-len-512/",
                    "model-roberta-base_finetuned-on-one-year-of-tsla-on-reddit_ftm-standard_checkpoint-2800_mask-no_masking/",
                    "layer-[-1]_agg-mean/norm-None/perplexity_dir/perplexity_results_list_new_format.pkl",
                ),
            ]
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    "path_list:\n%s",
                    path_list,
                )

            loaded_data_list = load_perplexity_containers_from_pickle_files(
                path_list=path_list,
                verbosity=verbosity,
                logger=logger,
            )
        elif perplexity_container_save_format == PerplexityContainerSaveFormat.LIST_AS_JSONL:
            path_list: list[pathlib.Path] = [
                pathlib.Path(
                    data_dir,
                    "embeddings/perplexity/",
                    dataset_subdir,
                    "lvl-token/add-prefix-space-False_max-len-512/model-roberta-base_task-MASKED_LM/layer--1_agg-mean/norm-None/perplexity_dir/perplexity_results_list.jsonl",
                ),
                pathlib.Path(
                    data_dir,
                    "embeddings/perplexity/",
                    dataset_subdir,
                    "lvl-token/add-prefix-space-False_max-len-512/model-model-roberta-base_task-MASKED_LM_multiwoz21-train-10000-ner_tags_ftm-standard_lora-None_5e-05-linear-0.01-5_ckpt-2800_task-MASKED_LM/layer--1_agg-mean/norm-None/perplexity_dir/perplexity_results_list.jsonl",
                ),
            ]
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    "path_list:\n%s",
                    path_list,
                )

            loaded_data_list = load_multiple_perplexity_containers_from_jsonl_files(
                path_list=path_list,
                verbosity=verbosity,
                logger=logger,
            )
        else:
            msg = f"Unknown {perplexity_container_save_format = }"
            raise ValueError(msg)

        compute_averages_over_loaded_data_list(
            loaded_data_list=loaded_data_list,
            verbosity=verbosity,
            logger=logger,
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                "============= END dataset_subdir:\n%s",
                dataset_subdir,
            )

    logger.info("Running script DONE")


if __name__ == "__main__":
    main()
