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
import pickle
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import numpy as np
import omegaconf
import torch
from tqdm import tqdm

from topollm.config_classes.setup_OmegaConf import setup_OmegaConf
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.log_list_info import log_list_info
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.model_inference.perplexity.sentence_perplexity_container import SentencePerplexityContainer

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig
    from topollm.model_inference.perplexity.compute_perplexity_over_dataset import (
        PerplexityResultsList,
    )

default_device = torch.device("cpu")
default_logger = logging.getLogger(__name__)

global_logger = logging.getLogger(__name__)

setup_exception_logging(
    logger=global_logger,
)


setup_OmegaConf()


@hydra.main(
    config_path="../../../configs",
    config_name="main_config",
    version_base="1.2",
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

    data_dir = main_config.paths.data_dir
    logger.info(
        "data_dir:\n%s",
        data_dir,
    )

    # dataset_subdir = pathlib.Path("data-multiwoz21_split-validation_ctxt-dataset_entry_samples-3000")
    dataset_subdir = pathlib.Path("data-one-year-of-tsla-on-reddit_split-validation_ctxt-dataset_entry_samples-3000")

    path_list: list[pathlib.Path] = [
        pathlib.Path(
            data_dir,
            "embeddings/perplexity/",
            dataset_subdir,
            "lvl-token/add-prefix-space-False_max-len-512/model-roberta-base_finetuned-on-one-year-of-tsla-on-reddit_ftm-standard_checkpoint-400_mask-no_masking/layer-[-1]_agg-mean/norm-None/perplexity_dir/perplexity_results_list.pkl",
        ),
        pathlib.Path(
            data_dir,
            "embeddings/perplexity/",
            dataset_subdir,
            "lvl-token/add-prefix-space-False_max-len-512/model-roberta-base_finetuned-on-one-year-of-tsla-on-reddit_ftm-standard_checkpoint-2800_mask-no_masking/layer-[-1]_agg-mean/norm-None/perplexity_dir/perplexity_results_list.pkl",
        ),
    ]
    logger.info(
        "path_list:\n%s",
        path_list,
    )

    loaded_data_list: list[PerplexityResultsList] = []
    for path in tqdm(
        path_list,
        desc="Iterating over path_list",
    ):
        with pathlib.Path(path).open(
            mode="rb",
        ) as file:
            loaded_data = pickle.load(  # noqa: S301 - trusted source
                file,
            )
            loaded_data_list.append(
                loaded_data,
            )

    averages_list = []
    for loaded_data in tqdm(
        loaded_data_list,
        desc="Iterating over loaded_data_list",
    ):
        averages = []
        for _, sentence_perplexity_container in tqdm(
            loaded_data,
            desc="Iterating over loaded_data",
        ):
            average_perplexity = compute_average_sequence_perplexity(
                sentence_perplexity_container,
            )
            averages.append(
                average_perplexity,
            )
        averages_list.append(
            averages,
        )

        log_list_info(
            averages,
            list_name="averages",
            logger=logger,
        )

    differences = [(b - a) for a, b in zip(averages_list[0], averages_list[1])]
    average_difference = sum(differences) / len(differences)

    log_list_info(
        differences,
        list_name="differences",
        logger=logger,
    )
    logger.info(
        "average_difference:\n%s",
        average_difference,
    )

    # # # #
    # Take exponential of token level losses before computing the average
    defferences_of_exps = [(np.exp(b) - np.exp(a)) for a, b in zip(averages_list[0], averages_list[1])]
    average_difference_of_exps = sum(defferences_of_exps) / len(defferences_of_exps)
    logger.info(
        "average_difference_of_exps:\n%s",
        average_difference_of_exps,
    )

    logger.info("Running script DONE")


def compute_average_sequence_perplexity(
    sentence_perplexity_container: SentencePerplexityContainer,
) -> float:
    """Compute the average perplexity of a sequence."""
    perplexity_list = sentence_perplexity_container.token_perplexities
    average_perplexity = sum(perplexity_list) / len(perplexity_list)
    return average_perplexity


if __name__ == "__main__":
    main()
