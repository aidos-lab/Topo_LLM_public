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

import logging

import numpy as np
from tqdm import tqdm

from topollm.logging.log_list_info import log_list_info
from topollm.model_inference.perplexity.sentence_perplexity_container import SentencePerplexityContainer
from topollm.typing.enums import Verbosity
from topollm.typing.types import PerplexityResultsList

default_logger = logging.getLogger(__name__)


def compute_average_sequence_perplexity(
    sentence_perplexity_container: SentencePerplexityContainer,
) -> float:
    """Compute the average perplexity of a sequence."""
    perplexity_list = sentence_perplexity_container.token_perplexities
    average_perplexity = sum(perplexity_list) / len(perplexity_list)
    return average_perplexity


def compute_averages_over_loaded_data_list(
    loaded_data_list: list[PerplexityResultsList],
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Compute the average perplexity of a sequence list[PerplexityResultsList]."""
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

        if verbosity >= Verbosity.NORMAL:
            log_list_info(
                averages,
                list_name="averages",
                logger=logger,
            )

    differences = [
        (b - a)
        for a, b in zip(
            averages_list[0],
            averages_list[1],
            strict=True,
        )
    ]
    average_difference = sum(differences) / len(differences)

    if verbosity >= Verbosity.NORMAL:
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
    defferences_of_exps = [
        (np.exp(b) - np.exp(a))
        for a, b in zip(
            averages_list[0],
            averages_list[1],
            strict=True,
        )
    ]
    average_difference_of_exps = sum(defferences_of_exps) / len(defferences_of_exps)
    logger.info(
        "average_difference_of_exps:\n%s",
        average_difference_of_exps,
    )
