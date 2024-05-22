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
import os
import pathlib
import pickle

from topollm.model_inference.perplexity.sentence_perplexity_container import SentencePerplexityContainer
from topollm.typing.enums import Verbosity
from topollm.typing.types import PerplexityResultsList

default_logger = logging.getLogger(__name__)


def save_perplexity_results_list(
    perplexity_results_list: PerplexityResultsList,
    perplexity_dir: os.PathLike,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Save the perplexity results list to a file."""
    # # # #
    # Save in pickle format
    save_file_path_pickle = pathlib.Path(
        perplexity_dir,
        "perplexity_results_list_new_format.pkl",
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(f"Saving perplexity results to {save_file_path_pickle = } ...")  # noqa: G004 - low overhead
    with pathlib.Path(save_file_path_pickle).open(
        mode="wb",
    ) as file:
        pickle.dump(
            obj=perplexity_results_list,
            file=file,
        )
    if verbosity >= Verbosity.NORMAL:
        logger.info(f"Saving perplexity results to {save_file_path_pickle = } DONE")  # noqa: G004 - low overhead

    # # # #
    # Save in jsonl format
    save_file_path_josnl = pathlib.Path(
        perplexity_dir,
        "perplexity_results_list.jsonl",
    )
    # Iterate over the list and save each item as a jsonl line
    if verbosity >= Verbosity.NORMAL:
        logger.info(f"Saving perplexity results to {save_file_path_josnl = } ...")  # noqa: G004 - low overhead
    with save_file_path_josnl.open(
        mode="w",
    ) as file:
        for _, sentence_perplexity_container in perplexity_results_list:
            if not isinstance(
                sentence_perplexity_container,
                SentencePerplexityContainer,
            ):
                msg = "Expected a SentencePerplexityContainer."
                raise TypeError(msg)

            model_dump: str = sentence_perplexity_container.model_dump_json()
            file.write(model_dump)
            file.write("\n")
    if verbosity >= Verbosity.NORMAL:
        logger.info(f"Saving perplexity results to {save_file_path_josnl = } DONE")  # noqa: G004 - low overhead
