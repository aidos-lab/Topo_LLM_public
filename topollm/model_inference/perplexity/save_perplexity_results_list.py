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

"""Save the perplexity results list to a file."""

import logging
import pathlib
import pickle

from topollm.model_inference.perplexity.sentence_perplexity_container import SentencePerplexityContainer
from topollm.path_management.embeddings.protocol import EmbeddingsPathManager
from topollm.typing.enums import PerplexityContainerSaveFormat, Verbosity
from topollm.typing.types import PerplexityResultsList

default_logger = logging.getLogger(__name__)


def save_perplexity_results_list_as_pickle(
    perplexity_results_list: PerplexityResultsList,
    save_file_path: pathlib.Path,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Save the perplexity results list as pickle."""
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"Saving perplexity results to {save_file_path = } ...",  # noqa: G004 - low overhead
        )
    with pathlib.Path(
        save_file_path,
    ).open(
        mode="wb",
    ) as file:
        pickle.dump(
            obj=perplexity_results_list,
            file=file,
        )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"Saving perplexity results to {save_file_path = } DONE",  # noqa: G004 - low overhead
        )


def save_perplexity_results_list_as_jsonl(
    perplexity_results_list: PerplexityResultsList,
    save_file_path: pathlib.Path,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Save the perplexity results list as jsonl."""
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"Saving perplexity results to {save_file_path = } ...",  # noqa: G004 - low overhead
        )
    with pathlib.Path(
        save_file_path,
    ).open(
        mode="w",
    ) as file:
        # Iterate over the list and save each item as a jsonl line
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
        logger.info(
            f"Saving perplexity results to {save_file_path = } DONE",  # noqa: G004 - low overhead
        )


def save_perplexity_results_list_in_multiple_formats(
    perplexity_results_list: PerplexityResultsList,
    embeddings_path_manager: EmbeddingsPathManager,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Save the perplexity results list to a file."""
    for perplexity_container_save_format in [
        PerplexityContainerSaveFormat.LIST_AS_PICKLE,
        PerplexityContainerSaveFormat.LIST_AS_JSONL,
    ]:
        save_file_path = embeddings_path_manager.get_perplexity_container_save_file_absolute_path(
            perplexity_container_save_format=perplexity_container_save_format,
        )

        save_file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        match perplexity_container_save_format:
            case PerplexityContainerSaveFormat.LIST_AS_PICKLE:
                save_perplexity_results_list_as_pickle(
                    perplexity_results_list=perplexity_results_list,
                    save_file_path=save_file_path,
                    verbosity=verbosity,
                    logger=logger,
                )
            case PerplexityContainerSaveFormat.LIST_AS_JSONL:
                save_perplexity_results_list_as_jsonl(
                    perplexity_results_list=perplexity_results_list,
                    save_file_path=save_file_path,
                    verbosity=verbosity,
                    logger=logger,
                )
            case _:
                msg = "Unsupported perplexity container save format."
                raise ValueError(msg)
