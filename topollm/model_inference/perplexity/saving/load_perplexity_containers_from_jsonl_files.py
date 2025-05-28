# Copyright 2024
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
# AUTHOR_2 (author2@example.com)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#


"""Load perplexity containers from jsonl files."""

import json
import logging
import pathlib

from tqdm import tqdm

from topollm.model_inference.perplexity.saving.sentence_perplexity_container import SentencePerplexityContainer
from topollm.typing.enums import Verbosity
from topollm.typing.types import PerplexityResultsList

default_logger = logging.getLogger(__name__)


def load_multiple_perplexity_containers_from_jsonl_files(
    path_list: list[pathlib.Path],
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> list[PerplexityResultsList]:
    """Load perplexity containers from pickle files."""
    loaded_data_list: list[PerplexityResultsList] = []

    for current_path in tqdm(
        path_list,
        desc="Iterating over path_list",
    ):
        current_perplexity_results_list = load_single_perplexity_container_from_jsonl_file(
            path=current_path,
        )
        loaded_data_list.append(
            current_perplexity_results_list,
        )

    if verbosity >= Verbosity.NORMAL:
        logger.debug(
            f"Loaded {len(loaded_data_list) = } perplexity results lists "  # noqa: G004 - low overhead
            f"from {len(path_list) = } jsonl files.",
        )

    return loaded_data_list


def load_single_perplexity_container_from_jsonl_file(
    path: pathlib.Path,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> PerplexityResultsList:
    """Load single perplexity container from jsonl file."""
    perplexity_results_list: PerplexityResultsList = []

    if verbosity >= Verbosity.NORMAL:
        logger.debug(
            f"Loading from {path = } ...",  # noqa: G004 - low overhead
        )

    with pathlib.Path(path).open(
        mode="r",
    ) as file:
        # Iterate over lines in file
        for line_idx, line in enumerate(
            file,
        ):
            line_json = json.loads(
                line,
            )
            loaded_data = SentencePerplexityContainer.model_validate(
                obj=line_json,
            )
            perplexity_results_list.append(
                (
                    line_idx,
                    loaded_data,
                ),
            )

    if verbosity >= Verbosity.NORMAL:
        logger.debug(
            f"Loading from {path = } DONE",  # noqa: G004 - low overhead
        )

    return perplexity_results_list
