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
import pathlib
import pickle

from tqdm import tqdm

from topollm.typing.enums import Verbosity
from topollm.typing.types import PerplexityResultsList

default_logger = logging.getLogger(__name__)


def load_perplexity_containers_from_pickle_files(
    path_list: list[pathlib.Path],
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> list[PerplexityResultsList]:
    """Load perplexity containers from pickle files."""
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"Loading perplexity containers from {path_list = } ...",  # noqa: G004 - low overhead
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

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"Loading perplexity containers from {path_list = } DONE",  # noqa: G004 - low overhead
        )

    return loaded_data_list
