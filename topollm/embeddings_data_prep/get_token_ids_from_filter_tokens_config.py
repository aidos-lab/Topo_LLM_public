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

from topollm.config_classes.embeddings_data_prep.embeddings_data_prep_config import FilterTokensConfig
from topollm.typing.enums import Verbosity
from topollm.typing.types import TransformersTokenizer

default_logger = logging.getLogger(__name__)


def get_token_ids_from_filter_tokens_config(
    tokenizer: TransformersTokenizer,
    filter_tokens_config: FilterTokensConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> list[int]:
    """Get the token ids to filter out from the tokenizer based on the filter tokens config."""
    token_ids_to_filter = []
    if filter_tokens_config.remove_bos_token:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                f"{tokenizer.bos_token_id = }",  # noqa: G004 - low overhead
            )
        if tokenizer.bos_token_id is None:
            msg = "The tokenizer beginning of sequence token id is None."
            raise ValueError(
                msg,
            )
        token_ids_to_filter.append(
            tokenizer.bos_token_id,
        )
    if filter_tokens_config.remove_eos_token:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                f"{tokenizer.eos_token_id = }",  # noqa: G004 - low overhead
            )
        if tokenizer.eos_token_id is None:
            msg = "The tokenizer end of sequence token id is None."
            raise ValueError(
                msg,
            )
        token_ids_to_filter.append(
            tokenizer.eos_token_id,
        )
    if filter_tokens_config.remove_pad_token:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                f"{tokenizer.pad_token_id = }",  # noqa: G004 - low overhead
            )
        if tokenizer.pad_token_id is None:
            msg = "The tokenizer padding token id is None."
            raise ValueError(
                msg,
            )
        token_ids_to_filter.append(
            tokenizer.pad_token_id,
        )

    return token_ids_to_filter
