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

"""Fill mask in a masked language model."""

import logging
import pprint

import torch
import transformers

default_device = torch.device("cpu")
logger = logging.getLogger(__name__)


def do_fill_mask(
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
    model: transformers.PreTrainedModel,
    prompts: list[str],
    device: torch.device = default_device,
    logger: logging.Logger = logger,
) -> list[list[dict]]:
    """Fill mask in a masked language model."""
    fill_pipeline = transformers.pipeline(
        task="fill-mask",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    logger.info(
        msg=f"prompts:\n{pprint.pformat(prompts)}",  # noqa: G004 - low overhead
    )

    result = fill_pipeline(
        prompts,
    )
    logger.info(
        f"result:\n{pprint.pformat(result)}",  # noqa: G004 - low overhead
    )

    return result  # type: ignore - problem with matching return type of pipeline
