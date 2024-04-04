# coding=utf-8
#
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
import pprint

import torch
import transformers
from tqdm import tqdm


def do_text_generation(
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
    model: transformers.PreTrainedModel,
    prompts: list[str],
    max_length: int = 50,
    num_return_sequences: int = 3,
    device: torch.device = torch.device("cpu"),
    logger: logging.Logger = logging.getLogger(__name__),
) -> list[list[str]]:
    """
    Generates text based on the provided prompts using causal language modeling,
    with support for multiple generations per prompt.

    Args:
        tokenizer: A tokenizer instance compatible with the provided model.
        model: A pre-trained model instance capable of text generation.
        prompts: A list of strings, each being a prompt to generate text from.
        max_length: Maximum length of the generated text.
        num_return_sequences: Number of sequences to generate for each prompt.
        device: The device to run the model on.
        logger: Logger instance for logging information.

    Returns:
        A list of lists containing generated text sequences for each prompt.
    """

    text_generation_pipeline = transformers.pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=(
            device.index if device else -1
        ),  # Use device index for compatibility, -1 for CPU
        max_length=max_length,
        num_return_sequences=num_return_sequences,
    )

    logger.info(f"prompts:\n{pprint.pformat(prompts)}")

    all_generated_texts: list[list[str]] = []

    for prompt in tqdm(
        prompts,
        desc="Iterating over prompts",
    ):
        results: list[dict] = text_generation_pipeline(
            prompt,
        )  # type: ignore

        if results is None:
            raise ValueError("No results were generated.")
        if not isinstance(
            results,
            list,
        ):
            raise ValueError(f"{results = } must be a list.")

        generated_texts: list[str] = [result["generated_text"] for result in results]

        # Appending the generated texts for the current prompt to the list of lists
        all_generated_texts.append(
            generated_texts,
        )

        # Logging each generated text for the current prompt
        logger.info(f"{prompt = }")
        logger.info(f"generated_texts:\n" f"{pprint.pformat(generated_texts)}")
    return all_generated_texts
