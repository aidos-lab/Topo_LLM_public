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

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from topollm.config_classes.tokenizer.tokenizer_config import TokenizerConfig
from topollm.model_inference.perplexity.compute_perplexity_over_dataset import pseudoperplexity_per_token_of_sentence
from topollm.typing.enums import (
    MLMPseudoperplexityGranularity,
    Verbosity,
)


def test_pseudoperplexity_per_token_of_sentence(
    tokenizer_config: TokenizerConfig,
    device_fixture: torch.device,
    verbosity: Verbosity,
    logger_fixture: logging.Logger,
) -> None:
    model_name = "roberta-base"
    model = AutoModelForMaskedLM.from_pretrained(
        model_name,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
    )

    model.to(
        device_fixture,
    )
    model.eval()

    sentences_list = [
        "Paris is in France.",
        "Berlin is in France",
        "London is the capital of Great Britain.",
        "London is the capital of South America.",
    ]

    for sentence in sentences_list:
        result = pseudoperplexity_per_token_of_sentence(
            sentence=sentence,
            tokenizer=tokenizer,
            tokenizer_config=tokenizer_config,
            model=model,
            mlm_pseudoperplexity_granularity=MLMPseudoperplexityGranularity.TOKEN,
            device=device_fixture,
            verbosity=verbosity,
            logger=logger_fixture,
        )
        logger_fixture.info(
            "sentence:\n%s",
            sentence,
        )
        logger_fixture.info(
            "result:\n%s",
            result,
        )
