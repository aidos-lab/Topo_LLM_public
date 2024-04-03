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
from transformers import AutoModelForMaskedLM

from topollm.config_classes.MainConfig import MainConfig
from topollm.model_handling.tokenizer.load_tokenizer import load_tokenizer


def do_fill_mask(
    main_config: MainConfig,
    device: torch.device = torch.device("cpu"),
    logger: logging.Logger = logging.getLogger(__name__),
) -> None:
    tokenizer = load_tokenizer(
        pretrained_model_name_or_path=main_config.embeddings.language_model.pretrained_model_name_or_path,
        tokenizer_config=main_config.embeddings.tokenizer,
        verbosity=main_config.verbosity,
        logger=logger,
    )

    # Note that you cannot use `AutoModel.from_pretrained` here,
    # because it would lead to the error:
    # `KeyError: 'logits'`
    #
    # See also: https://github.com/huggingface/transformers/issues/16569
    pretrained_model_name_or_path = (
        main_config.embeddings.language_model.pretrained_model_name_or_path
    )
    logger.info(f"pretrained_model_name_or_path:\n" f"{pretrained_model_name_or_path}")

    model = AutoModelForMaskedLM.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
    )

    model.to(
        device,
    )
    logger.info(f"model:\n" f"{model}")

    fill_pipeline = transformers.pipeline(
        task="fill-mask",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    prompts: list[str] = [
        f"I am looking for a {tokenizer.mask_token}",
        f"Can you find me a {tokenizer.mask_token}?",
        f"I would like a {tokenizer.mask_token} hotel in the center of town, please.",
        tokenizer.mask_token + " is a cheap restaurant in the south of town.",
        "The train should go to " + tokenizer.mask_token + ".",
        "No, it should be " + tokenizer.mask_token + ", look again!",
    ]
    logger.info(f"prompts:\n" f"{pprint.pformat(prompts)}")

    result = fill_pipeline(prompts)
    logger.info(f"result:\n" f"{pprint.pformat(result)}")

    return None
