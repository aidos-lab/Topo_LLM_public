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

"""Run inference with a language model."""

import logging
import pprint

import transformers

from topollm.config_classes.enums import LMmode
from topollm.config_classes.main_config import MainConfig
from topollm.model_handling.get_torch_device import get_torch_device
from topollm.model_handling.model.load_model import load_model
from topollm.model_handling.tokenizer.load_tokenizer import load_tokenizer
from topollm.model_inference.causal_language_modeling.do_text_generation import (
    do_text_generation,
)
from topollm.model_inference.default_prompts import (
    get_default_clm_prompts,
    get_default_mlm_prompts,
)
from topollm.model_inference.masked_language_modeling.do_fill_mask import do_fill_mask

logger = logging.getLogger(__name__)


def do_inference(
    main_config: MainConfig,
    prompts: list[str] | None = None,
    logger: logging.Logger = logger,
) -> list[list]:
    """Run inference with a language model.

    If `prompts` is `None`, default prompts are used.
    Make sure to not accidentally use an empty list as the default argument.
    """
    device = get_torch_device(
        preferred_torch_backend=main_config.preferred_torch_backend,
        verbosity=main_config.verbosity,
        logger=logger,
    )

    tokenizer = load_tokenizer(
        pretrained_model_name_or_path=main_config.language_model.pretrained_model_name_or_path,
        tokenizer_config=main_config.tokenizer,
        verbosity=main_config.verbosity,
        logger=logger,
    )

    # Note that you cannot use `AutoModel.from_pretrained` to load a model for inference,
    # because it would lead to the error:
    # `KeyError: 'logits'`
    # See also: https://github.com/huggingface/transformers/issues/16569
    #
    # We use the `AutoModelFor...` class instead,
    # which will load the model correctly for inference.
    # Note: The `AutoModelForPreTraining` class appears to work for the
    # "roberta"-models, but not for the "bert"-models.

    # Case distinction for different language model modes
    # (Masked Language Modeling, Causal Language Modeling).
    lm_mode = main_config.language_model.lm_mode

    if lm_mode == LMmode.MLM:
        model_loading_class = transformers.AutoModelForMaskedLM
    elif lm_mode == LMmode.CLM:
        model_loading_class = transformers.AutoModelForCausalLM
    else:
        msg = f"Invalid lm_mode: {lm_mode = }"
        raise ValueError(msg)

    model = load_model(
        pretrained_model_name_or_path=main_config.language_model.pretrained_model_name_or_path,
        model_loading_class=model_loading_class,
        device=device,
        verbosity=main_config.verbosity,
        logger=logger,
    )
    model.eval()

    if lm_mode == LMmode.MLM:
        if prompts is None:
            prompts = get_default_mlm_prompts(
                mask_token=tokenizer.mask_token,
            )
        logger.info(
            "prompts:\n%s",
            pprint.pformat(prompts),
        )

        results = do_fill_mask(
            tokenizer=tokenizer,
            model=model,
            prompts=prompts,
            device=device,
            logger=logger,
        )
    elif lm_mode == LMmode.CLM:
        if prompts is None:
            prompts = get_default_clm_prompts()
        logger.info(
            "prompts:\n%s",
            pprint.pformat(prompts),
        )

        results = do_text_generation(
            tokenizer=tokenizer,
            model=model,
            prompts=prompts,
            max_length=main_config.inference.max_length,
            num_return_sequences=main_config.inference.num_return_sequences,
            device=device,
            logger=logger,
        )
    else:
        msg = f"Invalid lm_mode: {lm_mode = }"
        raise ValueError(msg)

    return results
