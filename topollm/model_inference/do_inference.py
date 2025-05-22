# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Matthias Ruppik (mail@ruppik.net)
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
from typing import TYPE_CHECKING

from topollm.config_classes.main_config import MainConfig
from topollm.model_handling.prepare_loaded_model_container import (
    prepare_device_and_tokenizer_and_model_from_main_config,
)
from topollm.model_inference.causal_language_modeling.do_text_generation import (
    do_text_generation,
)
from topollm.model_inference.default_prompts import (
    get_default_clm_prompts,
    get_default_mlm_prompts,
)
from topollm.model_inference.masked_language_modeling.do_fill_mask import do_fill_mask
from topollm.typing.enums import LMmode, Verbosity

if TYPE_CHECKING:
    from topollm.model_handling.loaded_model_container import LoadedModelContainer

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def do_inference(
    main_config: MainConfig,
    prompts: list[str] | None = None,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> list[list]:
    """Run inference with a language model.

    If `prompts` is `None`, default prompts are used.
    Make sure to not accidentally use an empty list as the default argument.
    """
    loaded_model_container: LoadedModelContainer = prepare_device_and_tokenizer_and_model_from_main_config(
        main_config=main_config,
        verbosity=verbosity,
        logger=logger,
    )
    device = loaded_model_container.device
    tokenizer = loaded_model_container.tokenizer
    lm_mode = loaded_model_container.lm_mode
    model = loaded_model_container.model

    # Set up the model for evaluation.
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
