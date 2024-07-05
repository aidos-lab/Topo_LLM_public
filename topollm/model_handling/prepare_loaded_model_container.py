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

"""Prepare device, tokenizer, and model."""

import logging
from typing import TYPE_CHECKING

import transformers

from topollm.config_classes.main_config import MainConfig
from topollm.model_handling.get_torch_device import get_torch_device
from topollm.model_handling.loaded_model_container import LoadedModelContainer
from topollm.model_handling.model.load_model import load_model
from topollm.model_handling.tokenizer.load_tokenizer import load_modified_tokenizer
from topollm.typing.enums import LMmode

if TYPE_CHECKING:
    import torch

default_logger = logging.getLogger(__name__)


def prepare_device_and_tokenizer_and_model(
    main_config: MainConfig,
    logger: logging.Logger = default_logger,
) -> LoadedModelContainer:
    """Prepare device, tokenizer, and model."""
    device: torch.device = get_torch_device(
        preferred_torch_backend=main_config.preferred_torch_backend,
        verbosity=main_config.verbosity,
        logger=logger,
    )

    tokenizer, tokenizer_modifier = load_modified_tokenizer(
        main_config=main_config,
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

    # TODO: Update this so that it supports the same model loading classes as the finetuning code.

    # Case distinction for different language model modes
    # (Masked Language Modeling, Causal Language Modeling).
    lm_mode: LMmode = main_config.language_model.lm_mode

    if lm_mode == LMmode.MLM:
        model_loading_class = transformers.AutoModelForMaskedLM
    elif lm_mode == LMmode.CLM:
        model_loading_class = transformers.AutoModelForCausalLM
    else:
        msg = f"Invalid lm_mode: {lm_mode = }"
        raise ValueError(msg)

    model: transformers.PreTrainedModel = load_model(
        pretrained_model_name_or_path=main_config.language_model.pretrained_model_name_or_path,
        model_loading_class=model_loading_class,
        device=device,
        verbosity=main_config.verbosity,
        logger=logger,
    )

    loaded_model_container = LoadedModelContainer(
        device=device,
        tokenizer=tokenizer,
        tokenizer_config=main_config.tokenizer,
        tokenizer_modifier=tokenizer_modifier,
        lm_mode=lm_mode,
        model=model,
    )

    return loaded_model_container
