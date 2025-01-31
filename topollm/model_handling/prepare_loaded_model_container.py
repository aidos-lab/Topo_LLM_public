# Copyright 2024-2025
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (mail@ruppik.net)
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

from topollm.config_classes.language_model.language_model_config import LanguageModelConfig
from topollm.config_classes.main_config import MainConfig
from topollm.config_classes.tokenizer.tokenizer_config import TokenizerConfig
from topollm.model_handling.get_torch_device import get_torch_device
from topollm.model_handling.loaded_model_container import LoadedModelContainer
from topollm.model_handling.model.load_model import load_model
from topollm.model_handling.tokenizer.load_tokenizer import load_modified_tokenizer
from topollm.typing.enums import LMmode, PreferredTorchBackend, Verbosity

if TYPE_CHECKING:
    import torch

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def prepare_device_and_tokenizer_and_model_from_language_model_config(
    language_model_config: LanguageModelConfig,
    tokenizer_config: TokenizerConfig,
    preferred_torch_backend: PreferredTorchBackend,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> LoadedModelContainer:
    """Prepare the loaded container from the individual config files."""
    device: torch.device = get_torch_device(
        preferred_torch_backend=preferred_torch_backend,
        verbosity=verbosity,
        logger=logger,
    )

    (
        tokenizer,
        tokenizer_modifier,
    ) = load_modified_tokenizer(
        language_model_config=language_model_config,
        tokenizer_config=tokenizer_config,
        verbosity=verbosity,
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
    # (Masked Language Modeling, Causal Language Modeling or other tasks).
    lm_mode: LMmode = language_model_config.lm_mode

    match lm_mode:
        case LMmode.MLM:
            model_loading_class = transformers.AutoModelForMaskedLM
        case LMmode.CLM:
            model_loading_class = transformers.AutoModelForCausalLM
        # For the SETSUMBT and Trippy models, we use the AutoModel class without a specific head on top of the encoder.
        case LMmode.SETSUMBT:
            model_loading_class = transformers.AutoModel
        case LMmode.TRIPPY:
            model_loading_class = transformers.AutoModel
        case _:
            msg: str = f"Invalid lm_mode: {lm_mode = }"
            raise ValueError(
                msg,
            )

    model: transformers.PreTrainedModel = load_model(
        pretrained_model_name_or_path=language_model_config.pretrained_model_name_or_path,
        model_loading_class=model_loading_class,
        device=device,
        verbosity=verbosity,
        logger=logger,
    )

    # Potential modification of the tokenizer and the model if this is necessary for compatibility.
    # For instance, for some autoregressive models, the tokenizer needs to be modified to add a padding token.
    model: transformers.PreTrainedModel = tokenizer_modifier.update_model(
        model=model,
    )

    loaded_model_container = LoadedModelContainer(
        device=device,
        model=model,
        language_model_config=language_model_config,
        lm_mode=lm_mode,
        tokenizer=tokenizer,
        tokenizer_config=tokenizer_config,
        tokenizer_modifier=tokenizer_modifier,
    )

    return loaded_model_container


def prepare_device_and_tokenizer_and_model_from_main_config(
    main_config: MainConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> LoadedModelContainer:
    """Prepare device, tokenizer, and model from a main config file.

    This is a convenience function that extracts the necessary information from the main config file.
    """
    loaded_model_container: LoadedModelContainer = prepare_device_and_tokenizer_and_model_from_language_model_config(
        language_model_config=main_config.language_model,
        tokenizer_config=main_config.tokenizer,
        preferred_torch_backend=main_config.preferred_torch_backend,
        verbosity=verbosity,
        logger=logger,
    )

    return loaded_model_container
