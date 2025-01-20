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

"""Gradient modifier that freezes layers."""

import logging

import peft.peft_model
from transformers import PreTrainedModel

from topollm.logging.log_model_info import log_param_requires_grad_for_model
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


class GradientModifierFreezeLayers:
    """Gradient modifier that freezes layers."""

    def __init__(
        self,
        target_modules_to_freeze: list[str] | None = None,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Initialize the model modifier."""
        self.verbosity: Verbosity = verbosity
        self.logger: logging.Logger = logger

        if target_modules_to_freeze is None:
            self.target_modules_to_freeze: list = []
        else:
            self.target_modules_to_freeze: list = target_modules_to_freeze

    def modify_gradients(
        self,
        model: PreTrainedModel | peft.peft_model.PeftModel,
    ) -> PreTrainedModel | peft.peft_model.PeftModel:
        """Freeze layers of the model."""
        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg="Freezing layers ...",
            )

        for name, param in model.named_parameters():
            name: str

            should_be_frozen: bool = self.check_if_layer_should_be_frozen(
                name=name,
            )

            if should_be_frozen:
                param.requires_grad = False

        if self.verbosity >= Verbosity.NORMAL:
            log_param_requires_grad_for_model(
                model=model,
                logger=self.logger,
            )

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg="Freezing layers DONE.",
            )

        return model

    def check_if_layer_should_be_frozen(
        self,
        name: str,
    ) -> bool:
        """Check if a layer should be frozen.

        The decision is based on the name of the parameter,
        if it contains any of the target modules to freeze as a substring,
        the layer should be frozen.

        Example names of named parameters:
            - For model 'bert-base-uncased':
                'bert.encoder.layer.11.attention.self.key.bias'
            - For model 'gpt2-medium':
                'transformer.h.23.ln_1.weight'
                'transformer.h.23.ln_1.bias'
                'transformer.h.23.attn.c_attn.weight'
                'transformer.h.23.attn.c_attn.bias'
                'transformer.h.23.attn.c_proj.weight'
                'transformer.h.23.attn.c_proj.bias'
                'transformer.h.23.ln_2.weight'
                'transformer.h.23.ln_2.bias'
                'transformer.h.23.mlp.c_fc.weight'
                'transformer.h.23.mlp.c_fc.bias'
                'transformer.h.23.mlp.c_proj.weight'
                'transformer.h.23.mlp.c_proj.bias'
            - For model 'roberta-base':
                'roberta.encoder.layer.11.attention.self.query.weight'
                'roberta.encoder.layer.11.attention.self.query.bias'
        """
        return any(target_module in name for target_module in self.target_modules_to_freeze)
