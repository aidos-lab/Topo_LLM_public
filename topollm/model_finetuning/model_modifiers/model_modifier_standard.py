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

"""Model modifier that does not modify the model, i.e., leads to standard fine-tuning behavior."""

import logging

import peft.peft_model
from transformers import PreTrainedModel

from topollm.typing.enums import Verbosity

default_logger = logging.getLogger(__name__)


class ModelModifierStandard:
    """Model modifier that does not modify the model, i.e., leads to standard fine-tuning behavior."""

    def __init__(
        self,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Initialize the model modifier."""
        self.verbosity = verbosity
        self.logger = logger

    def modify_model(
        self,
        model: PreTrainedModel,
    ) -> peft.peft_model.PeftModel | PreTrainedModel:
        if self.verbosity >= 1:
            self.logger.info("Using base model without modifications.")
            self.logger.info("Returning unmodified model.")

        return model
