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

"""Gradient modifier that freezes layers."""

import logging

from transformers import PreTrainedModel

from topollm.typing.enums import Verbosity

default_logger = logging.getLogger(__name__)


class GradientModifierFreezeLayers:
    """Gradient modifier that freezes layers."""

    def __init__(
        self,
        target_modules_to_freeze: list[str] | None = None,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Initialize the model modifier."""
        self.verbosity = verbosity
        self.logger = logger

        if target_modules_to_freeze is None:
            self.target_modules_to_freeze = []
        else:
            self.target_modules_to_freeze = target_modules_to_freeze

    def modify_gradients(
        self,
        model: PreTrainedModel,
    ) -> PreTrainedModel:
        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info("Freezing layers ...")

        for name, param in model.named_parameters():
            # TODO(Ben): Modify the criterion here
            if name.startswith("encoder.layer.0"):
                param.requires_grad = False

        if self.verbosity >= Verbosity.NORMAL:
            for name, param in model.named_parameters():
                self.logger.info(
                    f"{name = }, {param.requires_grad = }",  # noqa: G004 - low overhead
                )

        return model
