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
import pathlib

from topollm.config_classes.finetuning.peft.PEFTConfig import PEFTConfig
from topollm.config_classes.enums import FinetuningMode
from topollm.config_classes.constants import NAME_PREFIXES


# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Globals

# END Globals
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class PEFTPathManagerBasic:
    def __init__(
        self,
        peft_config: PEFTConfig,
        verbosity: int = 1,
        logger: logging.Logger = logging.getLogger(__name__),
    ) -> None:
        self.peft_config = peft_config

        self.verbosity = verbosity
        self.logger = logger

        return None

    @property
    def peft_description(
        self,
    ) -> str:

        if self.peft_config.finetuning_mode == FinetuningMode.STANDARD:
            description = f"{NAME_PREFIXES['FinetuningMode']}" f"standard"
        elif self.peft_config.finetuning_mode == FinetuningMode.LORA:
            description = f"{NAME_PREFIXES['FinetuningMode']}" f"lora"
            # TODO: Update this
        else:
            raise ValueError(
                f"Unknown finetuning_mode: " f"{self.peft_config.finetuning_mode = }"
            )

        return description
