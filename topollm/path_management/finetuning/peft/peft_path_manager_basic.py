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

from topollm.config_classes.constants import ITEM_SEP, KV_SEP, NAME_PREFIXES
from topollm.config_classes.finetuning.peft.peft_config import PEFTConfig
from topollm.path_management.convert_object_to_valid_path_part import convert_list_to_path_part
from topollm.typing.enums import FinetuningMode

default_logger = logging.getLogger(__name__)


class PEFTPathManagerBasic:
    def __init__(
        self,
        peft_config: PEFTConfig,
        verbosity: int = 1,
        logger: logging.Logger = default_logger,
    ) -> None:
        self.peft_config = peft_config

        self.verbosity = verbosity
        self.logger = logger

    @property
    def peft_description_subdir(
        self,
    ) -> pathlib.Path:
        path = pathlib.Path(
            self.finetuning_mode_description,
            self.lora_description,
        )

        return path

    @property
    def finetuning_mode_description(
        self,
    ) -> str:
        if self.peft_config.finetuning_mode == FinetuningMode.STANDARD:
            desc = f"{NAME_PREFIXES['FinetuningMode']}{KV_SEP}standard"
        elif self.peft_config.finetuning_mode == FinetuningMode.LORA:
            desc = f"{NAME_PREFIXES['FinetuningMode']}{KV_SEP}lora"
        else:
            msg = f"Unknown finetuning_mode: {self.peft_config.finetuning_mode = }"
            raise ValueError(msg)

        return desc

    @property
    def lora_description(
        self,
    ) -> str:
        if self.peft_config.finetuning_mode == FinetuningMode.STANDARD:
            description = "lora-None"
        elif self.peft_config.finetuning_mode == FinetuningMode.LORA:
            description = (
                f"{NAME_PREFIXES['lora_r']}"
                f"{KV_SEP}"
                f"{self.peft_config.r}"
                f"{ITEM_SEP}"
                f"{NAME_PREFIXES['lora_alpha']}"
                f"{KV_SEP}"
                f"{self.peft_config.lora_alpha}"
                f"{ITEM_SEP}"
                f"{NAME_PREFIXES['lora_target_modules']}"
                f"{KV_SEP}"
                f"{target_modules_to_path_part(self.peft_config.target_modules)}"
                f"{ITEM_SEP}"
                f"{NAME_PREFIXES['lora_dropout']}"
                f"{KV_SEP}"
                f"{self.peft_config.lora_dropout}"
            )
        else:
            msg = f"Unknown finetuning_mode: {self.peft_config.finetuning_mode = }"
            raise ValueError(msg)

        return description


def target_modules_to_path_part(
    target_modules: list[str] | str | None,
) -> str:
    """Convert the target_modules to a path part."""
    if target_modules is None:
        return "None"
    elif isinstance(
        target_modules,
        str,
    ):
        return target_modules
    else:
        return convert_list_to_path_part(
            input_list=target_modules,
        )
