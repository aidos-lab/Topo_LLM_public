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

from topollm.config_classes.finetuning.peft.peft_config import PEFTConfig
from topollm.path_management.finetuning.peft.PEFTPathManagerBasic import (
    PEFTPathManagerBasic,
)
from topollm.path_management.finetuning.peft.PEFTPathManagerProtocol import (
    PEFTPathManager,
)

logger = logging.getLogger(__name__)


def get_peft_path_manager(
    peft_config: PEFTConfig,
    verbosity: int = 1,
    logger: logging.Logger = logger,
) -> PEFTPathManager:
    """Create a PEFTPathManager instance."""
    path_manger = PEFTPathManagerBasic(
        peft_config=peft_config,
        verbosity=verbosity,
        logger=logger,
    )

    return path_manger
