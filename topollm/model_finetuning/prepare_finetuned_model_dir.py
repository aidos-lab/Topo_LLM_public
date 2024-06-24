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
import os

from topollm.path_management.finetuning.protocol import (
    FinetuningPathManager,
)
from topollm.typing.enums import Verbosity

default_logger = logging.getLogger(__name__)


def prepare_finetuned_model_dir(
    finetuning_path_manager: FinetuningPathManager,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> os.PathLike:
    """Prepare the directory for the finetuned model."""
    finetuned_model_dir = finetuning_path_manager.finetuned_model_dir

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"{finetuned_model_dir = }",  # noqa: G004 - low overhead
        )

    # Create the output directory if it does not exist
    finetuned_model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return finetuned_model_dir
