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

import logging
import os
import pathlib

from topollm.path_management.finetuning.protocol import (
    FinetuningPathManager,
)

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def prepare_logging_dir(
    finetuning_path_manager: FinetuningPathManager,
    logger: logging.Logger = default_logger,
) -> pathlib.Path | None:
    """Prepare the logging directory for the finetuning process."""
    logging_dir: pathlib.Path | None = finetuning_path_manager.logging_dir
    logger.info(
        msg=f"{logging_dir = }",  # noqa: G004 - low overhead
    )

    # Create the logging directory if it does not exist
    if logging_dir is not None:
        logging_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
    else:
        logger.info(
            msg="No logging directory specified. Using default logging from transformers.Trainer.",
        )

    return logging_dir
