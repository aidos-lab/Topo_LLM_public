# coding=utf-8
#
# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (ruppik@hhu.de)
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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Imports

# System imports
import logging

# External imports

# Local imports
from convlab.tda.config_classes.constants import FILE_NAME_TRUNCATION_LENGTH

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def truncate_length_of_desc(
    desc: str,
    truncation_length: int = FILE_NAME_TRUNCATION_LENGTH,
    logger: logging.Logger = logging.getLogger(__name__),
):
    """
    Truncate the length of the description string if it is too long.
    """
    if len(desc) > truncation_length:
        logger.warning(
            f"Too long:\n"
            f"{desc = }\n"
            f"{len(desc) = }\n"
            f"Will be truncated to {truncation_length = } characters.",
        )
        desc = desc[:truncation_length]

    return desc
