# Copyright 2024-2025
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

"""Setting up logging for the transformers library."""

from typing import TYPE_CHECKING

import transformers

if TYPE_CHECKING:
    import logging


def setup_transformers_logger() -> None:
    """Set up the transformers logger for propagating to the root logger."""
    # Set the transformers logging level
    transformers.logging.set_verbosity_info()

    # Make the transformers logger propagate to the root logger
    transformers_logger: logging.Logger = transformers.logging.get_logger()
    transformers_logger.handlers = []  # This avoids duplicate logging
    transformers_logger.propagate = True
