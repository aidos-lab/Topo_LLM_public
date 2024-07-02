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

import logging
import subprocess

from topollm.typing.enums import Verbosity

default_logger = logging.getLogger(__name__)


def call_command(
    command: list[str],
    *,
    dry_run: bool = False,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Call a command in a subprocess."""
    if verbosity >= Verbosity.NORMAL:
        # Add separator line to log
        logger.info(
            30 * "-",
        )

    if dry_run:
        # Logging always enabled in dry run mode
        logger.info(
            "Dry run enabled. Command not executed.",
        )
        logger.info(
            "** Dry run ** command:\n%s",
            command,
        )
    else:
        # Calling submit_job
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                "Calling command ...",
            )
            logger.info(
                "command:\n%s",
                command,
            )

        subprocess.run(
            args=command,
            shell=False,
            check=True,
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                "Calling command DONE",
            )

    if verbosity >= Verbosity.NORMAL:
        # Add separator line to log
        logger.info(
            30 * "-",
        )
