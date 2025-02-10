# Copyright 2024-2025
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (mail@ruppik.net)
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

"""Log information about a list."""

import logging
import pprint

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def log_list_info(
    list_: list,
    list_name: str,
    max_log_elements: int = 20,
    logger: logging.Logger = default_logger,
) -> None:
    """Log information about a list.

    Args:
        list_ (list):
            The list to log information about.
        list_name (str):
            The name of the list.
        max_log_elements (int, optional):
            The maximum number of elements to log for the head and tail of the list.
            Defaults to 20.
        logger (logging.Logger, optional):
            The logger to log information to.

    Returns:
        None

    Side effects:
        Logs information about the list to the logger.

    """
    logger.info(
        msg=f"len({list_name}):\n{len(list_)}",  # noqa: G004 - low overhead
    )
    logger.info(
        msg=f"{list_name}[:{max_log_elements}]:\n{pprint.pformat(list_[:max_log_elements])}",  # noqa: G004 - low overhead
    )
    logger.info(
        msg=f"{list_name}[-{max_log_elements}:]:\n{pprint.pformat(list_[-max_log_elements:])}",  # noqa: G004 - low overhead
    )
