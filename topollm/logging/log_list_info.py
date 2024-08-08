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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Imports

# Standard library imports
import logging
import pprint

# Third party imports

# Local imports

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def log_list_info(
    list_: list,
    list_name: str,
    max_log_elements: int = 20,
    logger: logging.Logger = logging.getLogger(__name__),
) -> None:
    """
    Logs information about a list.

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
            Defaults to logging.getLogger(__name__).

    Returns:
        None

    Side effects:
        Logs information about the list to the logger.
    """

    logger.info(f"len({list_name}):\n" f"{len(list_)}")
    logger.info(f"{list_name}[:{max_log_elements}]:\n" f"{pprint.pformat(list_[:max_log_elements])}")
    logger.info(f"{list_name}[-{max_log_elements}:]:\n" f"{pprint.pformat(list_[-max_log_elements:])}")

    return
