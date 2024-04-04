# coding=utf-8
#
# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Julius von Rohrscheidt (julius.rohrscheidt@helmholtz-muenchen.de)
# Benjamin Ruppik (ruppik@hhu.de)
#
# This code was generated with the help of AI writing assistants
# including GitHub Copilot, ChatGPT, Bing Chat.
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
import os
import pprint
import sys

# Third party imports

# Local imports

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def setup_exception_logging(
    logger: logging.Logger = logging.getLogger(__name__),
) -> None:
    """
    Sets up a custom exception handler that logs uncaught exceptions using the provided logger.

    Args:
        logger: An instance of a logger to be used for logging exceptions.

    Side effects:
        - Sets the HYDRA_FULL_ERROR environment variable to "1".
        - Sets the sys.excepthook to a custom exception handler that logs uncaught exceptions using the provided logger.
    """

    # Setting this environment variable to "1" makes Hydra print the full stack trace of exceptions.
    # This is necessary to set here, because otherwise the exceptions would not be correctly logged.
    #
    # We use print here instead of logging, since this function is usually called before the logging is set up,
    # and we want to make sure that this message is printed.
    print(f"Setting HYDRA_FULL_ERROR environment variable to '1'.")
    os.environ["HYDRA_FULL_ERROR"] = "1"
    print(f"{os.environ['HYDRA_FULL_ERROR'] = }")

    def handle_exception(
        exc_type,
        exc_value,
        exc_traceback,
    ):
        """
        Handles uncaught exceptions by logging them, except for KeyboardInterrupt.

        This function is designed to be compatible with sys.excepthook.
        Thus, you should not call this function directly, but rather set it as the sys.excepthook.
        Also make sure you do not change the signature of this function, as it is called by sys.excepthook.

        Args:
            exc_type:
                The exception type.
            exc_value:
                The exception value.
            exc_traceback:
                The traceback object.
        """
        if issubclass(
            exc_type,
            KeyboardInterrupt,
        ):
            sys.__excepthook__(
                exc_type,
                exc_value,
                exc_traceback,
            )
            return
        else:
            logger.critical(
                "Uncaught exception",
                exc_info=(
                    exc_type,
                    exc_value,
                    exc_traceback,
                ),
            )

    sys.excepthook = handle_exception

    return None
