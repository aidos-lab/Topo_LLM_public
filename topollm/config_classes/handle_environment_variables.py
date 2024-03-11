# coding=utf-8
#
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

# System imports
import logging
import pathlib
import os

from traitlets import default

# Local imports


# Third-party imports


# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def get_data_dir_path_from_environment_variable(
    set_variables_in_this_script: bool = True,
    data_dir_env_variable_name: str = "DATA_DIR",
    default_data_dir: str = "$HOME/git-source/Topo_LLM/data",
) -> pathlib.Path:
    if set_variables_in_this_script:
        # Optional: Set environment variables in the script.
        logging.info("Setting environment variables in script.")

        os.environ[data_dir_env_variable_name] = default_data_dir

    # # # #
    # Get the base path from the environment variable
    data_dir_env = os.environ.get(
        data_dir_env_variable_name,
    )
    if data_dir_env is None:
        raise ValueError(
            f"Environment variable {data_dir_env_variable_name = !r} is not set"
        )
    else:
        data_dir_env = os.path.expandvars(
            data_dir_env,
        )  # Replace the $HOME part with the user's home directory
        data_dir = pathlib.Path(
            data_dir_env,
        ).resolve()  # Compute the canonical, absolute form
        logging.info(f"{data_dir = }")

    # check if the paths are valid directories
    if not data_dir.is_dir():
        raise ValueError(f"{data_dir = }" f"is not a directory")

    return data_dir
