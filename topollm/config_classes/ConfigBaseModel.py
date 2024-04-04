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

import json
import pathlib
import pprint
from abc import ABC
from os import PathLike
from typing import IO

from pydantic import BaseModel

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Globals

# END Globals
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class ConfigBaseModel(
    BaseModel,
    ABC,
):
    """
    This class is the base class for all our configuration classes.
    It is inherited from Pydantic's BaseModel and adds some functionality
    for saving and loading the configuration to and from a json file.
    """

    def save(
        self,
        io: IO[str],
    ) -> None:
        """Save the configuration to a general IO object."""
        io.write(
            json.dumps(
                self.model_dump(
                    mode="json",
                ),
                indent=4,
            )
        )

    @classmethod
    def load(
        cls,
        io: IO[str],
    ) -> "ConfigBaseModel":
        """Load the configuration from a general IO object."""
        return cls.model_validate(
            json.load(
                io,
            )
        )

    @classmethod
    def load_from_path(
        cls,
        file_path: PathLike,
    ) -> "ConfigBaseModel":
        """Load the configuration from the specified file path."""
        with open(
            file_path,
            "r",
        ) as file:
            return cls.load(file)

    def save_to_path(
        self,
        file_path: PathLike,
    ) -> None:
        """Save the configuration to the specified file path."""

        file_path = pathlib.Path(
            file_path,
        )

        # Create the parent directories if they do not exist
        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            file_path,
            "w",
        ) as file:
            self.save(file)

    def __repr__(
        self,
    ) -> str:
        return pprint.pformat(
            self.model_dump(
                mode="python",
            )
        )
