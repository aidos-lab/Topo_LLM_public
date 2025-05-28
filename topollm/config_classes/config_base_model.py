# Copyright 2024
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
# AUTHOR_2 (author2@example.com)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#


"""Module for the ConfigBaseModel class."""

import json
import pathlib
import pprint
from abc import ABC
from os import PathLike
from typing import IO

from pydantic import BaseModel


class ConfigBaseModel(
    BaseModel,
    ABC,
):
    """Base class for all our configuration classes.

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
            ),
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
            ),
        )

    @classmethod
    def load_from_path(
        cls,
        file_path: PathLike,
    ) -> "ConfigBaseModel":
        """Load the configuration from the specified file path."""
        with pathlib.Path(
            file_path,
        ).open(
            mode="r",
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

        with pathlib.Path(
            file_path,
        ).open(
            mode="w",
        ) as file:
            self.save(file)

    def __repr__(
        self,
    ) -> str:
        """Return a pretty printed representation of the configuration."""
        return pprint.pformat(
            self.model_dump(
                mode="python",
            ),
        )
