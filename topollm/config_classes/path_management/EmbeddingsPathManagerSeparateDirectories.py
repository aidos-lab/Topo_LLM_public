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
import os
import pathlib

# Local imports
from topollm.config_classes.DataConfig import DataConfig
from topollm.config_classes.EmbeddingsConfig import EmbeddingsConfig
from topollm.config_classes.finetuning.FinetuningConfig import FinetuningConfig
from topollm.config_classes.PathsConfig import PathsConfig
from topollm.config_classes.path_management.truncate_length_of_desc import (
    truncate_length_of_desc,
)
from topollm.config_classes.TransformationsConfig import (
    TransformationsConfig,
)
from topollm.config_classes.constants import NAME_PREFIXES

# Third-party imports


# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Globals

# END Globals
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class EmbeddingsPathManagerSeparateDirectories:
    def __init__(
        self,
        data_config: DataConfig,
        embeddings_config: EmbeddingsConfig,
        paths_config: PathsConfig,
        transformations_config: TransformationsConfig,
        verbosity: int = 1,
        logger: logging.Logger = logging.getLogger(__name__),
    ):
        self.data_config = data_config
        self.embeddings_config = embeddings_config
        self.paths_config = paths_config
        self.transformations_config = transformations_config

        self.verbosity = verbosity
        self.logger = logger

    @property
    def data_dir(
        self,
    ) -> pathlib.Path:
        return self.paths_config.data_dir

    def get_nested_subfolder_path(
        self,
    ) -> pathlib.Path:
        """
        Constructs a nested subfolder path based on specific attributes.

        Args:
            include_data:
                Whether to include the data prefix in the path.

        Returns:
            pathlib.Path: The constructed nested subfolder path.
        """

        path = pathlib.Path(
            self.data_config.data_config_description,
            self.embeddings_config.embeddings_config_description,
            self.embeddings_config.tokenizer.tokenizer_config_description,
            self.embeddings_config.language_model.lanugage_model_config_description,
            self.embeddings_config.embedding_extraction.embedding_extraction_config_description,
            self.transformations_config.transformation_config_description,
        )

        return path

    # # # #
    # array_dir

    @property
    def array_dir_absolute_path(
        self,
    ) -> pathlib.Path:
        path = pathlib.Path(
            self.data_dir,
            self.array_dir_relative_path,
        )

        if self.verbosity >= 1:
            self.logger.info(f"array_dir_absolute_path:\n" f"{path}")

        return path

    @property
    def array_dir_relative_path(
        self,
    ) -> pathlib.Path:
        path = pathlib.Path(
            "embeddings",
            "arrays",
            self.get_nested_subfolder_path(),
            self.array_dir_name,
        )

        return path

    @property
    def array_dir_name(
        self,
    ) -> str:
        return "array_dir"

    # # # #
    # metadata_dir

    @property
    def metadata_dir_absolute_path(
        self,
    ) -> pathlib.Path:
        path = pathlib.Path(
            self.data_dir,
            self.metadata_dir_relative_path,
        )

        if self.verbosity >= 1:
            self.logger.info(f"metadata_dir_absolute_path:\n" f"{path}")

        return path

    @property
    def metadata_dir_relative_path(
        self,
    ) -> pathlib.Path:
        path = pathlib.Path(
            "embeddings",
            "metadata",
            self.get_nested_subfolder_path(),
            self.metadata_dir_name,
        )

        return path

    @property
    def metadata_dir_name(
        self,
    ) -> str:
        return "metadata_dir"
