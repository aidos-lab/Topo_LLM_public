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

# Standard library imports
import pathlib

# Third party imports
from pydantic import Field

# Local imports
from topollm.config_classes.ConfigBaseModel import ConfigBaseModel
from topollm.config_classes.constants import NAME_PREFIXES
from topollm.config_classes.enums import DatasetType, Split

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class DataConfig(ConfigBaseModel):
    """
    Configurations for specifying data.
    """

    column_name: str = Field(
        ...,
        title="Column name to use for computing embeddings.",
        description="The column name to use for computing embeddings.",
    )

    context: str = Field(
        ...,
        title="Context to use for computing embeddings.",
        description="The context to use for computing embeddings.",
    )

    dataset_description_string: str = Field(
        ...,
        title="Dataset description string.",
        description=f"The dataset description string. "
        f"This will be used for creating the file paths",
    )

    data_dir: pathlib.Path | None = Field(
        None,
        title="data_dir argument will be passed to huggingface datasets.",
    )

    dataset_path: str = Field(
        ...,
        title="Dataset identifier for huggingface datasets.",
        description="The dataset identifier for the huggingface datasets "
        "to use for computing embeddings.",
    )

    dataset_name: str | None = Field(
        None,
        title="Dataset name.",
        description="The dataset name.",
    )

    dataset_type: DatasetType = Field(
        ...,
        title="Dataset type.",
        description="The dataset type.",
    )

    number_of_samples: int = Field(
        ...,
        title="Number of samples to use for computing embeddings.",
        description="The number of samples to use for computing embeddings.",
    )

    split: Split = Field(
        ...,
        title="Split to use for computing embeddings.",
        description="The split to use for computing embeddings.",
    )

    @property
    def data_config_description(
        self,
    ) -> str:
        return (
            f"{NAME_PREFIXES['data']}{self.dataset_description_string}"
            f"_"
            f"{NAME_PREFIXES['split']}{self.split}"
            f"_"
            f"{NAME_PREFIXES['context']}{self.context}"
        )