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

"""Configuration for specifying data."""

import pathlib

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.config_classes.constants import ITEM_SEP, KV_SEP, NAME_PREFIXES
from topollm.config_classes.data.data_splitting_config import DataSplittingConfig
from topollm.typing.enums import DatasetType, DescriptionType, Split


class DataConfig(ConfigBaseModel):
    """Configuration for specifying data."""

    column_name: str = Field(
        default="body",
        title="Column name to use for computing embeddings.",
        description="The column name to use for computing embeddings.",
    )

    context: str = Field(
        default="dataset_entry",
        title="Context to use for computing embeddings.",
        description="The context to use for computing embeddings.",
    )

    dataset_description_string: str = Field(
        default="one-year-of-tsla-on-reddit",
        title="Dataset description string.",
        description=("The dataset description string. This will be used for creating the file paths"),
    )

    data_dir: pathlib.Path | None = Field(
        default=None,
        title="data_dir argument will be passed to huggingface datasets.",
    )

    dataset_path: str = Field(
        default="SocialGrep/one-year-of-tsla-on-reddit",
        title="Dataset identifier for huggingface datasets.",
        description="The dataset identifier for the huggingface datasets to use for computing embeddings.",
    )

    dataset_name: str | None = Field(
        default="comments",
        title="Dataset name.",
        description="The dataset name.",
    )

    dataset_type: DatasetType = Field(
        default=DatasetType.HUGGINGFACE_DATASET,
        title="Dataset type.",
        description="The dataset type.",
    )

    data_splitting: DataSplittingConfig = Field(
        default=DataSplittingConfig(),
        title="Data splitting configuration.",
        description="The data splitting configuration. This is useful if the dataset is not already split.",
    )

    feature_column_name: str = Field(
        default="ner_tags",
        title="Feature column name, will be used when finetuning a model on a specific tagging or classification task.",
    )

    # Config options for the data subset
    split: Split = Field(
        default=Split.TRAIN,
        title="Split to use for computing embeddings.",
        description="The split to use for computing embeddings.",
    )

    number_of_samples: int = Field(
        default=5_000,
        title="Number of samples to use for computing embeddings.",
        description="The number of samples to use for computing embeddings.",
    )

    def get_partial_path(
        self,
    ) -> pathlib.Path:
        """Return the partial path which can be used in the path manager.

        This partial path can be used to determine the save location of objects derived from the data configuration."""
        partial_path: pathlib.Path = pathlib.Path(
            self.long_config_description_with_data_splitting_without_data_subset,
            # TODO: Include the subset information in the path once it is implemented as a config group
        )

        return partial_path

    @property
    def long_config_description_with_data_splitting_without_data_subset(
        self,
    ) -> str:
        description = (
            f"{NAME_PREFIXES['data']}"
            f"{KV_SEP}"
            f"{self.dataset_description_string}"
            f"{ITEM_SEP}"
            f"{self.data_splitting.config_description}"  # Description of the data splitting configuration
            f"{ITEM_SEP}"
            f"{NAME_PREFIXES['context']}"
            f"{KV_SEP}"
            f"{self.context}"
            f"{ITEM_SEP}"
            f"{NAME_PREFIXES['feature_column_name']}"
            f"{KV_SEP}"
            f"{self.feature_column_name}"
        )

        # TODO: Move this into a separate part of the path
        #
        # f"{ITEM_SEP}"
        # f"{NAME_PREFIXES['split']}"
        # f"{KV_SEP}"
        # f"{self.split}"
        # f"{ITEM_SEP}"
        # f"{NAME_PREFIXES['number_of_samples']}"
        # f"{KV_SEP}"
        # f"{self.number_of_samples}"
        # f"{ITEM_SEP}"

        return description

    def get_config_description(
        self,
        description_type: DescriptionType = DescriptionType.LONG,
        short_description_separator: str = "-",
    ) -> str:
        """Return the config description."""
        match description_type:
            case DescriptionType.LONG:
                return self.long_config_description_with_data_splitting_without_data_subset
            case DescriptionType.SHORT:
                # This should be a combined description which is short enough to be used in the model name
                short_description: str = (
                    self.dataset_description_string
                    + short_description_separator
                    + self.split
                    + self.data_splitting.config_description
                    + short_description_separator
                    + self.feature_column_name
                    # TODO: Move this into a separate part of the path
                    # + short_description_separator
                    # + str(object=self.number_of_samples)
                )
                return short_description
            case _:
                msg: str = f"Unknown {description_type = }"
                raise ValueError(msg)
