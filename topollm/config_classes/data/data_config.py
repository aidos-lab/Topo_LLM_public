# Copyright 2024-2025
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


"""Configuration for specifying data."""

import pathlib

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.config_classes.constants import ITEM_SEP, KV_SEP, NAME_PREFIXES
from topollm.config_classes.data.data_filtering_config import DataFilteringConfig
from topollm.config_classes.data.data_splitting_config import DataSplittingConfig
from topollm.config_classes.data.data_subsampling_config import DataSubsamplingConfig
from topollm.typing.enums import DatasetsLoadingFunction, DatasetType, DescriptionType


class DataNormalizationConfig(ConfigBaseModel):
    """Configuration for normalizing data."""

    apply_string_strip: bool = Field(
        default=False,
        title="Apply string.strip().",
        description="Apply string.strip() to the data.",
    )


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
        title="data_dir argument will be passed to `data_dir=` argument in huggingface datasets; "
        "currently unused for other dataset types.",
    )

    dataset_path: str = Field(
        default="SocialGrep/one-year-of-tsla-on-reddit",
        title="Dataset identifier for huggingface datasets; "
        "path to the saved processed dataset for the SETSUMBT datasets.",
        description="The dataset identifier to use for computing embeddings; "
        "this is used for the `path=` argument for the huggingface datasets; "
        "for the SETSUMBT datasets, this is the path to the saved processed dataset.",
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

    datasets_loading_function: DatasetsLoadingFunction = Field(
        default=DatasetsLoadingFunction.LOAD_DATASET,
        title="HuggingFace datasets loading function",
        description="The HuggingFace datasets loading function to use.",
    )

    data_splitting: DataSplittingConfig = Field(
        default_factory=DataSplittingConfig,
        title="Data splitting configuration.",
        description="The data splitting configuration. This is useful if the dataset is not already split.",
    )

    data_subsampling: DataSubsamplingConfig = Field(
        default_factory=DataSubsamplingConfig,
        title="Data subsampling configuration.",
        description="The data subsampling configuration.",
    )

    filtering: DataFilteringConfig = Field(
        default_factory=DataFilteringConfig,
        title="Data filtering configuration.",
        description="The data filtering configuration.",
    )

    normalization: DataNormalizationConfig = Field(
        default_factory=DataNormalizationConfig,
        title="Data normalization configuration.",
        description="The data normalization configuration.",
    )

    feature_column_name: str = Field(
        default="ner_tags",
        title="Feature column name, will be used when finetuning a model on a specific tagging or classification task.",
    )

    def get_partial_path(
        self,
    ) -> pathlib.Path:
        """Return the partial path which can be used in the path manager.

        This partial path can be used to determine the save location of objects derived from the data configuration.
        """
        partial_path: pathlib.Path = pathlib.Path(
            self.long_config_description_with_data_splitting_without_data_subsampling,
            self.data_subsampling.config_description,
        )

        return partial_path

    @property
    def long_config_description_with_data_splitting_without_data_subsampling(
        self,
    ) -> str:
        description: str = (
            f"{NAME_PREFIXES['data']}"
            f"{KV_SEP}"
            f"{self.dataset_description_string}"
            f"{ITEM_SEP}"
            f"{
                self.filtering.get_config_description(
                    description_type=DescriptionType.LONG,
                )
            }"  # Description of the data filtering configuration
            f"{ITEM_SEP}"
            f"{
                self.data_splitting.get_config_description(
                    description_type=DescriptionType.LONG,
                )
            }"  # Description of the data splitting configuration
            f"{ITEM_SEP}"
            f"{NAME_PREFIXES['context']}"
            f"{KV_SEP}"
            f"{self.context}"
            f"{ITEM_SEP}"
            f"{NAME_PREFIXES['feature_column_name']}"
            f"{KV_SEP}"
            f"{self.feature_column_name}"
        )

        return description

    def get_short_config_description_with_data_splitting_without_data_subsampling(
        self,
        short_description_separator: str = "-",
    ) -> str:
        """Return a short description of the configuration without the data subsampling information."""
        description: str = (
            self.dataset_description_string
            + short_description_separator
            + self.filtering.get_config_description(
                description_type=DescriptionType.SHORT,
            )
            + short_description_separator
            + self.data_splitting.get_config_description(
                description_type=DescriptionType.SHORT,
            )
            + short_description_separator
            + self.feature_column_name
        )
        return description

    def get_config_description(
        self,
        description_type: DescriptionType = DescriptionType.LONG,
        short_description_separator: str = "-",
    ) -> str:
        """Return the config description."""
        match description_type:
            case DescriptionType.LONG:
                return (
                    self.long_config_description_with_data_splitting_without_data_subsampling
                    + ITEM_SEP
                    + self.data_subsampling.config_description
                )
            case DescriptionType.SHORT:
                # This should be a combined description which is short enough to be used in the model name
                short_description: str = (
                    self.get_short_config_description_with_data_splitting_without_data_subsampling(
                        short_description_separator=short_description_separator,
                    )
                    + ITEM_SEP
                    + self.data_subsampling.get_short_config_description(
                        short_description_separator=short_description_separator,
                    )
                )
                return short_description
            case _:
                msg: str = f"Unknown {description_type = }"
                raise ValueError(msg)
