"""Configuration class for the finetuning datasets."""

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.config_classes.data.data_config import DataConfig


class FinetuningDatasetsConfig(ConfigBaseModel):
    """Configurations for the finetuning datasets."""

    train_dataset: DataConfig = Field(
        ...,
        description="The configuration for the training dataset.",
    )

    eval_dataset: DataConfig = Field(
        ...,
        description="The configuration for the evaluation dataset.",
    )
