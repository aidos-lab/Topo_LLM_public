"""Configuration class for specifying batch sizes."""

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel


class BatchSizesConfig(ConfigBaseModel):
    """Configurations for specifying batch sizes."""

    train: int = Field(
        default=16,
        description="The batch size for training.",
    )

    eval: int = Field(
        default=16,
        description="The batch size for evaluation.",
    )
