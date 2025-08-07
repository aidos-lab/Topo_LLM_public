"""Configuration class for specifying dataset map."""

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel


class DatasetMapConfig(ConfigBaseModel):
    """Configurations for specifying dataset map."""

    batch_size: int = Field(
        default=1_000,
        title="Batch size for mapping tokenization on dataset.",
        description="The batch size for mapping tokenization on dataset.",
    )

    num_proc: int = Field(
        default=1,
        title="Number of processes for mapping tokenization on dataset.",
        description="The number of processes for mapping tokenization on dataset.",
    )
