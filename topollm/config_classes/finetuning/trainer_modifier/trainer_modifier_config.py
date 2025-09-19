"""Configuration class for the trainer modifier."""

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.typing.enums import TrainerModifierMode


class TrainerModifierConfig(ConfigBaseModel):
    """Configurations for the trainer modifier."""

    mode: TrainerModifierMode = Field(
        default=TrainerModifierMode.DO_NOTHING,
        description="The trainer modifier mode.",
    )

    num_samples: int = Field(
        default=10,
        description="The number of samples to use when logging predictions.",
    )

    frequency: int = Field(
        default=400,
        description="The frequency of logging predictions.",
    )
