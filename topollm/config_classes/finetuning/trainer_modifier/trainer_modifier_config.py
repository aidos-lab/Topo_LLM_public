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
