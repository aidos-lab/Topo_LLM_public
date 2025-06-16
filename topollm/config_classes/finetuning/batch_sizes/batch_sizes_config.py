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
