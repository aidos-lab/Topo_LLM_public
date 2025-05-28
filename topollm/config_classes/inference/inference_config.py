# Copyright 2024
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#


"""Configuration class for inference."""

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel


class InferenceConfig(ConfigBaseModel):
    """Configurations for running model inference."""

    max_length: int = Field(
        default=100,
        title="Maximum length of generated text.",
        description="The maximum length of generated text.",
    )

    num_return_sequences: int = Field(
        default=3,
        title="Number of returned sequences.",
        description="The number of returned sequences.",
    )
