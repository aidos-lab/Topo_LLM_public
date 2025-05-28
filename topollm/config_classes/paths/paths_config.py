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


"""Configuration class for specifying paths."""

import pathlib

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.config_classes.constants import TOPO_LLM_REPOSITORY_BASE_PATH


class PathsConfig(ConfigBaseModel):
    """Configurations for specifying paths."""

    data_dir: pathlib.Path = Field(
        default=pathlib.Path(
            TOPO_LLM_REPOSITORY_BASE_PATH,
            "data",
        ),
        title="Data path.",
        description="The path to the data.",
    )

    repository_base_path: pathlib.Path = Field(
        default=TOPO_LLM_REPOSITORY_BASE_PATH,
        title="Repository base path.",
        description="The base path to the repository.",
    )
