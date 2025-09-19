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
