"""Configuration class for specifying transformations."""

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.config_classes.constants import KV_SEP, NAME_PREFIXES


class TransformationsConfig(ConfigBaseModel):
    """Configurations for specifying transformations."""

    normalization: str = Field(
        default="None",
        title="Normalization method.",
        description="The normalization method.",
    )

    @property
    def config_description(
        self,
    ) -> str:
        desc = f"{NAME_PREFIXES['normalization']}"
        desc += f"{KV_SEP}"
        desc += f"{self.normalization}"

        return desc
