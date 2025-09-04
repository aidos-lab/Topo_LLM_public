"""Configuration class for filtering tokens."""

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel


class FilterTokensConfig(ConfigBaseModel):
    """Configurations for filtering tokens."""

    remove_bos_token: bool = Field(
        default=False,
        title="Remove beginning of sequence token.",
        description="Whether to remove the beginning of sequence token.",
    )

    remove_eos_token: bool = Field(
        default=True,
        title="Remove end of sequence token.",
        description="Whether to remove the end of sequence token.",
    )

    remove_pad_token: bool = Field(
        default=True,
        title="Remove padding token.",
        description="Whether to remove the padding token.",
    )
