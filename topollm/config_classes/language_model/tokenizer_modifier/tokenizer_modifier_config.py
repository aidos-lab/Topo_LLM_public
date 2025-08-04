"""Configuration class for modifying the tokenizer."""

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.typing.enums import TokenizerModifierMode


class TokenizerModifierConfig(ConfigBaseModel):
    """Configurations for modifying the tokenizer."""

    mode: TokenizerModifierMode = Field(
        default=TokenizerModifierMode.DO_NOTHING,
        description="The mode of the tokenizer modifier.",
    )

    padding_token: str | None = Field(
        default="<|pad|>",
        description="The string representation of the padding token. "
        "Can be set to None if not needed or if the tokenizer does not have a padding token.",
    )

    replace_pad_token_with_other_special_token_identifier: str = Field(
        default="eos_token",
        description="The identifier of the other special token to replace the padding token with. "
        "Only used if the mode is REPLACE_PAD_TOKEN_WITH_OTHER_SPECIAL_TOKEN.",
    )
