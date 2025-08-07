"""Tokenizer configuration class."""

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.config_classes.constants import ITEM_SEP, KV_SEP, NAME_PREFIXES
from topollm.config_classes.values_to_short_string import bool_to_short_string
from topollm.typing.enums import DescriptionType


class TokenizerConfig(ConfigBaseModel):
    """Configurations for the tokenizer."""

    add_prefix_space: bool = Field(
        default=False,
        title="Add prefix space.",
        description="Whether to add prefix space.",
    )

    max_length: int = Field(
        default=512,
        title="Maximum length of the input sequence.",
        description="The maximum length of the input sequence.",
    )

    return_special_tokens_mask: bool = Field(
        default=True,
        title="Return special tokens mask.",
        description="Whether to return special tokens mask.",
    )

    def get_config_description(
        self,
        description_type: DescriptionType = DescriptionType.LONG,
        short_description_separator: str = "-",
    ) -> str:
        """Return the config description."""
        match description_type:
            case DescriptionType.LONG:
                description: str = (
                    f"{NAME_PREFIXES['add_prefix_space']}"
                    f"{KV_SEP}"
                    f"{str(object=self.add_prefix_space)}"
                    f"{ITEM_SEP}"
                    f"{NAME_PREFIXES['max_length']}"
                    f"{KV_SEP}"
                    f"{str(object=self.max_length)}"
                )
            case DescriptionType.SHORT:
                # This should be a combined description which is short enough to be used in the model name
                description: str = (
                    f"{NAME_PREFIXES['add_prefix_space_short']}"
                    f"{short_description_separator}"
                    f"{bool_to_short_string(value=self.add_prefix_space)}"
                    f"{short_description_separator}"
                    f"{NAME_PREFIXES['max_length_short']}"
                    f"{short_description_separator}"
                    f"{str(object=self.max_length)}"
                )
            case _:
                msg: str = f"Unknown {description_type = }"
                raise ValueError(msg)

        return description
