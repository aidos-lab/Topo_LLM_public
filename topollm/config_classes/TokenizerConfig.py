from topollm.config_classes.ConfigBaseModel import ConfigBaseModel
from topollm.config_classes.constants import NAME_PREFIXES


from pydantic import Field


class TokenizerConfig(ConfigBaseModel):
    add_prefix_space: bool = Field(
        ...,
        title="Add prefix space.",
        description="Whether to add prefix space.",
    )

    max_length: int = Field(
        ...,
        title="Maximum length of the input sequence.",
        description="The maximum length of the input sequence.",
    )

    @property
    def tokenizer_config_description(
        self,
    ) -> str:
        """
        Get the description of the tokenizer config.

        Returns:
            str: The description of the tokenizer.
        """
        return (
            f"{NAME_PREFIXES['add_prefix_space']}{str(self.add_prefix_space)}"
            f"_"
            f"{NAME_PREFIXES['max_length']}{str(self.max_length)}"
        )