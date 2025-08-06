"""Configuration for filtering data."""

from pydantic import Field

from topollm.config_classes.config_base_model import ConfigBaseModel
from topollm.config_classes.constants import KV_SEP, NAME_PREFIXES
from topollm.config_classes.values_to_short_string import bool_to_short_string
from topollm.typing.enums import DescriptionType


class DataFilteringConfig(ConfigBaseModel):
    """Configuration for filtering data."""

    remove_empty_sequences: bool = Field(
        default=True,
        title="Remove empty sequences.",
        description="Remove empty sequences.",
    )

    remove_sequences_with_starting_segment_in_this_blocklist: list[str] = Field(
        default_factory=list,
        title="Remove sequences with starting segment in this blocklist.",
        description="Remove sequences with starting segment in this blocklist. Empty list by default.",
    )

    def get_config_description(
        self,
        description_type: DescriptionType = DescriptionType.LONG,
        short_description_separator: str = "-",
    ) -> str:
        """Return the config description.

        Note that the sequence starting segment blocklist is not included in the description.
        """
        match description_type:
            case DescriptionType.LONG:
                description: str = (
                    f"{NAME_PREFIXES['data_filtering_remove_empty_sequences']}{KV_SEP}{self.remove_empty_sequences}"
                )
            case DescriptionType.SHORT:
                description = (
                    f"{NAME_PREFIXES['data_filtering_remove_empty_sequences_short']}{short_description_separator}"
                    f"{bool_to_short_string(value=self.remove_empty_sequences)}"
                )
            case _:
                msg: str = f"Unknown {description_type = }"
                raise ValueError(
                    msg,
                )

        return description
