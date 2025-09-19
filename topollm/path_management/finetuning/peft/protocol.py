"""Protocol for managing the paths for the PEFT process."""

import pathlib
from typing import Protocol

from topollm.typing.enums import DescriptionType


class PEFTPathManager(Protocol):
    """Manages the paths for the PEFT process."""

    @property
    def peft_description_subdir(
        self,
    ) -> pathlib.Path: ...  # pragma: no cover

    def get_config_description(
        self,
        description_type: DescriptionType = DescriptionType.LONG,
        short_description_separator: str = "-",
    ) -> str: ...  # pragma: no cover
