import pathlib
from typing import Protocol

from topollm.typing.enums import DescriptionType


class FinetuningPathManager(Protocol):
    """Manages the paths for the finetuning process."""

    def get_finetuned_model_relative_dir(
        self,
    ) -> pathlib.Path: ...  # pragma: no cover

    def get_finetuned_short_model_name(
        self,
    ) -> str: ...  # pragma: no cover

    def get_finetuning_parameters_description_for_short_model_name(
        self,
        short_description_separator: str = "-",
    ) -> str: ...  # pragma: no cover

    @property
    def finetuned_model_dir(
        self,
    ) -> pathlib.Path: ...  # pragma: no cover

    @property
    def logging_dir(
        self,
    ) -> pathlib.Path | None: ...  # pragma: no cover
