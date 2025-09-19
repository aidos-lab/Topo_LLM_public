"""Base class for dataclasses that should be hashable."""

from pydantic import BaseModel, ConfigDict


class BaseModelHashable(BaseModel):
    """Base class for dataclasses that should be hashable."""

    model_config = ConfigDict(
        frozen=True,
    )

    def __hash__(
        self,
    ) -> int:
        """Hash the object."""
        return hash(
            (type(self), *tuple(self.__dict__.values())),
        )
