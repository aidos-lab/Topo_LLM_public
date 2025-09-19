"""Protocol for modifying how the gradients behave during fune-tuning."""

from typing import Protocol

from topollm.typing.types import ModifiedModel


class GradientModifier(Protocol):
    """Modify the gradient behaviour during fine-tuning."""

    def modify_gradients(
        self,
        model: ModifiedModel,
    ) -> ModifiedModel: ...  # pragma: no cover
