"""Protocol for model modifiers which can be used to influence the fine-tuning."""

from typing import Protocol

from transformers import PreTrainedModel

from topollm.typing.types import ModifiedModel


class ModelModifier(Protocol):
    """Modify a model for finetuning."""

    def modify_model(
        self,
        model: PreTrainedModel,
    ) -> ModifiedModel: ...  # pragma: no cover
