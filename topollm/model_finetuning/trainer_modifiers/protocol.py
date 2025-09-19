"""Protocol for modifying how the gradients behave during fune-tuning."""

from typing import Protocol

import transformers


class TrainerModifier(Protocol):
    """Modify Trainer during fine-tuning."""

    def modify_trainer(
        self,
        trainer: transformers.Trainer,
    ) -> transformers.Trainer: ...  # pragma: no cover
