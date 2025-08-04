"""Protocol for a tokenizer modifier."""

from typing import Protocol

from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast


class TokenizerModifier(Protocol):
    """Protocol for a tokenizer modifier."""

    def modify_tokenizer(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    ) -> PreTrainedTokenizer | PreTrainedTokenizerFast: ...  # pragma: no cover

    def update_model(
        self,
        model: PreTrainedModel,
    ) -> PreTrainedModel:
        """Return the updated model after modifying the tokenizer, to make it compatible with the new tokenizer.

        When modifying the tokenizer, the model might need to be updated as well.
        """
        ...  # pragma: no cover
