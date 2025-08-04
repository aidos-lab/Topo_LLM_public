"""Modify a tokenizer by adding a padding token."""

import logging
from typing import TYPE_CHECKING

from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from topollm.logging.log_model_info import log_model_info
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    from torch.nn.modules.sparse import Embedding

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


class TokenizerModifierAddPaddingToken:
    """Modify a tokenizer by adding a padding token."""

    def __init__(
        self,
        padding_token: str = "<|pad|>",  # noqa: S107 - Not a magic number.
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Initialize the tokenizer modifier."""
        self.padding_token: str = padding_token
        self.verbosity: Verbosity = verbosity
        self.logger: logging.Logger = logger

        self.modified_tokenizer: None | PreTrainedTokenizer | PreTrainedTokenizerFast = None

    def modify_tokenizer(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    ) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg=f"Modifying tokenizer {tokenizer = } by adding padding token {self.padding_token = } ...",  # noqa: G004 - low overhead
            )

        # Check if the tokenizer already has the padding token.
        if self.padding_token in tokenizer.all_special_tokens:
            if self.verbosity >= Verbosity.NORMAL:
                self.logger.warning(
                    msg=f"The tokenizer already has the padding token {self.padding_token = }. Nothing to do.",  # noqa: G004 - low overhead
                )
        else:
            num_added_tokens: int = tokenizer.add_special_tokens(
                special_tokens_dict={"pad_token": self.padding_token},
            )

            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    msg=f"Added {num_added_tokens = } token(s).",  # noqa: G004 - low overhead
                )
                self.logger.info(
                    msg=f"{tokenizer = }",  # noqa: G004 - low overhead
                )
                self.logger.info(
                    "Important: Make sure to also resize "
                    "the token embedding matrix "
                    "of the model so that its embedding matrix "
                    "matches the tokenizer.",
                )

        self.modified_tokenizer = tokenizer

        return tokenizer

    def update_model(
        self,
        model: PreTrainedModel,
    ) -> PreTrainedModel:
        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg=f"Updating model to match the modified tokenizer {self.modified_tokenizer = } ...",  # noqa: G004 - low overhead
            )
            log_model_info(
                model=model,
                model_name="model",
                logger=self.logger,
            )

        if self.modified_tokenizer is None:
            msg = "The tokenizer has not been modified yet. Please call 'modify_tokenizer' first."
            raise ValueError(msg)

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg=f"{len(self.modified_tokenizer) = }",  # noqa: G004 - low overhead
            )

        # The return value from 'resize_token_embeddings' is a pointer
        # to the model's token embeddings module,
        # which we only need for logging.
        #
        # Note: We could use 'pad_to_multiple_of'
        # in the future to speed up training.
        embeddings_module: Embedding = model.resize_token_embeddings(
            new_num_tokens=len(self.modified_tokenizer),
            pad_to_multiple_of=None,
        )

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg="Logging 'model' after potentially resizing token embeddings:",
            )
            self.logger.info(
                msg=f"embeddings_module:\n{embeddings_module}",  # noqa: G004 - low overhead
            )
            log_model_info(
                model=model,
                model_name="model",
                logger=self.logger,
            )

        return model
