# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (ruppik@hhu.de)
# Julius von Rohrscheidt (julius.rohrscheidt@helmholtz-muenchen.de)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Modify a tokenizer by adding a padding token."""

import logging

from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from topollm.logging.log_model_info import log_model_info
from topollm.typing.enums import Verbosity

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
        self.padding_token = padding_token
        self.verbosity = verbosity
        self.logger = logger

        self.modified_tokenizer: None | PreTrainedTokenizer | PreTrainedTokenizerFast = None

    def modify_tokenizer(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    ) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                f"Modifying tokenizer {tokenizer = } " f"by adding padding token {self.padding_token = } ..."
            )

        # Check if the tokenizer already has the padding token.
        if self.padding_token in tokenizer.all_special_tokens:
            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    f"The tokenizer already has the padding token " f"{self.padding_token = }. " f"Nothing to do."
                )
        else:
            num_added_tokens = tokenizer.add_special_tokens(
                {"pad_token": self.padding_token},
            )

            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(f"Added {num_added_tokens = } token(s).")
                self.logger.info(f"{tokenizer = }")
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
        if self.verbosity >= 1:
            self.logger.info(f"Updating model " f"to match the modified tokenizer " f"{self.modified_tokenizer = } ...")
            log_model_info(
                model=model,
                model_name="model",
                logger=self.logger,
            )

        if self.modified_tokenizer is None:
            msg = "The tokenizer has not been modified yet. Please call 'modify_tokenizer' first."
            raise ValueError(msg)

        if self.verbosity >= 1:
            self.logger.info(f"{len(self.modified_tokenizer) = }")

        # The return value from 'resize_token_embeddings' is a pointer
        # to the model's token embeddings module,
        # which we only need for logging.
        #
        # Note: We could use 'pad_to_multiple_of'
        # in the future to speed up training.
        embeddings_module = model.resize_token_embeddings(
            new_num_tokens=len(self.modified_tokenizer),
            pad_to_multiple_of=None,
        )

        if self.verbosity >= 1:
            self.logger.info("Logging 'model' after potentially resizing token embeddings:")
            self.logger.info(f"embeddings_module:\n" f"{embeddings_module}")
            log_model_info(
                model=model,
                model_name="model",
                logger=self.logger,
            )

        return model
