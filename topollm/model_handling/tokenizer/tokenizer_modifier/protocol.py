# Copyright 2024
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
# AUTHOR_2 (author2@example.com)
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
