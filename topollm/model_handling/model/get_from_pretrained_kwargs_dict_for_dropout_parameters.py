# Copyright 2024-2025
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

from topollm.config_classes.language_model.language_model_config import LanguageModelConfig
from topollm.typing.enums import DropoutMode


def get_from_pretrained_kwargs_dict_for_dropout_parameters(
    language_model_config: LanguageModelConfig,
) -> dict:
    """Get the keyword arguments for the dropout parameters."""
    match language_model_config.dropout.mode:
        case DropoutMode.DEFAULTS:
            # Empty dictionary for default values.
            from_pretrained_kwargs_dict: dict = {}
        case DropoutMode.MODIFY_ROBERTA_DROPOUT_PARAMETERS:
            from_pretrained_kwargs_dict: dict = {
                "hidden_dropout_prob": language_model_config.dropout.probabilities.hidden_dropout_prob,
                "attention_probs_dropout_prob": language_model_config.dropout.probabilities.attention_probs_dropout_prob,
                "classifier_dropout": language_model_config.dropout.probabilities.classifier_dropout,
            }
        # Note: We can extend this switch statement with additional cases for other dropout modes
        # if we add support for other language models later.
        case _:
            msg: str = f"Unknown {language_model_config.dropout.mode = }"
            raise ValueError(
                msg,
            )

    return from_pretrained_kwargs_dict
