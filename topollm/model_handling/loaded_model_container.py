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

"""Container for a loaded device, tokenizer, model."""

from dataclasses import dataclass

import torch
import transformers

from topollm.config_classes.language_model.language_model_config import LanguageModelConfig
from topollm.config_classes.tokenizer.tokenizer_config import TokenizerConfig
from topollm.model_handling.tokenizer.tokenizer_modifier.protocol import TokenizerModifier
from topollm.typing.enums import LMmode


@dataclass
class LoadedModelContainer:
    """Container for a loaded device, tokenizer, model."""

    device: torch.device

    model: transformers.PreTrainedModel
    language_model_config: LanguageModelConfig
    lm_mode: LMmode

    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast
    tokenizer_config: TokenizerConfig
    tokenizer_modifier: TokenizerModifier
