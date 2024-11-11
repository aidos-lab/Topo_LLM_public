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

"""Context for preparing dataloaders."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from topollm.config_classes.data.data_config import DataConfig
from topollm.config_classes.embeddings.embeddings_config import EmbeddingsConfig
from topollm.config_classes.tokenizer.tokenizer_config import TokenizerConfig
from topollm.typing.enums import Verbosity


@dataclass
class EmbeddingDataLoaderPreparerContext:
    """Encapsulates the context needed for preparing dataloaders."""

    data_config: DataConfig
    embeddings_config: EmbeddingsConfig
    tokenizer_config: TokenizerConfig
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
    collate_fn: Callable[[list], dict]
    verbosity: Verbosity = Verbosity.NORMAL
    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger(
            name=__name__,
        ),
    )
