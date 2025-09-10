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
