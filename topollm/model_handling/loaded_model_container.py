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
