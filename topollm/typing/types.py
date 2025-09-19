from typing import TypeAlias

import peft.peft_model
import transformers
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from topollm.model_inference.perplexity.saving.sentence_perplexity_container import SentencePerplexityContainer

PerplexityResultsList: TypeAlias = list[tuple[int, SentencePerplexityContainer]]

TransformersTokenizer: TypeAlias = PreTrainedTokenizer | PreTrainedTokenizerFast

ModifiedModel: TypeAlias = transformers.PreTrainedModel | peft.peft_model.PeftModel
