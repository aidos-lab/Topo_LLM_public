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


from typing import TypeAlias

import peft.peft_model
import transformers
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from topollm.model_inference.perplexity.saving.sentence_perplexity_container import SentencePerplexityContainer

PerplexityResultsList: TypeAlias = list[tuple[int, SentencePerplexityContainer]]

TransformersTokenizer: TypeAlias = PreTrainedTokenizer | PreTrainedTokenizerFast

ModifiedModel: TypeAlias = transformers.PreTrainedModel | peft.peft_model.PeftModel
